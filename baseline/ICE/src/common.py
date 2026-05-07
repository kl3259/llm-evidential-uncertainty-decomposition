import os
import sys
import json
import string
import re
import random
import time
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None
import openai

_azure_client = None
_azure_client_cfg = None
_DEFAULT_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "120"))
_DEFAULT_MAX_RETRY_ATTEMPTS = int(os.getenv("OPENAI_MAX_RETRY_ATTEMPTS", "6"))
_DEFAULT_RETRY_INITIAL_WAIT_SEC = float(os.getenv("OPENAI_RETRY_INITIAL_WAIT_SEC", "5"))
_DEFAULT_RETRY_MAX_WAIT_SEC = float(os.getenv("OPENAI_RETRY_MAX_WAIT_SEC", "60"))
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "azure_openai.json")
_CONTENT_FILTER_SKIP_MODE = "skip_question"


class ContentFilterSkipError(RuntimeError):
    def __init__(self, metadata):
        self.metadata = dict(metadata or {})
        message = self.metadata.get("message") or "Request blocked by content filter."
        super().__init__(message)


def _safe_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_response_json(response):
    if response is None:
        return None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            return json_method()
        except Exception:
            pass
    text = getattr(response, "text", None)
    if isinstance(text, str):
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


def _iter_error_payloads(exc):
    candidates = []
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        candidates.append(body)
    response_json = _safe_response_json(getattr(exc, "response", None))
    if isinstance(response_json, dict):
        candidates.append(response_json)
    error_attr = getattr(exc, "error", None)
    if isinstance(error_attr, dict):
        candidates.append(error_attr)
    for payload in candidates:
        yield payload
        nested_error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(nested_error, dict):
            yield nested_error


def _normalize_content_filter_metadata(exc):
    message = str(exc)
    best_match = None
    for payload in _iter_error_payloads(exc):
        if not isinstance(payload, dict):
            continue
        code = payload.get("code")
        innererror = payload.get("innererror")
        if not isinstance(innererror, dict):
            innererror = payload.get("innerError")
        inner_code = innererror.get("code") if isinstance(innererror, dict) else None
        payload_message = payload.get("message") or message
        content_filter_result = None
        if isinstance(innererror, dict):
            content_filter_result = innererror.get("content_filter_result")
            if content_filter_result is None:
                content_filter_result = innererror.get("contentFilterResult")
        if code == "content_filter" or inner_code == "ResponsibleAIPolicyViolation":
            best_match = {
                "reason": "content_filter",
                "error_code": code or "content_filter",
                "inner_error_code": inner_code,
                "message": payload_message,
                "content_filter_result": _safe_jsonable(content_filter_result)
                if content_filter_result is not None
                else None,
            }
            break

    if best_match is not None:
        return best_match

    lowered = message.lower()
    if "content filter" in lowered or "content management policy" in lowered:
        return {
            "reason": "content_filter",
            "error_code": "content_filter",
            "inner_error_code": None,
            "message": message,
            "content_filter_result": None,
        }
    return None


def _coerce_positive_float(value, default, key_name: str):
    if value is None:
        return float(default)
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{key_name}` must be a positive number.") from exc
    if coerced <= 0:
        raise ValueError(f"`{key_name}` must be > 0.")
    return coerced


def _coerce_positive_int(value, default, key_name: str):
    if value is None:
        return int(default)
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{key_name}` must be a positive integer.") from exc
    if coerced < 1:
        raise ValueError(f"`{key_name}` must be >= 1.")
    return coerced


def _get_status_code(exc):
    for attr_name in ("status_code", "status"):
        raw_value = getattr(exc, attr_name, None)
        if raw_value is not None:
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                pass
    response = getattr(exc, "response", None)
    if response is None:
        return None
    for attr_name in ("status_code", "status"):
        raw_value = getattr(response, attr_name, None)
        if raw_value is not None:
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                pass
    return None


def _is_retryable_status_code(status_code):
    return status_code in {408, 409, 429} or (status_code is not None and 500 <= status_code <= 599)


def _legacy_timeout_exception_type():
    if hasattr(openai, "error") and hasattr(openai.error, "Timeout"):
        return openai.error.Timeout
    if hasattr(openai, "Timeout"):
        timeout_type = getattr(openai, "Timeout")
        if isinstance(timeout_type, type):
            return timeout_type
    return None


def _transient_error_reason(exc):
    timeout_type = _legacy_timeout_exception_type()
    if timeout_type is not None and isinstance(exc, timeout_type):
        return "timeout"

    type_reason_pairs = (
        ("APITimeoutError", "timeout"),
        ("APIConnectionError", "connection error"),
        ("RateLimitError", "rate limit"),
        ("InternalServerError", "server error"),
    )
    for type_name, reason in type_reason_pairs:
        exc_type = getattr(openai, type_name, None)
        if isinstance(exc_type, type) and isinstance(exc, exc_type):
            return reason

    api_status_error_type = getattr(openai, "APIStatusError", None)
    if isinstance(api_status_error_type, type) and isinstance(exc, api_status_error_type):
        status_code = _get_status_code(exc)
        if _is_retryable_status_code(status_code):
            return f"http {status_code}"
    return None


def _should_retry_exception(exc):
    return (
        not isinstance(exc, ContentFilterSkipError)
        and _normalize_content_filter_metadata(exc) is None
        and _transient_error_reason(exc) is not None
    )


def _load_azure_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    api_key = cfg.get("api_key")
    endpoint = cfg.get("azure_endpoint")
    api_version = cfg.get("api_version", "2025-04-01-preview")
    deployment_name = cfg.get("deployment_name")
    request_timeout = _coerce_positive_float(
        cfg.get("request_timeout"),
        _DEFAULT_REQUEST_TIMEOUT,
        "request_timeout",
    )
    max_retry_attempts = _coerce_positive_int(
        cfg.get("max_retry_attempts"),
        _DEFAULT_MAX_RETRY_ATTEMPTS,
        "max_retry_attempts",
    )
    retry_initial_wait_sec = _coerce_positive_float(
        cfg.get("retry_initial_wait_sec"),
        _DEFAULT_RETRY_INITIAL_WAIT_SEC,
        "retry_initial_wait_sec",
    )
    retry_max_wait_sec = _coerce_positive_float(
        cfg.get("retry_max_wait_sec"),
        _DEFAULT_RETRY_MAX_WAIT_SEC,
        "retry_max_wait_sec",
    )
    if not api_key or not endpoint:
        raise ValueError("azure_config must provide api_key and azure_endpoint.")
    return {
        "api_key": api_key,
        "endpoint": endpoint,
        "api_version": api_version,
        "deployment_name": deployment_name,
        "request_timeout": request_timeout,
        "max_retry_attempts": max_retry_attempts,
        "retry_initial_wait_sec": retry_initial_wait_sec,
        "retry_max_wait_sec": retry_max_wait_sec,
    }


def _resolve_azure_config(config_path=None):
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    if config_path and os.path.exists(config_path):
        return _load_azure_config(config_path)

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not api_key or not endpoint:
        raise ValueError(
            "Azure OpenAI credentials not provided. "
            "Provide a config file or set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
        )
    return {
        "api_key": api_key,
        "endpoint": endpoint,
        "api_version": api_version,
        "deployment_name": deployment_name,
        "request_timeout": _DEFAULT_REQUEST_TIMEOUT,
        "max_retry_attempts": _DEFAULT_MAX_RETRY_ATTEMPTS,
        "retry_initial_wait_sec": _DEFAULT_RETRY_INITIAL_WAIT_SEC,
        "retry_max_wait_sec": _DEFAULT_RETRY_MAX_WAIT_SEC,
    }


def _get_azure_client(config_path=None):
    global _azure_client, _azure_client_cfg
    cfg = _resolve_azure_config(config_path)
    cfg_fingerprint = (
        cfg["api_key"],
        cfg["endpoint"],
        cfg["api_version"],
        cfg.get("deployment_name"),
    )
    if _azure_client is not None and _azure_client_cfg == cfg_fingerprint:
        return _azure_client, cfg
    if AzureOpenAI is None:
        openai.api_type = "azure"
        openai.api_key = cfg["api_key"]
        openai.api_base = cfg["endpoint"]
        openai.api_version = cfg["api_version"]
        _azure_client = None
        _azure_client_cfg = cfg_fingerprint
        return None, cfg
    _azure_client = AzureOpenAI(
        api_key=cfg["api_key"],
        azure_endpoint=cfg["endpoint"],
        api_version=cfg["api_version"],
    )
    _azure_client_cfg = cfg_fingerprint
    return _azure_client, cfg

def _compute_retry_delay(attempt_idx: int, initial_wait: float, max_wait: float):
    uncapped = initial_wait * (2 ** max(0, attempt_idx - 1))
    capped = min(max_wait, uncapped)
    jittered = capped * random.uniform(0.5, 1.5)
    return min(max_wait, jittered)


def _call_openai_completion(client, kwargs):
    if client is None:
        return openai.ChatCompletion.create(**kwargs)
    resp = client.chat.completions.create(**kwargs)
    return resp.model_dump() if hasattr(resp, "model_dump") else resp


def completion_with_backoff(**kwargs):
    config_path = kwargs.pop("config_path", None)
    content_filter_mode = kwargs.pop("content_filter_mode", None)
    if content_filter_mode not in {None, _CONTENT_FILTER_SKIP_MODE}:
        raise ValueError(
            f"Unsupported content_filter_mode: {content_filter_mode}. "
            f"Expected `{_CONTENT_FILTER_SKIP_MODE}` or None."
        )
    client, cfg = _get_azure_client(config_path=config_path)
    deployment_name = cfg.get("deployment_name")
    if deployment_name:
        if client is None:
            kwargs.pop("model", None)
            kwargs["engine"] = deployment_name
        else:
            kwargs["model"] = deployment_name
    else:
        kwargs.setdefault("model", "gpt-5-mini-2")
    kwargs.setdefault("top_p", 1)
    if client is None:
        kwargs.setdefault("request_timeout", cfg["request_timeout"])
    else:
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        kwargs.setdefault("timeout", cfg["request_timeout"])

    max_retry_attempts = cfg["max_retry_attempts"]
    retry_initial_wait_sec = cfg["retry_initial_wait_sec"]
    retry_max_wait_sec = cfg["retry_max_wait_sec"]

    for attempt_idx in range(1, max_retry_attempts + 1):
        try:
            return _call_openai_completion(client, kwargs)
        except Exception as e:
            content_filter_metadata = _normalize_content_filter_metadata(e)
            if content_filter_metadata is not None and content_filter_mode == _CONTENT_FILTER_SKIP_MODE:
                raise ContentFilterSkipError(content_filter_metadata) from e

            transient_reason = _transient_error_reason(e)
            if transient_reason is None or attempt_idx >= max_retry_attempts:
                raise

            delay = _compute_retry_delay(
                attempt_idx=attempt_idx,
                initial_wait=retry_initial_wait_sec,
                max_wait=retry_max_wait_sec,
            )
            print(
                f"[Warning] OpenAI request hit {transient_reason} "
                f"(attempt {attempt_idx}/{max_retry_attempts}); retrying in {delay:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError("completion_with_backoff exhausted retries without raising the original error.")


def majority_vote(answers):
    ans2freq = {}
    max_freq = 0
    max_ans = None
    for ans in answers:
        if ans not in ans2freq:
            ans2freq[ans] = 1
        else:
            ans2freq[ans] += 1
        if ans2freq[ans] > max_freq:
            max_ans = ans
            max_freq = ans2freq[ans]
    return max_ans, max_freq

def remove_punctuation(object):
    translator = str.maketrans('', '', string.punctuation)
    if type(object) == list:
        new_obj = []
        for item in object:
            new_obj.append(remove_punctuation(item))
        return new_obj
    elif type(object) == str:
        new_obj = object.strip().translate(translator).lower()
        return new_obj
    else:
        raise NotImplementedError


unk_word_pool = ['unknown', "I don't", "did not", "Not specified", "cannot be determined", "No Answer", "No final answer","I do not", "N/A", "No information", "It depends on", "I cannot","I can't ", "I am unable","don't know", "No answer", "Nobody", "enough information", "specific information", "There is no", "No specific","not provided", "None", "No character", "did not", "No output","Cannot answer", "Unavailable", "TBD", "To Be Determined", "I am asking about", "current year", "No translation", "depends on", "has not", 'unclear', "confusion", "incorrect", "not aware of", "invalid",'no one']

def check_answers(ans_list):
    unk_words = [x.lower() for x in unk_word_pool]
    purified_ans = []
    num_unk = 0
    for ans in ans_list:
        flag = False
        if ans.lower() in unk_words:
            flag = True
        for unk_w in unk_word_pool:
            if unk_w.lower() in ans.lower():
                flag = True
                break
        if flag:
            num_unk += 1
            purified_ans.append('unknown')
        else:
            purified_ans.append(ans)

    return purified_ans

def gsm8k_extract_ans(pred_str):
    ANS_RE = re.compile(r"\$?([0-9,]+)\.?\d*%?")
    pred = re.findall(ANS_RE, pred_str)
    if(len(pred) >= 1):
        pred = pred[-1]
        pred = pred.replace(",", "").replace(" ", "")
        try:
            return int(pred)
        except:
            return -1
    else:
        return -1

def ambiginst_extract_ans(model_ans):
    if "answer is" in model_ans:
        ext = model_ans.split('answer is')[-1].strip()
        return ext
    return model_ans
