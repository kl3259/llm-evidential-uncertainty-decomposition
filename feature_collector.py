'''
For generating and collecting hidden_states for TBG and SLT token positions.
'''

import os, re, string, pickle, sys, json, shutil, random, time, importlib.util
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Optional, Union
from distutils.util import strtobool as _bool
from pathlib import Path
import functools
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import torch, numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb, argparse

from openai import AzureOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, RateLimitError
from uncertainty import dissonance
from config import CollectConfig
from helper_functions import _save_preds_jsonl, _save_X_npz, StoppingCriteriaSub, \
    _save_questions_jsonl, _save_refs_jsonl, count_or_frac, load_X_npz, \
    load_pred_json, load_question_json, load_ref_json, load_resp_json, \
    load_cluster_embeddings_npz, save_cluster_embeddings_npz, \
    save_sequence_scores_npz, load_sequence_scores_npz, \
    get_sampled_sequence_scores_filename
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

import logging
for name in ("transformers", "openai", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)

WORKING_DIR = "./"

CASE_FULL_QIDS = None
CASE_SPLIT_QIDS = None

FIXED_SPLIT_SEEDS = {
    "fewshot_membership": 131071,
    "test_membership": 131101,
    "valid_membership": 131111,
    "train_membership": 131131,
    "train_order": 131149,
    "valid_order": 131173,
    "test_order": 131183,
}

OOD_TRIVIA_DATASET_NAMES = {"alcuna_trivia_qa", "repli_qa_trivia_qa", "temporalood_trivia_qa"}
AMBIG_QA_DATASET_NAME = "ambig_qa"
AMBIG_QA_DISAMBIG_DATASET_NAME = "ambig_qa_disambig"
AMBIG_QA_DISAMBIG_FEWSHOT_COUNT = 5
AMBIG_QA_FILTERED_DATASET_NAME = "ambig_qa_filtered"
AMBIG_QA_FILTERED_VER2_DATASET_NAME = "ambig_qa_filtered_ver2"
AMBIG_QA_FILTERED_VER3_DATASET_NAME = "ambig_qa_filtered_ver3"
SITUATED_QA_DATASET_NAME = "situated_qa"
SITUATED_QA_DISAMBIG_DATASET_NAME = "situated_qa_disambig"
SITUATED_QA_DATA_PATH = os.path.join(
    WORKING_DIR,
    "datasets",
    "SituatedQA",
    "data",
    "situated_qa.jsonl",
)
AMBIG_QA_FILTERED_VER3_FILTER_THRESHOLD = 0.5
AMBIG_QA_FILTERED_VER3_RESPONSE_OVERLAP_THRESHOLD = 0.30
OPENBOOK_QA_AMBIGUITY_THRESHOLD = 1.2
GPT5_MINI_AZURE_MODEL = "gpt-5-mini"
OPENAI_API_CONFIG_PATH = Path(__file__).resolve().with_name("openai_api.json")
AMBIG_QA_FILTERED_JUDGE_BATCH_SIZE = 64
DEFAULT_GPT5_MINI_JUDGE_WORKERS = 2
LLAMA_30_8B_SUFFIX = "llama-3-8b-instruct"
LLAMA_31_8B_SUFFIX = "llama-3.1-8b-instruct"
LLAMA_32_3B_SUFFIX = "llama-3.2-3b-instruct"
LLAMA_33_70B_SUFFIX = "llama-3.3-70b-instruct"
MISTRAL_SMALL_24B_INSTRUCT_ALIAS = "mistral-small-24b-instruct-2501"
MISTRAL_SMALL_24B_INSTRUCT_REPO = "mistralai/Mistral-Small-24B-Instruct-2501"
GEMMA_4_E4B_IT_ALIAS = "gemma-4-e4b-it"
GEMMA_4_E4B_IT_REPO = "google/gemma-4-E4B-it"
QWEN3_30B_A3B_ALIAS = "qwen3-30b-a3b"
QWEN3_30B_A3B_REPO = "Qwen/Qwen3-30B-A3B"
QWEN3_30B_A3B_INSTRUCT_2507_ALIAS = "qwen3-30b-a3b-instruct-2507"
QWEN3_30B_A3B_INSTRUCT_2507_REPO = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MISTRAL_SMALL_24B_SYSTEM_PROMPT = (
    "You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, "
    "a French startup headquartered in Paris.\n"
    "Your knowledge base was last updated on 2023-10-01. "
    "You are a helpful assistant that answer questions truthfully. "
)
_GPT5_MINI_CLIENT_TLS = threading.local()


def _split_seed(base_seed: int, fix_val_test_splits: bool, stage: str) -> int:
    if fix_val_test_splits:
        return int(FIXED_SPLIT_SEEDS[stage])
    return int(base_seed)

def remove_dup_questions(dataset):
    """Remove duplicate entries based on `question` field."""

    if isinstance(dataset, DatasetDict):
        deduped_splits = {}
        removed_total = 0
        for split, ds in dataset.items():
            deduped_ds, removed = remove_dup_questions(ds)
            deduped_splits[split] = deduped_ds
            removed_total += removed
        return DatasetDict(deduped_splits), removed_total

    if not isinstance(dataset, Dataset):
        return dataset, 0

    seen_questions = set()

    def _is_unique(example):
        question = example.get("question")
        if question in seen_questions:
            return False
        seen_questions.add(question)
        return True

    filtered = dataset.filter(_is_unique)
    removed = len(dataset) - len(filtered)

    return filtered, removed


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_case_debug_artifacts(dataset_name, output_dir, proc_datasets):
    if CASE_SPLIT_QIDS is None or CASE_FULL_QIDS is None:
        return

    case_dir = Path(WORKING_DIR) / "debug_cases" / dataset_name
    case_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset_name": dataset_name,
        "full_qids": [int(q) for q in CASE_FULL_QIDS],
        "split_qids": {split: [int(q) for q in qids] for split, qids in CASE_SPLIT_QIDS.items()},
        "feature_source_dir": str(output_dir),
    }
    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    split_dirnames = {"train": "train", "validation": "valid", "test": "test"}

    for split_name, disk_name in split_dirnames.items():
        ds = proc_datasets.get(split_name)
        if ds is None:
            continue

        qids = CASE_SPLIT_QIDS.get(split_name, [])
        questions = ds["question"] if "question" in ds.column_names else []
        references = ds["references"] if "references" in ds.column_names else []

        rows = []
        for idx, question in enumerate(questions):
            entry = {
                "case_qid": int(qids[idx]) if idx < len(qids) else None,
                "question": question,
                "references": references[idx] if idx < len(references) else [],
            }
            rows.append(entry)

        if rows:
            _write_jsonl(case_dir / f"{split_name}_cases.jsonl", rows)

        if questions:
            _save_questions_jsonl(str(case_dir / f"{split_name}_questions.jsonl"), questions)

        if references:
            _save_refs_jsonl(str(case_dir / f"{split_name}_references.jsonl"), references)

        split_dir = Path(output_dir) / disk_name
        if not split_dir.exists():
            continue

        feature_path = split_dir / "X.npz"
        if feature_path.exists():
            shutil.copy2(feature_path, case_dir / f"{split_name}_X.npz")

        se_resp_path = split_dir / "se_resp.jsonl"
        if se_resp_path.exists():
            shutil.copy2(se_resp_path, case_dir / f"{split_name}_se_resp.jsonl")

        for fname in ["questions.jsonl", "references.jsonl"]:
            src = split_dir / fname
            if src.exists():
                shutil.copy2(src, case_dir / f"{split_name}_{fname}")


def _is_gemma_model(model) -> bool:
    model_name = str(getattr(model, "name_or_path", "")).lower()
    return "gemma" in model_name


def _normalize_model_name(model_name: str) -> str:
    return str(model_name or "").strip().lower().rstrip("/")


def _is_mistral_small_24b_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    if not norm:
        return False
    return norm.split("/")[-1] == MISTRAL_SMALL_24B_INSTRUCT_ALIAS


def _is_gemma_4_e4b_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    if not norm:
        return False
    return GEMMA_4_E4B_IT_ALIAS in norm


def _resolve_hf_model_name(model_name: str) -> str:
    text = str(model_name or "").strip()
    basename = _normalize_model_name(text).split("/")[-1]
    if _is_mistral_small_24b_model_name(text):
        return MISTRAL_SMALL_24B_INSTRUCT_REPO
    if _is_gemma_4_e4b_model_name(text):
        return GEMMA_4_E4B_IT_REPO
    if basename == QWEN3_30B_A3B_INSTRUCT_2507_ALIAS:
        return QWEN3_30B_A3B_INSTRUCT_2507_REPO
    if basename == QWEN3_30B_A3B_ALIAS:
        return QWEN3_30B_A3B_REPO
    return text


def _is_gemma_chat_template_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    return "gemma-3" in norm or "gemma-4-e4b-it" in norm


def _instruction_tuned_system_prompt(model_name: str) -> Optional[str]:
    norm = _normalize_model_name(model_name)
    if not norm:
        return None
    if _is_mistral_small_24b_model_name(norm):
        return MISTRAL_SMALL_24B_SYSTEM_PROMPT
    basename = norm.split("/")[-1]
    if "instruct" in basename or basename.endswith("it"):
        return "You are a helpful assistant that answers question truthfully. "
    return None


def _normalize_dataset_name(dataset_name: str) -> str:
    return str(dataset_name or "").strip()


def _is_situated_qa_family_dataset(dataset_name: str) -> bool:
    norm = _normalize_dataset_name(dataset_name).lower()
    return norm in {SITUATED_QA_DATASET_NAME, SITUATED_QA_DISAMBIG_DATASET_NAME}


def _feature_output_dir(model_id: str, dataset_name: str, n_samples: int, judge_type: str) -> str:
    return os.path.join(
        WORKING_DIR,
        "input_hidden_states",
        f"{model_id}_{_normalize_dataset_name(dataset_name)}_{int(n_samples)}_{judge_type}",
    )


def _load_openai_api_config() -> Dict[str, str]:
    with open(OPENAI_API_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    required_keys = ("endpoint", "key", "api_version")
    missing = [key for key in required_keys if not config.get(key)]
    if missing:
        raise ValueError(
            f"Missing required OpenAI API config fields in {OPENAI_API_CONFIG_PATH}: {missing}"
        )
    return {key: str(config[key]) for key in required_keys}


def _create_gpt5_mini_client() -> AzureOpenAI:
    api_config = _load_openai_api_config()
    return AzureOpenAI(
        api_key=api_config["key"],
        azure_endpoint=api_config["endpoint"],
        api_version=api_config["api_version"],
    )


def _get_thread_local_gpt5_mini_client() -> AzureOpenAI:
    client = getattr(_GPT5_MINI_CLIENT_TLS, "client", None)
    if client is None:
        client = _create_gpt5_mini_client()
        _GPT5_MINI_CLIENT_TLS.client = client
    return client


def _get_retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None

    retry_after = None
    retry_after_ms = None
    try:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        retry_after_ms = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
    except AttributeError:
        retry_after = None
        retry_after_ms = None
    if retry_after is None and retry_after_ms is not None:
        try:
            value_ms = float(retry_after_ms)
        except (TypeError, ValueError):
            return None
        if value_ms < 0:
            return None
        return value_ms / 1000.0
    if retry_after is None:
        return None
    try:
        value = float(retry_after)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _is_retryable_gpt5_mini_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        return status_code in {408, 409, 429} or (
            isinstance(status_code, int) and status_code >= 500
        )
    return False


def _call_gpt5_mini_judge_with_retries(
    messages,
    *,
    purpose: str,
    sample_idx: Optional[int] = None,
    max_attempts: int = 6,
):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = _get_thread_local_gpt5_mini_client()
            return client.chat.completions.create(
                model=GPT5_MINI_AZURE_MODEL,
                messages=messages,
                top_p=1,
            )
        except Exception as exc:
            if not _is_retryable_gpt5_mini_error(exc):
                raise
            last_exc = exc
            if attempt >= max_attempts:
                break

            retry_after = _get_retry_after_seconds(exc)
            if retry_after is None:
                retry_after = min(60.0, 2.0 ** attempt)
            sleep_s = float(retry_after) + random.uniform(0.0, 0.5)
            print(
                "[gpt5-mini judge] retryable error "
                f"purpose={purpose} sample_idx={sample_idx} "
                f"attempt={attempt}/{max_attempts} "
                f"error={type(exc).__name__}; sleeping {sleep_s:.1f}s",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        "GPT-5-mini judge failed after retries "
        f"model={GPT5_MINI_AZURE_MODEL} purpose={purpose} sample_idx={sample_idx} "
        f"attempts={max_attempts} last_error={type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def _parse_true_false_response(text) -> Optional[bool]:
    normalized = str(text or "").strip().lower()
    if normalized.startswith("true"):
        return True
    if normalized.startswith("false"):
        return False
    return None


def _is_content_filter_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "content_filter" in msg
        or "ResponsibleAIPolicyViolation" in msg
        or "content management policy" in msg
    )


def _judge_feature_collection_single(chat, sample_idx: int) -> Tuple[float, bool]:
    try:
        resp = _call_gpt5_mini_judge_with_retries(
            chat,
            purpose="feature_collection",
            sample_idx=sample_idx,
        )
    except BadRequestError as exc:
        if _is_content_filter_error(exc):
            return 0.0, True
        raise

    parsed = _parse_true_false_response(resp.choices[0].message.content)
    if parsed is True:
        return 1.0, False
    return 0.0, False


def _choose_unique_file(split_dir: Path, pattern: str, description: str) -> Path:
    matches = sorted(split_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {description} in {split_dir} (pattern: {pattern}).")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Expected exactly one {description} in {split_dir}, found {len(matches)}: {names}"
        )
    return matches[0]


def _choose_sampled_response_file(split_dir: Path) -> Path:
    for name in ("se_resp_res.jsonl", "se_resp.jsonl"):
        path = split_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing sampled response jsonl in {split_dir}.")


def _build_cluster_display_text(semantic_ids_row, sampled_responses_row) -> str:
    ids_row = np.asarray(semantic_ids_row, dtype=np.int64).reshape(-1)
    if sampled_responses_row is None:
        responses_row = []
    elif isinstance(sampled_responses_row, list):
        responses_row = sampled_responses_row
    else:
        responses_row = list(sampled_responses_row)

    limit = min(len(ids_row), len(responses_row))
    if limit == 0:
        return "  <no sampled responses>"

    clusters: Dict[int, List[str]] = {}
    for sid, resp in zip(ids_row[:limit], responses_row[:limit]):
        sid_int = int(sid)
        if sid_int not in clusters:
            clusters[sid_int] = []
        clusters[sid_int].append(str(resp))

    if not clusters:
        return "  <no sampled responses>"

    lines: List[str] = []
    for display_sid, sid in enumerate(sorted(clusters), start=1):
        resp_list = clusters[sid]
        lines.append(f"  Cluster {display_sid} ({len(resp_list)} responses):")
        for j, resp in enumerate(resp_list, start=1):
            lines.append(f"    [{j}] {resp}")
    return "\n".join(lines)


def _format_references_for_prompt(references) -> str:
    if isinstance(references, list):
        return json.dumps([str(ref) for ref in references], ensure_ascii=False)
    return json.dumps(str(references), ensure_ascii=False)


def _filter_X_by_index(X: Dict[str, np.ndarray], idx_arr: np.ndarray, expected_size: int) -> Dict[str, np.ndarray]:
    filtered: Dict[str, np.ndarray] = {}
    for name, arr in X.items():
        if arr.ndim < 2:
            raise ValueError(f"Feature tensor `{name}` must be at least 2D, got shape {arr.shape}.")
        if arr.shape[1] != expected_size:
            raise ValueError(
                f"Feature tensor `{name}` has mismatched sample axis: expected {expected_size}, got {arr.shape[1]}."
            )
        filtered[name] = arr[:, idx_arr, ...].astype(np.float32, copy=False)
    return filtered


def _save_sampled_responses_jsonl(path: str, responses_per_question) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for responses in responses_per_question:
            f.write(json.dumps({"responses": responses}, ensure_ascii=False) + "\n")


def _build_gt_cluster_evidence_dicts(semantic_ids) -> List[Dict[int, float]]:
    evidence_all: List[Dict[int, float]] = []
    for row in semantic_ids:
        row_arr = np.asarray(row, dtype=np.int64).reshape(-1)
        if row_arr.size == 0:
            evidence_all.append({})
            continue
        counts = Counter(row_arr.tolist())
        evidence_all.append({int(cid): float(cnt) for cid, cnt in counts.items()})
    return evidence_all


def _compute_gt_dissonance(semantic_ids) -> np.ndarray:
    evidence_all = _build_gt_cluster_evidence_dicts(semantic_ids)
    return np.asarray([dissonance(e_dict) for e_dict in evidence_all], dtype=np.float32)


def _load_ambig_filtered_source_split(split_dir: Path) -> Dict[str, object]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    semantic_ids_path = _choose_unique_file(split_dir, "semantic_ids_*.npy", "semantic_ids file")
    semantic_ids_name = semantic_ids_path.name
    n_resp_suffix = semantic_ids_name[len("semantic_ids_"):-len(".npy")]
    se_resp_path = _choose_sampled_response_file(split_dir)

    response_embeddings_path = split_dir / f"response_embeddings_{n_resp_suffix}.npz"
    cluster_embeddings_path = split_dir / f"cluster_embeddings_{n_resp_suffix}.npz"
    response_slt_lastk_path = split_dir / f"response_slt_lastk_{n_resp_suffix}.npz"
    pred_sequence_scores_path = split_dir / "preds_sequence_scores.npz"
    sampled_response_sequence_scores_name = get_sampled_sequence_scores_filename(
        se_resp_path.name,
        n_resp_suffix,
    )
    sampled_response_sequence_scores_path = split_dir / sampled_response_sequence_scores_name

    X = load_X_npz(str(split_dir / "X.npz"))
    y = np.asarray(np.load(split_dir / "y.npy"), dtype=np.int64).reshape(-1)
    acc = np.asarray(np.load(split_dir / "acc.npy"), dtype=np.float32).reshape(-1)
    f1 = np.asarray(np.load(split_dir / "f1.npy"), dtype=np.float32).reshape(-1)
    ambig_label = np.asarray(np.load(split_dir / "ambig_label.npy"), dtype=np.int64).reshape(-1)
    preds = load_pred_json(str(split_dir / "preds.jsonl"))
    questions = load_question_json(str(split_dir / "questions.jsonl"))
    references = load_ref_json(str(split_dir / "references.jsonl"))
    se_scores = np.asarray(np.load(split_dir / "se.npy"), dtype=np.float32).reshape(-1)
    semantic_ids = np.load(semantic_ids_path, allow_pickle=True)
    sampled_responses = load_resp_json(str(se_resp_path))
    pred_sequence_scores = (
        load_sequence_scores_npz(str(pred_sequence_scores_path))
        if pred_sequence_scores_path.exists()
        else None
    )
    sampled_response_sequence_scores = (
        load_sequence_scores_npz(str(sampled_response_sequence_scores_path))
        if sampled_response_sequence_scores_path.exists()
        else None
    )

    response_embeddings = (
        load_cluster_embeddings_npz(str(response_embeddings_path))
        if response_embeddings_path.exists()
        else None
    )
    cluster_embeddings = (
        load_cluster_embeddings_npz(str(cluster_embeddings_path))
        if cluster_embeddings_path.exists()
        else None
    )
    response_slt_lastk = (
        load_cluster_embeddings_npz(str(response_slt_lastk_path))
        if response_slt_lastk_path.exists()
        else None
    )

    if response_embeddings is None and cluster_embeddings is None:
        raise FileNotFoundError(
            "Missing semantic embedding NPZ for filtered ambig_qa derivation in "
            f"{split_dir}. Expected either {response_embeddings_path.name} or {cluster_embeddings_path.name}."
        )

    expected_size = int(y.shape[0])
    lengths = {
        "y": expected_size,
        "acc": int(acc.shape[0]),
        "f1": int(f1.shape[0]),
        "ambig_label": int(ambig_label.shape[0]),
        "preds": len(preds),
        "questions": len(questions),
        "references": len(references),
        "se.npy": int(se_scores.shape[0]),
        "semantic_ids": len(semantic_ids),
        "sampled_responses": len(sampled_responses),
    }
    for name, arr in X.items():
        lengths[f"X.{name}"] = int(arr.shape[1])
    if response_embeddings is not None:
        lengths["response_embeddings"] = len(response_embeddings)
    if cluster_embeddings is not None:
        lengths["cluster_embeddings"] = len(cluster_embeddings)
    if response_slt_lastk is not None:
        lengths["response_slt_lastk"] = len(response_slt_lastk)
    if pred_sequence_scores is not None:
        lengths["pred_sequence_scores"] = int(pred_sequence_scores["N"])
    if sampled_response_sequence_scores is not None:
        lengths["sampled_response_sequence_scores"] = int(sampled_response_sequence_scores["N"])

    mismatched = {name: size for name, size in lengths.items() if size != expected_size}
    if mismatched:
        raise ValueError(
            f"Artifact length mismatch in {split_dir}: expected {expected_size}, mismatched={mismatched}"
        )
    if pred_sequence_scores is not None and pred_sequence_scores["sequence_logprob"].ndim != 1:
        raise ValueError(
            f"Primary sequence score NPZ in {pred_sequence_scores_path} must be rank-1."
        )
    if sampled_response_sequence_scores is not None:
        expected_n_responses = int(n_resp_suffix)
        if sampled_response_sequence_scores["sequence_logprob"].ndim != 2:
            raise ValueError(
                f"Sampled response sequence score NPZ in {sampled_response_sequence_scores_path} must be rank-2."
            )
        if int(sampled_response_sequence_scores["n_responses"]) != expected_n_responses:
            raise ValueError(
                "Sampled response sequence score NPZ response-count mismatch in "
                f"{sampled_response_sequence_scores_path}: "
                f"stored={sampled_response_sequence_scores['n_responses']}, expected={expected_n_responses}"
            )

    return {
        "n_total": expected_size,
        "n_resp_suffix": n_resp_suffix,
        "semantic_ids_name": semantic_ids_name,
        "se_resp_name": se_resp_path.name,
        "X": X,
        "y": y,
        "acc": acc,
        "f1": f1,
        "ambig_label": ambig_label,
        "preds": preds,
        "questions": questions,
        "references": references,
        "se_scores": se_scores,
        "semantic_ids": semantic_ids,
        "sampled_responses": sampled_responses,
        "pred_sequence_scores": pred_sequence_scores,
        "sampled_response_sequence_scores": sampled_response_sequence_scores,
        "sampled_response_sequence_scores_name": sampled_response_sequence_scores_name,
        "response_embeddings": response_embeddings,
        "cluster_embeddings": cluster_embeddings,
        "response_slt_lastk": response_slt_lastk,
    }


def _ambig_qa_filtered_prompt(
    question: str,
    answer: str,
    references,
    semantic_ids_row,
    sampled_responses_row,
) -> str:
    # cluster_display = _build_cluster_display_text(semantic_ids_row, sampled_responses_row)
    responses_display = "\n".join(sampled_responses_row)
    return (
        # "We are assessing whether a question is ambiguous. "
        # "Given the question, a primary response, i.i.d. sampled responses "
        # "(already clustered into semantic clusters), and the ground truth response, determine "
        # "whether the uncertainty in the responses is caused by ambiguity, answer with true "
        # "or false only.\n\n"
        # f"Question: {question}\n"
        # f"Primary response: {answer}\n"
        # f"References: {_format_references_for_prompt(references)}\n"
        # f"Sampled responses grouped by semantic clustering:\n{cluster_display}"
        "We are assessing whether a question is ambiguous. "
        "Given the question, a primary response, i.i.d. sampled responses, "
        "and the ground truth response, "
        "return true if there is a high uncertainty among the sampled responses and it is not caused by ambiguity in the question; otherwise, return false. "
        "Respond with true or false only. \n\n"
        f"Question: {question}\n"
        f"Primary response: {answer}\n"
        f"References: {_format_references_for_prompt(references)}\n"
        f"Sampled responses:\n{responses_display}"
    )


def _judge_ambig_qa_filtered_single(
    idx: int,
    question: str,
    answer: str,
    references,
    semantic_ids_row,
    sampled_responses_row,
):
    chat = [{
        "role": "user",
        "content": _ambig_qa_filtered_prompt(
            question=question,
            answer=answer,
            references=references,
            semantic_ids_row=semantic_ids_row,
            sampled_responses_row=sampled_responses_row,
        ),
    }]

    decision = False
    filtered_by_policy = False
    try:
        resp = _call_gpt5_mini_judge_with_retries(
            chat,
            purpose="ambig_qa_filtered",
            sample_idx=idx,
        )
        parsed = _parse_true_false_response(resp.choices[0].message.content)
        decision = bool(parsed) if parsed is not None else False
    except BadRequestError as exc:
        if _is_content_filter_error(exc):
            filtered_by_policy = True
            decision = False
        else:
            raise
    return idx, decision, filtered_by_policy


def _judge_ambig_qa_filtered_split(split_name: str, split_data: Dict[str, object]):

    y = np.asarray(split_data["y"], dtype=np.int64).reshape(-1)
    preds = split_data["preds"]
    questions = split_data["questions"]
    references = split_data["references"]
    semantic_ids = split_data["semantic_ids"]
    sampled_responses = split_data["sampled_responses"]
    candidate_mask = np.ones(y.shape[0], dtype=bool)

    keep_mask = np.ones(y.shape[0], dtype=bool)
    judge_true_mask = np.zeros(y.shape[0], dtype=bool)
    judged_mask = np.zeros(y.shape[0], dtype=bool)
    hallucination_idx = np.flatnonzero(y == 0)
    filtered_by_policy = 0

    idx_list = np.flatnonzero(candidate_mask).tolist()
    with tqdm(total=len(idx_list), desc=f"[ambig_qa_filtered] judge {split_name}") as pbar:
        with ThreadPoolExecutor(max_workers=AMBIG_QA_FILTERED_JUDGE_BATCH_SIZE) as executor:
            for start in range(0, len(idx_list), AMBIG_QA_FILTERED_JUDGE_BATCH_SIZE):
                batch_idx = idx_list[start:start + AMBIG_QA_FILTERED_JUDGE_BATCH_SIZE]
                futures = [
                    executor.submit(
                        _judge_ambig_qa_filtered_single,
                        int(idx),
                        questions[idx],
                        preds[idx],
                        references[idx],
                        semantic_ids[idx],
                        sampled_responses[idx],
                    )
                    for idx in batch_idx
                ]
                for future in as_completed(futures):
                    idx, decision, content_filtered = future.result()
                    judged_mask[idx] = True
                    if content_filtered:
                        filtered_by_policy += 1
                    keep_mask[idx] = not bool(decision)
                    if decision:
                        judge_true_mask[idx] = True
                    pbar.update(1)

    judge_true_rows = int(np.sum(judge_true_mask & judged_mask))
    judge_not_true_rows = int(np.sum((~judge_true_mask) & judged_mask))
    dropped_hallucinations_true = int(np.sum((y == 0) & (~keep_mask) & judge_true_mask))
    kept_hallucinations_not_true = int(
        np.sum((y == 0) & keep_mask & (~judge_true_mask) & judged_mask)
    )
    kept_hallucinations_unjudged = int(np.sum((y == 0) & keep_mask & (~judged_mask)))

    stats = {
        "source_total": int(y.shape[0]),
        "hallucination_candidates": int(hallucination_idx.shape[0]),
        "judged_candidates": int(np.sum(candidate_mask)),
        "judged_hallucination_candidates": int(np.sum(candidate_mask & (y == 0))),
        "judged_truthful_candidates": int(np.sum(candidate_mask & (y != 0))),
        "kept_non_hallucinations": int(np.sum((y != 0) & keep_mask.astype(bool))),
        "judge_true_rows": judge_true_rows,
        "judge_not_true_rows": judge_not_true_rows,
        "dropped_hallucinations_true": dropped_hallucinations_true,
        "kept_hallucinations_not_true": kept_hallucinations_not_true,
        "kept_hallucinations_unjudged": kept_hallucinations_unjudged,
        "judge_content_filter_kept": int(filtered_by_policy),
        "rows_submitted": int(len(idx_list)),
    }
    return keep_mask.astype(bool), stats


def _filter_ambig_qa_filtered_ver2_split(split_name: str, split_data: Dict[str, object]):
    y = np.asarray(split_data["y"], dtype=np.int64).reshape(-1)
    ambig_label = np.asarray(split_data["ambig_label"], dtype=np.int64).reshape(-1)
    gt_dissonance = _compute_gt_dissonance(split_data["semantic_ids"])

    finite_mask = np.isfinite(gt_dissonance)
    zero_dissonance_mask = finite_mask & np.isclose(gt_dissonance, 0.0)
    truthful_mask = y == 1
    annotated_ambig_mask = ambig_label == 1
    drop_mask = zero_dissonance_mask & truthful_mask & annotated_ambig_mask
    keep_mask = ~drop_mask

    stats = {
        "source_total": int(y.shape[0]),
        "finite_gt_dissonance_count": int(np.sum(finite_mask)),
        "gt_dissonance_zero_count": int(np.sum(zero_dissonance_mask)),
        "gt_dissonance_zero_truthful_count": int(np.sum(zero_dissonance_mask & truthful_mask)),
        "gt_dissonance_zero_ambig_count": int(np.sum(zero_dissonance_mask & annotated_ambig_mask)),
        "zero_dissonance_truthful_ambig_dropped": int(np.sum(drop_mask)),
    }
    return keep_mask.astype(bool), stats


@functools.lru_cache(maxsize=1)
def _get_ambig_qa_filtered_ver3_rouge():
    return load("rouge")


def _normalize_ambig_qa_filtered_ver3_references(reference_row) -> List[str]:
    if isinstance(reference_row, dict):
        reference_row = reference_row.get("ref", [])
    references = _dedupe_text_list(_flatten_nested_text_values(reference_row))
    return references


def _normalize_ambig_qa_filtered_ver3_responses(sampled_responses_row) -> List[str]:
    if sampled_responses_row is None:
        return []
    if isinstance(sampled_responses_row, (list, tuple, np.ndarray)):
        return ["" if response is None else str(response) for response in sampled_responses_row]
    return [str(sampled_responses_row)]


def _ambig_qa_filtered_ver3_max_rouge_l(
    prediction: str,
    references,
    rouge_metric,
) -> float:
    curr_refs = _normalize_ambig_qa_filtered_ver3_references(references)
    if not curr_refs:
        return 0.0

    max_score = 0.0
    for alias in curr_refs:
        rouge_scores = rouge_metric.compute(
            predictions=[prediction],
            references=[alias],
            use_aggregator=False,
        )["rougeL"]
        score = float(rouge_scores[0]) if len(rouge_scores) > 0 else 0.0
        if score > max_score:
            max_score = score
    return max_score


def _compute_ambig_qa_filtered_ver3_overlap_stats(
    split_name: str,
    split_data: Dict[str, object],
) -> Dict[str, np.ndarray]:
    rouge_metric = _get_ambig_qa_filtered_ver3_rouge()
    sampled_responses = split_data["sampled_responses"]
    references = split_data["references"]

    n_total = len(sampled_responses)
    overlap_ratio = np.zeros(n_total, dtype=np.float32)
    overlap_count = np.zeros(n_total, dtype=np.int64)
    n_sampled_responses = np.zeros(n_total, dtype=np.int64)

    for idx, (responses_row, reference_row) in enumerate(
        tqdm(
            zip(sampled_responses, references),
            total=n_total,
            desc=f"[{AMBIG_QA_FILTERED_VER3_DATASET_NAME}] overlap {split_name}",
        )
    ):
        curr_responses = _normalize_ambig_qa_filtered_ver3_responses(responses_row)
        n_resp = len(curr_responses)
        n_sampled_responses[idx] = int(n_resp)
        if n_resp == 0:
            continue

        max_scores = [
            _ambig_qa_filtered_ver3_max_rouge_l(response, reference_row, rouge_metric)
            for response in curr_responses
        ]
        n_overlap = int(
            sum(
                score >= AMBIG_QA_FILTERED_VER3_RESPONSE_OVERLAP_THRESHOLD
                for score in max_scores
            )
        )
        overlap_count[idx] = n_overlap
        overlap_ratio[idx] = float(n_overlap / n_resp)

    return {
        "overlap_ratio": overlap_ratio,
        "overlap_count": overlap_count,
        "n_sampled_responses": n_sampled_responses,
    }


def _ambig_qa_filtered_ver3_overlap_ratio_summary(
    overlap_ratio: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> Dict[str, Optional[float]]:
    values = np.asarray(overlap_ratio, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _filter_ambig_qa_filtered_ver3_split(split_name: str, split_data: Dict[str, object]):
    overlap_stats = _compute_ambig_qa_filtered_ver3_overlap_stats(split_name, split_data)
    overlap_ratio = np.asarray(overlap_stats["overlap_ratio"], dtype=np.float32).reshape(-1)
    n_sampled_responses = np.asarray(
        overlap_stats["n_sampled_responses"],
        dtype=np.int64,
    ).reshape(-1)
    ambig_label = np.asarray(split_data["ambig_label"], dtype=np.int64).reshape(-1)

    ambig_mask = ambig_label == 1
    clear_mask = ambig_label == 0
    keep_mask = clear_mask | (
        ambig_mask & (overlap_ratio >= float(AMBIG_QA_FILTERED_VER3_FILTER_THRESHOLD))
    )
    removed_mask = ~keep_mask
    removed_ambig_low_overlap_mask = removed_mask & ambig_mask

    stats = {
        "source_total": int(overlap_ratio.shape[0]),
        "filter_threshold": float(AMBIG_QA_FILTERED_VER3_FILTER_THRESHOLD),
        "response_overlap_threshold": float(AMBIG_QA_FILTERED_VER3_RESPONSE_OVERLAP_THRESHOLD),
        "dropped_ambig_overlap_ratio_lt_threshold": int(np.sum(removed_ambig_low_overlap_mask)),
        "zero_sampled_response_count": int(np.sum(n_sampled_responses == 0)),
        **_ambig_qa_filtered_ver3_overlap_ratio_summary(
            overlap_ratio,
            np.ones(overlap_ratio.shape[0], dtype=bool),
            "source_overlap_ratio",
        ),
        **_ambig_qa_filtered_ver3_overlap_ratio_summary(
            overlap_ratio,
            keep_mask,
            "kept_overlap_ratio",
        ),
        **_ambig_qa_filtered_ver3_overlap_ratio_summary(
            overlap_ratio,
            removed_mask,
            "removed_overlap_ratio",
        ),
    }
    return keep_mask.astype(bool), stats


def _summarize_ambig_row_groups(
    split_data: Dict[str, object],
    row_mask: np.ndarray,
) -> Dict[str, int]:
    y = np.asarray(split_data["y"], dtype=np.int64).reshape(-1)
    ambig_label = np.asarray(split_data["ambig_label"], dtype=np.int64).reshape(-1)
    row_mask = np.asarray(row_mask, dtype=bool)

    clear_mask = ambig_label == 0
    ambig_mask = ambig_label == 1
    hallu_mask = y == 0
    truthful_mask = y != 0

    return {
        "total": int(np.sum(row_mask)),
        "clear": int(np.sum(row_mask & clear_mask)),
        "ambig": int(np.sum(row_mask & ambig_mask)),
        "hallu_clear": int(np.sum(row_mask & hallu_mask & clear_mask)),
        "hallu_ambig": int(np.sum(row_mask & hallu_mask & ambig_mask)),
        "truthful_clear": int(np.sum(row_mask & truthful_mask & clear_mask)),
        "truthful_ambig": int(np.sum(row_mask & truthful_mask & ambig_mask)),
    }


def _summarize_kept_ambig_rows(
    split_data: Dict[str, object],
    keep_mask: np.ndarray,
) -> Dict[str, int]:
    kept_stats = _summarize_ambig_row_groups(split_data, keep_mask)
    return {
        "n_clear": kept_stats["clear"],
        "n_ambig": kept_stats["ambig"],
        "n_hallu_clear": kept_stats["hallu_clear"],
        "n_hallu_ambig": kept_stats["hallu_ambig"],
        "n_truthful_clear": kept_stats["truthful_clear"],
        "n_truthful_ambig": kept_stats["truthful_ambig"],
    }


def _summarize_removed_ambig_rows(
    split_data: Dict[str, object],
    keep_mask: np.ndarray,
) -> Dict[str, int]:
    removed_mask = ~np.asarray(keep_mask, dtype=bool)
    removed_stats = _summarize_ambig_row_groups(split_data, removed_mask)

    return {
        "removed_total": removed_stats["total"],
        "removed_clear_hallucination": removed_stats["hallu_clear"],
        "removed_ambig_hallucination": removed_stats["hallu_ambig"],
        "removed_clear_truthful": removed_stats["truthful_clear"],
        "removed_ambig_truthful": removed_stats["truthful_ambig"],
    }


def _save_ambig_filtered_split(
    split_name: str,
    split_dir: Path,
    split_data: Dict[str, object],
    keep_mask: np.ndarray,
    dataset_tag: str = AMBIG_QA_FILTERED_DATASET_NAME,
) -> Dict[str, object]:
    split_dir.mkdir(parents=True, exist_ok=True)

    idx_arr = np.flatnonzero(np.asarray(keep_mask, dtype=bool))
    X = _filter_X_by_index(split_data["X"], idx_arr, int(split_data["n_total"]))
    y = np.asarray(split_data["y"], dtype=np.int64).reshape(-1)[idx_arr]
    acc = np.asarray(split_data["acc"], dtype=np.float32).reshape(-1)[idx_arr]
    f1 = np.asarray(split_data["f1"], dtype=np.float32).reshape(-1)[idx_arr]
    preds = [split_data["preds"][i] for i in idx_arr.tolist()]
    questions = [split_data["questions"][i] for i in idx_arr.tolist()]
    references = [split_data["references"][i] for i in idx_arr.tolist()]
    se_scores = np.asarray(split_data["se_scores"], dtype=np.float32).reshape(-1)[idx_arr]
    semantic_ids = np.asarray(split_data["semantic_ids"])[idx_arr]
    sampled_responses = [split_data["sampled_responses"][i] for i in idx_arr.tolist()]
    ambig_label_filtered = np.asarray(split_data["ambig_label"], dtype=np.int64).reshape(-1)[idx_arr]

    _save_X_npz(str(split_dir / "X.npz"), X)
    np.save(split_dir / "y.npy", y.astype(np.int64))
    np.save(split_dir / "acc.npy", acc.astype(np.float32))
    np.save(split_dir / "f1.npy", f1.astype(np.float32))
    np.save(split_dir / "se.npy", se_scores.astype(np.float32))
    np.save(split_dir / "ambig_label.npy", ambig_label_filtered.astype(np.int64))
    np.save(split_dir / split_data["semantic_ids_name"], np.asarray(semantic_ids, dtype=np.int64))
    _save_preds_jsonl(str(split_dir / "preds.jsonl"), preds)
    _save_questions_jsonl(str(split_dir / "questions.jsonl"), questions)
    _save_refs_jsonl(str(split_dir / "references.jsonl"), references)
    _save_sampled_responses_jsonl(str(split_dir / split_data["se_resp_name"]), sampled_responses)
    if split_data["pred_sequence_scores"] is not None:
        pred_sequence_scores = split_data["pred_sequence_scores"]
        save_sequence_scores_npz(
            str(split_dir / "preds_sequence_scores.npz"),
            pred_sequence_scores["sequence_logprob"][idx_arr],
            pred_sequence_scores["sequence_prob"][idx_arr],
            pred_sequence_scores["length_normalized_sequence_logprob"][idx_arr],
        )
    if split_data["sampled_response_sequence_scores"] is not None:
        sampled_response_sequence_scores = split_data["sampled_response_sequence_scores"]
        save_sequence_scores_npz(
            str(split_dir / split_data["sampled_response_sequence_scores_name"]),
            sampled_response_sequence_scores["sequence_logprob"][idx_arr],
            sampled_response_sequence_scores["sequence_prob"][idx_arr],
            sampled_response_sequence_scores["length_normalized_sequence_logprob"][idx_arr],
        )

    if split_data["response_embeddings"] is not None:
        save_cluster_embeddings_npz(
            str(split_dir / f"response_embeddings_{split_data['n_resp_suffix']}.npz"),
            [split_data["response_embeddings"][i] for i in idx_arr.tolist()],
        )
    if split_data["cluster_embeddings"] is not None:
        save_cluster_embeddings_npz(
            str(split_dir / f"cluster_embeddings_{split_data['n_resp_suffix']}.npz"),
            [split_data["cluster_embeddings"][i] for i in idx_arr.tolist()],
        )
    if split_data["response_slt_lastk"] is not None:
        save_cluster_embeddings_npz(
            str(split_dir / f"response_slt_lastk_{split_data['n_resp_suffix']}.npz"),
            [split_data["response_slt_lastk"][i] for i in idx_arr.tolist()],
        )

    acc_mean = float(np.mean(acc)) if acc.size > 0 else 0.0
    f1_mean = float(np.mean(f1)) if f1.size > 0 else 0.0
    kept_stats = _summarize_kept_ambig_rows(split_data, keep_mask)
    removed_stats = _summarize_removed_ambig_rows(split_data, keep_mask)
    print(
        f"[{dataset_tag}] {split_name}: kept={len(idx_arr)}/{int(split_data['n_total'])} "
        f"| accuracy={acc_mean:.4f} | f1={f1_mean:.4f}"
    )
    if dataset_tag == AMBIG_QA_FILTERED_VER3_DATASET_NAME:
        print(
            f"[{dataset_tag}] {split_name}: kept_subgroups="
            f"n_clear={kept_stats['n_clear']} | n_ambig={kept_stats['n_ambig']} "
            f"| n_hallu_clear={kept_stats['n_hallu_clear']} | n_hallu_ambig={kept_stats['n_hallu_ambig']} "
            f"| n_truthful_clear={kept_stats['n_truthful_clear']} | n_truthful_ambig={kept_stats['n_truthful_ambig']}"
        )
    print(
        f"[{dataset_tag}] {split_name}: removed={removed_stats['removed_total']} "
        f"| clear+hallucination={removed_stats['removed_clear_hallucination']} "
        f"| ambig+hallucination={removed_stats['removed_ambig_hallucination']} "
        f"| clear+truthful={removed_stats['removed_clear_truthful']} "
        f"| ambig+truthful={removed_stats['removed_ambig_truthful']}"
    )
    return {
        "N": int(len(idx_arr)),
        "accuracy": acc_mean,
        "f1": f1_mean,
        "tbg_shape": tuple(X["tbg"].shape),
        "slt_shape": tuple(X["slt"].shape),
        "has_pred_sequence_scores": bool(split_data["pred_sequence_scores"] is not None),
        "sampled_response_sequence_scores_name": (
            split_data["sampled_response_sequence_scores_name"]
            if split_data["sampled_response_sequence_scores"] is not None
            else None
        ),
        **kept_stats,
        **removed_stats,
    }


def _print_ambig_filtered_final_stats(dataset_tag: str, split_stats) -> None:
    print(f"[{dataset_tag}] final split summary:")
    for stats in split_stats:
        split_name = stats["split"]
        print(
            f"[{dataset_tag}] {split_name}: kept={stats['kept_total']}/{stats['source_total']} "
            f"| accuracy={stats['accuracy']:.4f} | f1={stats['f1']:.4f}"
        )
        if {"n_clear", "n_ambig", "n_hallu_clear", "n_hallu_ambig", "n_truthful_clear", "n_truthful_ambig"} <= stats.keys():
            print(
                f"[{dataset_tag}] {split_name}: kept_subgroups="
                f"n_clear={stats['n_clear']} | n_ambig={stats['n_ambig']} "
                f"| n_hallu_clear={stats['n_hallu_clear']} | n_hallu_ambig={stats['n_hallu_ambig']} "
                f"| n_truthful_clear={stats['n_truthful_clear']} | n_truthful_ambig={stats['n_truthful_ambig']}"
            )
        print(
            f"[{dataset_tag}] {split_name}: removed={stats['removed_total']} "
            f"| clear+hallucination={stats['removed_clear_hallucination']} "
            f"| ambig+hallucination={stats['removed_ambig_hallucination']} "
            f"| clear+truthful={stats['removed_clear_truthful']} "
            f"| ambig+truthful={stats['removed_ambig_truthful']}"
        )


def _collect_ambig_qa_filtered_from_existing_run(args) -> None:
    model_id = args.model_name.split("/")[-1]
    source_dir = _feature_output_dir(
        model_id=model_id,
        dataset_name=AMBIG_QA_DATASET_NAME,
        n_samples=args.n_samples,
        judge_type=args.judge_type,
    )
    output_dir = _feature_output_dir(
        model_id=model_id,
        dataset_name=AMBIG_QA_FILTERED_DATASET_NAME,
        n_samples=args.n_samples,
        judge_type=args.judge_type,
    )

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            "ambig_qa_filtered requires an existing matching ambig_qa artifact directory. "
            f"Missing: {source_dir}"
        )

    print(f"[ambig_qa_filtered] source: {source_dir}")
    print(f"[ambig_qa_filtered] output: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    split_stats = []
    shapes = {}

    for split_name in ("train", "valid", "test"):
        source_split_dir = Path(source_dir) / split_name
        output_split_dir = Path(output_dir) / split_name
        split_data = _load_ambig_filtered_source_split(source_split_dir)
        keep_mask, judge_stats = _judge_ambig_qa_filtered_split(split_name, split_data)
        split_summary = _save_ambig_filtered_split(
            split_name=split_name,
            split_dir=output_split_dir,
            split_data=split_data,
            keep_mask=keep_mask,
        )
        shapes[split_name] = {
            "tbg_shape": split_summary["tbg_shape"],
            "slt_shape": split_summary["slt_shape"],
            "N": split_summary["N"],
            "has_pred_sequence_scores": split_summary["has_pred_sequence_scores"],
            "sampled_response_sequence_scores_name": split_summary["sampled_response_sequence_scores_name"],
        }
        split_stats.append({
            "split": split_name,
            "accuracy": split_summary["accuracy"],
            "f1": split_summary["f1"],
            **judge_stats,
            "kept_total": split_summary["N"],
            "removed_total": split_summary["removed_total"],
            "removed_clear_hallucination": split_summary["removed_clear_hallucination"],
            "removed_ambig_hallucination": split_summary["removed_ambig_hallucination"],
            "removed_clear_truthful": split_summary["removed_clear_truthful"],
            "removed_ambig_truthful": split_summary["removed_ambig_truthful"],
        })

    _write_jsonl(Path(output_dir) / "dataset_stat.jsonl", split_stats)
    _print_ambig_filtered_final_stats(AMBIG_QA_FILTERED_DATASET_NAME, split_stats)

    meta = {
        "model_name": args.model_name,
        "dataset_name": AMBIG_QA_FILTERED_DATASET_NAME,
        "source_dataset_name": AMBIG_QA_DATASET_NAME,
        "source_dir": source_dir,
        "n_samples": int(args.n_samples),
        "judge_type": args.judge_type,
        "ambig_filter_judge_model": GPT5_MINI_AZURE_MODEL,
        "artifact_usage": "direct_filtered_splits",
        "ambig_filter_policy": (
            "scope=all_questions; "
            "judge every row in train/valid/test; "
            "drop rows whose judge response starts with true; "
            "keep false/unparsed/content-filtered rows; "
            "save filtered train/valid/test splits for direct loading"
        ),
        "ambig_filter_scope": "all_questions",
        "shapes": shapes,
        "paths": {
            "base_dir": output_dir,
            "train": {
                k: os.path.join(output_dir, "train", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "valid": {
                k: os.path.join(output_dir, "valid", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "test": {
                k: os.path.join(output_dir, "test", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "dataset_stats": os.path.join(output_dir, "dataset_stat.jsonl"),
        },
    }
    for split_name in ("train", "valid", "test"):
        if not shapes[split_name]["has_pred_sequence_scores"]:
            meta["paths"][split_name].pop("preds_sequence_scores.npz", None)
        sampled_scores_name = shapes[split_name]["sampled_response_sequence_scores_name"]
        if sampled_scores_name is not None:
            meta["paths"][split_name][sampled_scores_name] = os.path.join(
                output_dir,
                split_name,
                sampled_scores_name,
            )
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Finished collecting features] Saved to:", output_dir)


def _collect_ambig_qa_filtered_ver2_from_existing_run(args) -> None:
    model_id = args.model_name.split("/")[-1]
    source_dir = _feature_output_dir(
        model_id=model_id,
        dataset_name=AMBIG_QA_DATASET_NAME,
        n_samples=args.n_samples,
        judge_type=args.judge_type,
    )
    output_dir = _feature_output_dir(
        model_id=model_id,
        dataset_name=AMBIG_QA_FILTERED_VER2_DATASET_NAME,
        n_samples=args.n_samples,
        judge_type=args.judge_type,
    )

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            "ambig_qa_filtered_ver2 requires an existing matching ambig_qa artifact directory. "
            f"Missing: {source_dir}"
        )

    print(f"[{AMBIG_QA_FILTERED_VER2_DATASET_NAME}] source: {source_dir}")
    print(f"[{AMBIG_QA_FILTERED_VER2_DATASET_NAME}] output: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    split_stats = []
    shapes = {}

    for split_name in ("train", "valid", "test"):
        source_split_dir = Path(source_dir) / split_name
        output_split_dir = Path(output_dir) / split_name
        split_data = _load_ambig_filtered_source_split(source_split_dir)
        keep_mask, filter_stats = _filter_ambig_qa_filtered_ver2_split(split_name, split_data)
        split_summary = _save_ambig_filtered_split(
            split_name=split_name,
            split_dir=output_split_dir,
            split_data=split_data,
            keep_mask=keep_mask,
            dataset_tag=AMBIG_QA_FILTERED_VER2_DATASET_NAME,
        )
        shapes[split_name] = {
            "tbg_shape": split_summary["tbg_shape"],
            "slt_shape": split_summary["slt_shape"],
            "N": split_summary["N"],
            "has_pred_sequence_scores": split_summary["has_pred_sequence_scores"],
            "sampled_response_sequence_scores_name": split_summary["sampled_response_sequence_scores_name"],
        }
        split_stats.append({
            "split": split_name,
            "accuracy": split_summary["accuracy"],
            "f1": split_summary["f1"],
            **filter_stats,
            "kept_total": split_summary["N"],
            "n_clear": split_summary["n_clear"],
            "n_ambig": split_summary["n_ambig"],
            "n_hallu_clear": split_summary["n_hallu_clear"],
            "n_hallu_ambig": split_summary["n_hallu_ambig"],
            "n_truthful_clear": split_summary["n_truthful_clear"],
            "n_truthful_ambig": split_summary["n_truthful_ambig"],
            "removed_total": split_summary["removed_total"],
            "removed_clear_hallucination": split_summary["removed_clear_hallucination"],
            "removed_ambig_hallucination": split_summary["removed_ambig_hallucination"],
            "removed_clear_truthful": split_summary["removed_clear_truthful"],
            "removed_ambig_truthful": split_summary["removed_ambig_truthful"],
        })

    _write_jsonl(Path(output_dir) / "dataset_stat.jsonl", split_stats)

    meta = {
        "model_name": args.model_name,
        "dataset_name": AMBIG_QA_FILTERED_VER2_DATASET_NAME,
        "source_dataset_name": AMBIG_QA_DATASET_NAME,
        "source_dir": source_dir,
        "n_samples": int(args.n_samples),
        "judge_type": args.judge_type,
        "ambig_filter_policy": (
            "drop rows where gt_dissonance == 0.0, y == 1, and ambig_label == 1 "
            "for train/valid/test"
        ),
        "shapes": shapes,
        "paths": {
            "base_dir": output_dir,
            "train": {
                k: os.path.join(output_dir, "train", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "valid": {
                k: os.path.join(output_dir, "valid", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "test": {
                k: os.path.join(output_dir, "test", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "dataset_stats": os.path.join(output_dir, "dataset_stat.jsonl"),
        },
    }
    for split_name in ("train", "valid", "test"):
        if not shapes[split_name]["has_pred_sequence_scores"]:
            meta["paths"][split_name].pop("preds_sequence_scores.npz", None)
        sampled_scores_name = shapes[split_name]["sampled_response_sequence_scores_name"]
        if sampled_scores_name is not None:
            meta["paths"][split_name][sampled_scores_name] = os.path.join(
                output_dir,
                split_name,
                sampled_scores_name,
            )
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Finished collecting features] Saved to:", output_dir)


def _collect_ambig_qa_filtered_ver3_from_existing_run(args) -> None:
    model_id = args.model_name.split("/")[-1]
    source_dir = _feature_output_dir(
        model_id=model_id,
        dataset_name=AMBIG_QA_DATASET_NAME,
        n_samples=args.n_samples,
        judge_type=args.judge_type,
    )
    output_dir = _feature_output_dir(
        model_id=model_id,
        dataset_name=AMBIG_QA_FILTERED_VER3_DATASET_NAME,
        n_samples=args.n_samples,
        judge_type=args.judge_type,
    )

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            "ambig_qa_filtered_ver3 requires an existing matching ambig_qa artifact directory. "
            f"Missing: {source_dir}"
        )

    print(f"[{AMBIG_QA_FILTERED_VER3_DATASET_NAME}] source: {source_dir}")
    print(f"[{AMBIG_QA_FILTERED_VER3_DATASET_NAME}] output: {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    split_stats = []
    shapes = {}

    for split_name in ("train", "valid", "test"):
        source_split_dir = Path(source_dir) / split_name
        output_split_dir = Path(output_dir) / split_name
        split_data = _load_ambig_filtered_source_split(source_split_dir)
        keep_mask, filter_stats = _filter_ambig_qa_filtered_ver3_split(split_name, split_data)
        split_summary = _save_ambig_filtered_split(
            split_name=split_name,
            split_dir=output_split_dir,
            split_data=split_data,
            keep_mask=keep_mask,
            dataset_tag=AMBIG_QA_FILTERED_VER3_DATASET_NAME,
        )
        shapes[split_name] = {
            "tbg_shape": split_summary["tbg_shape"],
            "slt_shape": split_summary["slt_shape"],
            "N": split_summary["N"],
            "has_pred_sequence_scores": split_summary["has_pred_sequence_scores"],
            "sampled_response_sequence_scores_name": split_summary["sampled_response_sequence_scores_name"],
        }
        split_stats.append({
            "split": split_name,
            "accuracy": split_summary["accuracy"],
            "f1": split_summary["f1"],
            **filter_stats,
            "kept_total": split_summary["N"],
            "n_clear": split_summary["n_clear"],
            "n_ambig": split_summary["n_ambig"],
            "n_hallu_clear": split_summary["n_hallu_clear"],
            "n_hallu_ambig": split_summary["n_hallu_ambig"],
            "n_truthful_clear": split_summary["n_truthful_clear"],
            "n_truthful_ambig": split_summary["n_truthful_ambig"],
            "removed_total": split_summary["removed_total"],
            "removed_clear_hallucination": split_summary["removed_clear_hallucination"],
            "removed_ambig_hallucination": split_summary["removed_ambig_hallucination"],
            "removed_clear_truthful": split_summary["removed_clear_truthful"],
            "removed_ambig_truthful": split_summary["removed_ambig_truthful"],
        })

    _write_jsonl(Path(output_dir) / "dataset_stat.jsonl", split_stats)
    _print_ambig_filtered_final_stats(AMBIG_QA_FILTERED_VER3_DATASET_NAME, split_stats)

    meta = {
        "model_name": args.model_name,
        "dataset_name": AMBIG_QA_FILTERED_VER3_DATASET_NAME,
        "source_dataset_name": AMBIG_QA_DATASET_NAME,
        "source_dir": source_dir,
        "n_samples": int(args.n_samples),
        "judge_type": args.judge_type,
        "artifact_usage": "direct_filtered_splits",
        "filter_threshold": float(AMBIG_QA_FILTERED_VER3_FILTER_THRESHOLD),
        "response_overlap_threshold": float(AMBIG_QA_FILTERED_VER3_RESPONSE_OVERLAP_THRESHOLD),
        "overlap_metric": "rougeL",
        "ambig_filter_policy": (
            "for each question in train/valid/test, compute overlap_ratio as the "
            "fraction of sampled responses whose max ROUGE-L against references is >= 0.30; "
            "keep all clear rows (ambig_label == 0); "
            "for ambiguous rows (ambig_label == 1), keep rows with overlap_ratio >= 0.5 "
            "and drop rows with overlap_ratio < 0.5"
        ),
        "shapes": shapes,
        "paths": {
            "base_dir": output_dir,
            "train": {
                k: os.path.join(output_dir, "train", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "valid": {
                k: os.path.join(output_dir, "valid", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "test": {
                k: os.path.join(output_dir, "test", k)
                for k in [
                    "X.npz",
                    "y.npy",
                    "acc.npy",
                    "f1.npy",
                    "preds.jsonl",
                    "preds_sequence_scores.npz",
                    "ambig_label.npy",
                ]
            },
            "dataset_stats": os.path.join(output_dir, "dataset_stat.jsonl"),
        },
    }
    for split_name in ("train", "valid", "test"):
        if not shapes[split_name]["has_pred_sequence_scores"]:
            meta["paths"][split_name].pop("preds_sequence_scores.npz", None)
        sampled_scores_name = shapes[split_name]["sampled_response_sequence_scores_name"]
        if sampled_scores_name is not None:
            meta["paths"][split_name][sampled_scores_name] = os.path.join(
                output_dir,
                split_name,
                sampled_scores_name,
            )
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Finished collecting features] Saved to:", output_dir)


def _is_llama_33_70b_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    if not norm:
        return False
    basename = norm.split("/")[-1]
    return basename.endswith(LLAMA_33_70B_SUFFIX) or norm.endswith(LLAMA_33_70B_SUFFIX)


def _is_qwen3_30b_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    if not norm:
        return False
    basename = norm.split("/")[-1]
    return (
        basename == QWEN3_30B_A3B_ALIAS
        or basename.startswith(f"{QWEN3_30B_A3B_ALIAS}-")
    )


def _is_legacy_fewshot_prompt_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    if not norm:
        return False
    basename = norm.split("/")[-1]
    return "llama-2-7b" in basename or "mistral-7b" in basename


def _needs_8bit_quantized_loading(model_name: str) -> bool:
    return (
        _is_llama_33_70b_model_name(model_name)
        or _is_qwen3_30b_model_name(model_name)
        or _is_mistral_small_24b_model_name(model_name)
    )


def _is_llama_chat_template_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    if not norm:
        return False
    basename = norm.split("/")[-1]
    return (
        basename.endswith(LLAMA_30_8B_SUFFIX)
        or basename.endswith(LLAMA_31_8B_SUFFIX)
        or basename.endswith(LLAMA_32_3B_SUFFIX)
        or _is_llama_33_70b_model_name(norm)
    )


def _uses_tokenizer_chat_template_model_name(model_name: str) -> bool:
    return (
        _is_gemma_chat_template_model_name(model_name)
        or _is_llama_chat_template_model_name(model_name)
        or _is_qwen3_30b_model_name(model_name)
        or _is_mistral_small_24b_model_name(model_name)
    )


def _build_chat_template_feature_payload(
    tokenizer,
    user_content: str,
    references,
    system_prompt: Optional[str],
    question: str,
    context: str,
    rewrite_source_question: Optional[str] = None,
):
    def _messages(assistant_content: Optional[str] = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        if assistant_content is not None:
            messages.append({"role": "assistant", "content": assistant_content})
        return messages

    prefix_str = tokenizer.apply_chat_template(
        _messages(),
        add_generation_prompt=True,
        tokenize=False,
    )
    tok = tokenizer(prefix_str, truncation=True)

    payload = {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "references": references,
        "question": question,
        "context": context,
    }
    if rewrite_source_question is not None:
        payload["rewrite_source_question"] = rewrite_source_question
    return payload


def _device_from_map_entry(device_like) -> Optional[torch.device]:
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, int):
        return torch.device(f"cuda:{int(device_like)}")
    if isinstance(device_like, str):
        text = device_like.strip().lower()
        if text.startswith("cuda"):
            return torch.device(text)
        if text == "cpu":
            return torch.device("cpu")
    return None


def _ensure_bitsandbytes_available() -> None:
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "8-bit quantized loading requires bitsandbytes in the active environment."
        )


def _resolve_single_gpu_device_map(
    accelerator: Optional[Accelerator] = None,
    fallback_device: Optional[object] = None,
) -> Dict[str, object]:
    target = accelerator.device if accelerator is not None else fallback_device

    if target is None:
        if torch.cuda.is_available():
            return {"": int(torch.cuda.current_device())}
        return {"": "cpu"}

    if isinstance(target, torch.device):
        if target.type == "cuda":
            index = target.index if target.index is not None else torch.cuda.current_device()
            return {"": int(index)}
        if target.type == "cpu":
            return {"": "cpu"}
        return {"": str(target)}

    if isinstance(target, int):
        return {"": int(target)}

    if isinstance(target, str):
        parsed = _device_from_map_entry(target)
        if parsed is not None:
            return _resolve_single_gpu_device_map(fallback_device=parsed)
        return {"": target}

    raise TypeError(f"Unsupported device spec for quantized model placement: {target!r}")


def _resolve_model_input_device(
    model,
    accelerator: Optional[Accelerator] = None,
    fallback_device: Optional[object] = None,
) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            device = _device_from_map_entry(mapped_device)
            if device is not None and device.type == "cuda":
                return device

    try:
        return next(model.parameters()).device
    except StopIteration:
        pass

    if accelerator is not None:
        return accelerator.device
    if fallback_device is not None:
        device = _device_from_map_entry(fallback_device)
        if device is not None:
            return device
        return torch.device(str(fallback_device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_num_hidden_layers(model) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return int(text_config.num_hidden_layers)
    if config is not None and hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    raise AttributeError(
        f"Unable to determine num_hidden_layers for model config type "
        f"{type(config).__name__ if config is not None else 'None'}."
    )


def _get_hidden_size(model) -> int:
    if _is_gemma_model(model):
        return int(model.config.text_config.hidden_size)
    return int(model.config.hidden_size)


def _split_indices_for_process(n_total: int, accelerator: Optional[Accelerator]) -> List[int]:
    if accelerator is None or accelerator.num_processes <= 1:
        return list(range(n_total))
    shards = np.array_split(np.arange(n_total, dtype=np.int64), accelerator.num_processes)
    return shards[accelerator.process_index].tolist()


def _sanitize_token(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text or "")
    return safe.strip("_") or "na"


def _dist_run_token() -> str:
    parts = [
        os.environ.get("TORCHELASTIC_RUN_ID"),
        os.environ.get("MASTER_ADDR"),
        os.environ.get("MASTER_PORT"),
    ]
    token = "_".join([p for p in parts if p])
    if not token:
        token = f"pid_{os.getppid()}"
    return _sanitize_token(token)


def _gather_records_to_main(
    local_records,
    accelerator: Optional[Accelerator],
    exchange_dir: Optional[str] = None,
    split_name: str = "split",
):
    if accelerator is None or accelerator.num_processes <= 1:
        return local_records
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed gather requested but torch.distributed is not initialized. "
            "Launch with `accelerate launch --num_processes N ...` for multi-GPU runs."
        )

    backend = (os.environ.get("FEATURE_DIST_GATHER_BACKEND", "file") or "file").strip().lower()
    rank = int(accelerator.process_index)
    world = int(accelerator.num_processes)
    split_tok = _sanitize_token(split_name)

    if backend == "torch":
        if accelerator.is_main_process:
            gathered = [None for _ in range(world)]
            torch.distributed.gather_object(local_records, gathered, dst=0)
            merged = []
            for shard in gathered:
                if shard:
                    merged.extend(shard)
            return merged
        torch.distributed.gather_object(local_records, None, dst=0)
        return None

    if exchange_dir is None:
        raise ValueError(
            "exchange_dir must be provided for file-based distributed gather. "
            "Either pass exchange_dir or set FEATURE_DIST_GATHER_BACKEND=torch."
        )

    exchange_path = Path(exchange_dir)
    exchange_path.mkdir(parents=True, exist_ok=True)
    token_file = exchange_path / "run_token.txt"
    wait_timeout_sec = int(os.environ.get("FEATURE_DIST_GATHER_FILE_TIMEOUT_SEC", "14400"))
    poll_sec = max(1.0, float(os.environ.get("FEATURE_DIST_GATHER_FILE_POLL_SEC", "5")))

    run_token = ""
    if token_file.exists():
        try:
            run_token = token_file.read_text(encoding="utf-8").strip()
        except OSError:
            run_token = ""
    if not run_token:
        run_token = _dist_run_token()

    shard_name = f"{run_token}_{split_tok}_rank{rank:03d}_of{world:03d}.pkl"
    shard_path = exchange_path / shard_name
    tmp_path = shard_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(local_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, shard_path)

    if not accelerator.is_main_process:
        return None

    expected = [
        exchange_path / f"{run_token}_{split_tok}_rank{r:03d}_of{world:03d}.pkl"
        for r in range(world)
    ]
    start = time.time()
    while True:
        missing = [str(p) for p in expected if not p.exists()]
        if not missing:
            break
        elapsed = time.time() - start
        if elapsed > wait_timeout_sec:
            raise RuntimeError(
                f"Timeout waiting for distributed shard files for split={split_tok} after {elapsed:.1f}s. "
                f"Missing {len(missing)}/{len(expected)} shards. "
                f"Example missing: {missing[:2]}"
            )
        time.sleep(poll_sec)

    merged = []
    for path in expected:
        with open(path, "rb") as f:
            shard = pickle.load(f)
        if shard:
            merged.extend(shard)
    if os.environ.get("FEATURE_DIST_GATHER_KEEP_SHARDS", "0") != "1":
        for path in expected:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return merged


def _collect_and_save(split_name, dataset, model, tokenizer, collator, cfg, outdir, accelerator: Optional[Accelerator] = None):
    is_main = accelerator is None or accelerator.is_main_process
    if is_main:
        print(f"[collect] {split_name}…")

    local_indices = _split_indices_for_process(len(dataset), accelerator)
    dataset_local = dataset.select(local_indices)

    X, y, scores, preds, pred_sequence_scores = collect_hidden_states(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset_local,
        collator=collator,
        cfg=cfg,
        accelerator=accelerator,
        global_indices=local_indices,
        expected_total=len(dataset),
        exchange_dir=os.path.join(outdir, ".dist_exchange"),
        split_name=split_name,
    )

    if not is_main:
        return None

    expected_n = len(dataset)
    n_tbg = int(X["tbg"].shape[1]) if isinstance(X, dict) and "tbg" in X else -1
    n_slt = int(X["slt"].shape[1]) if isinstance(X, dict) and "slt" in X else -1
    n_y = len(y)
    n_acc = len(scores["accuracy"])
    n_f1 = len(scores["f1"])
    n_pred = len(preds)
    n_pred_seq = int(pred_sequence_scores["sequence_logprob"].shape[0])
    if not (n_tbg == n_slt == n_y == n_acc == n_f1 == n_pred == n_pred_seq == expected_n):
        raise RuntimeError(
            "Output alignment check failed in feature collection: "
            f"expected={expected_n}, tbg={n_tbg}, slt={n_slt}, y={n_y}, "
            f"acc={n_acc}, f1={n_f1}, preds={n_pred}, pred_sequence_scores={n_pred_seq}."
        )

    split_dir = os.path.join(outdir, split_name)
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    _save_X_npz(os.path.join(split_dir, "X.npz"), X) # hidden states

    # labels / scores
    acc_mean = float(np.mean(scores["accuracy"]))
    f1_mean = float(np.mean(scores["f1"]))
    print(f"[Accuracy] {acc_mean:.4f}\n[F1] {f1_mean:.4f}")
    np.save(os.path.join(split_dir, "y.npy"),   np.asarray(y, dtype=np.int64))
    np.save(os.path.join(split_dir, "acc.npy"), np.asarray(scores["accuracy"], dtype=np.float32))
    np.save(os.path.join(split_dir, "f1.npy"),  np.asarray(scores["f1"], dtype=np.float32))
    if "ambig_label" in dataset.column_names:
        np.save(
            os.path.join(split_dir, "ambig_label.npy"),
            np.asarray(dataset["ambig_label"], dtype=np.int64),
        )
    if "ood_label" in dataset.column_names:
        np.save(
            os.path.join(split_dir, "ood_label.npy"),
            np.asarray(dataset["ood_label"], dtype=np.int64),
        )

    _save_preds_jsonl(os.path.join(split_dir, "preds.jsonl"), preds) # predictions
    save_sequence_scores_npz(
        os.path.join(split_dir, "preds_sequence_scores.npz"),
        pred_sequence_scores["sequence_logprob"],
        pred_sequence_scores["sequence_prob"],
        pred_sequence_scores["length_normalized_sequence_logprob"],
    )
    _save_refs_jsonl(os.path.join(split_dir, "references.jsonl"), dataset["references"]) # references
    _save_questions_jsonl(os.path.join(split_dir, "questions.jsonl"), dataset["question"]) # questions

    # Return shapes for meta
    return {
        "tbg_shape": tuple(X["tbg"].shape),
        "slt_shape": tuple(X["slt"].shape),
        "N": int(len(y)),
        "accuracy": acc_mean,
        "f1": f1_mean,
    }

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)               # remove articles
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s

def _f1_single(pred: str, gold: str) -> float:
    ptoks = _normalize_answer(pred).split()
    gtoks = _normalize_answer(gold).split()
    if len(ptoks) == 0 and len(gtoks) == 0:
        return 1.0
    if len(ptoks) == 0 or len(gtoks) == 0:
        return 0.0
    common = Counter(ptoks) & Counter(gtoks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(ptoks)
    recall    = num_same / len(gtoks)
    return 2 * precision * recall / (precision + recall)

def _f1_max_over_refs(pred: str, refs) -> float:
    # refs: list[str]
    return max((_f1_single(pred, r) for r in refs), default=0.0)

def _strip_harmony_markers(text: str) -> str:
    text = re.sub(r"<\|channel\|>\w+", "", text)
    text = text.replace("<|start|>", "")
    text = text.replace("<|message|>", "")
    text = text.replace("<|end|>", "")
    text = text.replace("<return>", "")
    return text.strip()

def _extract_harmony_text(messages) -> str:
    for msg in messages:
        if getattr(msg.author, "role", None) == Role.ASSISTANT and msg.channel == "final":
            if msg.content:
                return msg.content[0].text
    for msg in messages:
        if getattr(msg.author, "role", None) == Role.ASSISTANT:
            if msg.content:
                return msg.content[0].text
    for msg in messages:
        if msg.content:
            return msg.content[0].text
    return ""

def _decode_gpt_oss_completion(pred_ids) -> str:
    token_ids = pred_ids.tolist() if hasattr(pred_ids, "tolist") else list(pred_ids)
    if not token_ids:
        return ""
    try:
        messages = encoding.parse_messages_from_completion_tokens(
            token_ids, role=Role.ASSISTANT
        )
    except Exception:
        try:
            messages = encoding.parse_messages_from_completion_tokens(
                token_ids, role=Role.ASSISTANT, strict=False
            )
        except Exception:
            return _strip_harmony_markers(encoding.decode(token_ids))
    text = _extract_harmony_text(messages)
    if text:
        return text.strip()
    return _strip_harmony_markers(encoding.decode(token_ids))


def _stringify_if_nonempty(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _dedupe_text_list(values) -> List[str]:
    deduped = []
    seen = set()
    for value in values:
        text = _stringify_if_nonempty(value)
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def _load_jsonl_rows(path: Union[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _flatten_nested_text_values(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_nested_text_values(item))
        return flattened
    text = _stringify_if_nonempty(value)
    return [text] if text else []


def _extract_ambig_qa_reference_data(example) -> Tuple[List[str], str, int]:
    annotations = example.get("annotations")
    references: List[str] = []

    def _append_multiple_qa_answers(qa_pairs) -> None:
        nonlocal references
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        if isinstance(qa_pairs, list):
            for qa_pair in qa_pairs:
                if isinstance(qa_pair, dict):
                    references.extend(_flatten_nested_text_values(qa_pair.get("answer", [])))
                else:
                    references.extend(_flatten_nested_text_values(qa_pair))
        else:
            references.extend(_flatten_nested_text_values(qa_pairs))

    if isinstance(annotations, list):
        annotations = annotations[0] if len(annotations) > 0 and isinstance(annotations[0], dict) else None

    if isinstance(annotations, dict):
        annotation_types = _flatten_nested_text_values(annotations.get("type", []))
        annotation_type = annotation_types[0] if annotation_types else ""
        if annotation_type == "singleAnswer":
            references = _flatten_nested_text_values(annotations.get("answer", []))
        elif annotation_type == "multipleQAs":
            _append_multiple_qa_answers(annotations.get("qaPairs", []))
    else:
        annotation_type = ""

    references = _dedupe_text_list(references)
    if len(references) == 0:
        references = _dedupe_text_list(_flatten_nested_text_values(example.get("nq_answer", [])))

    primary_reference = references[0] if len(references) > 0 else ""
    if len(references) == 0:
        references = [""]

    ambig_label = int(annotation_type == "multipleQAs")
    return references, primary_reference, ambig_label


def _extract_ambig_qa_annotation_type(example) -> str:
    annotations = example.get("annotations")
    if isinstance(annotations, list):
        annotations = annotations[0] if annotations and isinstance(annotations[0], dict) else None
    if not isinstance(annotations, dict):
        return ""
    annotation_types = _flatten_nested_text_values(annotations.get("type", []))
    return annotation_types[0] if annotation_types else ""


def _extract_ambig_qa_gold_rewrites(
    example,
    original_question: str,
) -> Tuple[str, List[str]]:
    annotations = example.get("annotations")
    if isinstance(annotations, list):
        annotations = annotations[0] if annotations and isinstance(annotations[0], dict) else None

    annotation_type = _extract_ambig_qa_annotation_type(example)
    gold_rewrites: List[str] = []
    if isinstance(annotations, dict) and annotation_type == "multipleQAs":
        qa_pairs = annotations.get("qaPairs", [])
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        if isinstance(qa_pairs, list):
            for qa_pair in qa_pairs:
                if isinstance(qa_pair, dict):
                    gold_rewrites.extend(_flatten_nested_text_values(qa_pair.get("question", [])))
                else:
                    gold_rewrites.extend(_flatten_nested_text_values(qa_pair))
    elif annotation_type == "singleAnswer":
        gold_rewrites = [str(original_question)]

    return annotation_type, _dedupe_text_list(gold_rewrites)


def _build_ambig_qa_disambig_demo_entry(example) -> Optional[Dict[str, object]]:
    question = _stringify_if_nonempty(example.get("question"))
    if not question:
        return None
    annotation_type, gold_rewrites = _extract_ambig_qa_gold_rewrites(example, question)
    if annotation_type != "multipleQAs" or len(gold_rewrites) < 2:
        return None
    _, answer_value, _ = _extract_ambig_qa_reference_data(example)
    return {
        "question": question,
        "answer": answer_value,
        "disambiguated_questions": list(gold_rewrites),
    }


def _build_ambig_qa_disambig_fewshot_pools(
    dataset,
    *,
    seed: int,
    count: int = AMBIG_QA_DISAMBIG_FEWSHOT_COUNT,
) -> Tuple[List[int], List[Dict[str, str]], List[Dict[str, object]]]:
    eligible_entries: List[Tuple[int, Dict[str, object]]] = []
    for idx, example in enumerate(dataset):
        demo_entry = _build_ambig_qa_disambig_demo_entry(example)
        if demo_entry is not None:
            eligible_entries.append((int(idx), demo_entry))

    if len(eligible_entries) < int(count):
        raise ValueError(
            f"Need at least {int(count)} ambiguous AmbigQA demo rows with >=2 gold rewrites, "
            f"but found {len(eligible_entries)}."
        )

    rng = np.random.default_rng(int(seed))
    selected_positions = rng.choice(len(eligible_entries), size=int(count), replace=False).tolist()
    selected_rows = [eligible_entries[int(pos)] for pos in selected_positions]
    selected_rows.sort(key=lambda item: int(item[0]))

    selected_indices: List[int] = []
    answer_fewshot_pool: List[Dict[str, str]] = []
    disambig_fewshot_pool: List[Dict[str, object]] = []
    for raw_idx, demo_entry in selected_rows:
        selected_indices.append(int(raw_idx))
        answer_fewshot_pool.append(
            {
                "question": str(demo_entry["question"]),
                "answer": str(demo_entry["answer"]),
            }
        )
        disambig_fewshot_pool.append(
            {
                "question": str(demo_entry["question"]),
                "disambiguated_questions": list(demo_entry["disambiguated_questions"]),
            }
        )

    return selected_indices, answer_fewshot_pool, disambig_fewshot_pool


def _medqa_option_pairs(options) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    if isinstance(options, dict):
        iterable = list(options.items())
    elif isinstance(options, list):
        iterable = options
    else:
        iterable = []

    for idx, item in enumerate(iterable):
        key = ""
        value = ""
        if isinstance(item, dict):
            key = _stringify_if_nonempty(item.get("key"))
            value = _stringify_if_nonempty(item.get("value"))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            key = _stringify_if_nonempty(item[0])
            value = _stringify_if_nonempty(item[1])
        else:
            value = _stringify_if_nonempty(item)

        if not value:
            continue
        if not key:
            if idx < len(string.ascii_uppercase):
                key = string.ascii_uppercase[idx]
            else:
                key = str(idx + 1)
        pairs.append((key, value))

    return pairs


def _render_medqa_question(question, options) -> str:
    question_text = _stringify_if_nonempty(question)
    option_pairs = _medqa_option_pairs(options)
    if not option_pairs:
        return question_text

    option_lines = [f"{key}. {value}" for key, value in option_pairs]
    parts = [question_text, "Options:", *option_lines]
    return "\n".join(part for part in parts if part).strip()


def _extract_medqa_references(example) -> Tuple[str, List[str]]:
    option_pairs = _medqa_option_pairs(example.get("options", []))
    option_map = {key: value for key, value in option_pairs}

    answer_idx = _stringify_if_nonempty(example.get("answer_idx"))
    answer_text = _stringify_if_nonempty(example.get("answer"))
    if not answer_text and answer_idx in option_map:
        answer_text = option_map[answer_idx]

    references = [answer_text]
    if answer_idx:
        references.append(answer_idx)
        if answer_text:
            references.append(f"{answer_idx}. {answer_text}")
            references.append(f"{answer_idx}) {answer_text}")
    return answer_text, _dedupe_text_list(references)


def _normalize_medqa_example(example):
    answer_text, references = _extract_medqa_references(example)
    return {
        "question": _render_medqa_question(example.get("question", ""), example.get("options", [])),
        "answer": answer_text,
        "references": references,
    }


def _extract_gsm8k_final_answer(answer) -> str:
    answer_text = _stringify_if_nonempty(answer)
    if not answer_text:
        return ""

    match = re.search(r"####\s*(.+)$", answer_text, flags=re.MULTILINE | re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    return answer_text


def _normalize_gsm8k_example(example):
    final_answer = _extract_gsm8k_final_answer(example.get("answer", ""))
    return {
        "question": _stringify_if_nonempty(example.get("question", "")),
        "answer": final_answer,
        "references": [final_answer] if final_answer else [],
    }


def _openbookqa_option_pairs(choices) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        if not isinstance(texts, (list, tuple)):
            texts = [texts]

        for idx in range(min(len(labels), len(texts))):
            label = _stringify_if_nonempty(labels[idx])
            text = _stringify_if_nonempty(texts[idx])
            if not text:
                continue
            if not label:
                label = string.ascii_uppercase[idx] if idx < len(string.ascii_uppercase) else str(idx + 1)
            pairs.append((label, text))
        return pairs

    if isinstance(choices, list):
        for idx, item in enumerate(choices):
            label = ""
            text = ""
            if isinstance(item, dict):
                label = _stringify_if_nonempty(item.get("label"))
                text = _stringify_if_nonempty(item.get("text"))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                label = _stringify_if_nonempty(item[0])
                text = _stringify_if_nonempty(item[1])
            else:
                text = _stringify_if_nonempty(item)

            if not text:
                continue
            if not label:
                label = string.ascii_uppercase[idx] if idx < len(string.ascii_uppercase) else str(idx + 1)
            pairs.append((label, text))

    return pairs


def _render_openbookqa_question(question_stem, choices) -> str:
    question_text = _stringify_if_nonempty(question_stem)
    option_pairs = _openbookqa_option_pairs(choices)
    option_lines = [f"{label}. {text}" for label, text in option_pairs]
    return "\n".join(
        [
            f"Question: {question_text}",
            "Options:",
            *option_lines,
        ]
    ).strip()


def _normalize_openbookqa_example(example):
    clarity_raw = example.get("clarity")
    try:
        clarity = float(clarity_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OpenBookQA example has invalid clarity score: {clarity_raw!r}") from exc

    answer_key = _stringify_if_nonempty(example.get("answerKey"))
    return {
        "question": _render_openbookqa_question(
            example.get("question_stem", ""),
            example.get("choices", {}),
        ),
        "answer": answer_key,
        "references": [answer_key] if answer_key else [],
        "ambig_label": int(clarity <= OPENBOOK_QA_AMBIGUITY_THRESHOLD),
    }


def _normalize_situated_qa_is_dependent(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))

    normalized = _stringify_if_nonempty(value).lower()
    if normalized in {"true", "1", "yes"}:
        return 1
    if normalized in {"false", "0", "no"}:
        return 0
    raise ValueError(f"SituatedQA row has invalid is_dependent value: {value!r}")


def _normalize_situated_qa_example(example):
    question = _stringify_if_nonempty(example.get("question", ""))
    if not question:
        raise ValueError(f"SituatedQA row is missing question text: {example}")

    references = _dedupe_text_list(_flatten_nested_text_values(example.get("any_answer", [])))
    if len(references) == 0:
        references = _dedupe_text_list(_flatten_nested_text_values(example.get("answer", [])))
    if len(references) == 0:
        raise ValueError(
            "SituatedQA row is missing both any_answer and answer values: "
            f"{example.get('sample_id', '<missing sample_id>')}"
        )

    normalized = {
        "question": question,
        "answer": references[0],
        "references": references,
        "ambig_label": _normalize_situated_qa_is_dependent(example.get("is_dependent")),
    }
    sample_id = _stringify_if_nonempty(example.get("sample_id"))
    if sample_id:
        normalized["sample_id"] = sample_id
    return normalized


def _sample_dataset_rows(
    dataset: Dataset,
    size: int,
    *,
    shuffle_seed: int,
    order_seed: Optional[int] = None,
) -> Dataset:
    if size < 0:
        raise ValueError("Requested dataset sample size must be >= 0.")
    if size > len(dataset):
        raise ValueError(
            f"Requested {size} samples from a dataset split with only {len(dataset)} rows."
        )
    if size == 0:
        return dataset.select(range(0))

    sampled = dataset.shuffle(seed=shuffle_seed).select(range(size))
    if order_seed is not None:
        sampled = sampled.shuffle(seed=order_seed)
    return sampled


def _resolve_requested_split_size(value, total: int, name: str) -> int:
    if isinstance(value, (int, np.integer)):
        size = int(value)
        if size < 0:
            raise ValueError(f"{name} must be >= 0 when provided as a count.")
        return size

    frac = float(value)
    if not (0.0 <= frac < 1.0):
        raise ValueError(f"{name} as float must be in [0.0, 1.0).")
    return int(total * frac)


class EPNDataCollator:
    """Pad model inputs while preserving EPN metadata fields."""

    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self._base = DataCollatorWithPadding(
            tokenizer,
            padding=padding,
        )

    def __call__(self, features):
        features = [dict(f) for f in features]
        contexts = [f.pop("context") for f in features]
        refs = [f.pop("references") for f in features]
        qs = [f.pop("question") for f in features]
        rewrite_source_questions = None
        if "rewrite_source_question" in features[0]:
            rewrite_source_questions = [f.pop("rewrite_source_question") for f in features]
        disambig_fewshot_pools = None
        if "disambig_fewshot_pool" in features[0]:
            disambig_fewshot_pools = [f.pop("disambig_fewshot_pool") for f in features]
        ambig_labels = None
        ood_labels = None
        if "ambig_label" in features[0]:
            ambig_labels = [f.pop("ambig_label") for f in features]
        if "ood_label" in features[0]:
            ood_labels = [f.pop("ood_label") for f in features]

        batch = self._base(features)
        batch["references"] = refs
        batch["question"] = qs
        if rewrite_source_questions is not None:
            batch["rewrite_source_question"] = rewrite_source_questions
        if disambig_fewshot_pools is not None:
            batch["disambig_fewshot_pool"] = disambig_fewshot_pools
        batch["context"] = contexts
        if ambig_labels is not None:
            batch["ambig_label"] = torch.tensor(ambig_labels, dtype=torch.long)
        if ood_labels is not None:
            batch["ood_label"] = torch.tensor(ood_labels, dtype=torch.long)
        return batch


def build_features(dataset_name, tokenizer, ex):
    dataset_name = _normalize_dataset_name(dataset_name)
    context = ""
    model_name = getattr(tokenizer, "name_or_path", "").lower()
    ambig_label = None
    ood_label = None

    question = ex["question"]
    question_prompt = question
    answer_instruction = "Respond with the answer only, do not explain."

    if dataset_name in OOD_TRIVIA_DATASET_NAMES:
        answers = ex.get("answers", [])
        if isinstance(answers, list):
            references = [str(a) for a in answers if a is not None]
        elif answers is None:
            references = []
        else:
            references = [str(answers)]
        if not references:
            references = [""]
        primary_reference = references[0]
        ood_label = int(ex.get("ood_label", 0))
    elif dataset_name == "gsm8k":
        references = _dedupe_text_list(ex.get("references", []))
        primary_reference = _stringify_if_nonempty(ex.get("answer"))
        if not primary_reference and len(references) > 0:
            primary_reference = references[0]
        if not primary_reference:
            primary_reference = ""
        if len(references) == 0:
            references = [primary_reference] if primary_reference else [""]
    elif dataset_name == "med_qa":
        references = _dedupe_text_list(ex.get("references", []))
        primary_reference = _stringify_if_nonempty(ex.get("answer"))
        if not primary_reference and len(references) > 0:
            primary_reference = references[0]
        if not primary_reference:
            primary_reference = ""
        if len(references) == 0:
            references = [primary_reference] if primary_reference else [""]
        answer_instruction = "Choose the best option and respond with the option text only, do not explain."
    elif dataset_name == "openbook_qa":
        references = _dedupe_text_list(ex.get("references", []))
        primary_reference = _stringify_if_nonempty(ex.get("answer"))
        if not primary_reference and len(references) > 0:
            primary_reference = references[0]
        if not primary_reference:
            primary_reference = ""
        if len(references) == 0:
            references = [primary_reference] if primary_reference else [""]
        ambig_label = int(ex.get("ambig_label", 0))
        answer_instruction = "Think step by step about the science fact and any commonsense knowledge needed. Choose the best option and respond with the option text only, do not explain."
    elif _is_situated_qa_family_dataset(dataset_name):
        question_prompt = question + "? "
        references = _dedupe_text_list(ex.get("references", []))
        primary_reference = _stringify_if_nonempty(ex.get("answer"))
        if not primary_reference and len(references) > 0:
            primary_reference = references[0]
        if not primary_reference:
            primary_reference = ""
        if len(references) == 0:
            references = [primary_reference] if primary_reference else [""]
        ambig_label = int(ex.get("ambig_label", 0))
    elif "truthful_qa" in dataset_name:
        references = ex["correct_answers"]
        primary_reference = ex["best_answer"]
    elif "trivia_qa" in dataset_name:
        answers = ex["answer"]
        references = answers.get("aliases", [])
        if len(references) == 0:
            references = [answers.get("value", "")]
        primary_reference = answers.get("value") or references[0]
    elif "squad" in dataset_name:
        references = ex["answers"]["text"]
        primary_reference = references[0]
        context = ex.get("context", "")
    elif "nq_open" in dataset_name:
        question_prompt = question + "? "
        references = ex["answer"]
        primary_reference = references[0]
    elif "ambig_qa" in dataset_name:
        references, primary_reference, ambig_label = _extract_ambig_qa_reference_data(ex)
    elif dataset_name == "alcuna":
        answers = ex.get("answers", [])
        if isinstance(answers, list):
            references = [str(a) for a in answers if a is not None]
        elif answers is None:
            references = []
        else:
            references = [str(answers)]
        if not references:
            references = [""]
        primary_reference = references[0]
        if "ood_label" in ex:
            ood_label = int(ex.get("ood_label", 0))
    elif "boolq" in dataset_name:
        question_prompt = question + "? Answer True/False. "
        answer_value = ex["answer"]
        if isinstance(answer_value, (bool, np.bool_)):
            answer_text = "True" if bool(answer_value) else "False"
        else:
            answer_text = str(answer_value)
        references = [answer_text]
        primary_reference = answer_text
        context = ex.get("passage", "")
    elif "synthetic" in dataset_name or dataset_name == "temporalood":
        if isinstance(ex["answer"], list):
            references = ex["answer"]
        else:
            references = [ex["answer"]]
        primary_reference = references[0]
        if "ood_label" in ex:
            ood_label = int(ex.get("ood_label", 0))
    else:
        raise NotImplementedError

    use_context = ("squad" in dataset_name or "boolq" in dataset_name) and context != ""
    instruction_sep = "\n\n" if answer_instruction else ""

    if use_context:
        user_content = (
            "Context: {context}\n"
            "Question:{question}\n"
            "Answer the question depending on the context. {instruction}\n"
        ).format(context=context, question=question_prompt, instruction=answer_instruction)
    else:
        user_content = f"{question_prompt}{instruction_sep}{answer_instruction}\n"

    question_for_storage = question
    rewrite_source_question = None
    if dataset_name == AMBIG_QA_DISAMBIG_DATASET_NAME:
        rewrite_source_question = question_for_storage
    elif dataset_name == SITUATED_QA_DISAMBIG_DATASET_NAME:
        rewrite_source_question = question_prompt
    system_prompt = _instruction_tuned_system_prompt(model_name)

    def _attach_optional_labels(payload):
        if ambig_label is not None:
            payload["ambig_label"] = ambig_label
        if ood_label is not None:
            payload["ood_label"] = ood_label
        if "disambig_fewshot_pool" in ex:
            payload["disambig_fewshot_pool"] = ex["disambig_fewshot_pool"]
        return payload

    if _uses_tokenizer_chat_template_model_name(model_name):
        return _attach_optional_labels(
            _build_chat_template_feature_payload(
                tokenizer=tokenizer,
                user_content=user_content,
                references=references,
                system_prompt=system_prompt,
                question=question_for_storage,
                context=context,
                rewrite_source_question=rewrite_source_question,
            )
        )
    elif "gpt-oss-20b" in model_name:
        SYSTEM_LOW = (
            "You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\n"
            "Current date: 2026-01-11\n\n"
            "Reasoning: low\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
        )
        prompt = user_content
        messages_prefix = [
            {"role": "system", "content": SYSTEM_LOW},
            {"role": "user", "content": prompt},
        ]
        prefix_str = tokenizer.apply_chat_template(
            messages_prefix,
            add_generation_prompt=True,
            tokenize=False,
        )
        tok = tokenizer(prefix_str, truncation=True)

        payload = {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "references": references,
            "question": question_for_storage,
            "context": context,
        }
        if rewrite_source_question is not None:
            payload["rewrite_source_question"] = rewrite_source_question
        return _attach_optional_labels(payload)
    system_template_prefix = ""
    if system_prompt:
        system_template_prefix = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{system_prompt}\n"
        )
    prefix_template = (
        system_template_prefix
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        "{question}{instruction_sep}{instruction}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    prefix_template_w_context = (
        system_template_prefix
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        "Context: {context}\n"
        "Question:{question}\n"
        "Answer the question depending on the context. {instruction}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    if "llama-3" in model_name:
        prefix_base = prefix_template_w_context if use_context else prefix_template
    elif _is_legacy_fewshot_prompt_model_name(model_name):
        fewshot_pool = ex.get("fewshot_pool", [])
        if len(fewshot_pool) == 0:
            raise NotImplementedError("fewshot_pool required for llama-2/mistral-7b prompts.")

        def qa_line(sample):
            if use_context:
                return (
                    f"Context: {sample.get('context', '')}\n"
                    f"Question: {sample['question']}\n"
                    f"Answer: {sample['answer']}\n\n"
                )
            if dataset_name == "openbook_qa":
                return (
                    f"{sample['question']}\n"
                    f"{answer_instruction}\n"
                    f"Answer: {sample['answer']}\n\n"
                )
            return f"Question: {sample['question']}\nAnswer: {sample['answer']}\n\n"

        header = "You are a bot that answers questions in a short phrase.\n"
        if system_prompt:
            header = f"{system_prompt}\n{header}"
        fewshot_block = "".join(qa_line(d) for d in fewshot_pool)
        if use_context:
            prefix_base = (
                f"{header}{fewshot_block}"
                "Context: {context}\n"
                "Question: {question}\n"
                f"{answer_instruction}\n"
                "Answer:"
            )
        else:
            if dataset_name == "openbook_qa":
                prefix_base = (
                    f"{header}{fewshot_block}"
                    "{question}\n"
                    f"{answer_instruction}\n"
                    "Answer:"
                )
            else:
                prefix_base = (
                    f"{header}{fewshot_block}"
                    "Question: {question}\n"
                    f"{answer_instruction}\n"
                    "Answer:"
                )
    else:
        raise NotImplementedError

    if use_context:
        message = prefix_base.format(
            context=context,
            question=question_prompt,
            instruction=answer_instruction,
            system_prompt=system_prompt or "",
        )
    else:
        message = prefix_base.format(
            question=question_prompt,
            instruction=answer_instruction,
            instruction_sep=instruction_sep,
            system_prompt=system_prompt or "",
        )

    tok = tokenizer(message, truncation=True)

    payload = {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "references": references,
        "question": question_for_storage,
        "context": context,
    }
    if rewrite_source_question is not None:
        payload["rewrite_source_question"] = rewrite_source_question
    return _attach_optional_labels(payload)


def prepare_splits(
        dataset_name: str,
        tokenizer=None, 
        n_samples: int = 2000,
        n_fewshot: int = 5,
        test_frac: Union[float, int] = 0.10,
        val_frac: Union[float, int] = 0.10,
        seed: int = 42,
        fix_val_test_splits: bool = False,
        feature_fn=None,
):
    """
    Load dataset, remove duplicates when available, build a few-shot pool,
    then split into train/validation/test (fraction or count), and apply `feature_fn`.

    When `fix_val_test_splits=True`, validation/test memberships are deterministic
    across runs and independent of `n_samples`; only train size changes with
    `n_samples`. In this fixed mode, `test_frac` and `val_frac` must be int counts.

    Returns
    -------
    proc_train_ds, proc_valid_ds, proc_test_ds, fewshot_pool
    """

    if feature_fn is None:
        raise ValueError("You must pass a feature-building function.")

    dataset_name = _normalize_dataset_name(dataset_name)

    if fix_val_test_splits:
        if not isinstance(test_frac, (int, np.integer)) or not isinstance(val_frac, (int, np.integer)):
            raise ValueError("When fix_val_test_splits=True, test_frac and val_frac must be int counts.")
        fixed_test_size = int(test_frac)
        fixed_val_size = int(val_frac)
        if fixed_test_size < 0 or fixed_val_size < 0:
            raise ValueError("test_frac and val_frac must be >= 0 when fix_val_test_splits=True.")
        if int(n_samples) < (fixed_test_size + fixed_val_size):
            raise ValueError(
                "n_samples must be >= (test_frac + val_frac) when fix_val_test_splits=True. "
                f"Got n_samples={n_samples}, test_frac={fixed_test_size}, val_frac={fixed_val_size}."
            )

    n_removed = None
    full = None
    fixed_test = None
    stratify_column = None
    prebuilt_fewshot_pool = None
    disambig_fewshot_pool = None
    prebuilt_remainder = None

    if "boolq" in dataset_name:
        raw = load_dataset("google/boolq")
        full = concatenate_datasets([raw["train"], raw["validation"]])
        full, n_removed = remove_dup_questions(full)
        full = full.shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))
    elif "truthful_qa" in dataset_name:
        raw = load_dataset("truthful_qa", "generation")
        full = raw["validation"]
    elif "trivia_qa_split" in dataset_name:
        raw = load_dataset("mandarjoshi/trivia_qa", "rc")
        full = raw["train"]
        full, n_removed = remove_dup_questions(full)
        full = full.shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))
    elif "squad_split" in dataset_name:
        raw = load_dataset("squad_v2")
        # filter out the examples without answers
        full = raw["train"].filter(lambda x: len(x["answers"]["text"]) > 0)
        full, n_removed = remove_dup_questions(full)
        full = full.shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))
    elif "nq_open" in dataset_name:
        raw = load_dataset("google-research-datasets/nq_open")
        full = raw["train"].filter(lambda x: len(x["answer"]) > 0).shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))
    elif dataset_name == AMBIG_QA_DISAMBIG_DATASET_NAME:
        raw = load_dataset("sewon/ambig_qa", "full")
        if fix_val_test_splits:
            full = raw["train"].shuffle(seed=42)
        else:
            if isinstance(test_frac, (int, np.integer)):
                test_count = int(test_frac)
                if test_count < 0:
                    raise ValueError("test_frac must be >= 0 when provided as a count.")
                train_target = (n_samples - test_count) + AMBIG_QA_DISAMBIG_FEWSHOT_COUNT
            elif isinstance(test_frac, (float, np.floating)):
                if not (0.0 <= float(test_frac) < 1.0):
                    raise ValueError("test_frac as float must be in [0.0, 1.0).")
                train_target = n_samples + AMBIG_QA_DISAMBIG_FEWSHOT_COUNT
            else:
                raise TypeError("test_frac must be int or float.")
            if train_target < AMBIG_QA_DISAMBIG_FEWSHOT_COUNT:
                raise ValueError("n_samples is too small for the requested AmbigQA disambiguation demo pool.")
            train_len = len(raw["train"])
            if train_target > train_len:
                raise ValueError(
                    f"Requested {train_target} train samples but ambig_qa train has {train_len}."
                )
            full = raw["train"].shuffle(seed=42).select(range(train_target))
        fixed_test = raw["validation"]
        demo_indices, prebuilt_fewshot_pool, disambig_fewshot_pool = _build_ambig_qa_disambig_fewshot_pools(
            full,
            seed=_split_seed(seed, fix_val_test_splits, "fewshot_membership"),
            count=AMBIG_QA_DISAMBIG_FEWSHOT_COUNT,
        )
        demo_index_set = set(int(idx) for idx in demo_indices)
        keep_idx = [i for i in range(len(full)) if i not in demo_index_set]
        prebuilt_remainder = full.select(keep_idx)
    elif "ambig_qa" in dataset_name:
        raw = load_dataset("sewon/ambig_qa", "full")
        if fix_val_test_splits:
            full = raw["train"].shuffle(seed=42)
        else:
            if isinstance(test_frac, (int, np.integer)):
                test_count = int(test_frac)
                if test_count < 0:
                    raise ValueError("test_frac must be >= 0 when provided as a count.")
                train_target = (n_samples - test_count) + n_fewshot
            elif isinstance(test_frac, (float, np.floating)):
                if not (0.0 <= float(test_frac) < 1.0):
                    raise ValueError("test_frac as float must be in [0.0, 1.0).")
                train_target = n_samples + n_fewshot
            else:
                raise TypeError("test_frac must be int or float.")
            if train_target < n_fewshot:
                raise ValueError("n_samples is too small for the requested test_frac.")
            train_len = len(raw["train"])
            if train_target > train_len:
                raise ValueError(f"Requested {train_target} train samples but ambig_qa train has {train_len}.")
            full = raw["train"].shuffle(seed=42).select(range(train_target))
        fixed_test = raw["validation"]
    elif dataset_name == "gsm8k":
        raw = load_dataset("openai/gsm8k", "main")

        required_splits = ("train", "test")
        missing_splits = [split_name for split_name in required_splits if split_name not in raw]
        if missing_splits:
            raise ValueError(
                "openai/gsm8k is missing required splits: " + ", ".join(missing_splits)
            )

        train_split = raw["train"].map(
            _normalize_gsm8k_example,
            remove_columns=raw["train"].column_names,
        )
        train_split = train_split.filter(
            lambda x: (
                _stringify_if_nonempty(x.get("question")) != ""
                and _stringify_if_nonempty(x.get("answer")) != ""
                and len(_dedupe_text_list(x.get("references", []))) > 0
            )
        )

        test_split = raw["test"].map(
            _normalize_gsm8k_example,
            remove_columns=raw["test"].column_names,
        )
        test_split = test_split.filter(
            lambda x: (
                _stringify_if_nonempty(x.get("question")) != ""
                and _stringify_if_nonempty(x.get("answer")) != ""
                and len(_dedupe_text_list(x.get("references", []))) > 0
            )
        )

        if fix_val_test_splits:
            full = train_split.shuffle(seed=42)
        else:
            if isinstance(test_frac, (int, np.integer)):
                test_count = int(test_frac)
                if test_count < 0:
                    raise ValueError("test_frac must be >= 0 when provided as a count.")
                train_target = (n_samples - test_count) + n_fewshot
            elif isinstance(test_frac, (float, np.floating)):
                if not (0.0 <= float(test_frac) < 1.0):
                    raise ValueError("test_frac as float must be in [0.0, 1.0).")
                train_target = n_samples + n_fewshot
            else:
                raise TypeError("test_frac must be int or float.")
            if train_target < n_fewshot:
                raise ValueError("n_samples is too small for the requested test_frac.")
            train_len = len(train_split)
            if train_target > train_len:
                raise ValueError(f"Requested {train_target} train samples but gsm8k train has {train_len}.")
            full = train_split.shuffle(seed=42).select(range(train_target))
        fixed_test = test_split
    elif dataset_name == "alcuna":
        alcuna_path = os.path.join(WORKING_DIR, "datasets", "ALCUNA", "id2question.json")
        if not os.path.exists(alcuna_path):
            raise FileNotFoundError(f"ALCUNA id2question.json not found: {alcuna_path}")
        with open(alcuna_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = []
        for items in raw.values():
            for item in items:
                if item.get("form") != "fill-in-blank":
                    continue
                question = str(item.get("question", "")).strip()
                answers = item.get("answers", [])
                if isinstance(answers, list):
                    refs = [str(a).strip() for a in answers if a is not None and str(a).strip() != ""]
                elif answers is None:
                    refs = []
                else:
                    answer_text = str(answers).strip()
                    refs = [answer_text] if answer_text != "" else []
                if question != "" and refs:
                    rows.append({"question": question, "answers": refs, "ood_label": 1})
        full = Dataset.from_list(rows)
        if len(full) < n_samples + n_fewshot:
            raise ValueError(
                f"Requested {n_samples + n_fewshot} samples but alcuna has {len(full)} "
                "fill-in-blank entries."
            )
        full = full.shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))
    elif dataset_name == "med_qa":
        raw = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_source",
            trust_remote_code=True,
        )

        required_splits = ("train", "validation", "test")
        missing_splits = [split_name for split_name in required_splits if split_name not in raw]
        if missing_splits:
            raise ValueError(
                "bigbio/med_qa is missing required splits: " + ", ".join(missing_splits)
            )

        medqa_splits = {}
        for split_name in required_splits:
            split_ds = raw[split_name].map(
                _normalize_medqa_example,
                remove_columns=raw[split_name].column_names,
            )
            split_ds = split_ds.filter(
                lambda x: (
                    _stringify_if_nonempty(x.get("question")) != ""
                    and _stringify_if_nonempty(x.get("answer")) != ""
                    and len(_dedupe_text_list(x.get("references", []))) > 0
                )
            )
            medqa_splits[split_name] = split_ds

        train_pool = medqa_splits["train"]
        valid_pool = medqa_splits["validation"]
        test_pool = medqa_splits["test"]

        train_target = int(n_samples)
        test_size = _resolve_requested_split_size(test_frac, int(n_samples), "test_frac")
        val_size = _resolve_requested_split_size(val_frac, int(n_samples), "val_frac")
        train_target -= (test_size + val_size)
        if train_target < 0:
            raise ValueError(
                "n_samples must be >= (requested validation size + requested test size). "
                f"Got n_samples={n_samples}, validation={val_size}, test={test_size}."
            )

        k = min(int(n_fewshot), len(train_pool))
        if train_target > max(0, len(train_pool) - k):
            raise ValueError(
                f"Requested {train_target} MedQA train samples after reserving {k} few-shot examples, "
                f"but only {len(train_pool) - k} remain in the official train split."
            )
        if val_size > len(valid_pool):
            raise ValueError(
                f"Requested {val_size} MedQA validation samples, but the official validation split has {len(valid_pool)}."
            )
        if test_size > len(test_pool):
            raise ValueError(
                f"Requested {test_size} MedQA test samples, but the official test split has {len(test_pool)}."
            )

        if fix_val_test_splits:
            fewshot_seed = _split_seed(seed, True, "fewshot_membership")
            train_seed = _split_seed(seed, True, "train_membership")
            train_order_seed = _split_seed(seed, True, "train_order")
            valid_seed = _split_seed(seed, True, "valid_membership")
            valid_order_seed = _split_seed(seed, True, "valid_order")
            test_seed = _split_seed(seed, True, "test_membership")
            test_order_seed = _split_seed(seed, True, "test_order")
        else:
            fewshot_seed = seed
            train_seed = seed
            train_order_seed = seed
            valid_seed = seed
            valid_order_seed = seed
            test_seed = seed
            test_order_seed = seed

        shuffled_train_pool = train_pool.shuffle(seed=fewshot_seed)
        fewshot_ds = shuffled_train_pool.select(range(k)) if k > 0 else shuffled_train_pool.select(range(0))
        remaining_train_pool = shuffled_train_pool.select(range(k, len(shuffled_train_pool)))

        train_ds = _sample_dataset_rows(
            remaining_train_pool,
            train_target,
            shuffle_seed=train_seed,
            order_seed=train_order_seed,
        )
        valid_ds = _sample_dataset_rows(
            valid_pool,
            val_size,
            shuffle_seed=valid_seed,
            order_seed=valid_order_seed,
        )
        test_ds = _sample_dataset_rows(
            test_pool,
            test_size,
            shuffle_seed=test_seed,
            order_seed=test_order_seed,
        )

        fewshot_pool = [
            {
                "question": item.get("question", ""),
                "answer": _stringify_if_nonempty(item.get("answer")),
            }
            for item in fewshot_ds
        ]

        def attach_pool(ds):
            def add_pool(batch):
                n = len(batch["question"])
                return {"fewshot_pool": [fewshot_pool] * n}
            return ds.map(add_pool, batched=True)

        train_with_pool = attach_pool(train_ds)
        valid_with_pool = attach_pool(valid_ds)
        test_with_pool = attach_pool(test_ds)

        proc_train = train_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=train_with_pool.column_names,
            load_from_cache_file=False,
        )
        proc_valid = valid_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=valid_with_pool.column_names,
            load_from_cache_file=False,
        )
        proc_test = test_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=test_with_pool.column_names,
            load_from_cache_file=False,
        )

        return proc_train, proc_valid, proc_test, fewshot_pool
    elif dataset_name == "openbook_qa":
        raw = load_dataset("allenai/openbookqa", "additional")

        required_splits = ("train", "validation", "test")
        missing_splits = [split_name for split_name in required_splits if split_name not in raw]
        if missing_splits:
            raise ValueError(
                "allenai/openbookqa is missing required splits: " + ", ".join(missing_splits)
            )

        openbook_splits = {}
        for split_name in required_splits:
            split_ds = raw[split_name].map(
                _normalize_openbookqa_example,
                remove_columns=raw[split_name].column_names,
                load_from_cache_file=False,
            )
            split_ds = split_ds.filter(
                lambda x: (
                    _stringify_if_nonempty(x.get("question")) != ""
                    and _stringify_if_nonempty(x.get("answer")) != ""
                    and len(_dedupe_text_list(x.get("references", []))) > 0
                )
            )
            openbook_splits[split_name] = split_ds

        train_pool = openbook_splits["train"]
        valid_pool = openbook_splits["validation"]
        test_pool = openbook_splits["test"]

        train_target = int(n_samples)
        test_size = _resolve_requested_split_size(test_frac, int(n_samples), "test_frac")
        val_size = _resolve_requested_split_size(val_frac, int(n_samples), "val_frac")
        train_target -= (test_size + val_size)
        if train_target < 0:
            raise ValueError(
                "n_samples must be >= (requested validation size + requested test size). "
                f"Got n_samples={n_samples}, validation={val_size}, test={test_size}."
            )

        k = min(int(n_fewshot), len(train_pool))
        if train_target > max(0, len(train_pool) - k):
            raise ValueError(
                f"Requested {train_target} OpenBookQA train samples after reserving {k} few-shot examples, "
                f"but only {len(train_pool) - k} remain in the official train split."
            )
        if val_size > len(valid_pool):
            raise ValueError(
                f"Requested {val_size} OpenBookQA validation samples, but the official validation split has {len(valid_pool)}."
            )
        if test_size > len(test_pool):
            raise ValueError(
                f"Requested {test_size} OpenBookQA test samples, but the official test split has {len(test_pool)}."
            )

        if fix_val_test_splits:
            fewshot_seed = _split_seed(seed, True, "fewshot_membership")
            train_seed = _split_seed(seed, True, "train_membership")
            train_order_seed = _split_seed(seed, True, "train_order")
            valid_seed = _split_seed(seed, True, "valid_membership")
            valid_order_seed = _split_seed(seed, True, "valid_order")
            test_seed = _split_seed(seed, True, "test_membership")
            test_order_seed = _split_seed(seed, True, "test_order")
        else:
            fewshot_seed = seed
            train_seed = seed
            train_order_seed = seed
            valid_seed = seed
            valid_order_seed = seed
            test_seed = seed
            test_order_seed = seed

        shuffled_train_pool = train_pool.shuffle(seed=fewshot_seed)
        fewshot_ds = shuffled_train_pool.select(range(k)) if k > 0 else shuffled_train_pool.select(range(0))
        remaining_train_pool = shuffled_train_pool.select(range(k, len(shuffled_train_pool)))

        train_ds = _sample_dataset_rows(
            remaining_train_pool,
            train_target,
            shuffle_seed=train_seed,
            order_seed=train_order_seed,
        )
        valid_ds = _sample_dataset_rows(
            valid_pool,
            val_size,
            shuffle_seed=valid_seed,
            order_seed=valid_order_seed,
        )
        test_ds = _sample_dataset_rows(
            test_pool,
            test_size,
            shuffle_seed=test_seed,
            order_seed=test_order_seed,
        )

        fewshot_pool = [
            {
                "question": item.get("question", ""),
                "answer": _stringify_if_nonempty(item.get("answer")),
            }
            for item in fewshot_ds
        ]

        def attach_pool(ds):
            def add_pool(batch):
                n = len(batch["question"])
                return {"fewshot_pool": [fewshot_pool] * n}
            return ds.map(add_pool, batched=True)

        train_with_pool = attach_pool(train_ds)
        valid_with_pool = attach_pool(valid_ds)
        test_with_pool = attach_pool(test_ds)

        proc_train = train_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=train_with_pool.column_names,
        )
        proc_valid = valid_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=valid_with_pool.column_names,
        )
        proc_test = test_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=test_with_pool.column_names,
        )

        return proc_train, proc_valid, proc_test, fewshot_pool
    elif dataset_name in OOD_TRIVIA_DATASET_NAMES:
        if fix_val_test_splits:
            test_size = int(test_frac)
            val_size = int(val_frac)
            train_size = int(n_samples) - test_size - val_size
            if test_size < 0 or test_size > n_samples:
                raise ValueError("test_size out of range")
            if val_size < 0 or val_size > (n_samples - test_size):
                raise ValueError("val_size too large for remaining train split")
            if train_size < 0:
                raise ValueError("n_samples is too small for requested val/test sizes.")
        else:
            if isinstance(test_frac, (int, np.integer)):
                test_size = int(test_frac)
                assert 0 <= test_size <= n_samples, "test_size out of range"
                if isinstance(val_frac, (int, np.integer)):
                    val_size = int(val_frac)
                    assert 0 <= val_size <= (n_samples - test_size), "val_size too large for remaining train split"
                    train_size = n_samples - test_size - val_size
                else:
                    val_frac_float = float(val_frac)
                    if not (0.0 <= val_frac_float < 1.0):
                        raise ValueError("val_frac as float must be in [0.0, 1.0).")
                    remainder_dummy = Dataset.from_dict({"idx": list(range(n_samples - test_size))})
                    train_val_split = remainder_dummy.train_test_split(
                        test_size=val_frac_float,
                        seed=seed,
                    )
                    train_size = len(train_val_split["train"])
                    val_size = len(train_val_split["test"])
            else:
                test_frac_float = float(test_frac)
                if not (0.0 <= test_frac_float < 1.0):
                    raise ValueError("test_frac as float must be in [0.0, 1.0).")
                remainder_dummy = Dataset.from_dict({"idx": list(range(n_samples))})
                if test_frac_float > 0.0:
                    tv_split = remainder_dummy.train_test_split(test_size=test_frac_float, seed=seed)
                else:
                    tv_split = DatasetDict({
                        "train": remainder_dummy,
                        "test": remainder_dummy.select(range(0)),
                    })
                if isinstance(val_frac, (int, np.integer)):
                    val_size = int(val_frac)
                    assert 0 <= val_size <= len(tv_split["train"]), "val_size too large for remaining train split"
                    if val_size > 0:
                        train_val_split = tv_split["train"].train_test_split(
                            test_size=val_size,
                            seed=seed,
                        )
                    else:
                        train_val_split = DatasetDict({
                            "train": tv_split["train"],
                            "test": tv_split["train"].select(range(0)),
                        })
                else:
                    val_frac_float = float(val_frac)
                    if not (0.0 <= val_frac_float < 1.0):
                        raise ValueError("val_frac as float must be in [0.0, 1.0).")
                    val_frac_in_train = val_frac_float / (1.0 - test_frac_float)
                    if float(val_frac_in_train) > 0.0:
                        train_val_split = tv_split["train"].train_test_split(
                            test_size=float(val_frac_in_train),
                            seed=seed,
                        )
                    else:
                        train_val_split = DatasetDict({
                            "train": tv_split["train"],
                            "test": tv_split["train"].select(range(0)),
                        })
                train_size = len(train_val_split["train"])
                val_size = len(train_val_split["test"])
                test_size = len(tv_split["test"])

        test_ood_target = (test_size // 2) + (test_size % 2)
        test_trivia_target = test_size - test_ood_target
        train_trivia_target = train_size
        valid_trivia_target = val_size

        if dataset_name == "alcuna_trivia_qa":
            ood_source_name = "alcuna"
            path_candidates = [
                "/datasets/ALCUNA/id2question.json",
                os.path.join(WORKING_DIR, "datasets", "ALCUNA", "id2question.json"),
            ]
            alcuna_path = None
            for candidate in path_candidates:
                if os.path.exists(candidate):
                    alcuna_path = candidate
                    break
            if alcuna_path is None:
                raise FileNotFoundError(
                    "ALCUNA id2question.json not found. Tried: "
                    + ", ".join(path_candidates)
                )
            with open(alcuna_path, "r", encoding="utf-8") as f:
                raw_alcuna = json.load(f)

            ood_rows = []
            for items in raw_alcuna.values():
                for item in items:
                    if item.get("form") != "fill-in-blank":
                        continue
                    question = item.get("question", "")
                    answers = item.get("answers", [])
                    if isinstance(answers, list):
                        refs = [str(a) for a in answers if a is not None and str(a).strip() != ""]
                    elif answers is None:
                        refs = []
                    else:
                        refs = [str(answers)] if str(answers).strip() != "" else []
                    if refs:
                        ood_rows.append({"question": question, "answers": refs, "ood_label": 1})
        elif dataset_name == "temporalood_trivia_qa":
            ood_source_name = "temporalood"
            temporalood_path = os.path.join(
                WORKING_DIR,
                "datasets",
                "TemporalOOD",
                "post_nov_2025_open_domain_qa_2000.json",
            )
            if not os.path.exists(temporalood_path):
                raise FileNotFoundError(f"TemporalOOD dataset not found: {temporalood_path}")
            with open(temporalood_path, "r", encoding="utf-8") as f:
                raw_temporalood = json.load(f)

            ood_rows = []
            for item in raw_temporalood:
                question = item.get("question", "")
                answers = item.get("reference_answer", [])
                if isinstance(answers, list):
                    refs = [str(a) for a in answers if a is not None and str(a).strip() != ""]
                elif answers is None:
                    refs = []
                else:
                    refs = [str(answers)] if str(answers).strip() != "" else []
                if refs:
                    ood_rows.append({"question": question, "answers": refs, "ood_label": 1})
        else:
            ood_source_name = "repliqa"
            raw_repliqa = load_dataset("ServiceNow/repliqa", split="repliqa_0")
            raw_repliqa, n_repli_removed = remove_dup_questions(raw_repliqa)
            if n_repli_removed:
                print(f"Removed {n_repli_removed} duplicate question(s) from repliqa ({dataset_name} OOD source).")

            ood_rows = []
            for example in raw_repliqa:
                question = example.get("question", "")
                question_text = "" if question is None else str(question)
                answer = example.get("answer")
                refs = []
                if isinstance(answer, str):
                    if answer.strip() != "":
                        refs.append(answer)
                elif isinstance(answer, list):
                    for candidate in answer:
                        if isinstance(candidate, str):
                            if candidate.strip() != "" and candidate not in refs:
                                refs.append(candidate)
                        elif candidate is not None:
                            candidate_str = str(candidate)
                            if candidate_str.strip() != "" and candidate_str not in refs:
                                refs.append(candidate_str)
                elif isinstance(answer, dict):
                    for key in ("value", "text", "answer"):
                        val = answer.get(key)
                        if isinstance(val, str):
                            if val.strip() != "" and val not in refs:
                                refs.append(val)
                        elif isinstance(val, list):
                            for alias in val:
                                if isinstance(alias, str) and alias.strip() != "" and alias not in refs:
                                    refs.append(alias)
                elif answer is not None:
                    answer_str = str(answer)
                    if answer_str.strip() != "":
                        refs.append(answer_str)

                if refs:
                    ood_rows.append({"question": question_text, "answers": refs, "ood_label": 1})

        raw_trivia = load_dataset("mandarjoshi/trivia_qa", "rc")
        trivia_full = raw_trivia["train"]
        trivia_full, n_trivia_removed = remove_dup_questions(trivia_full)
        if n_trivia_removed:
            print(f"Removed {n_trivia_removed} duplicate question(s) from trivia_qa ({dataset_name} ID source).")
        trivia_full = trivia_full.filter(
            lambda x: (
                isinstance(x.get("answer"), dict)
                and (
                    (
                        isinstance(x["answer"].get("value"), str)
                        and x["answer"]["value"].strip() != ""
                    )
                    or (
                        isinstance(x["answer"].get("aliases"), list)
                        and any(
                            isinstance(alias, str) and alias.strip() != ""
                            for alias in x["answer"]["aliases"]
                        )
                    )
                )
            )
        )

        rng = np.random.default_rng(_split_seed(seed, fix_val_test_splits, "fewshot_membership"))
        max_k_ood = len(ood_rows) - test_ood_target
        max_k_trivia = len(trivia_full) - (
            train_trivia_target + valid_trivia_target + test_trivia_target
        )
        if max_k_ood < 0:
            raise ValueError(
                f"Need at least {test_ood_target} {ood_source_name} samples for test split, "
                f"but only {len(ood_rows)} are available."
            )
        if max_k_trivia < 0:
            raise ValueError(
                f"Need at least {train_trivia_target + valid_trivia_target + test_trivia_target} "
                f"trivia_qa samples for train/validation/test splits, "
                f"but only {len(trivia_full)} are available."
            )

        # Few-shot pool for OOD trivia datasets is trivia_qa-only by design.
        k_trivia = min(int(n_fewshot), max_k_trivia)

        ood_target = test_ood_target
        trivia_target = (
            train_trivia_target + valid_trivia_target + test_trivia_target
        ) + k_trivia

        ood_selected = Dataset.from_list(ood_rows).shuffle(seed=42).select(range(ood_target))
        trivia_selected = trivia_full.shuffle(seed=42).select(range(trivia_target))

        trivia_rows = []
        for example in trivia_selected:
            answer_obj = example.get("answer", {})
            refs = []
            answer_value = answer_obj.get("value") if isinstance(answer_obj, dict) else None
            if isinstance(answer_value, str) and answer_value.strip() != "":
                refs.append(answer_value)
            aliases = answer_obj.get("aliases", []) if isinstance(answer_obj, dict) else []
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip() != "" and alias not in refs:
                        refs.append(alias)
            if refs:
                trivia_rows.append({"question": example["question"], "answers": refs, "ood_label": 0})
        if len(trivia_rows) != trivia_target:
            raise ValueError(
                f"Constructed {len(trivia_rows)} trivia_qa rows, expected {trivia_target}."
            )
        trivia_selected_rows = Dataset.from_list(trivia_rows)

        fewshot_rows = [trivia_selected_rows[i] for i in range(k_trivia)]
        if fewshot_rows:
            fewshot_perm = rng.permutation(len(fewshot_rows))
            fewshot_rows = [fewshot_rows[i] for i in fewshot_perm]
        fewshot_pool = []
        for item in fewshot_rows:
            answers = item.get("answers", [])
            if isinstance(answers, list) and len(answers) > 0:
                answer_value = str(answers[0])
            elif answers is None:
                answer_value = ""
            else:
                answer_value = str(answers)
            fewshot_pool.append({"question": item.get("question", ""), "answer": answer_value})

        ood_test_pool = ood_selected
        trivia_nonfew_pool = trivia_selected_rows.select(
            range(
                k_trivia,
                k_trivia + train_trivia_target + valid_trivia_target + test_trivia_target,
            )
        )
        if fix_val_test_splits:
            valid_start = 0
            test_start = valid_trivia_target
            train_start = valid_trivia_target + test_trivia_target

            valid_ds = trivia_nonfew_pool.select(
                range(valid_start, valid_start + valid_trivia_target)
            )
            trivia_test_pool = trivia_nonfew_pool.select(
                range(test_start, test_start + test_trivia_target)
            )
            train_ds = trivia_nonfew_pool.select(
                range(train_start, train_start + train_trivia_target)
            )

            train_ds = train_ds.shuffle(seed=_split_seed(seed, True, "train_order"))
            valid_ds = valid_ds.shuffle(seed=_split_seed(seed, True, "valid_order"))
        else:
            train_ds = trivia_nonfew_pool.select(range(0, train_trivia_target)).shuffle(seed=seed)
            valid_ds = trivia_nonfew_pool.select(
                range(train_trivia_target, train_trivia_target + valid_trivia_target)
            ).shuffle(seed=seed)
            trivia_test_pool = trivia_nonfew_pool.select(
                range(
                    train_trivia_target + valid_trivia_target,
                    train_trivia_target + valid_trivia_target + test_trivia_target,
                )
            )

        if test_size > 0:
            # Test split remains stratified by class via class-specific sampling.
            test_ds = concatenate_datasets([ood_test_pool, trivia_test_pool]).shuffle(
                seed=_split_seed(seed, fix_val_test_splits, "test_order")
            )
        else:
            test_ds = train_ds.select(range(0))

        if len(train_ds) != train_size or len(valid_ds) != val_size or len(test_ds) != test_size:
            raise ValueError(
                f"Constructed {dataset_name} splits have unexpected sizes: "
                f"train={len(train_ds)} (expected {train_size}), "
                f"validation={len(valid_ds)} (expected {val_size}), "
                f"test={len(test_ds)} (expected {test_size})."
            )
        if len(train_ds) > 0 and np.any(np.asarray(train_ds["ood_label"], dtype=np.int64) != 0):
            raise ValueError(f"{dataset_name} train split must contain only trivia_qa samples (ood_label=0).")
        if len(valid_ds) > 0 and np.any(np.asarray(valid_ds["ood_label"], dtype=np.int64) != 0):
            raise ValueError(f"{dataset_name} validation split must contain only trivia_qa samples (ood_label=0).")
        if len(test_ds) > 0:
            test_ood = np.asarray(test_ds["ood_label"], dtype=np.int64)
            if int((test_ood == 1).sum()) != test_ood_target:
                raise ValueError(f"{dataset_name} test split must keep stratified OOD/ID sampling.")

        def attach_pool(ds):
            def add_pool(batch):
                n = len(batch["question"])
                return {"fewshot_pool": [fewshot_pool] * n}
            return ds.map(add_pool, batched=True)

        train_with_pool = attach_pool(train_ds)
        valid_with_pool = attach_pool(valid_ds)
        test_with_pool = attach_pool(test_ds)

        proc_train = train_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=train_with_pool.column_names,
        )
        proc_valid = valid_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=valid_with_pool.column_names,
        )
        proc_test = test_with_pool.map(
            functools.partial(feature_fn, dataset_name, tokenizer),
            remove_columns=test_with_pool.column_names,
        )

        return proc_train, proc_valid, proc_test, fewshot_pool
    elif _is_situated_qa_family_dataset(dataset_name):
        if not os.path.exists(SITUATED_QA_DATA_PATH):
            raise FileNotFoundError(f"SituatedQA combined dataset not found: {SITUATED_QA_DATA_PATH}")

        rows = _load_jsonl_rows(SITUATED_QA_DATA_PATH)
        if len(rows) == 0:
            raise ValueError(f"SituatedQA combined dataset is empty: {SITUATED_QA_DATA_PATH}")

        full = Dataset.from_list([_normalize_situated_qa_example(row) for row in rows]).shuffle(seed=42)
        stratify_column = "ambig_label"
        if not fix_val_test_splits:
            requested_total = int(n_samples) + int(n_fewshot)
            if requested_total > len(full):
                raise ValueError(
                    f"Requested {requested_total} SituatedQA rows (n_samples + n_fewshot), "
                    f"but only {len(full)} are available in {SITUATED_QA_DATA_PATH}."
                )
            full = full.select(range(requested_total))
    elif dataset_name in {"synthetic", "temporalood"}:
        temporalood_path = os.path.join(
            WORKING_DIR,
            "datasets",
            "TemporalOOD",
            "post_nov_2025_open_domain_qa_2000.json",
        )
        if not os.path.exists(temporalood_path):
            raise FileNotFoundError(f"TemporalOOD dataset not found: {temporalood_path}")
        with open(temporalood_path, "r", encoding="utf-8") as f:
            raw_temporalood = json.load(f)

        rows = []
        for item in raw_temporalood:
            question = str(item.get("question", "")).strip()
            answers = item.get("reference_answer", [])
            if isinstance(answers, list):
                refs = [str(a).strip() for a in answers if a is not None and str(a).strip() != ""]
            elif answers is None:
                refs = []
            else:
                answer_text = str(answers).strip()
                refs = [answer_text] if answer_text != "" else []
            if question != "" and refs:
                rows.append({"question": question, "answer": refs, "ood_label": 1})
        full = Dataset.from_list(rows)
        if len(full) < n_samples + n_fewshot:
            raise ValueError(
                f"Requested {n_samples + n_fewshot} samples but TemporalOOD has {len(full)} usable entries."
            )
        full = full.shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))
    elif "synthetic_no_instruction" in dataset_name:
        path = os.path.join(WORKING_DIR, "demos", "compare_no_note.json")
        raw = load_dataset("json", data_files=path, split='train')
        full = raw.shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))
    elif "synthetic_w_instruction" in dataset_name:
        path = os.path.join(WORKING_DIR, "demos", "compare_with_note.json")
        raw = load_dataset("json", data_files=path, split='train')
        full = raw.shuffle(seed=42)
        if not fix_val_test_splits:
            full = full.select(range(n_samples + n_fewshot))

    if n_removed is not None:
        print(f"Removed {n_removed} duplicate question(s) from {dataset_name}.")

    if full is None:
        raise NotImplementedError(f"Unsupported dataset_name: {dataset_name!r}")

    if prebuilt_fewshot_pool is not None:
        fewshot_pool = list(prebuilt_fewshot_pool)
        remainder = prebuilt_remainder
    else:
        rng = np.random.default_rng(_split_seed(seed, fix_val_test_splits, "fewshot_membership"))
        k = min(n_fewshot, len(full))
        if dataset_name in OOD_TRIVIA_DATASET_NAMES and "ood_label" in full.column_names:
            ood = np.asarray(full["ood_label"], dtype=np.int64)
            idx_ood = np.where(ood == 1)[0]
            idx_id = np.where(ood == 0)[0]
            rng.shuffle(idx_ood)
            rng.shuffle(idx_id)

            k_ood = min(len(idx_ood), (k // 2) + (k % 2))
            k_id = k - k_ood
            if k_id > len(idx_id):
                k_id = len(idx_id)
                k_ood = k - k_id
            if k_ood > len(idx_ood):
                raise ValueError("Not enough OOD samples to build stratified few-shot pool.")

            few_idx = set(np.concatenate([idx_ood[:k_ood], idx_id[:k_id]]).tolist())
        else:
            few_idx = set(rng.choice(len(full), size=k, replace=False).tolist())

        def _fewshot_entry(example):
            if dataset_name in OOD_TRIVIA_DATASET_NAMES:
                answers = example.get("answers", [])
                if isinstance(answers, list) and len(answers) > 0:
                    answer_value = str(answers[0])
                elif answers is None:
                    answer_value = ""
                else:
                    answer_value = str(answers)
                return {"question": example.get("question", ""), "answer": answer_value}
            if "trivia_qa" in dataset_name:
                answer_value = example["answer"].get("value") or example["answer"].get("aliases", [""])[0]
                return {"question": example["question"], "answer": answer_value}
            if "squad" in dataset_name:
                return {
                    "question": example["question"],
                    "answer": example["answers"]["text"][0],
                    "context": example.get("context", ""),
                }
            if "truthful_qa" in dataset_name:
                return {"question": example["question"], "answer": example["best_answer"]}
            if "nq_open" in dataset_name:
                return {"question": example["question"] + "? ", "answer": example["answer"][0]}
            if "ambig_qa" in dataset_name:
                _, answer_value, _ = _extract_ambig_qa_reference_data(example)
                return {"question": example.get("question", ""), "answer": answer_value}
            if _is_situated_qa_family_dataset(dataset_name):
                return {
                    "question": example.get("question", "") + "? ",
                    "answer": _stringify_if_nonempty(example.get("answer")),
                }
            if dataset_name == "gsm8k":
                return {
                    "question": example.get("question", ""),
                    "answer": _stringify_if_nonempty(example.get("answer")),
                }
            if dataset_name == "openbook_qa":
                return {
                    "question": example.get("question", ""),
                    "answer": _stringify_if_nonempty(example.get("answer")),
                }
            if dataset_name == "alcuna":
                answers = example.get("answers", [])
                if isinstance(answers, list) and len(answers) > 0:
                    answer_value = str(answers[0])
                elif answers is None:
                    answer_value = ""
                else:
                    answer_value = str(answers)
                return {"question": example.get("question", ""), "answer": answer_value}
            if "boolq" in dataset_name:
                answer_value = example["answer"]
                if isinstance(answer_value, (bool, np.bool_)):
                    answer_text = "True" if bool(answer_value) else "False"
                else:
                    answer_text = str(answer_value)
                return {
                    "question": example["question"] + "? Answer True/False. ",
                    "answer": answer_text,
                    "context": example.get("passage", ""),
                }
            if "synthetic" in dataset_name or dataset_name == "temporalood":
                answer = example["answer"][0] if isinstance(example["answer"], list) else example["answer"]
                return {"question": example["question"], "answer": answer}
            return {"question": example.get("question", ""), "answer": example.get("answer", "")}

        fewshot_pool = [_fewshot_entry(full[i]) for i in sorted(few_idx)]
        keep_idx = [i for i in range(len(full)) if i not in few_idx]
        remainder = full.select(keep_idx)

    N = len(remainder)
    if fix_val_test_splits:
        test_size = int(test_frac)
        val_size = int(val_frac)
        train_size_target = int(n_samples) - test_size - val_size
        if train_size_target < 0:
            raise ValueError("n_samples is too small for requested val/test sizes.")

        if fixed_test is not None:
            if val_size > N:
                raise ValueError(
                    f"val_size {val_size} exceeds available train pool size {N}."
                )
            if val_size > 0:
                train_val_split = remainder.train_test_split(
                    test_size=val_size,
                    seed=_split_seed(seed, True, "valid_membership"),
                )
            else:
                train_val_split = DatasetDict({
                    "train": remainder,
                    "test": remainder.select(range(0)),
                })

            train_pool = train_val_split["train"]
            valid_ds = train_val_split["test"]
            if train_size_target > len(train_pool):
                raise ValueError(
                    f"Requested train size {train_size_target} exceeds available pool {len(train_pool)} "
                    "after reserving validation samples."
                )
            if train_size_target > 0:
                train_ds = train_pool.shuffle(
                    seed=_split_seed(seed, True, "train_membership")
                ).select(range(train_size_target))
            else:
                train_ds = train_pool.select(range(0))

            if test_size > len(fixed_test):
                raise ValueError(f"test_size {test_size} exceeds fixed test split size {len(fixed_test)}")
            if test_size > 0:
                test_ds = fixed_test.shuffle(
                    seed=_split_seed(seed, True, "test_membership")
                ).select(range(test_size))
            else:
                test_ds = fixed_test.select(range(0))
        else:
            if stratify_column is not None:
                strat_remainder = remainder.class_encode_column(stratify_column)
                try:
                    if test_size > 0:
                        tv_split = strat_remainder.train_test_split(
                            test_size=test_size,
                            seed=_split_seed(seed, True, "test_membership"),
                            stratify_by_column=stratify_column,
                        )
                    else:
                        tv_split = DatasetDict({
                            "train": strat_remainder,
                            "test": strat_remainder.select(range(0)),
                        })

                    if val_size > 0:
                        train_val_split = tv_split["train"].train_test_split(
                            test_size=val_size,
                            seed=_split_seed(seed, True, "valid_membership"),
                            stratify_by_column=stratify_column,
                        )
                    else:
                        train_val_split = DatasetDict({
                            "train": tv_split["train"],
                            "test": tv_split["train"].select(range(0)),
                        })
                except ValueError as e:
                    raise ValueError(
                        f"Stratified split failed for {dataset_name} with column `{stratify_column}`: {e}"
                    ) from e
            else:
                if test_size > N:
                    raise ValueError(f"test_size {test_size} exceeds available pool size {N}.")
                if val_size > (N - test_size):
                    raise ValueError(
                        f"val_size {val_size} is too large after reserving test_size {test_size} "
                        f"from pool size {N}."
                    )
                if test_size > 0:
                    tv_split = remainder.train_test_split(
                        test_size=test_size,
                        seed=_split_seed(seed, True, "test_membership"),
                    )
                else:
                    tv_split = DatasetDict({
                        "train": remainder,
                        "test": remainder.select(range(0)),
                    })

                if val_size > 0:
                    train_val_split = tv_split["train"].train_test_split(
                        test_size=val_size,
                        seed=_split_seed(seed, True, "valid_membership"),
                    )
                else:
                    train_val_split = DatasetDict({
                        "train": tv_split["train"],
                        "test": tv_split["train"].select(range(0)),
                    })

            train_pool = train_val_split["train"]
            valid_ds = train_val_split["test"]
            test_ds = tv_split["test"]
            if train_size_target > len(train_pool):
                raise ValueError(
                    f"Requested train size {train_size_target} exceeds available pool {len(train_pool)} "
                    "after reserving validation and test samples."
                )
            if train_size_target > 0:
                train_ds = train_pool.shuffle(
                    seed=_split_seed(seed, True, "train_membership")
                ).select(range(train_size_target))
            else:
                train_ds = train_pool.select(range(0))

        train_ds = train_ds.shuffle(seed=_split_seed(seed, True, "train_order"))
        valid_ds = valid_ds.shuffle(seed=_split_seed(seed, True, "valid_order"))
        test_ds = test_ds.shuffle(seed=_split_seed(seed, True, "test_order"))
    else:
        if fixed_test is not None:
            if isinstance(test_frac, (int, np.integer)):
                test_size = int(test_frac)
                assert 0 <= test_size <= n_samples, "test_size out of range"
                if isinstance(val_frac, (int, np.integer)):
                    val_size = int(val_frac)
                    assert 0 <= val_size <= N, "val_size out of range"
                    train_val_split = remainder.train_test_split(test_size=val_size, seed=seed)
                else:
                    train_val_split = remainder.train_test_split(test_size=val_frac, seed=seed)

                train_ds = train_val_split["train"]
                valid_ds = train_val_split["test"]
                if test_size > len(fixed_test):
                    raise ValueError(f"test_size {test_size} exceeds fixed test split size {len(fixed_test)}")
                test_ds = fixed_test.shuffle(seed=seed).select(range(test_size)) if test_size > 0 else fixed_test.select(range(0))
            else:
                val_frac_in_train = val_frac / (1.0 - test_frac)
                tv_split = remainder.train_test_split(test_size=test_frac, seed=seed)
                train_val_split = tv_split["train"].train_test_split(test_size=val_frac_in_train, seed=seed)

                train_ds = train_val_split["train"]
                valid_ds = train_val_split["test"]
                test_size = len(tv_split["test"])
                if test_size > len(fixed_test):
                    raise ValueError(f"test_size {test_size} exceeds fixed test split size {len(fixed_test)}")
                test_ds = fixed_test.shuffle(seed=seed).select(range(test_size)) if test_size > 0 else fixed_test.select(range(0))
        else:
            if stratify_column is not None:
                strat_remainder = remainder.class_encode_column(stratify_column)
                try:
                    if isinstance(test_frac, (int, np.integer)):
                        test_size = int(test_frac)
                        val_size = int(val_frac)
                        assert 0 <= test_size <= N, "test_size out of range"
                        assert 0 <= val_size <= (N - test_size), "val_size too large for remaining train split"

                        if test_size > 0:
                            tv_split = strat_remainder.train_test_split(
                                test_size=test_size,
                                seed=seed,
                                stratify_by_column=stratify_column,
                            )
                        else:
                            tv_split = DatasetDict({
                                "train": strat_remainder,
                                "test": strat_remainder.select(range(0)),
                            })

                        if val_size > 0:
                            train_val_split = tv_split["train"].train_test_split(
                                test_size=val_size,
                                seed=seed,
                                stratify_by_column=stratify_column,
                            )
                        else:
                            train_val_split = DatasetDict({
                                "train": tv_split["train"],
                                "test": tv_split["train"].select(range(0)),
                            })
                    else:
                        val_frac_in_train = val_frac / (1.0 - test_frac)
                        if float(test_frac) > 0.0:
                            tv_split = strat_remainder.train_test_split(
                                test_size=test_frac,
                                seed=seed,
                                stratify_by_column=stratify_column,
                            )
                        else:
                            tv_split = DatasetDict({
                                "train": strat_remainder,
                                "test": strat_remainder.select(range(0)),
                            })

                        if float(val_frac_in_train) > 0.0:
                            train_val_split = tv_split["train"].train_test_split(
                                test_size=val_frac_in_train,
                                seed=seed,
                                stratify_by_column=stratify_column,
                            )
                        else:
                            train_val_split = DatasetDict({
                                "train": tv_split["train"],
                                "test": tv_split["train"].select(range(0)),
                            })
                except ValueError as e:
                    raise ValueError(
                        f"Stratified split failed for {dataset_name} with column `{stratify_column}`: {e}"
                    ) from e

                train_ds = train_val_split["train"]
                valid_ds = train_val_split["test"]
                test_ds = tv_split["test"]
            else:
                if isinstance(test_frac, (int, np.integer)):
                    test_size = int(test_frac)
                    val_size = int(val_frac)
                    assert 0 <= test_size <= N, "test_size out of range"
                    assert 0 <= val_size <= (N - test_size), "val_size too large for remaining train split"
                    tv_split = remainder.train_test_split(test_size=test_size, seed=seed)
                    train_val_split = tv_split["train"].train_test_split(test_size=val_size, seed=seed)
                else:
                    val_frac_in_train = val_frac / (1.0 - test_frac)
                    tv_split = remainder.train_test_split(test_size=test_frac, seed=seed)
                    train_val_split = tv_split["train"].train_test_split(test_size=val_frac_in_train, seed=seed)

                train_val_ds = DatasetDict({
                    "train":      train_val_split["train"],
                    "validation": train_val_split["test"],
                })
                train_ds = train_val_ds["train"]
                valid_ds = train_val_ds["validation"]
                test_ds = tv_split["test"]

    def attach_pool(ds):
        def add_pool(batch):
            n = len(batch["question"])
            payload = {"fewshot_pool": [fewshot_pool] * n}
            if disambig_fewshot_pool is not None:
                payload["disambig_fewshot_pool"] = [disambig_fewshot_pool] * n
            return payload
        return ds.map(add_pool, batched=True)

    train_with_pool = attach_pool(train_ds)
    valid_with_pool = attach_pool(valid_ds)
    test_with_pool = attach_pool(test_ds)

    proc_train = train_with_pool.map(
        functools.partial(feature_fn, dataset_name, tokenizer),
        remove_columns=train_with_pool.column_names,
    )
    proc_valid = valid_with_pool.map(
        functools.partial(feature_fn, dataset_name, tokenizer),
        remove_columns=valid_with_pool.column_names,
    )
    proc_test = test_with_pool.map(
        functools.partial(feature_fn, dataset_name, tokenizer),
        remove_columns=test_with_pool.column_names,
    )

    return proc_train, proc_valid, proc_test, fewshot_pool


def _extract_disambig_fewshot_pool_from_processed_splits(*datasets) -> Optional[List[Dict[str, object]]]:
    for dataset in datasets:
        if dataset is None or len(dataset) == 0:
            continue
        column_names = list(getattr(dataset, "column_names", []))
        if "disambig_fewshot_pool" not in column_names:
            continue
        sample = dataset[0]
        pool = sample.get("disambig_fewshot_pool")
        if isinstance(pool, list) and len(pool) > 0:
            return list(pool)
    return None


def collect_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer,
    dataset,                              # processed split from prepare_splits(...)[i]
    collator,                             # EPNDataCollator(tokenizer)
    cfg: CollectConfig = CollectConfig(),
    accelerator: Optional[Accelerator] = None,
    global_indices: Optional[List[int]] = None,
    expected_total: Optional[int] = None,
    exchange_dir: Optional[str] = None,
    split_name: str = "split",
):
    """
    Returns
          X    : dict with
            - "tbg": (N, L, H) all-layers hidden at last prompt token
            - "slt": (N, L, H) all-layers hidden at second-last generated token
          y    : (N,)  0/1 per-example correctness label (see label_via)
          score: dict with F1 score and gpt5-mini/rougeL accuracy score arrays
          gen  : list of generated responses
          sequence_scores : dict with per-response sequence score arrays

    cfg expectations (with safe defaults if absent):
        - device:               "cuda"
        - batch_size:           int
        - max_new_tokens:       int
        - do_sample:            bool
        - temperature:          float
        - top_p:                float
        - judge_type:           "gpt5-mini" | "rouge"
        - gpt_judge_workers:    int, only used for in-batch gpt5-mini judging
    """
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    local_records = []
    scores = {"accuracy": [], "f1": []}
    y = []

    if global_indices is None:
        global_indices = list(range(len(dataset)))

    is_main = accelerator is None or accelerator.is_main_process
    show_progress = accelerator is None or accelerator.is_local_main_process
    device = _resolve_model_input_device(model, accelerator=accelerator, fallback_device=cfg.device)

    # eos_ids = set() # eos id set
    # if tokenizer.eos_token_id is not None:
    #     eos_ids.add(int(tokenizer.eos_token_id))
    # try:
    #     eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #     if isinstance(eot_id, int) and eot_id >= 0:
    #         eos_ids.add(int(eot_id))
    # except Exception:
    #     pass

    rouge = None
    if getattr(cfg, "judge_type", "rouge") == "rouge":
        rouge = load("rouge")
        rouge_threshold = 0.30
    elif cfg.judge_type == "gpt5-mini":
        # user_msg = (
        #     'Given the question:\n{question}\n\n'
        #     'The ground truth reference answers are: {references}. '
        #     'The following is the answer from a LLM: {answer}\n'
        #     'Is the answer correct? Reply with exactly "True" if the LLM\'s answer '
        #     'matches the true answer, otherwise reply with exactly "False".'
        # )

        # user_msg_w_context = (
        #     'Given the context:\n{context}\n\n'
        #     'Given the question:\n{question}\n\n'
        #     'The ground truth reference answers are: {references}. '
        #     'The following is the answer from a LLM: {answer}\n'
        #     'Is the answer correct? Reply with exactly "True" if the LLM\'s answer '
        #     'matches the true answer, otherwise reply with exactly "False".'
        # )
        user_msg = (
            'We are assessing the quality of answers to the following question:\n{question}\n'
            'The expected answer is: {references}\n'
            'The proposed answer is: {answer}\n'
            'Within the context of the question, does the proposed answer mean the same as the expected answer? '
            'Respond only with True or False. '
        )

        user_msg_w_context = (
            'Given the context:\n{context}\n'
            'We are assessing the quality of answers to the following question:\n{question}\n'
            'The expected answer is: {references}\n'
            'The proposed answer is: {answer}\n'
            'Within the context of the question, does the proposed answer mean the same as the expected answer? '
            'Respond only with True or False. '
        )

    skipped = 0
    offset = 0
    for batch in tqdm(dl, disable=not show_progress):
        with torch.no_grad():
            input_ids   = batch["input_ids"].to(device)
            attn        = batch["attention_mask"].to(device)
            batch_size  = input_ids.size(0)
            references = batch["references"]
            prompt_lens = attn.sum(dim=1)
            input_seq_len = input_ids.size(1)

            # collecting TBG hidden_states
            preds_tbg = model(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states: tuple [emb, layer1, ..., layerN], each (B, Lp, H)
            hidden_states_tbg = preds_tbg.hidden_states[1:] # remove input embeddings
            tbg_batch = torch.stack(
                [layer[:, -1, :] for layer in hidden_states_tbg], dim=0
            ).to(torch.float32).cpu() # [n_layer, n_batch, n_hidden] at TBG token

            n_layers_cfg = _get_num_hidden_layers(model)
            assert tbg_batch.shape[0] == n_layers_cfg

            # Keep the legacy stop criteria on the older few-shot prompt backbones only.
            stopping_criteria = None
            if _is_legacy_fewshot_prompt_model_name(model.name_or_path):
                stopping_criteria = StoppingCriteriaList([
                    StoppingCriteriaSub(
                        stops=['\n\n\n\n\n', '\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:'],
                        initial_length=len(input_ids[0]),
                        tokenizer=tokenizer
                    )
                ])
                print(f"[Stop Criteria] Applying stopping criteria for {model.name_or_path}.")

            # collecting SLT hidden_states
            if "gpt-oss" in model.name_or_path.lower():
                stop_token_ids = encoding.stop_tokens_for_assistant_actions()
            else:
                stop_token_ids = tokenizer.eos_token_id
            
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=cfg.max_new_tokens if "gpt-oss" not in model.name_or_path.lower() else 200,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )
            preds = gen.sequences
            transition_scores = model.compute_transition_scores(
                gen.sequences,
                gen.scores,
                normalize_logits=True,
            )
            hidden_states_slt = getattr(gen, "hidden_states", None) # tuple [n_generated_tokens][n_layer + 1]
            n_layers = len(hidden_states_slt[0]) - 1

            # decoding & hallucination labeling
            batch_judge_records = []
            batch_judge_jobs = []
            for i in range(batch_size):
                row = preds[i]
                curr_prompt_len = int(prompt_lens[i])
                # Previous slicing logic (kept for reference):
                # mask_non_padding = (
                #     (row != tokenizer.pad_token_id)
                #     if tokenizer.pad_token_id is not None
                #     else (row != tokenizer.eos_token_id)
                # )
                # idx_non_padding = mask_non_padding.nonzero(as_tuple=False).squeeze(1)
                # prompt_ids = row.index_select(0, idx_non_padding[:curr_prompt_len])
                # pred_ids = row.index_select(0, idx_non_padding[curr_prompt_len:])
                # Slice generated tokens by absolute input length. This avoids boundary drift
                # when pad_token_id == eos_token_id and chat templates include EOS-like tokens.
                pred_ids = row[input_seq_len:]
                if "gpt-oss" in model.name_or_path.lower():
                    pred_txt = _decode_gpt_oss_completion(pred_ids)
                else:
                    pred_txt = tokenizer.decode(pred_ids, skip_special_tokens=True)

                pred_token_count = int(pred_ids.numel())
                effective_len = min(pred_token_count, len(hidden_states_slt))
                slt_idx = max(effective_len - 2, 0)
                ts_row = transition_scores[i]
                valid_ts = torch.isfinite(ts_row)
                seq_logprob_tensor = ts_row.masked_fill(~valid_ts, 0).sum()
                seq_len_tensor = valid_ts.sum().clamp_min(1).to(torch.float32)
                length_normalized_sequence_logprob_tensor = seq_logprob_tensor / seq_len_tensor
                sequence_prob_tensor = torch.exp(seq_logprob_tensor)
                slt_row = torch.stack(
                    [hidden_states_slt[slt_idx][l + 1][i, -1, :].to(torch.float32) for l in range(n_layers)],
                    dim=0
                ).cpu() # [n_layers, n_hidden]
                f1_value = float(_f1_max_over_refs(pred_txt, references[i]))
                global_idx = int(global_indices[offset + i])
                record = {
                    "sample_idx": global_idx,
                    "tbg": tbg_batch[:, i, :].numpy().astype(np.float32),
                    "slt": slt_row.numpy().astype(np.float32),
                    "f1": f1_value,
                    "pred": pred_txt,
                    "sequence_logprob": float(seq_logprob_tensor.item()),
                    "sequence_prob": float(sequence_prob_tensor.item()),
                    "length_normalized_sequence_logprob": float(
                        length_normalized_sequence_logprob_tensor.item()
                    ),
                }

                # estimate accu & token-level F1 score
                if cfg.judge_type == "rouge":
                    rouge_scores = [
                        rouge.compute(
                            predictions=[pred_txt],
                            references=[alias],
                            use_aggregator=False  # returns list[dict]
                        )["rougeL"]
                        for alias in references[i]
                    ]
                    max_r = max(rouge_scores)[0]
                    judge_value = float(max_r >= rouge_threshold)
                    record["y"] = int(judge_value)
                    record["accuracy"] = float(max_r)
                    local_records.append(record)
                    y.append(judge_value)
                    scores["accuracy"].append(float(max_r))
                    scores["f1"].append(f1_value)
                elif cfg.judge_type == "gpt5-mini":
                    question = batch["question"][i]
                    context = batch.get("context", [""] * batch_size)[i]
                    if context == "":
                        chat = [{"role": "user", "content": user_msg.format(
                            question=question, references="\n".join(references[i]), answer=pred_txt
                        )}]
                    else:
                        chat = [{"role": "user", "content": user_msg_w_context.format(
                            context=context, question=question, references="\n".join(references[i]), answer=pred_txt
                        )}]
                    batch_judge_jobs.append((len(batch_judge_records), chat, global_idx))
                    batch_judge_records.append(record)

            if cfg.judge_type == "gpt5-mini" and batch_judge_records:
                max_workers = max(
                    1,
                    int(getattr(cfg, "gpt_judge_workers", DEFAULT_GPT5_MINI_JUDGE_WORKERS)),
                )
                max_workers = min(max_workers, len(batch_judge_jobs))
                judge_results = [None] * len(batch_judge_records)
                if max_workers == 1:
                    for record_pos, chat, sample_idx in batch_judge_jobs:
                        judge_results[record_pos] = _judge_feature_collection_single(chat, sample_idx)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_pos = {
                            executor.submit(
                                _judge_feature_collection_single,
                                chat,
                                sample_idx,
                            ): record_pos
                            for record_pos, chat, sample_idx in batch_judge_jobs
                        }
                        for future in as_completed(future_to_pos):
                            judge_results[future_to_pos[future]] = future.result()

                for record, result in zip(batch_judge_records, judge_results):
                    if result is None:
                        raise RuntimeError("Missing gpt5-mini judge result for one batch record.")
                    judge_value, content_filtered = result
                    if content_filtered:
                        skipped += 1
                    record["y"] = int(judge_value)
                    record["accuracy"] = float(judge_value)
                    local_records.append(record)
                    y.append(float(judge_value))
                    scores["accuracy"].append(float(judge_value))
                    scores["f1"].append(float(record["f1"]))
        offset += batch_size

    gathered_records = _gather_records_to_main(
        local_records,
        accelerator,
        exchange_dir=exchange_dir,
        split_name=split_name,
    )
    if accelerator is not None and not accelerator.is_main_process:
        return None, None, None, None, None
    gathered_records = sorted(gathered_records, key=lambda row: row["sample_idx"])

    sample_indices = [int(row["sample_idx"]) for row in gathered_records]
    if expected_total is None:
        expected_total = len(sample_indices)
    expected_indices = list(range(int(expected_total)))
    if sample_indices != expected_indices:
        raise RuntimeError(
            "Sample index alignment failed in feature collection. "
            f"Expected indices 0..{int(expected_total) - 1}, "
            f"got {len(sample_indices)} samples with first indices {sample_indices[:10]}."
        )

    if skipped > 0 and is_main:
        print(f"[collect] skipped {skipped} samples due to content filter.")

    if len(gathered_records) == 0:
        n_layers = _get_num_hidden_layers(model)
        hidden_dim = _get_hidden_size(model)
        X = {
            "tbg": np.zeros((n_layers, 0, hidden_dim), dtype=np.float32),
            "slt": np.zeros((n_layers, 0, hidden_dim), dtype=np.float32),
        }
        scores = {
            "accuracy": np.asarray([], dtype=np.float32),
            "f1": np.asarray([], dtype=np.float32),
        }
        y = np.asarray([], dtype=np.int64)
        pred_sequence_scores = {
            "sequence_logprob": np.asarray([], dtype=np.float32),
            "sequence_prob": np.asarray([], dtype=np.float32),
            "length_normalized_sequence_logprob": np.asarray([], dtype=np.float32),
        }
        return X, y, scores, [], pred_sequence_scores

    X = {
        "tbg": np.stack([row["tbg"] for row in gathered_records], axis=1).astype(np.float32), # (n_layers, n_questions, n_hidden)
        "slt": np.stack([row["slt"] for row in gathered_records], axis=1).astype(np.float32), # (n_layers, n_questions, n_hidden)
    }
    scores = {
        "accuracy": np.asarray([row["accuracy"] for row in gathered_records], dtype=np.float32),
        "f1": np.asarray([row["f1"] for row in gathered_records], dtype=np.float32),
    }
    y = np.asarray([row["y"] for row in gathered_records], dtype=np.int64)
    preds_decoded = [row["pred"] for row in gathered_records]
    pred_sequence_scores = {
        "sequence_logprob": np.asarray(
            [row["sequence_logprob"] for row in gathered_records],
            dtype=np.float32,
        ),
        "sequence_prob": np.asarray(
            [row["sequence_prob"] for row in gathered_records],
            dtype=np.float32,
        ),
        "length_normalized_sequence_logprob": np.asarray(
            [row["length_normalized_sequence_logprob"] for row in gathered_records],
            dtype=np.float32,
        ),
    }

    return X, y, scores, preds_decoded, pred_sequence_scores


def main(args):
    accelerator = Accelerator()
    set_seed(42, device_specific=False)
    dataset_name = _normalize_dataset_name(args.dataset_name)

    if dataset_name == AMBIG_QA_FILTERED_DATASET_NAME:
        if accelerator.is_main_process:
            _collect_ambig_qa_filtered_from_existing_run(args)
        return
    if dataset_name == AMBIG_QA_FILTERED_VER2_DATASET_NAME:
        if accelerator.is_main_process:
            _collect_ambig_qa_filtered_ver2_from_existing_run(args)
        return
    if dataset_name == AMBIG_QA_FILTERED_VER3_DATASET_NAME:
        if accelerator.is_main_process:
            _collect_ambig_qa_filtered_ver3_from_existing_run(args)
        return

    # loading model
    resolved_model_name = _resolve_hf_model_name(args.model_name)
    model_id = args.model_name.split("/")[-1]
    needs_8bit = _needs_8bit_quantized_loading(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name, torch_dtype=torch.bfloat16, padding_side='left', verbose=False,
    )
    try:
        if needs_8bit:
            _ensure_bitsandbytes_available()
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            device_map = _resolve_single_gpu_device_map(accelerator=accelerator)
            if accelerator.is_main_process:
                print(f"[int8] Loading {resolved_model_name} with device_map={device_map}.")
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                device_map=device_map,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(resolved_model_name, torch_dtype=torch.bfloat16)
            model.to(accelerator.device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                "Model must fit on a single GPU per process for Accelerator data-parallel collection. "
                "Use a smaller model or lower memory settings and retry."
            ) from e
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.pad_token_id  = tokenizer.eos_token_id

    proc_train_ds, proc_valid_ds, proc_test_ds, _ = prepare_splits(
        dataset_name      = dataset_name,
        tokenizer         = tokenizer,
        n_samples         = args.n_samples,
        n_fewshot         = args.n_fewshot,
        test_frac         = args.test_frac,
        val_frac          = args.valid_frac,
        seed              = 42,
        fix_val_test_splits = args.fix_val_test_splits,
        feature_fn        = build_features,
    )
    if accelerator.is_main_process:
        print(f"Dataset split: {len(proc_train_ds)}, {len(proc_valid_ds)}, {len(proc_test_ds)}.")
        print(
            f"Total references: {np.sum([len(item['references']) for item in proc_train_ds])}, "
            f"{np.sum([len(item['references']) for item in proc_valid_ds])}, "
            f"{np.sum([len(item['references']) for item in proc_test_ds])}"
        )

    data_collator = EPNDataCollator(tokenizer, True)

    # collecting hidden_states and hallucination labels & saving
    output_dir = _feature_output_dir(
        model_id=model_id,
        dataset_name=dataset_name,
        n_samples=args.n_samples,
        judge_type=args.judge_type,
    )
    exchange_dir = os.path.join(output_dir, ".dist_exchange")
    if accelerator.is_main_process:
        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)
        if not os.path.exists(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(exchange_dir).mkdir(parents=True, exist_ok=True)
        run_token = _sanitize_token(
            f"{_dist_run_token()}_{int(time.time())}_{os.getpid()}_{random.randint(0, 999999)}"
        )
        run_token_path = Path(exchange_dir) / "run_token.txt"
        run_token_tmp = run_token_path.with_suffix(".tmp")
        with open(run_token_tmp, "w", encoding="utf-8") as f:
            f.write(run_token)
        os.replace(run_token_tmp, run_token_path)
        print(f"[collect] run token: {run_token}")
    accelerator.wait_for_everyone()


    cfg = CollectConfig()
    cfg.device = str(accelerator.device)
    cfg.judge_type = args.judge_type
    cfg.gpt_judge_workers = max(1, int(args.gpt_judge_workers))
    if cfg.judge_type == "gpt5-mini" and accelerator.is_main_process:
        print(f"[gpt5-mini judge] in-batch workers per process: {cfg.gpt_judge_workers}")
    if needs_8bit and int(getattr(cfg, "batch_size", 1)) > 1:
        cfg.batch_size = 1
        if accelerator.is_main_process:
            print("[int8] Overriding feature collection batch_size to 1 for memory headroom.")
    shapes = {}
    dataset_stats = []
    train_stats = _collect_and_save("train", proc_train_ds, model, tokenizer, data_collator, cfg, output_dir, accelerator=accelerator)
    valid_stats = _collect_and_save("valid", proc_valid_ds, model, tokenizer, data_collator, cfg, output_dir, accelerator=accelerator)
    test_stats = _collect_and_save("test",  proc_test_ds,  model, tokenizer, data_collator, cfg, output_dir, accelerator=accelerator)

    if not accelerator.is_main_process:
        return

    for split_name, split_stats in [("train", train_stats), ("valid", valid_stats), ("test", test_stats)]:
        shapes[split_name] = {
            "tbg_shape": split_stats["tbg_shape"],
            "slt_shape": split_stats["slt_shape"],
            "N": split_stats["N"],
        }
        dataset_stats.append(
            {
                "split": split_name,
                "accuracy": split_stats["accuracy"],
                "f1": split_stats["f1"],
            }
        )

    _write_jsonl(Path(output_dir) / "dataset_stat.jsonl", dataset_stats)

    disambig_fewshot_pool_meta = _extract_disambig_fewshot_pool_from_processed_splits(
        proc_train_ds,
        proc_valid_ds,
        proc_test_ds,
    )
    meta = {
        "model_name": args.model_name,
        "dataset_name": dataset_name,
        "n_samples": int(args.n_samples),
        "judge_type": args.judge_type,
        "split_args": {
            "valid_frac": args.valid_frac,
            "test_frac": args.test_frac,
            "fix_val_test_splits": bool(args.fix_val_test_splits),
        },
        "gen_kwargs": {
            "max_new_tokens": getattr(cfg, "max_new_tokens", None),
            "do_sample": getattr(cfg, "do_sample", None),
            "temperature": getattr(cfg, "temperature", None),
            "top_p": getattr(cfg, "top_p", None),
        },
        "judge_kwargs": {
            "gpt_judge_workers": getattr(cfg, "gpt_judge_workers", None),
        },
        "tokenizer": {
            "padding_side": tokenizer.padding_side,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "shapes": shapes,
        "dtypes": {
            "hidden_states_saved_as": "float32 in X.npz",
            "scores_dtype": "float32",
            "labels_dtype": "int64",
        },
        "paths": {
            "base_dir": output_dir,
            "train": {k: os.path.join(output_dir, "train", k) for k in ["X.npz", "y.npy", "acc.npy", "f1.npy", "preds.jsonl", "preds_sequence_scores.npz"]},
            "valid": {k: os.path.join(output_dir, "valid", k) for k in ["X.npz", "y.npy", "acc.npy", "f1.npy", "preds.jsonl", "preds_sequence_scores.npz"]},
            "test":  {k: os.path.join(output_dir, "test",  k) for k in ["X.npz", "y.npy", "acc.npy", "f1.npy", "preds.jsonl", "preds_sequence_scores.npz"]},
            "dataset_stats": os.path.join(output_dir, "dataset_stat.jsonl"),
        },
    }
    if disambig_fewshot_pool_meta is not None:
        meta["disambig_fewshot_pool"] = disambig_fewshot_pool_meta
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Finished collecting features] Saved to:", output_dir)
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature Collector for EUP")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", type=str,
                        default="trivia_qa_split",
                        help="Dataset name (e.g., trivia_qa_split, squad_split, nq_open, boolq, ambig_qa, ambig_qa_disambig, ambig_qa_filtered, ambig_qa_filtered_ver2, ambig_qa_filtered_ver3, situated_qa, situated_qa_disambig, gsm8k, med_qa, openbook_qa, alcuna, temporalood, alcuna_trivia_qa, repli_qa_trivia_qa, temporalood_trivia_qa)")
    parser.add_argument("--judge_type", type=str, default="rouge")
    parser.add_argument(
        "--gpt_judge_workers",
        type=int,
        default=DEFAULT_GPT5_MINI_JUDGE_WORKERS,
        help="Per-process thread workers for in-batch gpt5-mini judging.",
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_fewshot", type=int, default=5)
    parser.add_argument("--valid_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--test_frac", type=count_or_frac, default=0.1)
    parser.add_argument(
        "--fix_val_test_splits",
        type=_bool,
        default=False,
        help="Use deterministic validation/test membership across runs; requires int --valid_frac/--test_frac.",
    )

    args = parser.parse_args()

    main(args)
