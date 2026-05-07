'''
For generating and collecting sampled responses during training and inference.
'''

import os, re, string, pickle, sys, json, random, time, importlib.util
from typing import Dict, List, Tuple, Optional, Union, Sequence
from pathlib import Path
from distutils.util import strtobool as _bool
import functools
from datetime import timedelta
from datasets import load_dataset, DatasetDict
import torch, numpy as np
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, \
    DataCollatorWithPadding, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb, argparse 
from openai import AzureOpenAI
from openai import BadRequestError
from config import SESamplingConfig
from feature_collector import prepare_splits, build_features, \
    AMBIG_QA_DISAMBIG_DATASET_NAME, \
    SITUATED_QA_DISAMBIG_DATASET_NAME, \
    EPNDataCollator, _instruction_tuned_system_prompt, \
    _is_legacy_fewshot_prompt_model_name, \
    _needs_8bit_quantized_loading, _resolve_hf_model_name, \
    _uses_tokenizer_chat_template_model_name, \
    _is_gemma_4_e4b_model_name, \
    _extract_disambig_fewshot_pool_from_processed_splits
from helper_functions import save_cluster_embeddings_npz, load_cluster_embeddings_npz, load_X_npz, \
    StoppingCriteriaSub, load_pred_json, load_question_json, load_resp_json, \
    count_or_frac, compute_ood_refusal_filter, combine_optional_eval_mask, \
    save_sequence_scores_npz, get_sampled_sequence_scores_filename
from eup_data import ClusterCountDataset, cluster_count_collate, build_dataset, weighted_cluster_count_collate
from eup import EvidentialUncertaintyProbe
from uncertainty import dissonance, vacuity
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from transformers import Gemma4ForCausalLM
except ImportError:
    Gemma4ForCausalLM = None

import logging
for name in ("transformers", "openai", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)

WORKING_DIR = "./"
OPENAI_API_CONFIG_PATH = Path(__file__).resolve().with_name("openai_api.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAMA_33_70B_SUFFIX = "llama-3.3-70b-instruct"
AMBIG_QA_DISAMBIG_REWRITE_SYSTEM_PROMPT = (
    "You are a helpful assistant who rewrites ambiguous questions into one unambiguous question."
)
AMBIG_QA_DISAMBIG_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant who answers questions truthfully."
)


def _is_llama_33_70b_model_name(model_name: str) -> bool:
    norm = str(model_name or "").strip().lower().rstrip("/")
    if not norm:
        return False
    basename = norm.split("/")[-1]
    return basename.endswith(LLAMA_33_70B_SUFFIX) or norm.endswith(LLAMA_33_70B_SUFFIX)


def _is_qwen3_30b_model_name(model_name: str) -> bool:
    norm = str(model_name or "").strip().lower().rstrip("/")
    return "qwen3-30b-a3b-instruct-2507" in norm


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


def _split_indices_for_process(
    n_total: int,
    accelerator: Optional[Accelerator],
    batch_size: int = 1,
    force_equal_batches: bool = False,
) -> List[int]:
    if accelerator is None or accelerator.num_processes <= 1:
        return list(range(n_total))
    world = int(accelerator.num_processes)
    rank = int(accelerator.process_index)

    if not force_equal_batches:
        shards = np.array_split(np.arange(n_total, dtype=np.int64), world)
        return shards[rank].tolist()

    bs = max(int(batch_size), 1)
    if n_total <= 0:
        return []

    # Preserve original sample->rank mapping from np.array_split, only add
    # padding slots (-1) so all ranks execute the same number of batches.
    shards = np.array_split(np.arange(n_total, dtype=np.int64), world)
    base = shards[rank].tolist()
    max_len = max(len(s) for s in shards)
    target_slots = int(np.ceil(float(max_len) / float(bs))) * bs
    if len(base) < target_slots:
        base.extend([-1] * (target_slots - len(base)))
    return base


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


def _resolve_feature_output_dir(
    model_id: str,
    dataset_name: str,
    n_samples: int,
    judge_type: str,
) -> str:
    return os.path.join(
        WORKING_DIR,
        "input_hidden_states",
        f"{model_id}_{dataset_name}_{n_samples}_{judge_type}",
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


def _create_azure_openai_client(timeout: float) -> AzureOpenAI:
    api_config = _load_openai_api_config()
    return AzureOpenAI(
        api_version=api_config["api_version"],
        azure_endpoint=api_config["endpoint"],
        api_key=api_config["key"],
        timeout=float(timeout),
        max_retries=0,
    )


def _normalize_question_text(question: object) -> str:
    return "" if question is None else str(question)


def _preview_alignment_value(value: object, limit: int = 240) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def itp_refusal_detection(response_text: object) -> bool:
    normalized = "" if response_text is None else str(response_text).strip().lower()
    return normalized.startswith("none")


def _pad_float_rows(rows: Sequence[Sequence[float]]) -> np.ndarray:
    row_list = [np.asarray(row, dtype=np.float32).reshape(-1) for row in rows]
    if not row_list:
        return np.zeros((0, 0), dtype=np.float32)
    width = max((int(row.shape[0]) for row in row_list), default=0)
    padded = np.zeros((len(row_list), width), dtype=np.float32)
    for idx, row in enumerate(row_list):
        if row.size > 0:
            padded[idx, : row.size] = row
    return padded


def _dataset_name_key(dataset_name: Optional[str]) -> str:
    return str(dataset_name or "").strip().lower()


def _is_rewrite_first_disambig_dataset(dataset_name: Optional[str]) -> bool:
    return _dataset_name_key(dataset_name) in {
        AMBIG_QA_DISAMBIG_DATASET_NAME,
        SITUATED_QA_DISAMBIG_DATASET_NAME,
    }


def _build_ambig_qa_disambig_rewrite_user_prompt(
    question: str,
    disambig_fewshot_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> str:
    if not disambig_fewshot_pool:
        return (
            f"{str(question)}\n\n"
            "Interpret this question so that it expresses exactly one plausible interpretation of the original question.\n"
            "If the question is already unambiguous, return it unchanged.\n"
            "If the question is ambiguous, add only the minimal information needed to remove the ambiguity. "
            "Do not list multiple interpretations. Do not ask a follow-up question. Do not explain your reasoning.\n\n"
            "Output only one rewritten question that expresses the chosen interpretation.\n"
        )

    parts: List[str] = []
    parts.append("Examples of ambiguous questions and their ground-truth disambiguated questions:")
    parts.append("")
    for demo in disambig_fewshot_pool:
        demo_question = str(demo.get("question", ""))
        demo_rewrites = demo.get("disambiguated_questions", [])
        if not isinstance(demo_rewrites, list):
            demo_rewrites = list(demo_rewrites) if demo_rewrites is not None else []
        parts.append("Original question:")
        parts.append(demo_question)
        parts.append("")
        parts.append("Ground-truth disambiguated questions:")
        for idx, rewrite in enumerate(demo_rewrites, start=1):
            parts.append(f"{idx}. {str(rewrite)}")
        parts.append("")
    parts.append("Now rewrite the following question.")
    parts.append("")

    parts.extend(
        [
            "Original question:",
            str(question),
            "",
            "Interpret this question so that it expresses exactly one plausible interpretation of the original question.",
            "If the question is already unambiguous, return it unchanged.",
            "If the question is ambiguous, add only the minimal information needed to remove the ambiguity.",
            "Do not list multiple interpretations. Do not ask a follow-up question. Do not explain your reasoning.",
            "",
            "Output only one rewritten question that expresses the chosen interpretation. Disambiguated question:",
            "",
        ]
    )
    return "\n".join(parts)


def _build_ambig_qa_disambig_answer_user_prompt(disambiguated_question: str) -> str:
    return (
        f"{str(disambiguated_question)}\n\n"
        "Respond with the answer only, do not explain.\n"
    )


def _can_apply_chat_template(tokenizer, model_name: str) -> bool:
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        return False
    if getattr(tokenizer, "chat_template", None):
        return True
    return _uses_tokenizer_chat_template_model_name(model_name)


def _build_generation_prompt_text(
    tokenizer,
    model_name: str,
    *,
    system_prompt: Optional[str],
    user_content: str,
) -> str:
    if _can_apply_chat_template(tokenizer, model_name):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass

    norm_model_name = str(model_name or "").strip().lower()
    if "llama-3" in norm_model_name:
        system_prefix = ""
        if system_prompt:
            system_prefix = (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}\n"
            )
        return (
            system_prefix
            + "<|start_header_id|>user<|end_header_id|>\n\n"
            + f"{user_content}\n"
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    if _is_legacy_fewshot_prompt_model_name(model_name):
        header = "You are a bot that follows instructions.\n"
        if system_prompt:
            header = f"{system_prompt}\n{header}"
        return f"{header}Instruction:\n{user_content}\nResponse:"

    blocks = []
    if system_prompt:
        blocks.append(f"System:\n{system_prompt}")
    blocks.append(f"User:\n{user_content}")
    blocks.append("Assistant:\n")
    return "\n\n".join(blocks)


def _legacy_generation_stopping_criteria(tokenizer, model_name: str, initial_length: int):
    if not _is_legacy_fewshot_prompt_model_name(model_name):
        return None
    return StoppingCriteriaList([
        StoppingCriteriaSub(
            stops=['\n\n\n\n\n', '\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:'],
            initial_length=int(initial_length),
            tokenizer=tokenizer,
        )
    ])


def _clean_disambiguated_question(text: object, fallback_question: object) -> str:
    candidate = "" if text is None else str(text).strip()
    if candidate:
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if lines:
            candidate = lines[0]
    for prefix in ("rewritten question:", "rewrite:", "output:", "question:"):
        if candidate.lower().startswith(prefix):
            candidate = candidate[len(prefix):].strip()
            break
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", "\"", "`"}:
        candidate = candidate[1:-1].strip()

    fallback = "" if fallback_question is None else str(fallback_question).strip()
    return candidate or fallback


def _identity_semantic_ids(n_responses: int) -> List[int]:
    return list(range(max(int(n_responses), 0)))


def _resolve_itp_semantic_ids(
    responses: Sequence[str],
    *,
    itp_semantic_clustering: bool,
    entailment_model,
    strict_entailment: bool,
) -> List[int]:
    n_responses = int(len(responses))
    if n_responses <= 1 or not bool(itp_semantic_clustering):
        return _identity_semantic_ids(n_responses)
    if entailment_model is None:
        raise ValueError(
            "itp_semantic_clustering=True requires an entailment_model for semantic clustering."
        )

    semantic_ids = [int(cid) for cid in get_semantic_ids(
        list(responses),
        model=entailment_model,
        strict_entailment=bool(strict_entailment),
        example=None,
    )]
    if len(semantic_ids) != n_responses:
        raise ValueError(
            "ITP semantic clustering returned an unexpected number of ids: "
            f"responses={n_responses}, semantic_ids={len(semantic_ids)}."
        )
    if any(int(cid) < 0 for cid in semantic_ids):
        raise ValueError(f"ITP semantic ids must be non-negative, got {semantic_ids}.")

    unique_ids = sorted(set(semantic_ids))
    remap = {int(cid): idx for idx, cid in enumerate(unique_ids)}
    return [int(remap[int(cid)]) for cid in semantic_ids]


def _sum_values_by_semantic_ids(
    semantic_ids: Sequence[int],
    values_by_response: Dict[int, float],
) -> Dict[int, float]:
    if not values_by_response:
        return {}
    ids_arr = np.asarray(semantic_ids, dtype=np.int64).reshape(-1)
    reduced: Dict[int, float] = {}
    for resp_id, value in sorted(values_by_response.items()):
        resp_idx = int(resp_id)
        if resp_idx < 0 or resp_idx >= int(ids_arr.shape[0]):
            raise ValueError(
                "ITP response id is out of range for semantic reduction: "
                f"resp_id={resp_idx}, semantic_ids_len={int(ids_arr.shape[0])}."
            )
        cid = int(ids_arr[resp_idx])
        reduced[cid] = reduced.get(cid, 0.0) + float(value)
    return reduced


def _mean_embeddings_by_semantic_ids(
    semantic_ids: Sequence[int],
    embeddings_by_response: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    if not embeddings_by_response:
        return {}
    ids_arr = np.asarray(semantic_ids, dtype=np.int64).reshape(-1)
    cluster_to_vecs: Dict[int, List[np.ndarray]] = {}
    for resp_id, emb in sorted(embeddings_by_response.items()):
        resp_idx = int(resp_id)
        if resp_idx < 0 or resp_idx >= int(ids_arr.shape[0]):
            raise ValueError(
                "ITP response id is out of range for embedding reduction: "
                f"resp_id={resp_idx}, semantic_ids_len={int(ids_arr.shape[0])}."
            )
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            continue
        cid = int(ids_arr[resp_idx])
        cluster_to_vecs.setdefault(cid, []).append(vec)

    reduced: Dict[int, np.ndarray] = {}
    for cid, vecs in cluster_to_vecs.items():
        reduced[int(cid)] = np.stack(vecs, axis=0).mean(axis=0, dtype=np.float32)
    return reduced


def _validate_split_against_feature_outputs(
    split_name: str,
    dataset,
    feature_split_dir: str,
) -> None:
    questions_path = os.path.join(feature_split_dir, "questions.jsonl")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(
            f"Missing feature_collector artifacts for split='{split_name}'. "
            f"Run feature_collector.py first with matching arguments. Missing: {questions_path}"
        )

    if "question" not in dataset.column_names:
        raise KeyError(
            f"Processed semantic split '{split_name}' is missing required columns. "
            f"Found columns: {list(dataset.column_names)}"
        )

    feature_questions = load_question_json(questions_path)
    current_questions = list(dataset["question"])

    n_current = len(current_questions)
    if len(feature_questions) != n_current:
        raise RuntimeError(
            f"Feature-output alignment failed for split='{split_name}': "
            f"feature questions={len(feature_questions)}, semantic questions={n_current}."
        )

    for idx, (feature_q, current_q) in enumerate(zip(feature_questions, current_questions)):
        if _normalize_question_text(feature_q) != _normalize_question_text(current_q):
            raise RuntimeError(
                f"Feature-output alignment failed for split='{split_name}' at qid={idx}: question mismatch. "
                f"feature_question={_preview_alignment_value(feature_q)} "
                f"semantic_question={_preview_alignment_value(current_q)}"
            )

    print(
        f"[SE validate] split={split_name} matched feature_collector outputs "
        f"({n_current} samples)."
    )


def _validate_feature_output_alignment(
    output_dir: str,
    split_datasets: Dict[str, object],
    accelerator: Optional[Accelerator] = None,
) -> None:
    split_dirnames = {"train": "train", "valid": "valid", "test": "test"}

    def _run_validation() -> None:
        for split_name, disk_name in split_dirnames.items():
            dataset = split_datasets.get(split_name)
            if dataset is None:
                continue
            feature_split_dir = os.path.join(output_dir, disk_name)
            _validate_split_against_feature_outputs(split_name, dataset, feature_split_dir)

    use_dist = (
        accelerator is not None
        and accelerator.num_processes > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not use_dist:
        _run_validation()
        return

    shared = [{"ok": True, "message": ""}]
    if accelerator.is_main_process:
        try:
            _run_validation()
        except Exception as exc:
            shared[0] = {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
    torch.distributed.broadcast_object_list(shared, src=0)
    result = shared[0]
    if not result["ok"]:
        raise RuntimeError(result["message"])


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

    backend = (os.environ.get("SE_DIST_GATHER_BACKEND", "file") or "file").strip().lower()
    rank = int(accelerator.process_index)
    world = int(accelerator.num_processes)
    split_tok = _sanitize_token(split_name)

    if backend == "torch":
        t0 = time.time()
        print(f"[SE gather][rank={rank}] backend=torch split={split_tok} local_records={len(local_records)}")
        if accelerator.is_main_process:
            gathered = [None for _ in range(world)]
            torch.distributed.gather_object(local_records, gathered, dst=0)
            merged = []
            for shard in gathered:
                if shard:
                    merged.extend(shard)
            print(
                f"[SE gather][rank={rank}] backend=torch split={split_tok} merged_records={len(merged)} "
                f"elapsed={time.time() - t0:.1f}s"
            )
            return merged
        torch.distributed.gather_object(local_records, None, dst=0)
        print(
            f"[SE gather][rank={rank}] backend=torch split={split_tok} sent_records={len(local_records)} "
            f"elapsed={time.time() - t0:.1f}s"
        )
        return None

    if exchange_dir is None:
        raise ValueError(
            "exchange_dir must be provided for file-based distributed gather. "
            "Either pass exchange_dir or set SE_DIST_GATHER_BACKEND=torch."
        )

    exchange_path = Path(exchange_dir)
    exchange_path.mkdir(parents=True, exist_ok=True)
    token_file = exchange_path / "run_token.txt"
    wait_timeout_sec = int(os.environ.get("SE_DIST_GATHER_FILE_TIMEOUT_SEC", "14400"))
    poll_sec = float(os.environ.get("SE_DIST_GATHER_FILE_POLL_SEC", "5"))
    poll_sec = max(1.0, poll_sec)

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

    t0 = time.time()
    print(f"[SE gather][rank={rank}] backend=file split={split_tok} writing {len(local_records)} records to {shard_path}")
    with open(tmp_path, "wb") as f:
        pickle.dump(local_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, shard_path)
    print(
        f"[SE gather][rank={rank}] backend=file split={split_tok} shard_ready "
        f"elapsed={time.time() - t0:.1f}s"
    )

    if not accelerator.is_main_process:
        return None

    expected = [
        exchange_path / f"{run_token}_{split_tok}_rank{r:03d}_of{world:03d}.pkl"
        for r in range(world)
    ]

    last_log = 0.0
    while True:
        missing = [str(p) for p in expected if not p.exists()]
        if not missing:
            break
        elapsed = time.time() - t0
        if elapsed > wait_timeout_sec:
            raise RuntimeError(
                f"Timeout waiting for distributed shard files for split={split_tok} after {elapsed:.1f}s. "
                f"Missing {len(missing)}/{len(expected)} shards. "
                f"Example missing: {missing[:2]}"
            )
        if elapsed - last_log >= 60.0:
            last_log = elapsed
            print(
                f"[SE gather][rank=0] waiting for split={split_tok} shards "
                f"({len(expected) - len(missing)}/{len(expected)} ready, {elapsed:.1f}s elapsed)"
            )
        time.sleep(poll_sec)

    merged = []
    for path in expected:
        with open(path, "rb") as f:
            shard = pickle.load(f)
        if shard:
            merged.extend(shard)
    print(
        f"[SE gather][rank=0] backend=file split={split_tok} merged_records={len(merged)} "
        f"elapsed={time.time() - t0:.1f}s"
    )
    if os.environ.get("SE_DIST_GATHER_KEEP_SHARDS", "0") != "1":
        for path in expected:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return merged


def _is_response_level_mode(mode: str) -> bool:
    return (mode or "").lower() in {"response-wise", "ranking"}


def _normalize_feature_type(feature_type: str) -> str:
    norm = (feature_type or "tbg_concat").strip().lower()
    if norm == "tbg":
        norm = "tbg_concat"
    if norm not in {"tbg_concat", "slt", "tbg_slt_hidden", "tbg_slt_concat"}:
        raise ValueError(f"Unsupported feature_type: {feature_type}")
    return norm


def _layer_index_list(layer_range) -> List[int]:
    if torch.is_tensor(layer_range):
        arr = layer_range.detach().cpu().numpy()
    else:
        arr = np.asarray(layer_range)
    return np.atleast_1d(arr).astype(np.int64).tolist()


def _load_response_slt_lastk(
    load_dir: str,
    split: str,
    n_resp: int,
) -> List[Dict[int, np.ndarray]]:
    path = os.path.join(load_dir, split, f"response_slt_lastk_{n_resp}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing response-level SLT artifact: {path}. "
            "Regenerate semantic clusters to create response_slt_lastk artifacts."
        )
    return load_cluster_embeddings_npz(path)


def _load_uncertainty_labels(split_dir: str) -> Optional[np.ndarray]:
    ambig_path = os.path.join(split_dir, "ambig_label.npy")
    if os.path.exists(ambig_path):
        return np.asarray(np.load(ambig_path), dtype=np.int64)
    ood_path = os.path.join(split_dir, "ood_label.npy")
    if os.path.exists(ood_path):
        return np.asarray(np.load(ood_path), dtype=np.int64)
    return None


def _compose_tbg_slt_hidden_tensor(
    question_hidden: torch.Tensor,
    response_hidden: torch.Tensor,
    hidden_layer_count: int,
) -> torch.Tensor:
    """
    Compose hidden-only feature for one (question, response):
      [u, v, |u-v|, u*v, cos_layerwise(u,v)]
    where u/v are flattened over hidden_layer_count layers.
    """
    k_layers = int(hidden_layer_count)
    if k_layers <= 0:
        raise ValueError(f"hidden_layer_count must be > 0, got {k_layers}.")

    u_flat = question_hidden.to(torch.float32).reshape(-1)
    v_flat = response_hidden.to(torch.float32).reshape(-1)
    if u_flat.numel() != v_flat.numel():
        raise ValueError(
            "Question/response hidden dims mismatch for tbg_slt_hidden: "
            f"{u_flat.numel()} vs {v_flat.numel()}"
        )
    h_dim = int(u_flat.numel())
    if h_dim % k_layers != 0:
        raise ValueError(
            "Hidden dim is not divisible by hidden_layer_count for tbg_slt_hidden: "
            f"h_dim={h_dim}, hidden_layer_count={k_layers}"
        )

    hidden_size = h_dim // k_layers
    u_layer = u_flat.reshape(k_layers, hidden_size)
    v_layer = v_flat.reshape(k_layers, hidden_size)

    abs_diff = torch.abs(u_layer - v_layer).reshape(-1)
    prod = (u_layer * v_layer).reshape(-1)
    dot = torch.sum(u_layer * v_layer, dim=1)
    u_norm = torch.norm(u_layer, dim=1)
    v_norm = torch.norm(v_layer, dim=1)
    cos = dot / (u_norm * v_norm + 1e-12)
    cos = torch.clamp(cos, -1.0, 1.0)

    feat = torch.cat((u_flat, v_flat, abs_diff, prod, cos), dim=0)
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def _compose_tbg_slt_concat_tensor(
    question_hidden: torch.Tensor,
    slt_hidden: torch.Tensor,
    response_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Compose hidden-only feature for one (question, response):
      [question_tbg_hidden, question_slt_hidden, response_embedding]
    """
    u_flat = question_hidden.to(torch.float32).reshape(-1)
    v_flat = slt_hidden.to(torch.float32).reshape(-1)
    if u_flat.numel() != v_flat.numel():
        raise ValueError(
            "Question TBG/SLT hidden dims mismatch for tbg_slt_concat: "
            f"{u_flat.numel()} vs {v_flat.numel()}"
        )
    e_flat = response_emb.to(torch.float32).reshape(-1)
    feat = torch.cat((u_flat, v_flat, e_flat), dim=0)
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def _resolve_generated_layer_indices(
    layer_range: Union[torch.Tensor, Sequence[int]],
    n_layers_generated: int,
) -> List[int]:
    if n_layers_generated <= 0:
        raise ValueError(f"n_layers_generated must be positive, got {n_layers_generated}.")
    if torch.is_tensor(layer_range):
        requested = [int(x) for x in layer_range.detach().cpu().tolist()]
    else:
        requested = [int(x) for x in np.asarray(list(layer_range), dtype=np.int64).reshape(-1).tolist()]
    if not requested:
        return list(range(n_layers_generated))
    if min(requested) >= 0 and max(requested) < n_layers_generated:
        return requested
    k_layers = min(len(requested), n_layers_generated)
    return list(range(n_layers_generated - k_layers, n_layers_generated))


def _extract_generated_slt_vector(
    gen_hidden_states,
    pred_token_count: int,
    layer_indices: Sequence[int],
    global_resp_idx: int = 0,
) -> torch.Tensor:
    if not gen_hidden_states:
        raise ValueError("Generated hidden states are unavailable.")
    effective_len = min(int(pred_token_count), len(gen_hidden_states))
    slt_idx = max(effective_len - 2, 0)
    h_parts: List[torch.Tensor] = []
    for layer_idx in layer_indices:
        h_parts.append(
            gen_hidden_states[slt_idx][int(layer_idx) + 1][int(global_resp_idx), -1, :].to(torch.float32)
        )
    if not h_parts:
        raise ValueError("No generated SLT layers were selected.")
    return torch.cat(h_parts, dim=0).reshape(-1)


def _extract_prompt_tbg_vector(
    prompt_hidden_states,
    layer_indices: Sequence[int],
    batch_idx: int = 0,
) -> torch.Tensor:
    if not prompt_hidden_states:
        raise ValueError("Prompt hidden states are unavailable.")
    h_parts: List[torch.Tensor] = []
    for layer_idx in layer_indices:
        h_parts.append(
            prompt_hidden_states[int(layer_idx) + 1][int(batch_idx), -1, :].to(torch.float32)
        )
    if not h_parts:
        raise ValueError("No prompt TBG layers were selected.")
    return torch.cat(h_parts, dim=0).reshape(-1)


def _normalize_auxiliary_feature_vector(feature_vec: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if torch.is_tensor(feature_vec):
        arr = feature_vec.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    else:
        arr = np.asarray(feature_vec, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        return arr
    return (arr / max(norm, 1e-12)).astype(np.float32, copy=False)


def _predict_auxiliary_hallucination_probability(
    auxiliary_probe_clf,
    feature_vec: Union[np.ndarray, torch.Tensor],
) -> float:
    feature_batch = _normalize_auxiliary_feature_vector(feature_vec).reshape(1, -1)
    probas = auxiliary_probe_clf.predict_proba(feature_batch)
    pos_idx = int(np.where(auxiliary_probe_clf.classes_ == 1)[0][0]) if probas.shape[1] > 1 else 0
    return float(np.clip(probas[0, pos_idx], 0.0, 1.0))


def _collect_and_save(split_name, dataset, model, entailment_model, strict_entailment,
                      tokenizer, collator, n_responses, cfg, outdir,
                      MSE_type: str = "cluster-wise",
                      accelerator: Optional[Accelerator] = None,
                      dataset_name: str = ""):
    is_main = accelerator is None or accelerator.is_main_process
    if is_main:
        print(f"[SE collect] {split_name}…")

    force_equal_batches = bool(getattr(cfg, "force_equal_batches", False))
    local_indices = _split_indices_for_process(
        len(dataset),
        accelerator,
        batch_size=int(getattr(cfg, "batch_size", 1)),
        force_equal_batches=force_equal_batches,
    )
    local_select_indices = [int(idx) if int(idx) >= 0 else 0 for idx in local_indices]
    dataset_local = dataset.select(local_select_indices)

    (
        responses,
        semantic_ids,
        se_scores,
        semantic_cluster_embeddings,
        response_slt_lastk,
        response_sequence_scores,
        disambiguated_questions,
        disambig_question_semantic_ids,
        disambig_question_sequence_scores,
    ) = collect_semantic_cluster(
        model=model,
        entailment_model=entailment_model,
        strict_entailment=strict_entailment, 
        tokenizer=tokenizer,
        dataset=dataset_local,
        collator=collator,
        n_responses=n_responses, 
        cfg=cfg,
        MSE_type=MSE_type,
        accelerator=accelerator,
        global_indices=local_indices,
        expected_total=len(dataset),
        exchange_dir=os.path.join(outdir, ".dist_exchange"),
        split_name=split_name,
        dataset_name=dataset_name,
    )
    if not is_main:
        return None

    expected_n = len(dataset)
    n_resp = len(responses)
    n_sid = len(semantic_ids)
    n_se = len(se_scores)
    n_emb = len(semantic_cluster_embeddings)
    n_slt = len(response_slt_lastk)
    n_seq_scores = int(response_sequence_scores["sequence_logprob"].shape[0])
    n_seq_responses = int(response_sequence_scores["n_responses"])
    n_disambig = len(disambiguated_questions) if disambiguated_questions is not None else expected_n
    expect_disambig_metadata = _is_rewrite_first_disambig_dataset(dataset_name)
    n_disambig_sid = (
        len(disambig_question_semantic_ids)
        if disambig_question_semantic_ids is not None
        else expected_n
    )
    n_disambig_seq_scores = (
        int(disambig_question_sequence_scores["sequence_logprob"].shape[0])
        if disambig_question_sequence_scores is not None
        else expected_n
    )
    if not (n_resp == n_sid == n_se == n_emb == n_slt == n_seq_scores == expected_n):
        raise RuntimeError(
            "Output alignment check failed in semantic collection: "
            f"expected={expected_n}, responses={n_resp}, semantic_ids={n_sid}, "
            f"se_scores={n_se}, embeddings={n_emb}, response_slt_lastk={n_slt}, "
            f"response_sequence_scores={n_seq_scores}."
        )
    if n_disambig != expected_n:
        raise RuntimeError(
            "Disambiguated-question alignment check failed in semantic collection: "
            f"expected={expected_n}, disambiguated_questions={n_disambig}."
        )
    if expect_disambig_metadata and disambiguated_questions is None:
        raise RuntimeError(
            "Disambiguated-question metadata is missing in semantic collection for a rewrite-first disambiguation dataset."
        )
    if (not expect_disambig_metadata) and (
        disambiguated_questions is not None
        or disambig_question_semantic_ids is not None
        or disambig_question_sequence_scores is not None
    ):
        raise RuntimeError(
            "Unexpected disambiguated-question metadata found for a non-rewrite-first dataset."
        )
    if disambiguated_questions is None:
        if disambig_question_semantic_ids is not None or disambig_question_sequence_scores is not None:
            raise RuntimeError(
                "Disambiguated-question metadata is inconsistent in semantic collection: "
                "found semantic ids or sequence scores without disambiguated questions."
            )
    else:
        if disambig_question_semantic_ids is None or disambig_question_sequence_scores is None:
            raise RuntimeError(
                "Disambiguated-question metadata is incomplete in semantic collection: "
                "expected both semantic ids and sequence scores for a rewrite-first disambiguation dataset."
            )
    if n_disambig_sid != expected_n:
        raise RuntimeError(
            "Disambiguated-question semantic-id alignment check failed in semantic collection: "
            f"expected={expected_n}, disambig_question_semantic_ids={n_disambig_sid}."
        )
    if n_disambig_seq_scores != expected_n:
        raise RuntimeError(
            "Disambiguated-question sequence-score alignment check failed in semantic collection: "
            f"expected={expected_n}, disambig_question_sequence_scores={n_disambig_seq_scores}."
        )
    if disambiguated_questions is not None:
        bad_rows = [
            row_idx for row_idx, rewrites in enumerate(disambiguated_questions)
            if rewrites is None or len(rewrites) != int(n_responses)
        ]
        if bad_rows:
            raise RuntimeError(
                "Disambiguated-question row width mismatch in semantic collection: "
                f"expected each row to have {int(n_responses)} items, bad_rows={bad_rows[:10]}."
            )
    if disambig_question_semantic_ids is not None:
        bad_rows = [
            row_idx for row_idx, ids in enumerate(disambig_question_semantic_ids)
            if ids is None or len(ids) != int(n_responses)
        ]
        if bad_rows:
            raise RuntimeError(
                "Disambiguated-question semantic-id row width mismatch in semantic collection: "
                f"expected each row to have {int(n_responses)} items, bad_rows={bad_rows[:10]}."
            )
    if n_seq_responses != int(n_responses):
        raise RuntimeError(
            "Sampled response sequence-score alignment failed in semantic collection: "
            f"expected n_responses={int(n_responses)}, got {n_seq_responses}."
        )
    if disambig_question_sequence_scores is not None:
        n_disambig_seq_responses = int(disambig_question_sequence_scores["n_responses"])
        if n_disambig_seq_responses != int(n_responses):
            raise RuntimeError(
                "Sampled disambiguated-question sequence-score alignment failed in semantic collection: "
                f"expected n_responses={int(n_responses)}, got {n_disambig_seq_responses}."
            )
        seq_shapes = {
            tuple(disambig_question_sequence_scores["sequence_logprob"].shape),
            tuple(disambig_question_sequence_scores["sequence_prob"].shape),
            tuple(disambig_question_sequence_scores["length_normalized_sequence_logprob"].shape),
        }
        if len(seq_shapes) != 1:
            raise RuntimeError(
                "Disambiguated-question sequence-score shape mismatch in semantic collection: "
                f"shapes={sorted(seq_shapes)}."
            )

    split_dir = os.path.join(outdir, split_name)
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    # se_scores / semantic_ids
    np.save(os.path.join(split_dir, "se.npy"), np.asarray(se_scores, dtype=np.float32))
    np.save(os.path.join(split_dir, f"semantic_ids_{n_responses}.npy"), np.asarray(semantic_ids, dtype=np.int64))
    if disambig_question_semantic_ids is not None:
        np.save(
            os.path.join(split_dir, f"disambig_question_semantic_ids_{n_responses}.npy"),
            np.asarray(disambig_question_semantic_ids, dtype=np.int64),
        )

    # sampled responses
    resp_name = "se_resp_res.jsonl" if _is_response_level_mode(MSE_type) else "se_resp.jsonl"
    preds_path = os.path.join(split_dir, resp_name)
    with open(preds_path, "w", encoding="utf-8") as f:
        for row_idx, rs in enumerate(responses):  # rs: list[str] length n_responses
            payload = {"responses": rs}
            if disambiguated_questions is not None:
                payload["disambiguated_questions"] = disambiguated_questions[row_idx]
            if disambig_question_semantic_ids is not None:
                payload["disambig_question_semantic_ids"] = disambig_question_semantic_ids[row_idx]
            if disambig_question_sequence_scores is not None:
                payload["disambig_question_sequence_logprob"] = (
                    disambig_question_sequence_scores["sequence_logprob"][row_idx].tolist()
                )
                payload["disambig_question_sequence_prob"] = (
                    disambig_question_sequence_scores["sequence_prob"][row_idx].tolist()
                )
                payload["disambig_question_length_normalized_sequence_logprob"] = (
                    disambig_question_sequence_scores["length_normalized_sequence_logprob"][row_idx].tolist()
                )
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    save_sequence_scores_npz(
        os.path.join(
            split_dir,
            get_sampled_sequence_scores_filename(resp_name, n_responses),
        ),
        response_sequence_scores["sequence_logprob"],
        response_sequence_scores["sequence_prob"],
        response_sequence_scores["length_normalized_sequence_logprob"],
    )
    
    # semantic/response embeddings
    emb_name = (
        f"response_embeddings_{n_responses}.npz"
        if _is_response_level_mode(MSE_type)
        else f"cluster_embeddings_{n_responses}.npz"
    )
    emb_path = os.path.join(split_dir, emb_name)
    save_cluster_embeddings_npz(emb_path, semantic_cluster_embeddings)

    if _is_response_level_mode(MSE_type):
        slt_path = os.path.join(split_dir, f"response_slt_lastk_{n_responses}.npz")
        save_cluster_embeddings_npz(slt_path, response_slt_lastk)

    # Return shapes for meta
    slt_dim = 0
    for sample in response_slt_lastk:
        if sample:
            slt_dim = int(np.asarray(next(iter(sample.values())), dtype=np.float32).shape[0])
            break
    return {
        "n_responses": int(n_responses),
        "se_shape": tuple(np.asarray(se_scores, dtype=np.float32).shape), 
        "semantic_ids_shape": tuple(np.asarray(semantic_ids, dtype=np.int64).shape),
        "sequence_scores_shape": tuple(response_sequence_scores["sequence_logprob"].shape),
        "disambig_question_semantic_ids_shape": (
            tuple(np.asarray(disambig_question_semantic_ids, dtype=np.int64).shape)
            if disambig_question_semantic_ids is not None
            else None
        ),
        "disambig_question_sequence_scores_shape": (
            tuple(disambig_question_sequence_scores["sequence_logprob"].shape)
            if disambig_question_sequence_scores is not None
            else None
        ),
        "response_slt_lastk_dim": int(slt_dim),
    }

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli", use_safetensors=True
        ).to(self.device)
        self.model.eval()

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction

# def get_semantic_ids(responses, model, strict_entailment=False, example=None):
#     """Group list of predictions into semantic meaning."""
#
#     def are_equivalent(text1, text2):
#
#         implication_1 = model.check_implication(text1, text2, example=example)
#         implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
#         assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
#
#         if strict_entailment:
#             semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
#
#         else:
#             implications = [implication_1, implication_2]
#             # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
#             semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
#
#         return semantically_equivalent
#
#     # Initialise all ids with -1.
#     semantic_set_ids = [-1] * len(responses)
#     # Keep track of current id.
#     next_id = 0
#     for i, string1 in enumerate(responses):
#         # Check if string1 already has an id assigned.
#         if semantic_set_ids[i] == -1:
#             # If string1 has not been assigned an id, assign it next_id.
#             semantic_set_ids[i] = next_id
#             for j in range(i+1, len(responses)):
#                 # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
#                 if are_equivalent(string1, responses[j]):
#                     semantic_set_ids[j] = next_id
#             next_id += 1
#
#     assert -1 not in semantic_set_ids
#
#     return semantic_set_ids

def get_semantic_ids(responses, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    if not responses:
        return []
    if len(responses) == 1:
        return [0]

    can_batch = hasattr(model, "tokenizer") and hasattr(model, "model")
    n = len(responses)

    if not can_batch:
        def are_equivalent(text1, text2):
            implication_1 = model.check_implication(text1, text2, example=example)
            implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
            assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

            if strict_entailment:
                semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
            else:
                implications = [implication_1, implication_2]
                # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
                semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

            return semantically_equivalent

        # Initialise all ids with -1.
        semantic_set_ids = [-1] * n
        # Keep track of current id.
        next_id = 0
        for i, string1 in enumerate(responses):
            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                # If string1 has not been assigned an id, assign it next_id.
                semantic_set_ids[i] = next_id
                for j in range(i + 1, n):
                    # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                    if are_equivalent(string1, responses[j]):
                        semantic_set_ids[j] = next_id
                next_id += 1

        assert -1 not in semantic_set_ids
        return semantic_set_ids

    # Build all ordered pairs (i -> j) for i != j.
    pair_i = []
    pair_j = []
    texts1 = []
    texts2 = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair_i.append(i)
            pair_j.append(j)
            texts1.append(responses[i])
            texts2.append(responses[j])

    # Batched NLI inference.
    batch_size = 32
    entailment_device = getattr(model, "device", torch.device(DEVICE))
    preds = [None] * len(pair_i)
    for start in range(0, len(pair_i), batch_size):
        end = min(start + batch_size, len(pair_i))
        inputs = model.tokenizer(
            texts1[start:end],
            texts2[start:end],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(entailment_device) for k, v in inputs.items()}
        outputs = model.model(**inputs)
        logits = outputs.logits
        batch_preds = torch.argmax(F.softmax(logits, dim=1), dim=1).detach().cpu().tolist()
        preds[start:end] = batch_preds

    implication = [[None] * n for _ in range(n)]
    for i, j, pred in zip(pair_i, pair_j, preds):
        implication[i][j] = pred

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * n
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(responses):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, n):
                implication_1 = implication[i][j]
                implication_2 = implication[j][i]
                assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

                if strict_entailment:
                    semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
                else:
                    implications = [implication_1, implication_2]
                    # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
                    semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if semantically_equivalent:
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids

def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    log_likelihoods = torch.as_tensor(log_likelihoods, dtype=torch.float32)
    if log_likelihoods.ndim != 1:
        log_likelihoods = log_likelihoods.reshape(-1)
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    if not unique_ids:
        return log_likelihoods.new_empty((0,), dtype=torch.float32)
    log_likelihood_per_semantic_id: List[torch.Tensor] = []
    log_likelihood_norm = torch.logsumexp(log_likelihoods, dim=0)

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_index_tensor = torch.tensor(id_indices, dtype=torch.long, device=log_likelihoods.device)
        id_log_likelihoods = log_likelihoods.index_select(0, id_index_tensor)
        if agg == 'sum_normalized':
            log_lik_norm = id_log_likelihoods - log_likelihood_norm
            logsumexp_value = torch.logsumexp(log_lik_norm, dim=0)
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return torch.stack(log_likelihood_per_semantic_id, dim=0)

def predictive_entropy_rao(log_probs):
    arr = torch.as_tensor(log_probs, dtype=torch.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.numel() == 0:
        print(f"[SE] All log probs are nans!")
        return 0.0
    mask = torch.isfinite(arr)
    if not bool(mask.any().item()):
        print(f"[SE] All log probs are nans!")
        return 0.0
    masked = arr.masked_select(mask)
    entropy = -(torch.exp(masked) * masked).sum()
    return float(entropy.item())

def get_estimated_se(pred_evidence_all):
    '''
    Input
        pred_evidence_all: List[Dict[int, float]]
    '''
    entropies = []
    flag = False
    for item in pred_evidence_all:
        evidence = np.array(list(item.values()), dtype=np.float32)
        total_evidence = np.sum(evidence)

        if total_evidence <= 0.0:
            print(f"[Est SE] 0.0 total predicted evidence detected!")
            entropies.append(0.0)
            continue

        if not np.isfinite(total_evidence):
            print(f"[Est SE] NAN total predicted evidence detected!")
            entropies.append(0.0)
            continue

        p = evidence / (total_evidence + 1e-9)
        mask = (p > 0) & np.isfinite(p)

        if not np.any(mask):
            entropies.append(0.0)
            print(f"[Est SE] No valid probabilities!")
            continue

        flag = np.any(~mask)
        H = -(p[mask] * np.log(p[mask])).sum()
        entropies.append(H)
    print(f"[Est SE] Probabiltiy 0.0 or nan flag: {flag}")
    return entropies

def get_semantic_cluster_embedding(
    client, 
    semantic_ids, 
    responses, 
    aggregation: str="mean"
):
    """
    Compute one embedding vector per semantic cluster using OpenAI embeddings.

    Parameters
    ----------
    embedder : OpenAI client 
        - Should expose `.embeddings.create(model=..., input=...)`, returning `.data[i].embedding`.
    semantic_ids : Sequence[int]
        - Cluster id for each response; must be 0..K-1 contiguous (as produced by your `get_semantic_ids`).
    responses : Sequence[str]
        - Text responses aligned with `semantic_ids`.
    aggregation : str
        - "concatenation": join cluster texts -> single embedding (fastest, best for short responses).
        - "mean": embed each text in the cluster and take the arithmetic mean (length-robust).
        - "no": embed each text in the cluster and return all embeddings (slowest, per-response).

    Returns
    -------
    Dict[int, np.ndarray]
        Mapping {cluster_id: (3072,) float32 numpy vector} or {cluster_id: (3072,) list of float32 numpy vectors}.
    """

    embedder_name = "text-embedding-3-large"
    MAX_RETRIES   = 8
    BACKOFF_BASE  = 1.0   # seconds
    BACKOFF_CAP   = 60.0  # seconds

    try:
        from openai import (
            OpenAIError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIStatusError,
            AuthenticationError,
            BadRequestError,
        )
    except Exception:
        class _E(Exception): ...
        OpenAIError = _E
        APIConnectionError = _E
        APITimeoutError = _E
        RateLimitError = _E
        APIStatusError = _E
        AuthenticationError = _E
        BadRequestError = _E

    try:
        import httpx
        HTTPXError = httpx.HTTPError
    except Exception:
        HTTPXError = tuple()

    def _sleep_backoff(attempt: int):
        delay = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt)) + random.uniform(0.0, 0.25)
        time.sleep(delay)

    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (AuthenticationError, BadRequestError)):
            return False
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                resp = getattr(exc, "response", None)
                status_code = getattr(resp, "status_code", None)
            try:
                status_code = int(status_code) if status_code is not None else None
            except Exception:
                status_code = None
            # Retry unknown status, 429, and 5xx. Fail fast on other 4xx.
            if status_code is None:
                return True
            return status_code == 429 or status_code >= 500
        if HTTPXError and isinstance(exc, HTTPXError):
            return True
        # Unknown exceptions are treated as non-retryable to avoid long stalls.
        return False

    def _embed_one(text: str) -> np.ndarray:
        payload = text if (text and text.strip()) else " "
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=embedder_name, input=[payload])
                return np.asarray(resp.data[0].embedding, dtype=np.float32)
            except Exception as e:
                last_exc = e
                if (attempt < MAX_RETRIES - 1) and _is_retryable(e):
                    _sleep_backoff(attempt)
                else:
                    raise last_exc

    clusters: Dict[int, List[str]] = {} # grouping text by cluster id
    for cid, txt in zip(semantic_ids, responses):
        clusters.setdefault(int(cid), []).append(txt)

    out = {}

    if aggregation == "concatenation":
        for cid, texts in clusters.items(): # One API call per cluster
            paragraph = "\n".join(texts).strip()
            # resp = client.embeddings.create(model=embedder_name, input=[paragraph])  # list input; 1 result.  [oai_citation:1‡OpenAI Platform](https://platform.openai.com/docs/api-reference/embeddings/create?utm_source=chatgpt.com)
            # vec = np.array(resp.data[0].embedding, dtype=np.float32)
            # out[cid] = vec
            out[cid] = _embed_one(paragraph)
    elif aggregation == "mean":  # average of response-level embeddings within one cluster
        for cid, texts in clusters.items():
            vecs = []
            for txt in texts:
                vec = _embed_one(txt) # embed each response in the cluster individually
                vecs.append(vec)

            mat = np.stack(vecs, axis=0) # [num_texts, dim]
            out[cid] = mat.mean(axis=0, dtype=np.float32)
    elif aggregation == "no":  # no aggregation; return all response-level embeddings
        ...
    else:
        raise ValueError('aggregation must be "concatenation" or "mean"')

    return out


def _embed_response_texts_batch(
    client,
    responses: Sequence[str],
    embedder_name: str = "text-embedding-3-large",
) -> List[np.ndarray]:
    """Batch-embed responses while preserving order."""
    if not responses:
        return []

    max_retries = 8
    backoff_base = 1.0
    backoff_cap = 60.0
    chunk_size = max(1, int(os.environ.get("SE_EMBED_BATCH_SIZE", "64")))
    payloads = [txt if (txt and txt.strip()) else " " for txt in responses]

    def _sleep_backoff(attempt: int):
        delay = min(backoff_cap, backoff_base * (2 ** attempt)) + random.uniform(0.0, 0.25)
        time.sleep(delay)

    def _is_retryable(exc: Exception) -> bool:
        name = exc.__class__.__name__
        if name in {"AuthenticationError", "BadRequestError"}:
            return False
        if name in {"APIConnectionError", "APITimeoutError", "RateLimitError"}:
            return True
        if name == "APIStatusError":
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                resp = getattr(exc, "response", None)
                status_code = getattr(resp, "status_code", None)
            try:
                status_code = int(status_code) if status_code is not None else None
            except Exception:
                status_code = None
            if status_code is None:
                return True
            return status_code == 429 or status_code >= 500
        module_name = str(getattr(exc.__class__, "__module__", "")).lower()
        if "httpx" in module_name:
            return True
        return False

    vectors: List[np.ndarray] = []
    for start in range(0, len(payloads), chunk_size):
        chunk = payloads[start:start + chunk_size]
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=embedder_name, input=chunk)
                if len(resp.data) != len(chunk):
                    raise RuntimeError(
                        "Unexpected embedding response size for batch input: "
                        f"expected={len(chunk)}, got={len(resp.data)}."
                    )
                vectors.extend(
                    np.asarray(item.embedding, dtype=np.float32).reshape(-1)
                    for item in resp.data
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if (attempt < max_retries - 1) and _is_retryable(exc):
                    _sleep_backoff(attempt)
                else:
                    raise
        if last_exc is not None:
            raise last_exc

    if len(vectors) != len(payloads):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(payloads)}, got {len(vectors)}."
        )
    return vectors


def _generate_standard_sampled_answers(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_responses: int,
    cfg: SESamplingConfig,
    response_level_mode: bool,
    slt_last_k: int,
) -> Dict[str, object]:
    stopping_criteria = _legacy_generation_stopping_criteria(
        tokenizer,
        model.name_or_path,
        int(input_ids.shape[1]),
    )
    if stopping_criteria is not None:
        print(f"[Stop Criteria] Applying stopping criteria for {model.name_or_path}.")

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        num_return_sequences=n_responses,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=response_level_mode,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    transition_scores = model.compute_transition_scores(
        gen.sequences,
        gen.scores,
        normalize_logits=True,
    )
    preds = gen.sequences.detach().to("cpu")
    hidden_states_slt = getattr(gen, "hidden_states", None) if response_level_mode else None
    n_layers_slt = len(hidden_states_slt[0]) - 1 if hidden_states_slt else 0
    slt_layer_start = max(0, n_layers_slt - int(slt_last_k))
    del gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    batch_size = int(input_ids.size(0))
    input_seq_len = int(input_ids.size(1))
    responses_all: List[List[str]] = []
    slt_all: List[Dict[int, np.ndarray]] = []
    sequence_logprob_all: List[List[float]] = []
    sequence_prob_all: List[List[float]] = []
    mean_logprob_all: List[List[float]] = []

    for i in range(batch_size):
        rows = preds[i * n_responses:(i + 1) * n_responses]
        pred_ids_per_resp = [row[input_seq_len:] for row in rows]
        preds_txt = [tokenizer.decode(pred_ids, skip_special_tokens=True) for pred_ids in pred_ids_per_resp]
        resp_slt_dict: Dict[int, np.ndarray] = {}

        if response_level_mode and hidden_states_slt and n_layers_slt > 0:
            for resp_id, pred_ids in enumerate(pred_ids_per_resp):
                global_resp_idx = i * n_responses + resp_id
                pred_token_count = int(pred_ids.numel())
                effective_len = min(pred_token_count, len(hidden_states_slt))
                slt_idx = max(effective_len - 2, 0)
                slt_parts: List[torch.Tensor] = []
                for layer_idx in range(slt_layer_start, n_layers_slt):
                    slt_parts.append(
                        hidden_states_slt[slt_idx][layer_idx + 1][global_resp_idx, -1, :].to(torch.float32)
                    )
                if slt_parts:
                    resp_slt_dict[int(resp_id)] = (
                        torch.cat(slt_parts, dim=0).detach().cpu().numpy().astype(np.float32)
                    )

        seq_logp = transition_scores[i * n_responses:(i + 1) * n_responses, :]
        valid = torch.isfinite(seq_logp)
        seq_logp = seq_logp.masked_fill(~valid, 0).sum(1)
        seq_len = valid.sum(1).clamp_min(1)
        mean_logp = seq_logp / seq_len
        seq_prob = torch.exp(seq_logp)

        responses_all.append(preds_txt)
        slt_all.append(resp_slt_dict)
        sequence_logprob_all.append(seq_logp.detach().to(torch.float32).cpu().tolist())
        sequence_prob_all.append(seq_prob.detach().to(torch.float32).cpu().tolist())
        mean_logprob_all.append(mean_logp.detach().to(torch.float32).cpu().tolist())

    return {
        "responses": responses_all,
        "disambiguated_questions": None,
        "response_slt_lastk": slt_all,
        "sequence_logprob": sequence_logprob_all,
        "sequence_prob": sequence_prob_all,
        "length_normalized_sequence_logprob": mean_logprob_all,
        "disambig_question_sequence_logprob": None,
        "disambig_question_sequence_prob": None,
        "disambig_question_length_normalized_sequence_logprob": None,
    }


def _generate_disambiguated_sampled_answers(
    *,
    model,
    tokenizer,
    questions: Sequence[str],
    disambig_fewshot_pools: Optional[Sequence[Optional[Sequence[Dict[str, object]]]]] = None,
    n_responses: int,
    cfg: SESamplingConfig,
    response_level_mode: bool,
    slt_last_k: int,
    device: torch.device,
) -> Dict[str, object]:
    rewrite_prompts = [
        _build_generation_prompt_text(
            tokenizer,
            model.name_or_path,
            system_prompt=AMBIG_QA_DISAMBIG_REWRITE_SYSTEM_PROMPT,
            user_content=_build_ambig_qa_disambig_rewrite_user_prompt(
                question,
                None if disambig_fewshot_pools is None else disambig_fewshot_pools[q_idx],
            ),
        )
        for q_idx, question in enumerate(questions)
    ]
    rewrite_batch = tokenizer(
        rewrite_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    rewrite_input_ids = rewrite_batch["input_ids"].to(device)
    rewrite_attention_mask = rewrite_batch["attention_mask"].to(device)
    rewrite_stopping_criteria = _legacy_generation_stopping_criteria(
        tokenizer,
        model.name_or_path,
        int(rewrite_input_ids.shape[1]),
    )

    rewrite_gen = model.generate(
        input_ids=rewrite_input_ids,
        attention_mask=rewrite_attention_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        num_return_sequences=n_responses,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=False,
        stopping_criteria=rewrite_stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    rewrite_transition_scores = model.compute_transition_scores(
        rewrite_gen.sequences,
        rewrite_gen.scores,
        normalize_logits=True,
    )
    rewrite_sequences = rewrite_gen.sequences.detach().to("cpu")
    rewrite_input_seq_len = int(rewrite_input_ids.size(1))
    del rewrite_gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cleaned_questions_all: List[List[str]] = []
    rewrite_sequence_logprob_all: List[List[float]] = []
    rewrite_sequence_prob_all: List[List[float]] = []
    rewrite_mean_logprob_all: List[List[float]] = []
    answer_prompts: List[str] = []
    for q_idx, original_question in enumerate(questions):
        rows = rewrite_sequences[q_idx * n_responses:(q_idx + 1) * n_responses]
        pred_ids_per_resp = [row[rewrite_input_seq_len:] for row in rows]
        cleaned_questions = [
            _clean_disambiguated_question(
                tokenizer.decode(pred_ids, skip_special_tokens=True),
                original_question,
            )
            for pred_ids in pred_ids_per_resp
        ]
        cleaned_questions_all.append(cleaned_questions)
        rewrite_seq_logp = rewrite_transition_scores[q_idx * n_responses:(q_idx + 1) * n_responses, :]
        rewrite_valid = torch.isfinite(rewrite_seq_logp)
        rewrite_seq_logp = rewrite_seq_logp.masked_fill(~rewrite_valid, 0).sum(1)
        rewrite_seq_len = rewrite_valid.sum(1).clamp_min(1)
        rewrite_mean_logp = rewrite_seq_logp / rewrite_seq_len
        rewrite_seq_prob = torch.exp(rewrite_seq_logp)
        rewrite_sequence_logprob_all.append(
            rewrite_seq_logp.detach().to(torch.float32).cpu().tolist()
        )
        rewrite_sequence_prob_all.append(
            rewrite_seq_prob.detach().to(torch.float32).cpu().tolist()
        )
        rewrite_mean_logprob_all.append(
            rewrite_mean_logp.detach().to(torch.float32).cpu().tolist()
        )
        for disambiguated_question in cleaned_questions:
            answer_prompts.append(
                _build_generation_prompt_text(
                    tokenizer,
                    model.name_or_path,
                    system_prompt=AMBIG_QA_DISAMBIG_ANSWER_SYSTEM_PROMPT,
                    user_content=_build_ambig_qa_disambig_answer_user_prompt(disambiguated_question),
                )
            )

    answer_batch = tokenizer(
        answer_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    answer_input_ids = answer_batch["input_ids"].to(device)
    answer_attention_mask = answer_batch["attention_mask"].to(device)
    answer_stopping_criteria = _legacy_generation_stopping_criteria(
        tokenizer,
        model.name_or_path,
        int(answer_input_ids.shape[1]),
    )

    answer_gen = model.generate(
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=response_level_mode,
        stopping_criteria=answer_stopping_criteria,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    transition_scores = model.compute_transition_scores(
        answer_gen.sequences,
        answer_gen.scores,
        normalize_logits=True,
    )
    answer_sequences = answer_gen.sequences.detach().to("cpu")
    hidden_states_slt = getattr(answer_gen, "hidden_states", None) if response_level_mode else None
    n_layers_slt = len(hidden_states_slt[0]) - 1 if hidden_states_slt else 0
    slt_layer_start = max(0, n_layers_slt - int(slt_last_k))
    answer_input_seq_len = int(answer_input_ids.size(1))
    del answer_gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    responses_all: List[List[str]] = []
    slt_all: List[Dict[int, np.ndarray]] = []
    sequence_logprob_all: List[List[float]] = []
    sequence_prob_all: List[List[float]] = []
    mean_logprob_all: List[List[float]] = []

    for q_idx in range(len(questions)):
        start = q_idx * n_responses
        end = (q_idx + 1) * n_responses
        rows = answer_sequences[start:end]
        pred_ids_per_resp = [row[answer_input_seq_len:] for row in rows]
        preds_txt = [
            tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            for pred_ids in pred_ids_per_resp
        ]
        resp_slt_dict: Dict[int, np.ndarray] = {}

        if response_level_mode and hidden_states_slt and n_layers_slt > 0:
            for resp_id, pred_ids in enumerate(pred_ids_per_resp):
                global_resp_idx = start + resp_id
                pred_token_count = int(pred_ids.numel())
                effective_len = min(pred_token_count, len(hidden_states_slt))
                slt_idx = max(effective_len - 2, 0)
                slt_parts: List[torch.Tensor] = []
                for layer_idx in range(slt_layer_start, n_layers_slt):
                    slt_parts.append(
                        hidden_states_slt[slt_idx][layer_idx + 1][global_resp_idx, -1, :].to(torch.float32)
                    )
                if slt_parts:
                    resp_slt_dict[int(resp_id)] = (
                        torch.cat(slt_parts, dim=0).detach().cpu().numpy().astype(np.float32)
                    )

        seq_logp = transition_scores[start:end, :]
        valid = torch.isfinite(seq_logp)
        seq_logp = seq_logp.masked_fill(~valid, 0).sum(1)
        seq_len = valid.sum(1).clamp_min(1)
        mean_logp = seq_logp / seq_len
        seq_prob = torch.exp(seq_logp)

        responses_all.append(preds_txt)
        slt_all.append(resp_slt_dict)
        sequence_logprob_all.append(seq_logp.detach().to(torch.float32).cpu().tolist())
        sequence_prob_all.append(seq_prob.detach().to(torch.float32).cpu().tolist())
        mean_logprob_all.append(mean_logp.detach().to(torch.float32).cpu().tolist())

    return {
        "responses": responses_all,
        "disambiguated_questions": cleaned_questions_all,
        "response_slt_lastk": slt_all,
        "sequence_logprob": sequence_logprob_all,
        "sequence_prob": sequence_prob_all,
        "length_normalized_sequence_logprob": mean_logprob_all,
        "disambig_question_sequence_logprob": rewrite_sequence_logprob_all,
        "disambig_question_sequence_prob": rewrite_sequence_prob_all,
        "disambig_question_length_normalized_sequence_logprob": rewrite_mean_logprob_all,
    }


def collect_semantic_cluster(
    model: AutoModelForCausalLM,
    entailment_model, 
    strict_entailment, 
    tokenizer,
    dataset,                              # processed split
    collator,                             # EPNDataCollator(tokenizer)
    n_responses: int = 10,
    cfg: SESamplingConfig = SESamplingConfig(),
    MSE_type: str = "response-wise",
    slt_last_k: int = 5,
    accelerator: Optional[Accelerator] = None,
    global_indices: Optional[List[int]] = None,
    expected_total: Optional[int] = None,
    exchange_dir: Optional[str] = None,
    split_name: str = "split",
    dataset_name: str = "",
):
    """
    Sample multiple responses, estimate SE and cluster_ids
    Applying embedder models to get semantic cluster embedding

    Return 
        sampled responses, semantic_ids, SE, optional disambiguated-question metadata
    """
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    local_records = []
    response_level_mode = _is_response_level_mode(MSE_type)

    if global_indices is None:
        global_indices = list(range(len(dataset)))
    if accelerator is not None and accelerator.num_processes > 1:
        slots_local = len(global_indices)
        valid_local = int(sum(1 for idx in global_indices if int(idx) >= 0))
        bs = max(int(cfg.batch_size), 1)
        n_batches = int(np.ceil(float(slots_local) / float(bs))) if slots_local > 0 else 0
        print(
            f"[SE collect][rank={accelerator.process_index}] split={split_name} "
            f"slots={slots_local} valid={valid_local} batches={n_batches}"
        )

    show_progress = accelerator is None or accelerator.is_local_main_process
    device = _resolve_model_input_device(model, accelerator=accelerator, fallback_device=cfg.device)
    openai_timeout_sec = float(os.environ.get("SE_OPENAI_TIMEOUT_SEC", "60"))
    client = _create_azure_openai_client(timeout=openai_timeout_sec)

    offset = 0
    for batch in tqdm(dl, disable=not show_progress):
        batch_size_raw = int(batch["input_ids"].size(0))
        batch_global_indices_raw = [
            int(idx) for idx in global_indices[offset:offset + batch_size_raw]
        ]
        real_positions = [pos for pos, idx in enumerate(batch_global_indices_raw) if idx >= 0]
        if len(real_positions) == 0:
            offset += batch_size_raw
            continue

        with torch.no_grad():
            input_ids   = batch["input_ids"].to(device)
            attn        = batch["attention_mask"].to(device)
            rewrite_source_field = "rewrite_source_question" if "rewrite_source_question" in batch else "question"
            if len(real_positions) != batch_size_raw:
                select_idx = torch.as_tensor(real_positions, dtype=torch.long, device=input_ids.device)
                input_ids = input_ids.index_select(0, select_idx)
                attn = attn.index_select(0, select_idx)
                batch_global_indices = [batch_global_indices_raw[pos] for pos in real_positions]
                batch_questions = [batch["question"][pos] for pos in real_positions]
                batch_rewrite_questions = [batch[rewrite_source_field][pos] for pos in real_positions]
                batch_disambig_fewshot_pools = None
                if "disambig_fewshot_pool" in batch:
                    batch_disambig_fewshot_pools = [batch["disambig_fewshot_pool"][pos] for pos in real_positions]
            else:
                batch_global_indices = batch_global_indices_raw
                batch_questions = list(batch["question"])
                batch_rewrite_questions = list(batch[rewrite_source_field])
                batch_disambig_fewshot_pools = list(batch["disambig_fewshot_pool"]) if "disambig_fewshot_pool" in batch else None

            batch_size  = input_ids.size(0)
            if _is_rewrite_first_disambig_dataset(dataset_name):
                generation_outputs = _generate_disambiguated_sampled_answers(
                    model=model,
                    tokenizer=tokenizer,
                    questions=batch_rewrite_questions,
                    disambig_fewshot_pools=batch_disambig_fewshot_pools,
                    n_responses=n_responses,
                    cfg=cfg,
                    response_level_mode=response_level_mode,
                    slt_last_k=slt_last_k,
                    device=device,
                )
            else:
                generation_outputs = _generate_standard_sampled_answers(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attn,
                    n_responses=n_responses,
                    cfg=cfg,
                    response_level_mode=response_level_mode,
                    slt_last_k=slt_last_k,
                )

            for i in range(batch_size):
                preds_txt = generation_outputs["responses"][i]
                resp_slt_dict = generation_outputs["response_slt_lastk"][i]
                seq_logp = torch.as_tensor(
                    generation_outputs["sequence_logprob"][i],
                    dtype=torch.float32,
                )
                seq_prob = torch.as_tensor(
                    generation_outputs["sequence_prob"][i],
                    dtype=torch.float32,
                )
                mean_logp = torch.as_tensor(
                    generation_outputs["length_normalized_sequence_logprob"][i],
                    dtype=torch.float32,
                )
                disambiguated_questions = None
                if generation_outputs["disambiguated_questions"] is not None:
                    disambiguated_questions = generation_outputs["disambiguated_questions"][i]
                disambig_question_semantic_ids = None
                disambig_seq_logp = None
                disambig_seq_prob = None
                disambig_mean_logp = None
                if disambiguated_questions is not None:
                    disambig_question_semantic_ids = get_semantic_ids(
                        disambiguated_questions,
                        model=entailment_model,
                        strict_entailment=False,
                        example=None,
                    )
                    disambig_seq_logp = torch.as_tensor(
                        generation_outputs["disambig_question_sequence_logprob"][i],
                        dtype=torch.float32,
                    )
                    disambig_seq_prob = torch.as_tensor(
                        generation_outputs["disambig_question_sequence_prob"][i],
                        dtype=torch.float32,
                    )
                    disambig_mean_logp = torch.as_tensor(
                        generation_outputs["disambig_question_length_normalized_sequence_logprob"][i],
                        dtype=torch.float32,
                    )

                # get semantic clusters
                curr_semantic_ids = get_semantic_ids(
                    preds_txt, model=entailment_model,
                    strict_entailment=strict_entailment, example=None
                )

                # calculating SE
                semantic_cluster_logp = logsumexp_by_id(curr_semantic_ids, mean_logp, agg='sum_normalized')
                se = predictive_entropy_rao(semantic_cluster_logp)

                if _is_response_level_mode(MSE_type):
                    response_emb_dict: Dict[int, np.ndarray] = {}
                    response_embs = _embed_response_texts_batch(
                        client=client,
                        responses=preds_txt,
                    )
                    for resp_id, emb in enumerate(response_embs):
                        response_emb_dict[int(resp_id)] = np.asarray(emb, dtype=np.float32)
                    semantic_cluster_emb_dict = response_emb_dict
                else:
                    semantic_cluster_emb_dict = get_semantic_cluster_embedding(
                        client=client,
                        semantic_ids=curr_semantic_ids,
                        responses=preds_txt,
                        aggregation="mean",
                    )

                global_idx = int(batch_global_indices[i])
                local_records.append(
                    {
                        "sample_idx": global_idx,
                        "responses": preds_txt,
                        "semantic_ids": curr_semantic_ids,
                        "se_score": se,
                        "semantic_cluster_embeddings": semantic_cluster_emb_dict,
                        "response_slt_lastk": resp_slt_dict,
                        "sequence_logprob": seq_logp.detach().to(torch.float32).cpu().tolist(),
                        "sequence_prob": seq_prob.detach().to(torch.float32).cpu().tolist(),
                        "length_normalized_sequence_logprob": (
                            mean_logp.detach().to(torch.float32).cpu().tolist()
                        ),
                    }
                )
                if disambiguated_questions is not None:
                    local_records[-1]["disambiguated_questions"] = disambiguated_questions
                    local_records[-1]["disambig_question_semantic_ids"] = disambig_question_semantic_ids
                    local_records[-1]["disambig_question_sequence_logprob"] = (
                        disambig_seq_logp.detach().to(torch.float32).cpu().tolist()
                    )
                    local_records[-1]["disambig_question_sequence_prob"] = (
                        disambig_seq_prob.detach().to(torch.float32).cpu().tolist()
                    )
                    local_records[-1]["disambig_question_length_normalized_sequence_logprob"] = (
                        disambig_mean_logp.detach().to(torch.float32).cpu().tolist()
                    )
        offset += batch_size_raw

    gathered_records = _gather_records_to_main(
        local_records,
        accelerator,
        exchange_dir=exchange_dir,
        split_name=split_name,
    )
    if accelerator is not None and not accelerator.is_main_process:
        return None, None, None, None, None, None, None, None, None

    gathered_records = sorted(gathered_records, key=lambda row: row["sample_idx"])
    sample_indices = [int(row["sample_idx"]) for row in gathered_records]
    if expected_total is None:
        expected_total = len(sample_indices)
    expected_indices = list(range(int(expected_total)))
    if sample_indices != expected_indices:
        raise RuntimeError(
            "Sample index alignment failed in semantic collection. "
            f"Expected indices 0..{int(expected_total) - 1}, "
            f"got {len(sample_indices)} samples with first indices {sample_indices[:10]}."
        )

    responses = [row["responses"] for row in gathered_records]
    disambiguated_questions = (
        [row.get("disambiguated_questions") for row in gathered_records]
        if any("disambiguated_questions" in row for row in gathered_records)
        else None
    )
    disambig_question_semantic_ids = (
        [row.get("disambig_question_semantic_ids") for row in gathered_records]
        if any("disambig_question_semantic_ids" in row for row in gathered_records)
        else None
    )
    semantic_ids = [row["semantic_ids"] for row in gathered_records]
    se_scores = [float(row["se_score"]) for row in gathered_records]
    semantic_cluster_embeddings = [row["semantic_cluster_embeddings"] for row in gathered_records]
    response_slt_lastk = [row["response_slt_lastk"] for row in gathered_records]
    if gathered_records:
        response_sequence_scores = {
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
            "N": int(len(gathered_records)),
            "n_responses": int(n_responses),
        }
        disambig_question_sequence_scores = (
            {
                "sequence_logprob": np.asarray(
                    [row["disambig_question_sequence_logprob"] for row in gathered_records],
                    dtype=np.float32,
                ),
                "sequence_prob": np.asarray(
                    [row["disambig_question_sequence_prob"] for row in gathered_records],
                    dtype=np.float32,
                ),
                "length_normalized_sequence_logprob": np.asarray(
                    [row["disambig_question_length_normalized_sequence_logprob"] for row in gathered_records],
                    dtype=np.float32,
                ),
                "N": int(len(gathered_records)),
                "n_responses": int(n_responses),
            }
            if any("disambig_question_sequence_logprob" in row for row in gathered_records)
            else None
        )
    else:
        response_sequence_scores = {
            "sequence_logprob": np.zeros((0, int(n_responses)), dtype=np.float32),
            "sequence_prob": np.zeros((0, int(n_responses)), dtype=np.float32),
            "length_normalized_sequence_logprob": np.zeros((0, int(n_responses)), dtype=np.float32),
            "N": 0,
            "n_responses": int(n_responses),
        }
        disambig_question_sequence_scores = None
    return (
        responses,
        semantic_ids,
        se_scores,
        semantic_cluster_embeddings,
        response_slt_lastk,
        response_sequence_scores,
        disambiguated_questions,
        disambig_question_semantic_ids,
        disambig_question_sequence_scores,
    )


def collect_semantic_cluster_eval(
    eup: EvidentialUncertaintyProbe, 
    model: AutoModelForCausalLM,
    load_dir, 
    split, 
    layer_range,
    tokenizer,
    dataset,                              # processed split
    collator,                             # EPNDataCollator(tokenizer)
    fewshot_pool, 
    n_resp, 
    fix_n_eval_resp: bool = False,
    n_max_resp: int = 3,
    conf_thres: float = 6.0, 
    cfg: SESamplingConfig = SESamplingConfig(),
    output_feature: bool = False,
    MSE_type: str = "cluster-wise",
    sanity_check: bool = True,
    idx_subset: Optional[Sequence[int]] = None,
    uncertainty_labels: Optional[np.ndarray] = None,
    dataset_name: Optional[str] = None,
    remove_refusal: bool = False,
    feature_type: str = "tbg_concat",
    density_probe_feature: str = "tbg",
    extra_eval_mask: Optional[np.ndarray] = None,
    auxiliary_probe_clf: Optional[object] = None,
    accelerator: Optional[Accelerator] = None,
    global_indices: Optional[List[int]] = None,
    expected_total: Optional[int] = None,
    exchange_dir: Optional[str] = None,
    split_name: str = "split",
    need_response_embeddings: bool = True,
    itp_semantic_clustering: bool = False,
    entailment_model: Optional[object] = None,
    strict_entailment: bool = False,
):
    """
    Sample multiple semantically different responses, estimate SE and cluster_ids
    Applying embedder models to get semantic cluster embedding

    Return 
        sampled responses, semantic_ids, SE
    """
    eup.eval()
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
    if global_indices is None:
        global_indices = list(range(len(dataset)))
    else:
        global_indices = [int(idx) for idx in global_indices]
    if len(global_indices) != len(dataset):
        raise ValueError(
            "collect_semantic_cluster_eval global_indices length mismatch: "
            f"indices={len(global_indices)} vs dataset={len(dataset)}."
        )

    feature_type_norm = _normalize_feature_type(feature_type)
    density_probe_feature_norm = (density_probe_feature or "tbg").strip().lower()
    if density_probe_feature_norm not in {"tbg", "slt"}:
        raise ValueError(
            f"Unsupported density_probe_feature: {density_probe_feature}. Expected one of: tbg, slt."
        )
    hidden_only_feature = feature_type_norm in {"slt", "tbg_slt_hidden", "tbg_slt_concat"}
    needs_generated_hidden = feature_type_norm in {"slt", "tbg_slt_hidden"} or density_probe_feature_norm == "slt"

    # loading primary response hidden states, labels, ground truth semantic cluster embeddings
    split_dir = os.path.join(load_dir, split)

    X = load_X_npz(os.path.join(split_dir, "X.npz"))
    idx_arr = np.asarray(idx_subset, dtype=int) if idx_subset is not None else None
    if idx_arr is not None and idx_arr.size > 0:
        max_idx = int(idx_arr.max())
        if X["tbg"].shape[1] > max_idx:
            X["tbg"] = X["tbg"][:, idx_arr, :]
            if "slt" in X:
                X["slt"] = X["slt"][:, idx_arr, :]
    response_level_mode = _is_response_level_mode(MSE_type)
    if (hidden_only_feature or density_probe_feature_norm == "slt") and not response_level_mode:
        raise ValueError(
            f"feature_type={feature_type_norm} and density_probe_feature={density_probe_feature_norm} "
            "are only supported with response-wise (or ranking) evaluation mode."
        )

    feature_key = "slt" if feature_type_norm == "slt" else "tbg"
    layer_idx = _layer_index_list(layer_range)
    if feature_key not in X:
        raise KeyError(
            f"Requested feature '{feature_key}' is unavailable in {os.path.join(split_dir, 'X.npz')}."
        )
    X_input = torch.as_tensor(X[feature_key][layer_idx, :, :], dtype=torch.float32)
    X_flat = X_input.permute(1, 0, 2).contiguous().view(X_input.size(1), -1)
    X_flat = torch.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_flat = F.normalize(X_flat, p=2, dim=1)
    question_slt_flat: Optional[torch.Tensor] = None
    if feature_type_norm == "tbg_slt_concat":
        if "slt" not in X:
            raise KeyError(
                "feature_type=tbg_slt_concat requires X['slt'] in evaluation split: "
                f"{os.path.join(split_dir, 'X.npz')}."
            )
        X_slt_input = torch.as_tensor(X["slt"][layer_idx, :, :], dtype=torch.float32)
        X_slt_flat = X_slt_input.permute(1, 0, 2).contiguous().view(X_slt_input.size(1), -1)
        X_slt_flat = torch.nan_to_num(X_slt_flat, nan=0.0, posinf=0.0, neginf=0.0)
        X_slt_flat = F.normalize(X_slt_flat, p=2, dim=1)
        if X_slt_flat.shape != X_flat.shape:
            raise ValueError(
                "tbg/slt flattened shape mismatch for feature_type=tbg_slt_concat: "
                f"tbg={tuple(X_flat.shape)}, slt={tuple(X_slt_flat.shape)}"
            )
        question_slt_flat = X_slt_flat
    tbg_slt_layer_count = (
        int(len(layer_range)) if feature_type_norm == "tbg_slt_hidden" else None
    )
    y = np.load(os.path.join(split_dir, f"y.npy"))
    file_uncertainty_labels = _load_uncertainty_labels(split_dir)
    if idx_arr is not None and idx_arr.size > 0:
        max_idx = int(idx_arr.max())
        if y.shape[0] > max_idx:
            y = y[idx_arr]
        if file_uncertainty_labels is not None and file_uncertainty_labels.shape[0] > max_idx:
            file_uncertainty_labels = file_uncertainty_labels[idx_arr]
        if uncertainty_labels is not None and len(uncertainty_labels) > max_idx:
            uncertainty_labels = np.asarray(uncertainty_labels, dtype=int)[idx_arr]
    if uncertainty_labels is None and file_uncertainty_labels is not None:
        uncertainty_labels = file_uncertainty_labels

    semantic_cluster_embeddings_gt = None
    semantic_ids_gt = None
    if response_level_mode:
        semantic_ids_gt = np.load(
            os.path.join(split_dir, f"semantic_ids_{n_resp}.npy"), allow_pickle=True
        )
    else:
        emb_path = os.path.join(split_dir, f"cluster_embeddings_{n_resp}.npz")
        semantic_cluster_embeddings_gt = load_cluster_embeddings_npz(emb_path)
    if idx_arr is not None and idx_arr.size > 0:
        max_idx = int(idx_arr.max())
        if semantic_ids_gt is not None and len(semantic_ids_gt) > max_idx:
            semantic_ids_gt = semantic_ids_gt[idx_arr]
        if semantic_cluster_embeddings_gt is not None and len(semantic_cluster_embeddings_gt) > max_idx:
            semantic_cluster_embeddings_gt = [semantic_cluster_embeddings_gt[i] for i in idx_arr]

    responses: List[List[str]] = []
    se_scores: List[float] = []
    semantic_ids_all: List[List[int]] = []
    semantic_ids_itp_all: List[List[int]] = []
    response_sequence_probs_all: List[List[float]] = []
    response_norm_sequence_probs_all: List[List[float]] = []
    itp_refusal_counts_all: List[int] = []
    itp_attempt_counts_all: List[int] = []
    itp_valid_response_counts_all: List[int] = []
    itp_refusal_rates_all: List[float] = []
    semantic_cluster_embeddings_all: List[Dict[int, np.ndarray]] = []
    pred_evidence_all: List[Dict[int, float]] = []
    pred_evidence_discounted_all: Optional[List[Dict[int, float]]] = (
        [] if auxiliary_probe_clf is not None else None
    )
    dissonance_all: List[float] = []
    vacuity_all: List[float] = []
    dist_dissonance_all: Optional[List[float]] = [] if auxiliary_probe_clf is not None else None
    dist_vacuity_all: Optional[List[float]] = [] if auxiliary_probe_clf is not None else None
    aux_hallucination_probs_all: Optional[List[List[float]]] = (
        [] if auxiliary_probe_clf is not None else None
    )
    aux_reliability_scores_all: Optional[List[List[float]]] = (
        [] if auxiliary_probe_clf is not None else None
    )
    itp_features_all: List[Dict[int, Dict[str, List[float]]]] = [
        {} for _ in range(X_flat.shape[0])
    ] if output_feature else []
    pred_itp: Optional[Dict[int, Dict[str, object]]] = {} if sanity_check else None

    device = cfg.device

    embedding_client: Optional[AzureOpenAI] = None

    def _get_embedding_client() -> AzureOpenAI:
        nonlocal embedding_client
        if embedding_client is None:
            openai_timeout_sec = float(os.environ.get("SE_OPENAI_TIMEOUT_SEC", "60"))
            embedding_client = _create_azure_openai_client(timeout=openai_timeout_sec)
        return embedding_client

    model_system_prompt = _instruction_tuned_system_prompt(getattr(model, "name_or_path", ""))
    system_template_prefix = ""
    if model_system_prompt:
        system_template_prefix = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{system_prompt}\n"
        )
    prefix_template = (
        system_template_prefix
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        "{question} Respond with the answer only, do not explain.\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n")

    prefix_template_w_context = (
        system_template_prefix
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        "Context: {context}\n"
        "Question:{question}\n"
        "Answer the question depending on the context. Respond with the answer only, do not explain.\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n")

    offset = 0
    for batch in tqdm(dl):
        batch_input_ids = batch["input_ids"].to(device)
        batch_attn = batch["attention_mask"].to(device)
        batch_size = batch_input_ids.size(0)

        for b_idx in range(batch_size): # per question
            with torch.no_grad():
                attn_mask = batch_attn[b_idx]
                prompt_ids = batch_input_ids[b_idx][attn_mask.bool()].detach().clone()
                if prompt_ids.numel() == 0:
                    prompt_ids = batch_input_ids[b_idx].detach().clone()

                prompt_ids = prompt_ids.unsqueeze(0)
                prompt_attn = torch.ones_like(prompt_ids, device=device)
                # base_prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

                question_responses: List[str] = []
                logps: List[float] = []
                question_sequence_probs: List[float] = []
                question_norm_sequence_probs: List[float] = []
                question_refusal_count = 0
                question_attempt_count = 0
                response_embeddings_by_resp: Dict[int, np.ndarray] = {}
                response_evidence_dict: Dict[int, float] = {}
                discounted_response_evidence_dict: Optional[Dict[int, float]] = (
                    {} if auxiliary_probe_clf is not None else None
                )
                total_evidence = 0.0
                if sanity_check:
                    response_embs: List[Optional[np.ndarray]] = []
                    evidence_vals: List[float] = []
                aux_probs_q: Optional[List[float]] = [] if auxiliary_probe_clf is not None else None
                aux_reliability_q: Optional[List[float]] = [] if auxiliary_probe_clf is not None else None
                base_qid = offset + b_idx
                global_qid = int(global_indices[base_qid])

                if fix_n_eval_resp: # using fixed # of clusters
                    curr_n_max_resp = n_max_resp
                elif n_max_resp == -1: # using GT # of clusters
                    if response_level_mode:
                        if semantic_ids_gt is not None and offset + b_idx < len(semantic_ids_gt):
                            ids_row = np.asarray(semantic_ids_gt[offset + b_idx], dtype=int)
                            curr_n_max_resp = int(len(np.unique(ids_row)))
                        else:
                            print(f"[SE Eval] Missing GT semantic ids for sample {offset + b_idx}, using 0 clusters.")
                            curr_n_max_resp = 0
                    else:
                        curr_n_max_resp = len(semantic_cluster_embeddings_gt[offset + b_idx])
                else:
                    curr_n_max_resp = n_max_resp

                target_response_count = max(int(curr_n_max_resp), 0)
                max_attempt_count = 2 * target_response_count if target_response_count > 0 else 0

                while len(question_responses) < target_response_count:
                    if max_attempt_count > 0 and question_attempt_count >= max_attempt_count:
                        break

                    if len(question_responses) == 0: # primary response
                        curr_input_ids = prompt_ids.to(device)
                        curr_attn = prompt_attn.to(device)
                    else:
                        prev_resp_text = "\n".join(question_responses)
                        next_question = (
                            batch["question"][b_idx]
                            + f"\nHere are your previous responses:\n{prev_resp_text}\n"
                            + "\nGenerate a new response whose underlying semantic meaning or interpretation is genuinely different from all previous responses. "
                            + "Do not paraphrase, restate, or slightly reword an existing response. "
                            + "A different response must express a different interpretation, hypothesis, entity, value, or conclusion. "
                            + "If you cannot produce a genuinely different meaning, output `NONE`.\n"
                        )
                        

                        if _is_legacy_fewshot_prompt_model_name(model.name_or_path):
                            assert fewshot_pool is not None

                            def qa_line(q, a):
                                return f"Question: {q}\nAnswer: {a}\n\n"

                            header = "You are a bot that answers questions in a short phrase.\n"
                            if model_system_prompt:
                                header = f"{model_system_prompt}\n{header}"
                            fewshot_block = "".join(qa_line(d["question"], d["answer"]) for d in fewshot_pool)

                            prefix_template = (
                                f"{header}"
                                f"{fewshot_block}"
                                "Question: {question}\n"
                                "Answer:"
                            )
                            updated_prompt = prefix_template.format(question=next_question)
                        elif _uses_tokenizer_chat_template_model_name(model.name_or_path):
                            if batch["context"][b_idx] != "":
                                updated_user_content = (
                                    "Context: {context}\n"
                                    "Question:{question}\n"
                                    "Answer the question depending on the context. Respond with the answer only, do not explain.\n"
                                ).format(
                                    context=batch["context"][b_idx],
                                    question=next_question,
                                )
                            else:
                                updated_user_content = (
                                    f"{next_question} Respond with the answer only, do not explain.\n"
                                )
                            messages = []
                            if model_system_prompt:
                                messages.append({"role": "system", "content": model_system_prompt})
                            messages.append({"role": "user", "content": updated_user_content})
                            updated_prompt = tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                        else:
                            if batch["context"][b_idx] != "":
                                updated_prompt = prefix_template_w_context.format(
                                    context=batch['context'][b_idx],
                                    question=next_question,
                                    system_prompt=model_system_prompt or "",
                                )
                            else:
                                updated_prompt = prefix_template.format(
                                    question=next_question,
                                    system_prompt=model_system_prompt or "",
                                )
                        
                        enc = tokenizer(updated_prompt, return_tensors="pt", truncation=True)
                        curr_input_ids = enc["input_ids"].to(device)
                        curr_attn = enc["attention_mask"].to(device)

                    # Keep the legacy stop criteria on the older few-shot prompt backbones only.
                    stopping_criteria = None
                    if _is_legacy_fewshot_prompt_model_name(model.name_or_path):
                        stopping_criteria = StoppingCriteriaList([
                            StoppingCriteriaSub(
                                stops=['\n\n\n\n\n', '\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:'],
                                initial_length=curr_input_ids.shape[1],
                                tokenizer=tokenizer
                            )
                        ])

                    prompt_tbg_vec: Optional[torch.Tensor] = None
                    if auxiliary_probe_clf is not None:
                        prompt_outputs = model(
                            input_ids=curr_input_ids,
                            attention_mask=curr_attn,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        prompt_hidden_states = getattr(prompt_outputs, "hidden_states", None)
                        n_layers_prompt = len(prompt_hidden_states) - 1 if prompt_hidden_states else 0
                        if n_layers_prompt <= 0:
                            raise ValueError(
                                "Failed to collect prompt hidden states for response-wise auxiliary scoring."
                            )
                        prompt_layer_indices = _resolve_generated_layer_indices(
                            layer_range,
                            n_layers_prompt,
                        )
                        prompt_tbg_vec = _extract_prompt_tbg_vector(
                            prompt_hidden_states=prompt_hidden_states,
                            layer_indices=prompt_layer_indices,
                            batch_idx=0,
                        ).to(device)

                    gen = model.generate(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attn,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        num_return_sequences=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_hidden_states=needs_generated_hidden,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    gen_hidden_states = (
                        getattr(gen, "hidden_states", None)
                        if needs_generated_hidden
                        else None
                    )

                    seq = gen.sequences[0]
                    input_len = curr_input_ids.shape[1]
                    gen_tokens = seq[input_len:]
                    pred_txt = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    question_attempt_count += 1
                    if itp_refusal_detection(pred_txt):
                        question_refusal_count += 1
                        continue

                    curr_cid = len(question_responses)
                    question_responses.append(pred_txt)

                    transition_scores = model.compute_transition_scores(
                        gen.sequences, gen.scores, normalize_logits=True
                    )
                    ts = transition_scores[0]
                    valid = torch.isfinite(ts)
                    seq_logp = ts.masked_fill(~valid, 0).sum().item()
                    seq_len = max(int(valid.sum().item()), 1)
                    logps.append(seq_logp / seq_len)
                    question_sequence_probs.append(float(np.exp(seq_logp)))
                    question_norm_sequence_probs.append(float(np.exp(seq_logp / seq_len)))

                    emb_curr: Optional[np.ndarray] = None
                    if bool(need_response_embeddings):
                        resp_embedding_dict = get_semantic_cluster_embedding(
                            client=_get_embedding_client(),
                            semantic_ids=[0],
                            responses=[pred_txt],
                            aggregation="mean",
                        )
                        emb_curr = resp_embedding_dict.get(0)
                        if emb_curr is None:
                            raise ValueError(
                                f"Missing response embedding during eval (qid={global_qid}, resp_idx={curr_cid})."
                            )
                        emb_curr = np.asarray(emb_curr, dtype=np.float32)
                        response_embeddings_by_resp[int(curr_cid)] = emb_curr
                    if sanity_check:
                        response_embs.append(emb_curr)
                    generated_slt_vec: Optional[torch.Tensor] = None
                    generated_layer_indices: Optional[List[int]] = None
                    if gen_hidden_states is not None and (
                        feature_type_norm in {"slt", "tbg_slt_hidden"}
                        or auxiliary_probe_clf is not None
                        or density_probe_feature_norm == "slt"
                    ):
                        n_layers_generated = len(gen_hidden_states[0]) - 1 if gen_hidden_states else 0
                        if n_layers_generated <= 0:
                            raise ValueError(
                                "Failed to collect generated hidden states for response-wise auxiliary scoring."
                            )
                        generated_layer_indices = _resolve_generated_layer_indices(
                            layer_range,
                            n_layers_generated,
                        )
                        generated_slt_vec = _extract_generated_slt_vector(
                            gen_hidden_states=gen_hidden_states,
                            pred_token_count=int(gen_tokens.numel()),
                            layer_indices=generated_layer_indices,
                            global_resp_idx=0,
                        ).to(device)
                    density_slt_vec: Optional[np.ndarray] = None
                    if density_probe_feature_norm == "slt":
                        if generated_slt_vec is None:
                            raise ValueError(
                                "density_probe_feature=slt requires generated SLT hidden states for each response."
                            )
                        density_slt_vec = _normalize_auxiliary_feature_vector(generated_slt_vec)
                    if feature_type_norm == "slt":
                        if generated_slt_vec is None:
                            raise ValueError("Failed to collect generated hidden states for feature_type=slt.")
                        H_idx = generated_slt_vec.reshape(1, -1)
                        C_idx = torch.empty((1, 0), dtype=torch.float32, device=device)
                    elif feature_type_norm == "tbg_slt_hidden":
                        if generated_slt_vec is None:
                            raise ValueError(
                                "Failed to collect generated hidden states for feature_type=tbg_slt_hidden."
                            )
                        required_layers = int(tbg_slt_layer_count) if tbg_slt_layer_count is not None else 0
                        if generated_layer_indices is None or len(generated_layer_indices) < required_layers:
                            raise ValueError(
                                "Generated hidden layers are fewer than required layer_range for "
                                f"feature_type=tbg_slt_hidden: generated={0 if generated_layer_indices is None else len(generated_layer_indices)}, "
                                f"required={required_layers}"
                            )
                        q_vec = X_flat[base_qid].reshape(-1).to(device)
                        H_idx = _compose_tbg_slt_hidden_tensor(
                            q_vec,
                            generated_slt_vec,
                            hidden_layer_count=required_layers,
                        ).reshape(1, -1)
                        C_idx = torch.empty((1, 0), dtype=torch.float32, device=device)
                    elif feature_type_norm == "tbg_slt_concat":
                        if question_slt_flat is None:
                            raise ValueError(
                                "Missing question-level SLT features for feature_type=tbg_slt_concat."
                            )
                        q_vec = X_flat[base_qid].reshape(-1).to(device)
                        q_slt_vec = question_slt_flat[base_qid].reshape(-1).to(device)
                        if emb_curr is not None:
                            emb_tensor = torch.as_tensor(emb_curr, dtype=torch.float32, device=device).reshape(-1)
                            H_idx = _compose_tbg_slt_concat_tensor(
                                q_vec,
                                q_slt_vec,
                                emb_tensor,
                            ).reshape(1, -1)
                        else:
                            # When embeddings are intentionally disabled, keep only the hidden-state portion.
                            H_idx = torch.cat([q_vec, q_slt_vec], dim=0).reshape(1, -1)
                        C_idx = torch.empty((1, 0), dtype=torch.float32, device=device)
                    else:
                        H_idx = X_flat[base_qid].unsqueeze(0).to(device)
                        if emb_curr is not None:
                            C_idx = torch.as_tensor(
                                emb_curr.reshape(-1, emb_curr.shape[0])
                            ).to(torch.float32).to(device) # [1, c_dim]
                        else:
                            C_idx = torch.empty((1, 0), dtype=torch.float32, device=device)

                    pred_val = float(eup(H_idx, C_idx).item()) # per sample
                    response_evidence_dict[curr_cid] = response_evidence_dict.get(curr_cid, 0.0) + pred_val
                    if auxiliary_probe_clf is not None:
                        if prompt_tbg_vec is None:
                            raise ValueError(
                                "Auxiliary probe requires response-wise prompt TBG activations, but none were found."
                            )
                        aux_prob = _predict_auxiliary_hallucination_probability(
                            auxiliary_probe_clf,
                            prompt_tbg_vec,
                        )
                        aux_reliability = float(1.0 - aux_prob)
                        if aux_probs_q is not None:
                            aux_probs_q.append(aux_prob)
                        if aux_reliability_q is not None:
                            aux_reliability_q.append(aux_reliability)
                        if discounted_response_evidence_dict is not None:
                            discounted_val = aux_reliability * pred_val
                            discounted_response_evidence_dict[curr_cid] = (
                                discounted_response_evidence_dict.get(curr_cid, 0.0) + discounted_val
                            )
                    total_evidence += pred_val
                    if sanity_check:
                        evidence_vals.append(pred_val)

                    if output_feature:
                        # cache features for later analysis (hidden-only, cluster-only, full)
                        hidden_np = H_idx.squeeze(0).detach().cpu().numpy().tolist()
                        cluster_np = C_idx.squeeze(0).detach().cpu().numpy().tolist()
                        full_np = torch.cat([H_idx, C_idx], dim=1).squeeze(0).detach().cpu().numpy().tolist()
                        feature_payload = {
                            "hidden": hidden_np,
                            "cluster": cluster_np,
                            "full": full_np,
                        }
                        if density_slt_vec is not None:
                            feature_payload["density_slt"] = density_slt_vec.tolist()
                        itp_features_all[base_qid][curr_cid] = feature_payload

                    if (not fix_n_eval_resp) and n_max_resp != -1: # remove confidence threshold control if using ground truth # of clusters
                        if total_evidence >= conf_thres:
                            break

                curr_semantic_ids = _resolve_itp_semantic_ids(
                    question_responses,
                    itp_semantic_clustering=bool(itp_semantic_clustering),
                    entailment_model=entailment_model,
                    strict_entailment=bool(strict_entailment),
                )
                curr_cluster_embeddings = _mean_embeddings_by_semantic_ids(
                    curr_semantic_ids,
                    response_embeddings_by_resp,
                )
                evidence_dict = _sum_values_by_semantic_ids(
                    curr_semantic_ids,
                    response_evidence_dict,
                )
                discounted_evidence_dict: Optional[Dict[int, float]] = (
                    _sum_values_by_semantic_ids(curr_semantic_ids, discounted_response_evidence_dict)
                    if discounted_response_evidence_dict is not None
                    else None
                )
                dissonance_score = dissonance(evidence_dict)
                vacuity_score = vacuity(evidence_dict)

                responses.append(question_responses)
                semantic_ids_all.append(list(curr_semantic_ids))
                semantic_ids_itp_all.append(list(curr_semantic_ids))
                response_sequence_probs_all.append(question_sequence_probs)
                response_norm_sequence_probs_all.append(question_norm_sequence_probs)
                itp_refusal_counts_all.append(int(question_refusal_count))
                itp_attempt_counts_all.append(int(question_attempt_count))
                itp_valid_response_counts_all.append(int(len(question_responses)))
                refusal_rate = (
                    float(question_refusal_count) / float(question_attempt_count)
                    if question_attempt_count > 0
                    else 0.0
                )
                itp_refusal_rates_all.append(refusal_rate)
                semantic_cluster_embeddings_all.append(curr_cluster_embeddings)
                pred_evidence_all.append(evidence_dict)
                dissonance_all.append(dissonance_score)
                vacuity_all.append(vacuity_score)
                if pred_evidence_discounted_all is not None and discounted_evidence_dict is not None:
                    pred_evidence_discounted_all.append(discounted_evidence_dict)
                if dist_dissonance_all is not None and discounted_evidence_dict is not None:
                    dist_dissonance_all.append(float(dissonance(discounted_evidence_dict)))
                if dist_vacuity_all is not None and discounted_evidence_dict is not None:
                    dist_vacuity_all.append(float(vacuity(discounted_evidence_dict)))
                if aux_hallucination_probs_all is not None and aux_probs_q is not None:
                    aux_hallucination_probs_all.append(aux_probs_q)
                if aux_reliability_scores_all is not None and aux_reliability_q is not None:
                    aux_reliability_scores_all.append(aux_reliability_q)
                if sanity_check and pred_itp is not None:
                    pred_itp[base_qid] = {
                        "itp_question": batch["question"][b_idx],
                        "responses": question_responses,
                        "semantic_ids_itp": list(curr_semantic_ids),
                        "embeddings": response_embs,
                        "response_evidence": evidence_vals,
                        "cluster_evidence": dict(evidence_dict),
                        "itp_refusal_count": int(question_refusal_count),
                        "itp_attempt_count": int(question_attempt_count),
                    }

                if question_responses:
                    mean_logp_tensor = torch.tensor(logps, dtype=torch.float32)
                    semantic_cluster_logp = logsumexp_by_id(
                        curr_semantic_ids, mean_logp_tensor, agg='sum_normalized'
                    )
                    se_scores.append(
                        predictive_entropy_rao([float(v.item()) for v in semantic_cluster_logp])
                    )
                else:
                    se_scores.append(0.0)
        
        offset += batch_size

    # calculating estimated SE based on EUP prediçted evidence
    gathered_full_outputs = False
    if accelerator is not None and accelerator.num_processes > 1:
        local_records = []
        for local_qid in range(len(pred_evidence_all)):
            local_records.append(
                {
                    "sample_idx": int(global_indices[local_qid]),
                    "responses": responses[local_qid],
                    "semantic_ids": semantic_ids_all[local_qid],
                    "semantic_ids_itp": semantic_ids_itp_all[local_qid],
                    "response_sequence_probs": response_sequence_probs_all[local_qid],
                    "response_norm_sequence_probs": response_norm_sequence_probs_all[local_qid],
                    "itp_refusal_count": int(itp_refusal_counts_all[local_qid]),
                    "itp_attempt_count": int(itp_attempt_counts_all[local_qid]),
                    "itp_valid_response_count": int(itp_valid_response_counts_all[local_qid]),
                    "itp_refusal_rate": float(itp_refusal_rates_all[local_qid]),
                    "se_score": float(se_scores[local_qid]),
                    "semantic_cluster_embeddings": semantic_cluster_embeddings_all[local_qid],
                    "pred_evidence": pred_evidence_all[local_qid],
                    "pred_evidence_discounted": (
                        pred_evidence_discounted_all[local_qid]
                        if pred_evidence_discounted_all is not None
                        else None
                    ),
                    "dissonance": float(dissonance_all[local_qid]),
                    "vacuity": float(vacuity_all[local_qid]),
                    "dist_dissonance": (
                        float(dist_dissonance_all[local_qid])
                        if dist_dissonance_all is not None
                        else None
                    ),
                    "dist_vacuity": (
                        float(dist_vacuity_all[local_qid])
                        if dist_vacuity_all is not None
                        else None
                    ),
                    "aux_hallucination_probs": (
                        aux_hallucination_probs_all[local_qid]
                        if aux_hallucination_probs_all is not None
                        else None
                    ),
                    "aux_reliability_scores": (
                        aux_reliability_scores_all[local_qid]
                        if aux_reliability_scores_all is not None
                        else None
                    ),
                    "feature_map": (
                        itp_features_all[local_qid]
                        if output_feature
                        else None
                    ),
                    "pred_itp_item": (
                        pred_itp.get(local_qid)
                        if sanity_check and pred_itp is not None
                        else None
                    ),
                    "primary_truth": int(y[local_qid]) if local_qid < len(y) else 0,
                    "uncertainty_label": (
                        int(uncertainty_labels[local_qid])
                        if uncertainty_labels is not None and local_qid < len(uncertainty_labels)
                        else None
                    ),
                }
            )
        gathered_records = _gather_records_to_main(
            local_records,
            accelerator,
            exchange_dir=exchange_dir,
            split_name=split_name,
        )
        if not accelerator.is_main_process:
            return None
        gathered_records = sorted(gathered_records, key=lambda row: row["sample_idx"])
        sample_indices = [int(row["sample_idx"]) for row in gathered_records]
        if expected_total is None:
            expected_total = len(sample_indices)
        expected_indices = list(range(int(expected_total)))
        if sample_indices != expected_indices:
            raise RuntimeError(
                "Sample index alignment failed in semantic eval collection. "
                f"Expected indices 0..{int(expected_total) - 1}, "
                f"got {len(sample_indices)} samples with first indices {sample_indices[:10]}."
            )
        responses = [row["responses"] for row in gathered_records]
        semantic_ids_all = [row["semantic_ids"] for row in gathered_records]
        semantic_ids_itp_all = [row["semantic_ids_itp"] for row in gathered_records]
        response_sequence_probs_all = [row["response_sequence_probs"] for row in gathered_records]
        response_norm_sequence_probs_all = [row["response_norm_sequence_probs"] for row in gathered_records]
        itp_refusal_counts_all = [int(row["itp_refusal_count"]) for row in gathered_records]
        itp_attempt_counts_all = [int(row["itp_attempt_count"]) for row in gathered_records]
        itp_valid_response_counts_all = [int(row["itp_valid_response_count"]) for row in gathered_records]
        itp_refusal_rates_all = [float(row["itp_refusal_rate"]) for row in gathered_records]
        se_scores = [float(row["se_score"]) for row in gathered_records]
        semantic_cluster_embeddings_all = [row["semantic_cluster_embeddings"] for row in gathered_records]
        pred_evidence_all = [row["pred_evidence"] for row in gathered_records]
        dissonance_all = [float(row["dissonance"]) for row in gathered_records]
        vacuity_all = [float(row["vacuity"]) for row in gathered_records]
        if pred_evidence_discounted_all is not None:
            pred_evidence_discounted_all = [row["pred_evidence_discounted"] for row in gathered_records]
        if dist_dissonance_all is not None:
            dist_dissonance_all = [float(row["dist_dissonance"]) for row in gathered_records]
        if dist_vacuity_all is not None:
            dist_vacuity_all = [float(row["dist_vacuity"]) for row in gathered_records]
        if aux_hallucination_probs_all is not None:
            aux_hallucination_probs_all = [row["aux_hallucination_probs"] for row in gathered_records]
        if aux_reliability_scores_all is not None:
            aux_reliability_scores_all = [row["aux_reliability_scores"] for row in gathered_records]
        if output_feature:
            itp_features_all = [row["feature_map"] or {} for row in gathered_records]
        if sanity_check and pred_itp is not None:
            pred_itp = {
                int(row["sample_idx"]): row["pred_itp_item"]
                for row in gathered_records
                if row.get("pred_itp_item") is not None
            }
        y = np.asarray([int(row["primary_truth"]) for row in gathered_records], dtype=np.int64)
        if uncertainty_labels is not None:
            uncertainty_labels = np.asarray(
                [int(row["uncertainty_label"]) for row in gathered_records],
                dtype=np.int64,
            )
        gathered_full_outputs = True

    est_se_scores = get_estimated_se(pred_evidence_all)

    # calculating AUROC & AUPR
    if uncertainty_labels is not None:
        hallucination_labels = np.asarray(uncertainty_labels[:len(pred_evidence_all)], dtype=int)
    else:
        hallucination_labels = np.array(y[:len(pred_evidence_all)] == 0).astype(int) # 1 if hallucination, 0 if not
    metric_eval_mask, ood_refusal_eval = compute_ood_refusal_filter(
        labels=hallucination_labels,
        dataset_name=dataset_name or "",
        split=split,
        remove_refusal=remove_refusal,
        load_dir=load_dir,
        idx_subset=None if gathered_full_outputs else idx_arr,
    )
    metric_eval_mask = combine_optional_eval_mask(
        metric_eval_mask,
        extra_eval_mask=extra_eval_mask,
        idx_subset=None if gathered_full_outputs else idx_arr,
        label=f"collect_semantic_cluster_eval:{split}",
    )

    print(f"[EUP Eval] MSE between GT SE and Est SE: {np.mean((np.array(se_scores) - np.array(est_se_scores))**2):.4f}")

    vacuity_arr = np.array(vacuity_all)
    dissonance_arr = np.array(dissonance_all)
    eval_labels = np.asarray(hallucination_labels, dtype=int)[metric_eval_mask]
    eval_vacuity = vacuity_arr[metric_eval_mask]
    eval_dissonance = dissonance_arr[metric_eval_mask]
    eval_se = np.asarray(se_scores, dtype=float)[metric_eval_mask]
    eval_est_se = np.asarray(est_se_scores, dtype=float)[metric_eval_mask]
    eval_cluster_sizes = np.asarray([len(e_dict) for e_dict in pred_evidence_all], dtype=float)[metric_eval_mask]
    dist_vacuity_arr = (
        np.asarray(dist_vacuity_all, dtype=float)
        if dist_vacuity_all is not None
        else None
    )
    dist_dissonance_arr = (
        np.asarray(dist_dissonance_all, dtype=float)
        if dist_dissonance_all is not None
        else None
    )
    unique_labels = np.unique(eval_labels)
    if unique_labels.size < 2:
        print(f"[EUP Eval] Skipping AUROC/AUPR for {split} split (only one class present: {unique_labels.tolist()}).")
        auroc_cluster_size = float("nan")
        auroc_vacuity = float("nan")
        auroc_dissonance = float("nan")
        auroc_se = float("nan")
        auroc_est_se = float("nan")
        aupr_vacuity = float("nan")
        aupr_dissonance = float("nan")
        aupr_se = float("nan")
        aupr_est_se = float("nan")
        auroc_dist_vacuity = float("nan")
        auroc_dist_dissonance = float("nan")
        aupr_dist_vacuity = float("nan")
        aupr_dist_dissonance = float("nan")
    else:
        auroc_cluster_size = roc_auc_score(eval_labels, eval_cluster_sizes)
        print(f"[DEBUG] {split} AUROC by cluster size: {auroc_cluster_size:.4f}")
        auroc_vacuity = roc_auc_score(eval_labels, eval_vacuity)
        auroc_dissonance = roc_auc_score(eval_labels, eval_dissonance)
        auroc_se = roc_auc_score(eval_labels, eval_se)
        auroc_est_se = roc_auc_score(eval_labels, eval_est_se)
        aupr_vacuity = average_precision_score(eval_labels, eval_vacuity)
        aupr_dissonance = average_precision_score(eval_labels, eval_dissonance)
        aupr_se = average_precision_score(eval_labels, eval_se)
        aupr_est_se = average_precision_score(eval_labels, eval_est_se)
        if dist_vacuity_arr is not None and dist_dissonance_arr is not None:
            eval_dist_vacuity = dist_vacuity_arr[metric_eval_mask]
            eval_dist_dissonance = dist_dissonance_arr[metric_eval_mask]
            auroc_dist_vacuity = roc_auc_score(eval_labels, eval_dist_vacuity)
            auroc_dist_dissonance = roc_auc_score(eval_labels, eval_dist_dissonance)
            aupr_dist_vacuity = average_precision_score(eval_labels, eval_dist_vacuity)
            aupr_dist_dissonance = average_precision_score(eval_labels, eval_dist_dissonance)
        else:
            auroc_dist_vacuity = float("nan")
            auroc_dist_dissonance = float("nan")
            aupr_dist_vacuity = float("nan")
            aupr_dist_dissonance = float("nan")
        print(f"[EUP Eval] {split} AUROC vacuity: {auroc_vacuity:.4f}, dissonance: {auroc_dissonance:.4f}, se: {auroc_se:.4f}, est_se: {auroc_est_se:.4f}")
        print(f"[EUP Eval] {split} AUPR vacuity: {aupr_vacuity:.4f}, dissonance: {aupr_dissonance:.4f}, se: {aupr_se:.4f}, est_se: {aupr_est_se:.4f}")
        if np.isfinite(auroc_dist_vacuity) or np.isfinite(auroc_dist_dissonance):
            print(
                f"[Auxiliary Probe][{split}] Discounted AUROC vacuity: {auroc_dist_vacuity:.4f}, "
                f"dissonance: {auroc_dist_dissonance:.4f}"
            )

    return {
        "responses": responses, 
        "semantic_ids": semantic_ids_all, 
        "semantic_ids_itp": semantic_ids_itp_all,
        "se_scores": se_scores, 
        "est_se_scores": est_se_scores, 
        "y": y,
        "uncertainty_labels": hallucination_labels,
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "ood_refusal_eval": ood_refusal_eval,
        "semantic_cluster_embeddings": semantic_cluster_embeddings_all, 
        "pred_evidence": pred_evidence_all, 
        "pred_evidence_discounted": pred_evidence_discounted_all,
        "dissonance": dissonance_all, 
        "vacuity": vacuity_all,
        "dist_dissonance": dist_dissonance_all,
        "dist_vacuity": dist_vacuity_all,
        "aux_hallucination_probs": aux_hallucination_probs_all,
        "aux_reliability_scores": aux_reliability_scores_all,
        "response_sequence_probs": _pad_float_rows(response_sequence_probs_all),
        "response_norm_sequence_probs": _pad_float_rows(response_norm_sequence_probs_all),
        "itp_refusal_counts": np.asarray(itp_refusal_counts_all, dtype=np.int64),
        "itp_attempt_counts": np.asarray(itp_attempt_counts_all, dtype=np.int64),
        "itp_valid_response_counts": np.asarray(itp_valid_response_counts_all, dtype=np.int64),
        "itp_refusal_rates": np.asarray(itp_refusal_rates_all, dtype=np.float32),
        "features_itp": itp_features_all if output_feature else None,
        "pred_itp": pred_itp if sanity_check else None,
        "AUROC": {"vacuity": auroc_vacuity , "dissonance": auroc_dissonance, "se": auroc_se, "est_se": auroc_est_se},
        "AUPR": {"vacuity": aupr_vacuity , "dissonance": aupr_dissonance, "se": aupr_se, "est_se": aupr_est_se}, 
        "AUROC_discounted": {"vacuity": auroc_dist_vacuity, "dissonance": auroc_dist_dissonance},
        "AUPR_discounted": {"vacuity": aupr_dist_vacuity, "dissonance": aupr_dist_dissonance},
    }

def fix_n_existing_semantic_cluster_eval(
    eup: EvidentialUncertaintyProbe,
    load_dir,
    split,
    layer_range,
    n_resp,
    aug_cluster_embs: List[Dict[str, object]],
    n_fixed_clusters: int = 3,
    MSE_type: str = "cluster-wise",
    idx_subset: Optional[Sequence[int]] = None,
    uncertainty_labels: Optional[np.ndarray] = None,
    dataset_name: Optional[str] = None,
    remove_refusal: bool = False,
    feature_type: str = "tbg_concat",
    extra_eval_mask: Optional[np.ndarray] = None,
):
    """
    Evaluate EUP with a fixed number of semantic clusters using existing sampled embeddings.

    Clusters are selected by descending count from ``aug_cluster_embs``. If fewer clusters
    than ``n_fixed_clusters`` are available, all clusters are used.
    """

    eup.eval()
    feature_type_norm = _normalize_feature_type(feature_type)
    hidden_only_feature = feature_type_norm in {"slt", "tbg_slt_hidden", "tbg_slt_concat"}
    tbg_slt_layer_count = (
        int(len(layer_range)) if feature_type_norm == "tbg_slt_hidden" else None
    )

    split_dir = os.path.join(load_dir, split)

    X = load_X_npz(os.path.join(split_dir, "X.npz"))
    idx_arr = np.asarray(idx_subset, dtype=int) if idx_subset is not None else None
    if idx_arr is not None and idx_arr.size > 0:
        max_idx = int(idx_arr.max())
        if X["tbg"].shape[1] > max_idx:
            X["tbg"] = X["tbg"][:, idx_arr, :]
            if "slt" in X:
                X["slt"] = X["slt"][:, idx_arr, :]
    feature_key = "slt" if feature_type_norm == "slt" else "tbg"
    layer_idx = _layer_index_list(layer_range)
    if feature_key not in X:
        raise KeyError(
            f"Requested feature '{feature_key}' is unavailable in {os.path.join(split_dir, 'X.npz')}."
        )
    X_input = torch.as_tensor(X[feature_key][layer_idx, :, :], dtype=torch.float32)
    X_flat = X_input.permute(1, 0, 2).contiguous().view(X_input.size(1), -1)
    X_flat = torch.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_flat = F.normalize(X_flat, p=2, dim=1)
    question_slt_flat: Optional[torch.Tensor] = None
    if feature_type_norm == "tbg_slt_concat":
        if "slt" not in X:
            raise KeyError(
                "feature_type=tbg_slt_concat requires X['slt'] in evaluation split: "
                f"{os.path.join(split_dir, 'X.npz')}."
            )
        X_slt_input = torch.as_tensor(X["slt"][layer_idx, :, :], dtype=torch.float32)
        X_slt_flat = X_slt_input.permute(1, 0, 2).contiguous().view(X_slt_input.size(1), -1)
        X_slt_flat = torch.nan_to_num(X_slt_flat, nan=0.0, posinf=0.0, neginf=0.0)
        X_slt_flat = F.normalize(X_slt_flat, p=2, dim=1)
        if X_slt_flat.shape != X_flat.shape:
            raise ValueError(
                "tbg/slt flattened shape mismatch for feature_type=tbg_slt_concat: "
                f"tbg={tuple(X_flat.shape)}, slt={tuple(X_slt_flat.shape)}"
            )
        question_slt_flat = X_slt_flat
    y = np.load(os.path.join(split_dir, "y.npy"))
    file_uncertainty_labels = _load_uncertainty_labels(split_dir)
    if idx_arr is not None and idx_arr.size > 0:
        max_idx = int(idx_arr.max())
        if y.shape[0] > max_idx:
            y = y[idx_arr]
        if file_uncertainty_labels is not None and file_uncertainty_labels.shape[0] > max_idx:
            file_uncertainty_labels = file_uncertainty_labels[idx_arr]
        if uncertainty_labels is not None and len(uncertainty_labels) > max_idx:
            uncertainty_labels = np.asarray(uncertainty_labels, dtype=int)[idx_arr]
    if uncertainty_labels is None and file_uncertainty_labels is not None:
        uncertainty_labels = file_uncertainty_labels

    response_level_mode = _is_response_level_mode(MSE_type)
    if hidden_only_feature and not response_level_mode:
        raise ValueError(
            f"feature_type={feature_type_norm} is only supported with response-wise (or ranking) evaluation mode."
        )
    semantic_ids_arr = np.load(os.path.join(split_dir, f"semantic_ids_{n_resp}.npy"), allow_pickle=True)
    if not response_level_mode:
        se_responses = load_resp_json(os.path.join(split_dir, "se_resp.jsonl"))
    else:
        se_responses = load_resp_json(os.path.join(split_dir, "se_resp_res.jsonl"))
    response_embs_all: Optional[List[Dict[int, np.ndarray]]] = None
    if response_level_mode:
        emb_path = os.path.join(split_dir, f"response_embeddings_{n_resp}.npz")
        response_embs_all = load_cluster_embeddings_npz(emb_path)
    response_slt_all: Optional[List[Dict[int, np.ndarray]]] = None
    if response_level_mode and feature_type_norm in {"slt", "tbg_slt_hidden"}:
        response_slt_all = _load_response_slt_lastk(load_dir, split, n_resp)
    if idx_arr is not None and idx_arr.size > 0:
        max_idx = int(idx_arr.max())
        if len(semantic_ids_arr) > max_idx:
            semantic_ids_arr = semantic_ids_arr[idx_arr]
        if len(se_responses) > max_idx:
            se_responses = [se_responses[i] for i in idx_arr]
        if response_embs_all is not None and len(response_embs_all) > max_idx:
            response_embs_all = [response_embs_all[i] for i in idx_arr]
        if response_slt_all is not None and len(response_slt_all) > max_idx:
            response_slt_all = [response_slt_all[i] for i in idx_arr]
        if len(aug_cluster_embs) > max_idx:
            aug_cluster_embs = [aug_cluster_embs[i] for i in idx_arr]

    responses: List[List[str]] = []
    se_scores: List[float] = []
    semantic_ids_all: List[List[int]] = []
    semantic_cluster_embeddings_all: List[Dict[int, np.ndarray]] = []
    pred_evidence_all: List[Dict[int, float]] = []
    dissonance_all: List[float] = []
    vacuity_all: List[float] = []

    with torch.inference_mode():
        for qid in range(len(X_flat)): # per question
            H_qid = X_flat[qid].unsqueeze(0).to(DEVICE) if feature_type_norm == "tbg_concat" else None
            q_tbg_hidden = (
                X_flat[qid].reshape(-1).to(DEVICE)
                if feature_type_norm in {"tbg_slt_hidden", "tbg_slt_concat"}
                else None
            )
            q_slt_hidden = (
                question_slt_flat[qid].reshape(-1).to(DEVICE)
                if feature_type_norm == "tbg_slt_concat" and question_slt_flat is not None
                else None
            )
            evidence_dict: Dict[int, float] = {}

            if response_level_mode:
                ids_row = np.asarray(semantic_ids_arr[qid], dtype=int) if qid < len(semantic_ids_arr) else np.array([], dtype=int)
                if ids_row.size == 0:
                    responses.append([])
                    semantic_ids_all.append([])
                    semantic_cluster_embeddings_all.append({})
                    pred_evidence_all.append({})
                    dissonance_all.append(0.0)
                    vacuity_all.append(0.0)
                    se_scores.append(0.0)
                    continue

                unique_cids, counts = np.unique(ids_row, return_counts=True)
                order = np.argsort(-counts)
                if n_fixed_clusters > 0:
                    order = order[:n_fixed_clusters]
                sorted_clusters = [(int(unique_cids[i]), int(counts[i])) for i in order]
            else:
                aug_embeddings = aug_cluster_embs[qid] if qid < len(aug_cluster_embs) else {}
                sorted_clusters = sorted(
                    aug_embeddings.items(), key=lambda item: item[1].get("count", 0), reverse=True
                )
                if n_fixed_clusters > 0:
                    sorted_clusters = sorted_clusters[:n_fixed_clusters]

            selected_responses: List[str] = []
            curr_cluster_embeddings: Dict[int, np.ndarray] = {}
            selected_aug_semantic_ids: List[int] = []

            if response_level_mode:
                resp_embs_q = response_embs_all[qid] if response_embs_all is not None and qid < len(response_embs_all) else {}
                ids_row = np.asarray(semantic_ids_arr[qid], dtype=int) if qid < len(semantic_ids_arr) else np.array([], dtype=int)
                for cid_int, _cnt in sorted_clusters: # per cluster
                    resp_indices = np.where(ids_row == int(cid_int))[0].tolist()
                    resp_candidates = [int(rid) for rid in resp_indices if int(rid) in resp_embs_q]
                    if not resp_candidates:
                        continue
                    pred_vals: List[float] = []
                    emb_list: List[np.ndarray] = []
                    for resp_id in resp_candidates:
                        emb = resp_embs_q.get(int(resp_id))
                        if emb is None:
                            print(f"Empty embedding for response id {resp_id} in question {qid}.")
                            continue
                        if feature_type_norm == "slt":
                            if response_slt_all is None or qid >= len(response_slt_all):
                                continue
                            slt_vec = response_slt_all[qid].get(int(resp_id))
                            if slt_vec is None:
                                continue
                            H_idx = torch.as_tensor(slt_vec, dtype=torch.float32, device=DEVICE).reshape(1, -1)
                            C_idx = torch.empty((1, 0), dtype=torch.float32, device=DEVICE)
                        elif feature_type_norm == "tbg_slt_hidden":
                            if response_slt_all is None or qid >= len(response_slt_all):
                                continue
                            slt_vec = response_slt_all[qid].get(int(resp_id))
                            if slt_vec is None:
                                continue
                            slt_tensor = torch.as_tensor(
                                slt_vec,
                                dtype=torch.float32,
                                device=DEVICE,
                            ).reshape(-1)
                            if q_tbg_hidden is None:
                                continue
                            H_idx = _compose_tbg_slt_hidden_tensor(
                                q_tbg_hidden,
                                slt_tensor,
                                hidden_layer_count=int(tbg_slt_layer_count) if tbg_slt_layer_count is not None else 0,
                            ).reshape(1, -1)
                            C_idx = torch.empty((1, 0), dtype=torch.float32, device=DEVICE)
                        elif feature_type_norm == "tbg_slt_concat":
                            if q_tbg_hidden is None or q_slt_hidden is None:
                                continue
                            emb_tensor = torch.as_tensor(
                                emb,
                                dtype=torch.float32,
                                device=DEVICE,
                            ).reshape(-1)
                            H_idx = _compose_tbg_slt_concat_tensor(
                                q_tbg_hidden,
                                q_slt_hidden,
                                emb_tensor,
                            ).reshape(1, -1)
                            C_idx = torch.empty((1, 0), dtype=torch.float32, device=DEVICE)
                        else:
                            H_idx = H_qid
                            C_idx = torch.as_tensor(emb.reshape(-1, emb.shape[0])).to(torch.float32).to(DEVICE)
                        pred_vals.append(float(eup(H_idx, C_idx).item()))
                        emb_list.append(np.asarray(emb, dtype=np.float32))
                    if not pred_vals:
                        continue
                    selected_aug_semantic_ids.append(int(cid_int))
                    evidence_dict[int(cid_int)] = float(np.mean(pred_vals))
                    if emb_list:
                        curr_cluster_embeddings[int(cid_int)] = np.stack(emb_list, axis=0).mean(axis=0, dtype=np.float32)

                    if qid < len(semantic_ids_arr) and qid < len(se_responses):
                        semantic_ids_q = semantic_ids_arr[qid].tolist()
                        responses_q = se_responses[qid]
                        selected_responses.extend(
                            [resp for sid, resp in zip(semantic_ids_q, responses_q) if int(sid) == int(cid_int)]
                        )
            else:
                for cid, meta in sorted_clusters: # per cluster
                    emb = meta.get("emb")
                    if emb is None:
                        continue
                    cid_int = int(cid)
                    selected_aug_semantic_ids.append(cid_int)
                    curr_cluster_embeddings[cid_int] = emb

                    C_idx = torch.as_tensor(emb.reshape(-1, emb.shape[0])).to(torch.float32).to(DEVICE)
                    pred_val = float(eup(H_qid, C_idx).item())
                    evidence_dict[cid_int] = pred_val

                    # gather responses matching this cid if available
                    if qid < len(semantic_ids_arr) and qid < len(se_responses):
                        semantic_ids_q = semantic_ids_arr[qid].tolist()
                        responses_q = se_responses[qid]
                        selected_responses.extend(
                            [resp for sid, resp in zip(semantic_ids_q, responses_q) if int(sid) == cid_int]
                        )

            responses.append(selected_responses)
            semantic_ids_all.append(selected_aug_semantic_ids) # not meaningful, no metadata map back
            semantic_cluster_embeddings_all.append(curr_cluster_embeddings)
            pred_evidence_all.append(evidence_dict)

            dissonance_all.append(dissonance(evidence_dict))
            vacuity_all.append(vacuity(evidence_dict))

            if evidence_dict:
                evidence_vals = np.array(list(evidence_dict.values()), dtype=np.float32)
                prob = evidence_vals / (np.sum(evidence_vals) + 1e-9)
                se_scores.append(predictive_entropy_rao(np.log(prob + 1e-9)))
            else:
                se_scores.append(0.0)

    est_se_scores = get_estimated_se(pred_evidence_all)

    if uncertainty_labels is not None:
        hallucination_labels = np.asarray(uncertainty_labels, dtype=int)
    else:
        hallucination_labels = np.array(y == 0).astype(int)
    metric_eval_mask, ood_refusal_eval = compute_ood_refusal_filter(
        labels=hallucination_labels,
        dataset_name=dataset_name or "",
        split=split,
        remove_refusal=remove_refusal,
        load_dir=load_dir,
        idx_subset=idx_arr,
    )
    metric_eval_mask = combine_optional_eval_mask(
        metric_eval_mask,
        extra_eval_mask=extra_eval_mask,
        idx_subset=idx_arr,
        label=f"fix_n_existing_semantic_cluster_eval:{split}",
    )
    print(f"[EUP Eval] MSE between GT SE and Est SE: {np.mean((np.array(se_scores) - np.array(est_se_scores))**2):.4f}")

    vacuity_arr = np.array(vacuity_all)
    dissonance_arr = np.array(dissonance_all)
    eval_labels = np.asarray(hallucination_labels, dtype=int)[metric_eval_mask]
    eval_vacuity = vacuity_arr[metric_eval_mask]
    eval_dissonance = dissonance_arr[metric_eval_mask]
    eval_se = np.asarray(se_scores, dtype=float)[metric_eval_mask]
    eval_est_se = np.asarray(est_se_scores, dtype=float)[metric_eval_mask]
    eval_cluster_sizes = np.asarray([len(e_dict) for e_dict in pred_evidence_all], dtype=float)[metric_eval_mask]
    unique_labels = np.unique(eval_labels)
    if unique_labels.size < 2:
        print(
            f"[EUP Eval][Fix N Existing Clusters] Skipping AUROC/AUPR for {split} split "
            f"(only one class present: {unique_labels.tolist()})."
        )
        auroc_cluster_size = float("nan")
        auroc_vacuity = float("nan")
        auroc_dissonance = float("nan")
        auroc_se = float("nan")
        auroc_est_se = float("nan")
        aupr_vacuity = float("nan")
        aupr_dissonance = float("nan")
        aupr_se = float("nan")
        aupr_est_se = float("nan")
    else:
        auroc_cluster_size = roc_auc_score(eval_labels, eval_cluster_sizes)
        print(f"[DEBUG] {split} AUROC by cluster size: {auroc_cluster_size:.4f}")
        auroc_vacuity = roc_auc_score(eval_labels, eval_vacuity)
        auroc_dissonance = roc_auc_score(eval_labels, eval_dissonance)
        auroc_se = roc_auc_score(eval_labels, eval_se)
        auroc_est_se = roc_auc_score(eval_labels, eval_est_se)
        aupr_vacuity = average_precision_score(eval_labels, eval_vacuity)
        aupr_dissonance = average_precision_score(eval_labels, eval_dissonance)
        aupr_se = average_precision_score(eval_labels, eval_se)
        aupr_est_se = average_precision_score(eval_labels, eval_est_se)
        print(
            f"[EUP Eval][Fix N Existing Clusters] {split} AUROC vacuity: {auroc_vacuity:.4f}, dissonance: {auroc_dissonance:.4f}, se: {auroc_se:.4f}, est_se: {auroc_est_se:.4f}"
        )
        print(
            f"[EUP Eval][Fix N Existing Clusters] {split} AUPR vacuity: {aupr_vacuity:.4f}, dissonance: {aupr_dissonance:.4f}, se: {aupr_se:.4f}, est_se: {aupr_est_se:.4f}"
        )

    return {
        "responses": responses,
        "semantic_ids": semantic_ids_all,
        "se_scores": se_scores,
        "est_se_scores": est_se_scores,
        "y": y,
        "uncertainty_labels": hallucination_labels,
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "ood_refusal_eval": ood_refusal_eval,
        "semantic_cluster_embeddings": semantic_cluster_embeddings_all,
        "pred_evidence": pred_evidence_all,
        "dissonance": dissonance_all,
        "vacuity": vacuity_all,
        "AUROC": {"vacuity": auroc_vacuity , "dissonance": auroc_dissonance, "se": auroc_se, "est_se": auroc_est_se},
        "AUPR": {"vacuity": aupr_vacuity , "dissonance": aupr_dissonance, "se": aupr_se, "est_se": aupr_est_se},
    }


def main(args):
    timeout_min = int(os.environ.get("SE_DDP_TIMEOUT_MINUTES", "180"))
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=timeout_min))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(42, device_specific=False)
    if accelerator.is_main_process:
        print(f"[DDP] process group timeout set to {timeout_min} minutes")

    resolved_model_name = _resolve_hf_model_name(args.model_name)
    model_id = args.model_name.split("/")[-1]
    needs_8bit = _needs_8bit_quantized_loading(args.model_name)
    dtype = torch.float16 if "llama-2-7b" in args.model_name.lower() else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name, torch_dtype=dtype, padding_side='left', verbose=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.pad_token_id  = tokenizer.eos_token_id

    proc_train_ds, proc_valid_ds, proc_test_ds, _ = prepare_splits(
        dataset_name      = args.dataset_name,
        tokenizer         = tokenizer,
        n_samples         = args.n_samples,
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

    output_dir = _resolve_feature_output_dir(
        model_id=model_id,
        dataset_name=args.dataset_name,
        n_samples=int(args.n_samples),
        judge_type=args.judge_type,
    )
    _validate_feature_output_alignment(
        output_dir=output_dir,
        split_datasets={
            "train": proc_train_ds,
            "valid": proc_valid_ds,
            "test": proc_test_ds,
        },
        accelerator=accelerator,
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
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        elif "gemma-3" in args.model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                torch_dtype=dtype,
                local_files_only=True,
            )
        elif _is_gemma_4_e4b_model_name(args.model_name):
            if Gemma4ForCausalLM is None:
                raise ImportError("Gemma4ForCausalLM is unavailable in the installed transformers package.")
            model = Gemma4ForCausalLM.from_pretrained(resolved_model_name, torch_dtype=dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(resolved_model_name, torch_dtype=dtype)
        if not needs_8bit:
            model.to(accelerator.device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                "Model must fit on a single GPU per process for Accelerator data-parallel collection. "
                "Use a smaller model or lower memory settings and retry."
            ) from e
        raise

    data_collator = EPNDataCollator(tokenizer, True)

    # collecting & saving
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
        print(f"[SE gather] run token: {run_token}")
    accelerator.wait_for_everyone()

    entailment_model = (
        EntailmentDeberta(device=accelerator.device)
        if args.entailment_model == "Deberta"
        else None
    )
    
    cfg = SESamplingConfig()
    cfg.device = str(accelerator.device)
    cfg.force_equal_batches = bool(args.force_equal_batches)
    if accelerator.is_main_process:
        print(f"[DDP] force_equal_batches={cfg.force_equal_batches}")

    shapes = {}
    shapes["train"] = _collect_and_save(
        "train", proc_train_ds, model, entailment_model, args.strict_entailment,
        tokenizer, data_collator, args.n_resp, cfg, output_dir,
        MSE_type=args.loss_type,
        accelerator=accelerator,
        dataset_name=args.dataset_name,
    )
    shapes["valid"] = _collect_and_save(
        "valid", proc_valid_ds, model, entailment_model, args.strict_entailment,
        tokenizer, data_collator, args.n_resp, cfg, output_dir,
        MSE_type=args.loss_type,
        accelerator=accelerator,
        dataset_name=args.dataset_name,
    )
    shapes["test"] = _collect_and_save(
        "test", proc_test_ds, model, entailment_model, args.strict_entailment,
        tokenizer, data_collator, args.n_resp, cfg, output_dir,
        MSE_type=args.loss_type,
        accelerator=accelerator,
        dataset_name=args.dataset_name,
    )

    if not accelerator.is_main_process:
        return

    disambig_fewshot_pool = _extract_disambig_fewshot_pool_from_processed_splits(
        proc_train_ds,
        proc_valid_ds,
        proc_test_ds,
    )
    meta = {
        "collector": "semantic_entropy",
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "n_samples": int(args.n_samples),
        "n_responses": int(args.n_resp),
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
        "tokenizer": {
            "padding_side": tokenizer.padding_side,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "shapes": shapes,
    }
    if disambig_fewshot_pool is not None:
        meta["disambig_fewshot_pool"] = disambig_fewshot_pool
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Finished collecting SE] Saved to:", output_dir)
    pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Semantic Cluster Collector for EUP")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", type=str,
                        default="trivia_qa_split",
                        help="Dataset name (e.g., trivia_qa_split, squad_split, nq_open, boolq, ambig_qa, ambig_qa_disambig, situated_qa, situated_qa_disambig)")
    parser.add_argument("--judge_type", type=str, default="rouge")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--valid_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--test_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--n_resp", type=int, default=10)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="response-wise",
        help="cluster-wise | response-wise | ranking",
    )
    parser.add_argument("--entailment_model", type=str, default="Deberta")
    parser.add_argument("--strict_entailment", type=_bool, default=False)
    parser.add_argument(
        "--fix_val_test_splits",
        type=_bool,
        default=False,
        help="Use deterministic validation/test memberships to match feature_collector when enabled.",
    )
    parser.add_argument(
        "--force_equal_batches",
        type=_bool,
        default=True,
        help="Pad per-rank sample slots so each process runs the same number of dataloader batches.",
    )

    args = parser.parse_args()

    main(args)
