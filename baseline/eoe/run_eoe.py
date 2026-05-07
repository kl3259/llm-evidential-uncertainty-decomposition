from __future__ import annotations

import argparse
import copy
import os
import sys
import types
from datetime import timedelta
from distutils.util import strtobool as _bool
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from accelerate import Accelerator
except Exception:  # pragma: no cover
    Accelerator = None

try:
    from accelerate.utils import InitProcessGroupKwargs
except Exception:  # pragma: no cover
    InitProcessGroupKwargs = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _install_feature_collector_stubs() -> None:
    try:
        import evaluate  # noqa: F401
    except Exception:
        evaluate_stub = types.ModuleType("evaluate")

        def _missing_evaluate_load(*args, **kwargs):
            raise ModuleNotFoundError(
                "evaluate package is required for judge-related paths in feature_collector."
            )

        evaluate_stub.load = _missing_evaluate_load
        sys.modules["evaluate"] = evaluate_stub

    try:
        import openai  # noqa: F401
    except Exception:
        openai_stub = types.ModuleType("openai")

        class _DummyAzureOpenAI:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise ModuleNotFoundError(
                    "openai package is required for GPT-judge paths in feature_collector."
                )

        class _DummyBadRequestError(Exception):
            pass

        openai_stub.AzureOpenAI = _DummyAzureOpenAI
        openai_stub.BadRequestError = _DummyBadRequestError
        sys.modules["openai"] = openai_stub

    try:
        import openai_harmony  # noqa: F401
    except Exception:
        harmony_stub = types.ModuleType("openai_harmony")

        class _DummyEncoding:
            def parse_messages_from_completion_tokens(self, *args, **kwargs):
                del args, kwargs
                return []

            def decode(self, token_ids):
                del token_ids
                return ""

            def stop_tokens_for_assistant_actions(self):
                return None

        class _HarmonyEncodingName:
            HARMONY_GPT_OSS = "HARMONY_GPT_OSS"

        class _Role:
            ASSISTANT = "assistant"

        def _load_harmony_encoding(name):
            del name
            return _DummyEncoding()

        class _Placeholder:
            pass

        harmony_stub.HarmonyEncodingName = _HarmonyEncodingName
        harmony_stub.load_harmony_encoding = _load_harmony_encoding
        harmony_stub.Conversation = _Placeholder
        harmony_stub.Message = _Placeholder
        harmony_stub.Role = _Role
        harmony_stub.SystemContent = _Placeholder
        harmony_stub.DeveloperContent = _Placeholder
        sys.modules["openai_harmony"] = harmony_stub


_install_feature_collector_stubs()

from helper_functions import count_or_frac, load_question_json, load_ref_json  # noqa: E402

from generate_paraphrase import BartParaphraser, ParaphraseConfig  # noqa: E402
from split_helpers import eoe_feature_fn  # noqa: E402
from utils import (  # noqa: E402
    TextEmbedder,
    TextEmbedderConfig,
    build_summary_metrics,
    l2_normalize,
    log_summary_to_wandb,
    set_all_seeds,
)


WORKING_DIR = PROJECT_ROOT
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PARAPHRASE_MODEL = "eugenesiow/bart-paraphrase"
LLAMA_33_70B_SUFFIX = "llama-3.3-70b-instruct"

DATASET_ALIAS_MAP = {
    "trivia_qa_split": "trivia_qa_split",
    "nq_open": "nq_open",
    "alcuna_trivia_qa": "alcuna_trivia_qa",
    "temporalood_trivia_qa": "temporalood_trivia_qa",
    "ambig_qa": "ambig_qa",
    "ambig_qa_filtered": "ambig_qa_filtered",
    "ambig_qa_filtered_ver2": "ambig_qa_filtered_ver2",
    "ambig_qa_filtered_ver3": "ambig_qa_filtered_ver3",
    "situated_qa": "situated_qa",
    "boolq": "boolq",
    "squad": "squad_split",
    "squad_split": "squad_split",
}

AMBIG_QA_FILTERED_DATASET_NAMES = {
    "ambig_qa_filtered",
    "ambig_qa_filtered_ver2",
    "ambig_qa_filtered_ver3",
}

SPLIT_ALIAS_MAP = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "dev": "valid",
    "test": "test",
}

_build_features = None
_prepare_splits = None
_decode_gpt_oss_completion_fn = None
_ensure_bitsandbytes_available_fn = None
_needs_8bit_quantized_loading_fn = None
_resolve_hf_model_name_fn = None
_resolve_model_input_device_fn = None
_resolve_single_gpu_device_map_fn = None


class _SingleProcessAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_processes = 1
        self.process_index = 0
        self.is_main_process = True
        self.is_local_main_process = True

    def wait_for_everyone(self) -> None:
        return


def _load_feature_collector_functions() -> None:
    global _build_features
    global _prepare_splits
    global _decode_gpt_oss_completion_fn
    global _ensure_bitsandbytes_available_fn
    global _needs_8bit_quantized_loading_fn
    global _resolve_hf_model_name_fn
    global _resolve_model_input_device_fn
    global _resolve_single_gpu_device_map_fn
    if _build_features is not None:
        return
    try:
        from feature_collector import (
            _ensure_bitsandbytes_available,
            _decode_gpt_oss_completion,
            _needs_8bit_quantized_loading,
            _resolve_hf_model_name,
            _resolve_model_input_device,
            _resolve_single_gpu_device_map,
            build_features,
            prepare_splits,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import feature_collector."
        ) from exc
    _build_features = build_features
    _prepare_splits = prepare_splits
    _decode_gpt_oss_completion_fn = _decode_gpt_oss_completion
    _ensure_bitsandbytes_available_fn = _ensure_bitsandbytes_available
    _needs_8bit_quantized_loading_fn = _needs_8bit_quantized_loading
    _resolve_hf_model_name_fn = _resolve_hf_model_name
    _resolve_model_input_device_fn = _resolve_model_input_device
    _resolve_single_gpu_device_map_fn = _resolve_single_gpu_device_map


def _canonical_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    if key not in DATASET_ALIAS_MAP:
        supported = sorted(DATASET_ALIAS_MAP.keys())
        raise ValueError(f"Unsupported dataset_name `{dataset_name}`. Supported: {supported}")
    return DATASET_ALIAS_MAP[key]


def _canonical_split_name(split_name: str) -> str:
    key = str(split_name).strip().lower()
    if key not in SPLIT_ALIAS_MAP:
        supported = sorted(SPLIT_ALIAS_MAP.keys())
        raise ValueError(f"Unsupported eval_split `{split_name}`. Supported aliases: {supported}")
    return SPLIT_ALIAS_MAP[key]


def _normalize_model_name(model_name: str) -> str:
    return str(model_name or "").strip().lower().rstrip("/")


def _is_llama_33_70b_model_name(model_name: str) -> bool:
    norm = _normalize_model_name(model_name)
    if not norm:
        return False
    basename = norm.split("/")[-1]
    return basename.endswith(LLAMA_33_70B_SUFFIX) or norm.endswith(LLAMA_33_70B_SUFFIX)


def _is_distributed(accelerator) -> bool:
    return accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1


def _init_accelerator():
    if Accelerator is None:
        return _SingleProcessAccelerator(), None

    timeout_min = int(
        os.environ.get(
            "EOE_DDP_TIMEOUT_MINUTES",
            os.environ.get("SE_DDP_TIMEOUT_MINUTES", "180"),
        )
    )
    kwargs_handlers = []
    if InitProcessGroupKwargs is not None:
        kwargs_handlers.append(InitProcessGroupKwargs(timeout=timedelta(minutes=timeout_min)))
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
    return accelerator, timeout_min


def _split_indices_for_process(n_total: int, accelerator) -> List[int]:
    if not _is_distributed(accelerator):
        return list(range(n_total))
    shards = np.array_split(np.arange(n_total, dtype=np.int64), int(accelerator.num_processes))
    return shards[int(accelerator.process_index)].tolist()


def _gather_records_to_main(local_records: List[Dict[str, object]], accelerator):
    if not _is_distributed(accelerator):
        return local_records
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed gather requested but torch.distributed is not initialized. "
            "Launch with `accelerate launch --multi_gpu --num_processes N ...`."
        )

    world = int(accelerator.num_processes)
    if accelerator.is_main_process:
        gathered = [None for _ in range(world)]
        torch.distributed.gather_object(local_records, gathered, dst=0)
        merged: List[Dict[str, object]] = []
        for shard in gathered:
            if shard:
                merged.extend(shard)
        return merged

    torch.distributed.gather_object(local_records, None, dst=0)
    return None


def _load_split_dataset(
    *,
    dataset_name: str,
    tokenizer,
    model_name: str,
    judge_type: str,
    n_samples: int,
    valid_frac,
    test_frac,
    eval_split: str,
    seed: int,
    fix_val_test_splits: bool,
):
    if dataset_name in AMBIG_QA_FILTERED_DATASET_NAMES:
        return _load_filtered_ambig_qa_split(
            dataset_name=dataset_name,
            model_name=model_name,
            judge_type=judge_type,
            n_samples=n_samples,
            eval_split=eval_split,
        )

    train_ds, valid_ds, test_ds, _ = _prepare_splits(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        n_samples=int(n_samples),
        test_frac=test_frac,
        val_frac=valid_frac,
        seed=int(seed),
        fix_val_test_splits=bool(fix_val_test_splits),
        feature_fn=eoe_feature_fn,
    )
    split_map = {"train": train_ds, "valid": valid_ds, "test": test_ds}
    return split_map[eval_split]


def _normalize_reference_list(references) -> List[str]:
    if isinstance(references, list):
        ref_list = [str(ref) for ref in references if ref is not None and str(ref).strip() != ""]
    elif references is None:
        ref_list = []
    else:
        ref_list = [str(references)] if str(references).strip() != "" else []
    return ref_list or [""]


def _build_ambig_annotations_for_eoe(*, references: List[str], ambig_label: int):
    if int(ambig_label) == 1:
        return [{
            "type": ["multipleQAs"],
            "qaPairs": [{"answer": references}],
        }]
    return [{
        "type": ["singleAnswer"],
        "answer": references,
    }]


def _load_filtered_ambig_qa_split(
    *,
    dataset_name: str,
    model_name: str,
    judge_type: str,
    n_samples: int,
    eval_split: str,
):
    model_id = str(model_name).split("/")[-1]
    split_dir = os.path.join(
        WORKING_DIR,
        "input_hidden_states",
        f"{model_id}_{dataset_name}_{int(n_samples)}_{judge_type}",
        eval_split,
    )
    questions_path = os.path.join(split_dir, "questions.jsonl")
    references_path = os.path.join(split_dir, "references.jsonl")
    ambig_label_path = os.path.join(split_dir, "ambig_label.npy")

    missing_paths = [
        path
        for path in (questions_path, references_path, ambig_label_path)
        if not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Filtered ambig_qa EoE split is missing required artifacts in "
            f"{split_dir}: {missing_paths}"
        )

    questions = load_question_json(questions_path)
    references = load_ref_json(references_path)
    ambig_labels = np.asarray(np.load(ambig_label_path), dtype=np.int64).reshape(-1)

    n_questions = len(questions)
    if len(references) != n_questions or int(ambig_labels.shape[0]) != n_questions:
        raise ValueError(
            "Filtered ambig_qa EoE split length mismatch: "
            f"questions={n_questions}, references={len(references)}, "
            f"ambig_label={int(ambig_labels.shape[0])} in {split_dir}"
        )

    records: List[Dict[str, object]] = []
    for question, reference_row, ambig_label in zip(questions, references, ambig_labels.tolist()):
        ref_list = _normalize_reference_list(reference_row)
        records.append({
            "question": str(question),
            "annotations": _build_ambig_annotations_for_eoe(
                references=ref_list,
                ambig_label=int(ambig_label),
            ),
            "nq_answer": ref_list,
        })
    return records


def _validate_split_args(valid_frac, test_frac, fix_val_test_splits: bool) -> None:
    if not bool(fix_val_test_splits):
        return
    if not isinstance(valid_frac, int) or not isinstance(test_frac, int):
        raise ValueError(
            "When --fix_val_test_splits=True, --valid_frac and --test_frac must be integer counts "
            "(for example, --valid_frac=500 --test_frac=500)."
        )


def _build_prompt_payload(dataset_name: str, tokenizer, example: Dict, question_text: str) -> Dict:
    payload_example = copy.deepcopy(example)
    payload_example["question"] = question_text
    return _build_features(dataset_name, tokenizer, payload_example)


def _load_lm(*, model_name: str, dtype, accelerator):
    resolved_model_name = (
        _resolve_hf_model_name_fn(model_name)
        if _resolve_hf_model_name_fn is not None
        else model_name
    )
    needs_8bit = bool(
        _needs_8bit_quantized_loading_fn(model_name)
        if _needs_8bit_quantized_loading_fn is not None
        else False
    )
    is_llama_33_70b = _is_llama_33_70b_model_name(model_name)
    if is_llama_33_70b and _is_distributed(accelerator):
        raise RuntimeError(
            "Llama-3.3-70B-Instruct must run as a single process for EoE because "
            "device_map='auto' requires one process to own the full sharded model. "
            "Launch a single process with all target GPUs visible."
        )

    try:
        if needs_8bit:
            if _ensure_bitsandbytes_available_fn is None:
                raise RuntimeError(
                    "feature_collector quantization helpers were not loaded before EoE model init."
                )
            _ensure_bitsandbytes_available_fn()
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            if _is_distributed(accelerator):
                device_map = _resolve_single_gpu_device_map_fn(accelerator=accelerator)
            else:
                device_map = "auto"
            if accelerator.is_main_process:
                print(f"[EoE][int8] Loading {resolved_model_name} with device_map={device_map}.")
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                device_map=device_map,
                quantization_config=quant_config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        elif _is_distributed(accelerator):
            model = AutoModelForCausalLM.from_pretrained(resolved_model_name, torch_dtype=dtype)
            model.to(accelerator.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                device_map="auto",
                torch_dtype=dtype,
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "Model must fit on a single GPU per process for accelerate multi-GPU EoE runs. "
                "Use a smaller model, reduce memory usage, or run single-process with device_map='auto'."
            ) from exc
        raise

    model.eval()
    return model


def _decode_generation(
    *,
    seq_tokens: torch.Tensor,
    input_seq_len: int,
    tokenizer,
    model_name: str,
) -> str:
    pred_ids = seq_tokens[input_seq_len:]
    if "gpt-oss" in model_name.lower() and _decode_gpt_oss_completion_fn is not None:
        return _decode_gpt_oss_completion_fn(pred_ids).strip()
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


def _generate_sampled_responses(
    *,
    model,
    tokenizer,
    payload: Dict,
    inner_m: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    model_name: str,
) -> List[str]:
    if "input_ids" not in payload or "attention_mask" not in payload:
        raise KeyError("build_features payload missing input_ids/attention_mask.")

    model_device = (
        _resolve_model_input_device_fn(model)
        if _resolve_model_input_device_fn is not None
        else next(model.parameters()).device
    )
    input_ids = torch.as_tensor(payload["input_ids"], dtype=torch.long, device=model_device).unsqueeze(0)
    attention_mask = torch.as_tensor(
        payload["attention_mask"], dtype=torch.long, device=model_device
    ).unsqueeze(0)
    input_ids = input_ids.repeat(int(inner_m), 1)
    attention_mask = attention_mask.repeat(int(inner_m), 1)
    input_seq_len = int(input_ids.size(1))

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = [
        _decode_generation(
            seq_tokens=outputs[i],
            input_seq_len=input_seq_len,
            tokenizer=tokenizer,
            model_name=model_name,
        )
        for i in range(outputs.size(0))
    ]
    return responses


def _resolve_hallucination_labels(
    *,
    split_dir: str,
    n_questions: int,
) -> np.ndarray:
    y_path = os.path.join(split_dir, "y.npy")
    labels = np.full(n_questions, -1, dtype=np.int64)
    if not os.path.exists(y_path):
        print(f"[warn] Missing {y_path}; hallucination_label will be -1.")
        return labels

    y = np.asarray(np.load(y_path), dtype=np.int64).reshape(-1)
    if y.shape[0] != n_questions:
        print(
            f"[warn] y.npy length mismatch (y={y.shape[0]} vs eval={n_questions}); "
            "hallucination_label will be -1."
        )
        return labels
    return np.asarray(1 - y, dtype=np.int64)


def _resolve_primary_labels(
    *,
    ambig_label: np.ndarray,
    ood_label: np.ndarray,
    hallucination_label: np.ndarray,
) -> np.ndarray:
    out = np.full(ambig_label.shape[0], -1, dtype=np.int64)
    mask_ambig = ambig_label >= 0
    out[mask_ambig] = ambig_label[mask_ambig]
    mask_ood = np.logical_and(out < 0, ood_label >= 0)
    out[mask_ood] = ood_label[mask_ood]
    mask_hall = np.logical_and(out < 0, hallucination_label >= 0)
    out[mask_hall] = hallucination_label[mask_hall]
    return out


def _save_uncertainty_npz(
    *,
    out_path: str,
    au_trace: np.ndarray,
    eu_trace: np.ndarray,
    total_uncertainty: np.ndarray,
    label_resolved: np.ndarray,
    hallucination_label: np.ndarray,
    ood_label: np.ndarray,
    ambig_label: np.ndarray,
    sample_idx: np.ndarray,
    metadata: Dict[str, object],
    num_processed: int,
) -> None:
    np.savez_compressed(
        out_path,
        au_trace=np.asarray(au_trace, dtype=np.float32),
        eu_trace=np.asarray(eu_trace, dtype=np.float32),
        total_uncertainty=np.asarray(total_uncertainty, dtype=np.float32),
        label_resolved=np.asarray(label_resolved, dtype=np.int64),
        hallucination_label=np.asarray(hallucination_label, dtype=np.int64),
        ood_label=np.asarray(ood_label, dtype=np.int64),
        ambig_label=np.asarray(ambig_label, dtype=np.int64),
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        dataset_name=np.array(str(metadata["dataset_name"])),
        model_name=np.array(str(metadata["model_name"])),
        eval_split=np.array(str(metadata["eval_split"])),
        outer_n=np.array(int(metadata["outer_n"]), dtype=np.int64),
        inner_m=np.array(int(metadata["inner_m"]), dtype=np.int64),
        sigma=np.array(float(metadata["sigma"]), dtype=np.float32),
        paraphrase_pool_size=np.array(int(metadata["paraphrase_pool_size"]), dtype=np.int64),
        seed=np.array(int(metadata["seed"]), dtype=np.int64),
        num_processed=np.array(int(num_processed), dtype=np.int64),
    )


def _record_to_values(record: Dict[str, object]):
    return (
        int(record["qid"]),
        np.float32(float(record["au"])),
        np.float32(float(record["eu"])),
        np.float32(float(record["total"])),
        int(record["ambig_label"]),
        int(record["ood_label"]),
    )


def _populate_traces_from_records(
    *,
    n_questions: int,
    records: List[Dict[str, object]],
):
    ambig_label = np.full(n_questions, -1, dtype=np.int64)
    ood_label = np.full(n_questions, -1, dtype=np.int64)
    au_trace = np.full(n_questions, np.nan, dtype=np.float32)
    eu_trace = np.full(n_questions, np.nan, dtype=np.float32)
    total_uncertainty = np.full(n_questions, np.nan, dtype=np.float32)
    sample_idx = np.arange(n_questions, dtype=np.int64)

    for record in records:
        qid, au_val, eu_val, total_val, ambig_val, ood_val = _record_to_values(record)
        ambig_label[qid] = ambig_val
        ood_label[qid] = ood_val
        au_trace[qid] = au_val
        eu_trace[qid] = eu_val
        total_uncertainty[qid] = total_val

    return ambig_label, ood_label, au_trace, eu_trace, total_uncertainty, sample_idx


def _compute_question_record(
    *,
    qid: int,
    example: Dict,
    dataset_name: str,
    tokenizer,
    model,
    paraphraser,
    embedder,
    args,
) -> Optional[Dict[str, object]]:
    question = str(example.get("question", "")).strip()
    if question == "":
        return None

    base_payload = _build_prompt_payload(dataset_name, tokenizer, example, question)
    ambig_val = int(base_payload["ambig_label"]) if "ambig_label" in base_payload else -1
    ood_val = int(base_payload["ood_label"]) if "ood_label" in base_payload else -1

    paraphrase_pool = paraphraser.generate_pool(
        question=question,
        pool_size=int(args.paraphrase_pool_size),
        include_original=True,
        seed=int(args.seed + qid),
    )
    if len(paraphrase_pool) == 0:
        paraphrase_pool = [question]

    pool_embeddings = embedder.encode(paraphrase_pool, batch_size=args.batch_size_embed)
    question_embedding = embedder.encode([question], batch_size=args.batch_size_embed)[0]

    rng = np.random.default_rng(int(args.seed + qid))
    selected_questions: List[str] = []
    for _ in range(int(args.outer_n)):
        noisy = question_embedding + rng.normal(
            loc=0.0,
            scale=float(args.sigma),
            size=question_embedding.shape,
        ).astype(np.float32)
        noisy = l2_normalize(noisy)
        cosine_scores = np.matmul(pool_embeddings, noisy)
        best_idx = int(np.argmax(cosine_scores))
        selected_questions.append(paraphrase_pool[best_idx])

    outer_means: List[np.ndarray] = []
    outer_inner_traces: List[float] = []
    for outer_idx, para_question in enumerate(selected_questions):
        payload = _build_prompt_payload(dataset_name, tokenizer, example, para_question)
        responses = _generate_sampled_responses(
            model=model,
            tokenizer=tokenizer,
            payload=payload,
            inner_m=int(args.inner_m),
            max_new_tokens=int(args.response_max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            seed=int(args.seed + qid * 10000 + outer_idx),
            model_name=args.model_name,
        )
        response_embeddings = embedder.encode(responses, batch_size=args.batch_size_embed)
        response_embeddings = l2_normalize(response_embeddings)

        mu_i = response_embeddings.mean(axis=0)
        outer_means.append(mu_i.astype(np.float32, copy=False))
        if response_embeddings.shape[0] > 1:
            centered = response_embeddings - mu_i
            cov_i = (centered.T @ centered) / float(response_embeddings.shape[0] - 1)
            outer_inner_traces.append(float(np.trace(cov_i)))
        else:
            outer_inner_traces.append(0.0)

    if len(outer_means) == 0:
        return None

    au_val = float(np.mean(np.asarray(outer_inner_traces, dtype=np.float32)))
    means_arr = np.stack(outer_means, axis=0).astype(np.float32, copy=False)
    if means_arr.shape[0] > 1:
        centered_means = means_arr - means_arr.mean(axis=0, keepdims=True)
        cov_e = (centered_means.T @ centered_means) / float(means_arr.shape[0] - 1)
        eu_val = float(np.trace(cov_e))
    else:
        eu_val = 0.0

    total_val = float(au_val + eu_val)
    return {
        "qid": int(qid),
        "au": au_val,
        "eu": eu_val,
        "total": total_val,
        "ambig_label": ambig_val,
        "ood_label": ood_val,
    }


def run_eoe(args) -> None:
    accelerator, timeout_min = _init_accelerator()
    _load_feature_collector_functions()
    dataset_name = _canonical_dataset_name(args.dataset_name)
    eval_split = _canonical_split_name(args.eval_split)
    _validate_split_args(args.valid_frac, args.test_frac, args.fix_val_test_splits)
    set_all_seeds(args.seed)
    if timeout_min is not None and accelerator.is_main_process and _is_distributed(accelerator):
        print(f"[EoE][DDP] process group timeout set to {timeout_min} minutes")
        if int(args.save_every) > 0:
            print("[EoE][DDP] save_every partial checkpoints are disabled; only the final merged NPZ will be written.")

    if not torch.cuda.is_available():
        dtype = torch.float32
    else:
        dtype = torch.float16 if "llama-2-7b" in args.model_name.lower() else torch.bfloat16
    resolved_model_name = (
        _resolve_hf_model_name_fn(args.model_name)
        if _resolve_hf_model_name_fn is not None
        else args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        torch_dtype=dtype,
        padding_side="left",
        verbose=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = _load_lm(model_name=args.model_name, dtype=dtype, accelerator=accelerator)

    eval_dataset = _load_split_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        model_name=args.model_name,
        judge_type=args.judge_type,
        n_samples=args.n_samples,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        eval_split=eval_split,
        seed=args.seed,
        fix_val_test_splits=args.fix_val_test_splits,
    )
    n_questions = len(eval_dataset)
    if accelerator.is_main_process:
        print(f"[EoE] Loaded {n_questions} questions from dataset={dataset_name}, split={eval_split}.")

    model_id = args.model_name.split("/")[-1]
    split_dir = os.path.join(
        WORKING_DIR,
        "input_hidden_states",
        f"{model_id}_{dataset_name}_{args.n_samples}_{args.judge_type}",
        eval_split,
    )
    if accelerator.is_main_process:
        os.makedirs(split_dir, exist_ok=True)
    out_path = os.path.join(split_dir, "eoe_uncertainty.npz")
    run_name = (
        f"baseline_{dataset_name}_{int(args.n_samples)}_"
        f"{eval_split}_{args.baseline_method}_{model_id}"
    )
    wandb_config = {
        "dataset_name": dataset_name,
        "model_name": args.model_name,
        "baseline_method": args.baseline_method,
        "n_samples": int(args.n_samples),
        "eval_split": eval_split,
        "output_npz_path": out_path,
    }
    if accelerator.is_main_process and bool(args.wandb_log) and wandb.run is None:
        if wandb is None:
            raise ImportError("wandb is not installed but --wandb_log=True was requested.")
        init_kwargs = {"project": args.wandb_project, "name": run_name, "config": wandb_config}
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        wandb.init(**init_kwargs)

    paraphrase_config = ParaphraseConfig(
        model_name=args.paraphrase_model_name,
        batch_size=max(1, int(args.batch_size_paraphrase)),
    )
    helper_device = str(accelerator.device) if _is_distributed(accelerator) else None
    paraphraser = BartParaphraser(paraphrase_config, device=helper_device)
    embedder = TextEmbedder(
        TextEmbedderConfig(
            model_name=args.embedding_model_name,
            batch_size=max(1, int(args.batch_size_embed)),
            normalize=True,
            device=helper_device,
        )
    )
    if accelerator.is_main_process:
        print(f"[EoE] Text embedding backend: {embedder.backend}")

    metadata = {
        "dataset_name": dataset_name,
        "model_name": args.model_name,
        "eval_split": eval_split,
        "outer_n": args.outer_n,
        "inner_m": args.inner_m,
        "sigma": args.sigma,
        "paraphrase_pool_size": args.paraphrase_pool_size,
        "seed": args.seed,
    }

    if not _is_distributed(accelerator):
        hallucination_label = _resolve_hallucination_labels(split_dir=split_dir, n_questions=n_questions)
        ambig_label = np.full(n_questions, -1, dtype=np.int64)
        ood_label = np.full(n_questions, -1, dtype=np.int64)
        au_trace = np.full(n_questions, np.nan, dtype=np.float32)
        eu_trace = np.full(n_questions, np.nan, dtype=np.float32)
        total_uncertainty = np.full(n_questions, np.nan, dtype=np.float32)
        sample_idx = np.arange(n_questions, dtype=np.int64)

        for qid in tqdm(range(n_questions), total=n_questions, desc="EoE questions"):
            record = _compute_question_record(
                qid=int(qid),
                example=eval_dataset[int(qid)],
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                model=model,
                paraphraser=paraphraser,
                embedder=embedder,
                args=args,
            )
            if record is None:
                continue

            qid_idx, au_val, eu_val, total_val, ambig_val, ood_val = _record_to_values(record)
            ambig_label[qid_idx] = ambig_val
            ood_label[qid_idx] = ood_val
            au_trace[qid_idx] = au_val
            eu_trace[qid_idx] = eu_val
            total_uncertainty[qid_idx] = total_val

            if args.save_every > 0 and (qid_idx + 1) % int(args.save_every) == 0:
                label_resolved_partial = _resolve_primary_labels(
                    ambig_label=ambig_label[: qid_idx + 1],
                    ood_label=ood_label[: qid_idx + 1],
                    hallucination_label=hallucination_label[: qid_idx + 1],
                )
                _save_uncertainty_npz(
                    out_path=out_path,
                    au_trace=au_trace[: qid_idx + 1],
                    eu_trace=eu_trace[: qid_idx + 1],
                    total_uncertainty=total_uncertainty[: qid_idx + 1],
                    label_resolved=label_resolved_partial,
                    hallucination_label=hallucination_label[: qid_idx + 1],
                    ood_label=ood_label[: qid_idx + 1],
                    ambig_label=ambig_label[: qid_idx + 1],
                    sample_idx=sample_idx[: qid_idx + 1],
                    metadata=metadata,
                    num_processed=qid_idx + 1,
                )
    else:
        local_indices = _split_indices_for_process(n_questions, accelerator)
        local_records: List[Dict[str, object]] = []
        for qid in tqdm(
            local_indices,
            total=len(local_indices),
            desc=f"EoE questions rank {int(accelerator.process_index)}",
            disable=not accelerator.is_main_process,
        ):
            record = _compute_question_record(
                qid=int(qid),
                example=eval_dataset[int(qid)],
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                model=model,
                paraphraser=paraphraser,
                embedder=embedder,
                args=args,
            )
            if record is not None:
                local_records.append(record)

        gathered_records = _gather_records_to_main(local_records, accelerator)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return

        hallucination_label = _resolve_hallucination_labels(split_dir=split_dir, n_questions=n_questions)
        (
            ambig_label,
            ood_label,
            au_trace,
            eu_trace,
            total_uncertainty,
            sample_idx,
        ) = _populate_traces_from_records(
            n_questions=n_questions,
            records=gathered_records or [],
        )

    label_resolved = _resolve_primary_labels(
        ambig_label=ambig_label,
        ood_label=ood_label,
        hallucination_label=hallucination_label,
    )
    _save_uncertainty_npz(
        out_path=out_path,
        au_trace=au_trace,
        eu_trace=eu_trace,
        total_uncertainty=total_uncertainty,
        label_resolved=label_resolved,
        hallucination_label=hallucination_label,
        ood_label=ood_label,
        ambig_label=ambig_label,
        sample_idx=sample_idx,
        metadata=metadata,
        num_processed=n_questions,
    )
    print(f"[EoE] Saved uncertainty NPZ: {out_path}")

    summary_metrics = build_summary_metrics(
        dataset_name=dataset_name,
        model_name=args.model_name,
        baseline_method=args.baseline_method,
        n_samples=int(args.n_samples),
        eval_split=eval_split,
        label_resolved=label_resolved,
        au_trace=au_trace,
        eu_trace=eu_trace,
        total_trace=total_uncertainty,
    )
    print(
        "[EoE] AUROC/AUPR "
        f"aleatoric=({summary_metrics['aleatoric_auroc']}, {summary_metrics['aleatoric_aupr']}), "
        f"epistemic=({summary_metrics['epistemic_auroc']}, {summary_metrics['epistemic_aupr']}), "
        f"total=({summary_metrics['total_auroc']}, {summary_metrics['total_aupr']})"
    )

    run_url = None
    if bool(args.wandb_log):
        run_url = log_summary_to_wandb(
            summary_metrics=summary_metrics,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=run_name,
            config=wandb_config,
        )
    print(f"[EoE] wandb_project: {args.wandb_project}")
    print(f"[EoE] wandb_logged: {bool(args.wandb_log)}")
    if bool(args.wandb_log):
        print(f"[EoE] wandb_run_url: {run_url or 'unavailable'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone EoE baseline.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help=(
            "One of: trivia_qa_split, nq_open, alcuna_trivia_qa, temporalood_trivia_qa, "
            "ambig_qa, ambig_qa_filtered, ambig_qa_filtered_ver2, "
            "ambig_qa_filtered_ver3, situated_qa, boolq, squad."
        ),
    )
    parser.add_argument("--judge_type", type=str, default="rouge")
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument(
        "--valid_frac",
        type=count_or_frac,
        default=500,
        help="Validation split size as count or fraction. Must be an integer count when --fix_val_test_splits=True.",
    )
    parser.add_argument(
        "--test_frac",
        type=count_or_frac,
        default=500,
        help="Test split size as count or fraction. Must be an integer count when --fix_val_test_splits=True.",
    )
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument(
        "--fix_val_test_splits",
        type=_bool,
        default=True,
        help="Use deterministic val/test memberships shared with the main pipeline. Requires integer counts for valid/test split sizes.",
    )

    parser.add_argument("--outer_n", type=int, default=5)
    parser.add_argument("--inner_m", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--paraphrase_pool_size", type=int, default=500)

    parser.add_argument("--response_max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--top_p", type=float, default=0.99)

    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--paraphrase_model_name", type=str, default=DEFAULT_PARAPHRASE_MODEL)
    parser.add_argument("--batch_size_embed", type=int, default=32)
    parser.add_argument("--batch_size_paraphrase", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--baseline_method", type=str, default="EoE")
    parser.add_argument("--wandb_project", type=str, default="EPN-Baselines")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log", type=_bool, default=True)

    parsed_args = parser.parse_args()
    run_eoe(parsed_args)
