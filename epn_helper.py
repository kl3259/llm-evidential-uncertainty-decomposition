from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eud_helper import (
    build_probe as build_eud_probe,
    compute_binary_ece,
    fit_temperature_scaler,
    positive_probability_from_logits,
)
from feature_collector import _is_gemma_4_e4b_model_name, _needs_8bit_quantized_loading, _resolve_hf_model_name
from helper_functions import load_sequence_scores_npz
from uncertainty import dissonance, theils_u, vacuity

try:
    from eup import EvidentialUncertaintyProbe
except ImportError:
    EvidentialUncertaintyProbe = None

try:
    from transformers import Gemma4ForCausalLM
except ImportError:
    Gemma4ForCausalLM = None


WORKING_DIR = "./"
EPS = 1e-6
REFUSAL_ANS = [
    "I cannot help with",
    "I cannot assist with",
    "I cannot answer",
    "I cannot provide",
    "I couldn't find",
    "I cannot verify",
    "I couldn't verify",
    "I can't recall",
    "I don't have information",
    "I can't find any information",
    "Not found in",
    "I don't know",
    "I cannot recall",
    "I'm sorry",
    "I am sorry",
    "I apologize",
]


@dataclass
class LoadedSplitArtifacts:
    split: str
    split_dir: str
    X: Dict[str, np.ndarray]
    y: np.ndarray
    semantic_ids: np.ndarray
    uncertainty_labels: np.ndarray
    uncertainty_label_source: str
    response_embeddings: Optional[List[Dict[int, np.ndarray]]]
    response_slt_lastk: Optional[List[Dict[int, np.ndarray]]]
    paths: Dict[str, Optional[str]]


@dataclass
class SplitArtifacts:
    load_dir: str
    split_dir: str
    split: str
    responses: List[List[str]]
    semantic_ids: List[np.ndarray]
    response_sequence_probs: np.ndarray
    response_norm_sequence_probs: np.ndarray
    se_scores: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    label_source: str
    paths: Dict[str, Optional[str]]
    disambig_question_semantic_ids: Optional[List[np.ndarray]] = None
    disambig_question_sequence_probs: Optional[np.ndarray] = None
    disambig_question_norm_sequence_probs: Optional[np.ndarray] = None
    disambig_question_metrics_available: bool = False
    disambig_question_metrics_skip_reason: Optional[str] = None


@dataclass
class PrimaryHallTable:
    split: str
    qid: np.ndarray
    features: np.ndarray
    hall_labels: np.ndarray
    nonhall_labels: np.ndarray


@dataclass
class SampledResponseTable:
    split: str
    qid: np.ndarray
    resp_id: np.ndarray
    cid: np.ndarray
    K: np.ndarray
    features: np.ndarray
    semantic_prob_target: np.ndarray
    response_count: np.ndarray
    question_count: int
    uncertainty_labels: np.ndarray
    uncertainty_label_source: str
    primary_truth_labels: np.ndarray


@dataclass
class QuestionRhoTable:
    split: str
    qid: np.ndarray
    features: np.ndarray
    K: np.ndarray
    question_count: int
    uncertainty_labels: np.ndarray
    uncertainty_label_source: str
    primary_truth_labels: np.ndarray


@dataclass
class QuestionDensityTable:
    split: str
    qid: np.ndarray
    features: np.ndarray
    probe_labels: np.ndarray
    probe_target: str
    probe_label_source: str
    question_count: int
    uncertainty_labels: np.ndarray
    uncertainty_label_source: str
    primary_truth_labels: np.ndarray


class EPNScalarProbe(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        activation: str,
        n_hidden_layer: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        n_hidden = int(n_hidden_layer)
        self.input_dim = int(input_dim)
        self.activation = str(activation)
        self.n_hidden_layer = n_hidden
        self.hidden_dim = int(hidden_dim)
        if EvidentialUncertaintyProbe is None:
            raise ImportError("EvidentialUncertaintyProbe requires the optional eup module.")
        self.base = EvidentialUncertaintyProbe(
            h_dim=self.input_dim,
            c_dim=0,
            activation=self.activation,
            has_hidden=n_hidden > 0,
            n_hidden=n_hidden,
            hidden_dim=self.hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        empty = x.new_zeros((x.shape[0], 0))
        return self.base(x, empty)


class BinaryMLPProbe(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        in_dim = int(input_dim)
        hidden = int(hidden_dim)
        if hidden <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.input_dim = in_dim
        self.hidden_dim = hidden
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        self.hidden_act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.nn.functional.layer_norm(x, x.shape[-1:])
        hidden = self.hidden_act(self.fc1(x_norm))
        logits = self.fc2(hidden)
        return logits.squeeze(-1)


class _ZeroEvidenceProbe(torch.nn.Module):
    def forward(self, H: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        del C
        return torch.zeros(H.shape[0], dtype=torch.float32, device=H.device)


def seed_everything(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def artifact_load_dir(model_name: str, dataset_name: str, n_samples: int, judge_type: str) -> str:
    model_id = str(model_name).rstrip("/").split("/")[-1]
    return os.path.join(
        WORKING_DIR,
        "input_hidden_states",
        f"{model_id}_{dataset_name}_{int(n_samples)}_{judge_type}",
    )


def _artifact_load_dir(model_id: str, dataset_name: str, n_samples: int, judge_type: str) -> str:
    return os.path.join(
        WORKING_DIR,
        "input_hidden_states",
        f"{model_id}_{dataset_name}_{n_samples}_{judge_type}",
    )


def _response_slt_lastk_path(load_dir: str, split: str, n_resp: int) -> str:
    return os.path.join(load_dir, split, f"response_slt_lastk_{n_resp}.npz")


def _split_artifact_paths(load_dir: str, split: str, n_resp: int) -> Dict[str, Optional[str]]:
    from helper_functions import get_sampled_sequence_scores_filename

    split_dir = os.path.join(load_dir, split)
    return {
        "responses": os.path.join(split_dir, "se_resp_res.jsonl"),
        "semantic_ids": os.path.join(split_dir, f"semantic_ids_{n_resp}.npy"),
        "disambig_question_semantic_ids": os.path.join(split_dir, f"disambig_question_semantic_ids_{n_resp}.npy"),
        "sampled_response_sequence_scores": os.path.join(
            split_dir,
            get_sampled_sequence_scores_filename("se_resp_res.jsonl", n_resp),
        ),
        "se": os.path.join(split_dir, "se.npy"),
        "y": os.path.join(split_dir, "y.npy"),
        "response_slt_lastk": _response_slt_lastk_path(load_dir, split, n_resp),
        "preds": os.path.join(split_dir, "preds.jsonl"),
        "questions": os.path.join(split_dir, "questions.jsonl"),
        "references": os.path.join(split_dir, "references.jsonl"),
    }


def count_or_frac(s: str):
    text = s.strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Must be int (count) or float (fraction)") from exc


def detect_refusal_responses(
    preds: Sequence[str],
    refusal_answers: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = list(refusal_answers) if refusal_answers is not None else list(REFUSAL_ANS)
    candidates = [("" if c is None else str(c)).lower() for c in candidates]
    preds_list = [("" if p is None else str(p)).lower() for p in preds]
    match_scores = np.zeros(len(preds_list), dtype=np.float32)
    refusal_mask = np.zeros(len(preds_list), dtype=bool)
    if not candidates:
        return refusal_mask, match_scores
    for idx, pred in enumerate(preds_list):
        matched = any(candidate and candidate in pred for candidate in candidates)
        refusal_mask[idx] = bool(matched)
        match_scores[idx] = 1.0 if matched else 0.0
    return refusal_mask, match_scores


def compute_ood_refusal_filter(
    labels: Sequence[int],
    dataset_name: str,
    split: str,
    remove_refusal: bool,
    load_dir: Optional[str] = None,
    preds: Optional[Sequence[str]] = None,
    idx_subset: Optional[Sequence[int]] = None,
    refusal_answers: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    total = int(labels_arr.shape[0])
    mask_all = np.ones(total, dtype=bool)
    stats = {
        "applied": False,
        "accuracy": float("nan"),
        "excluded_ood_count": 0,
        "excluded_id_count": 0,
        "excluded_total_count": 0,
        "remaining_count": total,
        "total_count": total,
    }

    ds_name = str(dataset_name or "").strip().lower()
    split_name = str(split or "").strip().lower()
    if not (
        bool(remove_refusal)
        and ds_name in {"alcuna_trivia_qa", "repli_qa_trivia_qa", "temporalood_trivia_qa"}
        and split_name == "test"
    ):
        return mask_all, stats

    preds_list = None
    if preds is not None:
        preds_list = ["" if item is None else str(item) for item in preds]
    elif load_dir:
        preds_path = os.path.join(load_dir, split_name, "preds.jsonl")
        if os.path.exists(preds_path):
            try:
                from helper_functions import load_pred_json

                preds_list = load_pred_json(preds_path)
            except Exception as exc:
                print(f"[Refusal Filter] Failed to read {preds_path}: {exc}. Skipping refusal filtering.")
                return mask_all, stats
        else:
            print(f"[Refusal Filter] Missing {preds_path}. Skipping refusal filtering.")
            return mask_all, stats
    else:
        print("[Refusal Filter] Missing both preds and load_dir. Skipping refusal filtering.")
        return mask_all, stats

    if preds_list is None:
        return mask_all, stats

    if idx_subset is not None:
        idx_arr = np.asarray(idx_subset, dtype=np.int64).reshape(-1)
        if idx_arr.size > 0 and len(preds_list) > int(idx_arr.max()) and len(preds_list) != total:
            preds_list = [preds_list[int(i)] for i in idx_arr.tolist()]

    if len(preds_list) != total:
        print(
            "[Refusal Filter] preds/labels length mismatch "
            f"(preds={len(preds_list)}, labels={total}). Skipping refusal filtering."
        )
        return mask_all, stats

    try:
        refusal_mask, _ = detect_refusal_responses(
            preds=preds_list,
            refusal_answers=refusal_answers,
        )
    except Exception as exc:
        print(f"[Refusal Filter] Failed refusal substring matching: {exc}. Skipping refusal filtering.")
        return mask_all, stats
    refusal_mask = np.asarray(refusal_mask, dtype=bool)
    eval_mask = ~refusal_mask
    excluded_total = int(refusal_mask.sum())
    excluded_ood = int(np.logical_and(refusal_mask, labels_arr == 1).sum())
    excluded_id = int(np.logical_and(refusal_mask, labels_arr == 0).sum())
    acc = float(excluded_ood / excluded_total) if excluded_total > 0 else float("nan")
    stats.update(
        {
            "applied": True,
            "accuracy": acc,
            "excluded_ood_count": excluded_ood,
            "excluded_id_count": excluded_id,
            "excluded_total_count": excluded_total,
            "remaining_count": int(eval_mask.sum()),
            "total_count": total,
        }
    )
    return eval_mask, stats


def _prepare_exchange_dir(exchange_dir: str) -> None:
    Path(exchange_dir).mkdir(parents=True, exist_ok=True)
    run_token = f"epn_{int(time.time())}_{os.getpid()}_{random.randint(0, 999999)}"
    run_token_path = Path(exchange_dir) / "run_token.txt"
    tmp_path = run_token_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(run_token)
    os.replace(tmp_path, run_token_path)


def _build_itp_sampling_cfg(args, device: torch.device):
    class _RuntimeSamplingConfig:
        pass

    cfg = _RuntimeSamplingConfig()
    cfg.max_new_tokens = int(args.itp_max_new_tokens)
    cfg.batch_size = int(args.itp_batch_size)
    cfg.temperature = float(args.itp_temperature)
    cfg.top_p = float(args.itp_top_p)
    cfg.do_sample = bool(args.itp_do_sample)
    cfg.device = str(device)
    cfg.judge_type = str(args.judge_type)
    return cfg


def _normalize_question_key(question: object) -> str:
    return "" if question is None else str(question)


def _reference_signature(references: object) -> str:
    if isinstance(references, np.ndarray):
        references = references.tolist()
    if references is None:
        values = []
    elif isinstance(references, (list, tuple)):
        values = [str(ref) for ref in references]
    else:
        values = [str(references)]
    return json.dumps(values, ensure_ascii=False)


def _first_unused_index(candidates, used_indices) -> int | None:
    for idx in candidates:
        if int(idx) not in used_indices:
            return int(idx)
    return None


def _build_saved_order_split_datasets(load_dir: str, prepared_splits: Dict[str, object]) -> Dict[str, object]:
    from datasets import concatenate_datasets
    from helper_functions import load_question_json, load_ref_json

    combined_dataset = concatenate_datasets(
        [prepared_splits["train"], prepared_splits["valid"], prepared_splits["test"]]
    )
    question_to_indices = {}
    question_ref_to_indices = {}

    all_questions = list(combined_dataset["question"])
    all_references = list(combined_dataset["references"])
    for idx, (question, references) in enumerate(zip(all_questions, all_references)):
        question_key = _normalize_question_key(question)
        question_to_indices.setdefault(question_key, []).append(int(idx))
        pair_key = (question_key, _reference_signature(references))
        question_ref_to_indices.setdefault(pair_key, []).append(int(idx))

    split_datasets = {}
    used_indices = set()
    for split in ("train", "valid", "test"):
        questions_path = os.path.join(load_dir, split, "questions.jsonl")
        if not os.path.exists(questions_path):
            split_datasets[split] = prepared_splits[split]
            continue

        saved_questions = load_question_json(questions_path)
        references_path = os.path.join(load_dir, split, "references.jsonl")
        if os.path.exists(references_path):
            saved_references = load_ref_json(references_path)
        else:
            saved_references = [None] * len(saved_questions)

        if len(saved_references) != len(saved_questions):
            raise RuntimeError(
                f"Saved artifact length mismatch for {split}: "
                f"questions={len(saved_questions)}, references={len(saved_references)}."
            )

        matched_indices = []
        for qid, (question, references) in enumerate(zip(saved_questions, saved_references)):
            question_key = _normalize_question_key(question)
            matched_idx = _first_unused_index(
                question_ref_to_indices.get((question_key, _reference_signature(references)), ()),
                used_indices,
            )
            if matched_idx is None:
                matched_idx = _first_unused_index(
                    question_to_indices.get(question_key, ()),
                    used_indices,
                )
            if matched_idx is None:
                raise RuntimeError(
                    f"Unable to reconstruct saved prompt split for {split} at qid={qid}: "
                    f"question={question!r}. The saved artifact question was not found in the "
                    "processed dataset pool built for inference."
                )
            used_indices.add(int(matched_idx))
            matched_indices.append(int(matched_idx))

        split_datasets[split] = combined_dataset.select(matched_indices)

    return split_datasets


def _load_language_model_and_datasets(
    args,
    load_dir: str,
    device: torch.device,
    accelerator=None,
) -> Tuple[torch.nn.Module, object, Dict[str, object], object, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from feature_collector import (
        EPNDataCollator,
        build_features,
        prepare_splits,
        _ensure_bitsandbytes_available,
        _is_gemma_4_e4b_model_name,
        _needs_8bit_quantized_loading,
        _resolve_hf_model_name,
        _resolve_single_gpu_device_map,
    )

    try:
        from transformers import Gemma4ForCausalLM
    except ImportError:
        Gemma4ForCausalLM = None

    resolved_model_name = _resolve_hf_model_name(args.model_name)
    dtype = torch.float16 if "llama-2-7b" in args.model_name.lower() else torch.bfloat16
    distributed = accelerator is not None and int(accelerator.num_processes) > 1
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        torch_dtype=dtype,
        padding_side="left",
        verbose=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    needs_8bit = _needs_8bit_quantized_loading(args.model_name)
    if needs_8bit:
        _ensure_bitsandbytes_available()
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = _resolve_single_gpu_device_map(accelerator=accelerator, fallback_device=device)
        if accelerator is None or bool(accelerator.is_main_process):
            print(f"[EPN][int8] Loading {resolved_model_name} with device_map={device_map}.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map=device_map,
            quantization_config=quant_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    elif _is_gemma_4_e4b_model_name(args.model_name):
        if Gemma4ForCausalLM is None:
            raise ImportError("Gemma4ForCausalLM is unavailable in the installed transformers package.")
        if distributed:
            model = Gemma4ForCausalLM.from_pretrained(
                resolved_model_name,
                torch_dtype=dtype,
            )
            model.to(device)
        else:
            model = Gemma4ForCausalLM.from_pretrained(
                resolved_model_name,
                device_map="auto",
                torch_dtype=dtype,
            )
    else:
        if distributed:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                torch_dtype=dtype,
            )
            model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                device_map="auto",
                torch_dtype=dtype,
            )

    proc_train_ds, proc_valid_ds, proc_test_ds, fewshot_pool = prepare_splits(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        test_frac=args.test_frac,
        val_frac=args.valid_frac,
        seed=42,
        fix_val_test_splits=args.fix_val_test_splits,
        feature_fn=build_features,
    )
    prepared_splits = {
        "train": proc_train_ds,
        "valid": proc_valid_ds,
        "test": proc_test_ds,
    }
    split_datasets = _build_saved_order_split_datasets(load_dir, prepared_splits)
    data_collator = EPNDataCollator(tokenizer, True)
    return model, tokenizer, split_datasets, fewshot_pool, data_collator


def normalize_feature_type(feature_type: str) -> str:
    norm = str(feature_type or "tbg_concat").strip().lower()
    if norm == "tbg":
        norm = "tbg_concat"
    if norm not in {"tbg_concat", "slt", "tbg_slt_hidden", "tbg_slt_concat"}:
        raise ValueError(f"Unsupported --feature_type: {feature_type}")
    return norm


def normalize_density_probe_feature(density_probe_feature: str) -> str:
    norm = str(density_probe_feature or "tbg").strip().lower()
    if norm not in {"tbg", "slt"}:
        raise ValueError(
            f"Unsupported --density_probe_feature: {density_probe_feature}. Expected one of: tbg, slt."
        )
    return norm


def normalize_probe_target(probe_target: str) -> str:
    norm = str(probe_target or "hallucination").strip().lower()
    if norm not in {"hallucination", "ambiguity"}:
        raise ValueError(
            f"Unsupported --probe_target: {probe_target}. Expected one of: hallucination, ambiguity."
        )
    return norm


def layer_index_list(layer_range: Sequence[int]) -> List[int]:
    arr = np.asarray(layer_range, dtype=np.int64).reshape(-1)
    return arr.astype(np.int64).tolist()


def resolve_layer_range_indices(X_key: np.ndarray, layer_range: str) -> np.ndarray:
    x_arr = np.asarray(X_key, dtype=np.float32)
    if x_arr.ndim != 3:
        raise ValueError(f"Expected hidden states shaped [n_layers, n_items, hidden_dim], got {x_arr.shape}.")
    n_layers = int(x_arr.shape[0])
    mode = str(layer_range or "last_5").strip().lower()
    if mode == "all":
        return np.arange(0, n_layers, dtype=np.int64)
    if mode == "last_1":
        return np.arange(max(0, n_layers - 1), n_layers, dtype=np.int64)
    if mode == "last_5":
        k = min(5, n_layers)
        return np.arange(n_layers - k, n_layers, dtype=np.int64)
    raise ValueError(
        f"Unsupported --layer_range: {layer_range}. Expected one of: last_1, last_5, all."
    )


def load_X_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].astype(np.float32, copy=False) for key in payload.files}


def load_cluster_embeddings_npz(path: str) -> List[Dict[int, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        offsets = payload["offsets"].astype(np.int64)
        cluster_ids = payload["cluster_ids"].astype(np.int64)
        emb = payload["emb"].astype(np.float32)
        n_examples = int(payload["N"])
        dim = int(payload["dim"])

    out: List[Dict[int, np.ndarray]] = []
    for idx in range(n_examples):
        start = int(offsets[idx])
        stop = int(offsets[idx + 1])
        row: Dict[int, np.ndarray] = {}
        if stop > start:
            row_ids = cluster_ids[start:stop]
            row_emb = emb[start:stop, :dim]
            for cid, vec in zip(row_ids, row_emb):
                row[int(cid)] = np.asarray(vec, dtype=np.float32).reshape(-1)
        out.append(row)
    return out


def resolve_dataset_labels(split_dir: str, y: np.ndarray) -> Tuple[np.ndarray, str]:
    for filename in ("ambig_label.npy", "ood_label.npy"):
        path = os.path.join(split_dir, filename)
        if os.path.exists(path):
            return np.asarray(np.load(path), dtype=np.int64).reshape(-1), filename
    return np.asarray(1 - np.asarray(y, dtype=np.int64).reshape(-1), dtype=np.int64), "1-y"


def binary_ce_np(target: np.ndarray, pred: np.ndarray) -> float:
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    if target_arr.shape != pred_arr.shape:
        raise ValueError(f"Target/prediction shape mismatch: {target_arr.shape} vs {pred_arr.shape}.")
    pred_arr = np.clip(pred_arr, EPS, 1.0 - EPS)
    loss = -(target_arr * np.log(pred_arr) + (1.0 - target_arr) * np.log(1.0 - pred_arr))
    return float(np.mean(loss)) if loss.size > 0 else float("nan")


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=np.float32)
    out = np.empty_like(logits_arr, dtype=np.float32)
    pos_mask = logits_arr >= 0.0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-logits_arr[pos_mask]))
    exp_logits = np.exp(logits_arr[~pos_mask])
    out[~pos_mask] = exp_logits / (1.0 + exp_logits)
    return out.astype(np.float32, copy=False)


def safe_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(scores_arr)
    labels_arr = labels_arr[finite_mask]
    scores_arr = scores_arr[finite_mask]
    if labels_arr.size == 0 or np.unique(labels_arr).size < 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels_arr, scores_arr)),
        "aupr": float(average_precision_score(labels_arr, scores_arr)),
    }


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_split_artifacts(
    load_dir: str,
    split: str,
    n_resp: int,
    feature_type: str,
) -> LoadedSplitArtifacts:
    feature_type_norm = normalize_feature_type(feature_type)
    split_dir = os.path.join(load_dir, split)
    X_path = os.path.join(split_dir, "X.npz")
    y_path = os.path.join(split_dir, "y.npy")
    semantic_ids_path = os.path.join(split_dir, f"semantic_ids_{int(n_resp)}.npy")

    missing = [path for path in (X_path, y_path, semantic_ids_path) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    X = load_X_npz(X_path)
    y = np.asarray(np.load(y_path), dtype=np.int64).reshape(-1)
    semantic_ids = np.asarray(np.load(semantic_ids_path, allow_pickle=True), dtype=np.int64)
    uncertainty_labels, uncertainty_label_source = resolve_dataset_labels(split_dir, y)

    response_embeddings = None
    response_slt_lastk = None
    response_embeddings_path = None
    response_slt_lastk_path = None

    if feature_type_norm in {"tbg_concat", "tbg_slt_concat"}:
        response_embeddings_path = os.path.join(split_dir, f"response_embeddings_{int(n_resp)}.npz")
        if not os.path.exists(response_embeddings_path):
            raise FileNotFoundError(
                f"Missing response-level embedding artifact: {response_embeddings_path}."
            )
        response_embeddings = load_cluster_embeddings_npz(response_embeddings_path)

    if feature_type_norm in {"slt", "tbg_slt_hidden"}:
        response_slt_lastk_path = os.path.join(split_dir, f"response_slt_lastk_{int(n_resp)}.npz")
        if not os.path.exists(response_slt_lastk_path):
            raise FileNotFoundError(
                f"Missing response-level SLT artifact: {response_slt_lastk_path}."
            )
        response_slt_lastk = load_cluster_embeddings_npz(response_slt_lastk_path)

    return LoadedSplitArtifacts(
        split=split,
        split_dir=split_dir,
        X=X,
        y=y,
        semantic_ids=semantic_ids,
        uncertainty_labels=uncertainty_labels,
        uncertainty_label_source=uncertainty_label_source,
        response_embeddings=response_embeddings,
        response_slt_lastk=response_slt_lastk,
        paths={
            "X": X_path,
            "y": y_path,
            "semantic_ids": semantic_ids_path,
            "response_embeddings": response_embeddings_path,
            "response_slt_lastk": response_slt_lastk_path,
        },
    )


def normalize_flat_hidden(X_flat: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X_flat, dtype=np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X_arr / norms


def flatten_selected_question_hidden(
    X_key: np.ndarray,
    selected_layers: Sequence[int],
) -> np.ndarray:
    idx = np.asarray(selected_layers, dtype=np.int64).reshape(-1)
    X_input = np.asarray(X_key, dtype=np.float32)[idx, :, :]
    X_flat = np.transpose(X_input, (1, 0, 2)).reshape(X_input.shape[1], -1)
    return normalize_flat_hidden(X_flat).astype(np.float32, copy=False)


def validate_response_map_alignment(
    response_maps: Sequence[Dict[int, np.ndarray]],
    semantic_ids: np.ndarray,
    split: str,
    artifact_name: str,
) -> None:
    if len(response_maps) != int(semantic_ids.shape[0]):
        raise ValueError(
            f"{artifact_name} size mismatch for split={split}: "
            f"maps={len(response_maps)}, semantic_ids={semantic_ids.shape[0]}."
        )
    for qid, (row_map, ids_row) in enumerate(zip(response_maps, semantic_ids)):
        ids_arr = np.asarray(ids_row, dtype=np.int64).reshape(-1)
        expected = list(range(int(ids_arr.shape[0])))
        actual = sorted(int(key) for key in row_map.keys())
        if actual != expected:
            raise ValueError(
                f"{artifact_name} response-id mismatch at split={split}, qid={qid}: "
                f"expected={expected[:10]}{'...' if len(expected) > 10 else ''}, "
                f"got={actual[:10]}{'...' if len(actual) > 10 else ''}."
            )


def resolve_response_slt_selection(
    response_slt_all: Sequence[Dict[int, np.ndarray]],
    X_slt: np.ndarray,
    selected_layers: Sequence[int],
    split: str,
) -> List[Dict[int, np.ndarray]]:
    X_arr = np.asarray(X_slt, dtype=np.float32)
    total_layers = int(X_arr.shape[0])
    hidden_size = int(X_arr.shape[2])

    stored_dim = None
    for row_map in response_slt_all:
        if row_map:
            first_vec = next(iter(row_map.values()))
            stored_dim = int(np.asarray(first_vec, dtype=np.float32).reshape(-1).shape[0])
            break
    if stored_dim is None:
        return [dict() for _ in response_slt_all]
    if stored_dim % hidden_size != 0:
        raise ValueError(
            f"response_slt_lastk dim {stored_dim} is not divisible by hidden size {hidden_size}."
        )
    stored_layers = int(stored_dim // hidden_size)
    selected_arr = np.asarray(selected_layers, dtype=np.int64).reshape(-1)

    if stored_layers == total_layers:
        local_indices = selected_arr.tolist()
    else:
        start_global = total_layers - stored_layers
        if np.any(selected_arr < start_global):
            raise ValueError(
                "Requested sampled-response SLT layers are unavailable in response_slt_lastk artifacts "
                f"for split={split}: requested={selected_arr.tolist()}, available_global="
                f"{list(range(start_global, total_layers))}."
            )
        local_indices = (selected_arr - start_global).astype(np.int64).tolist()

    selected_maps: List[Dict[int, np.ndarray]] = []
    for row_map in response_slt_all:
        row_selected: Dict[int, np.ndarray] = {}
        for resp_id, vec in row_map.items():
            vec_arr = np.asarray(vec, dtype=np.float32).reshape(stored_layers, hidden_size)
            row_selected[int(resp_id)] = vec_arr[local_indices, :].reshape(-1).astype(np.float32, copy=False)
        selected_maps.append(row_selected)
    return selected_maps


def build_tbg_slt_hidden_features(
    H: np.ndarray,
    response_hidden_embs: List[Dict[int, np.ndarray]],
    semantic_ids: np.ndarray,
    hidden_layer_count: int,
) -> List[Dict[int, np.ndarray]]:
    H_np = np.asarray(H, dtype=np.float32)
    semantic_ids_np = np.asarray(semantic_ids, dtype=np.int64)
    if H_np.ndim != 2:
        raise ValueError(f"H must be 2D, got {H_np.shape}.")
    if semantic_ids_np.ndim != 2:
        raise ValueError(f"semantic_ids must be 2D, got {semantic_ids_np.shape}.")
    if int(H_np.shape[0]) != int(semantic_ids_np.shape[0]):
        raise ValueError(f"H/semantic_ids size mismatch: {H_np.shape[0]} vs {semantic_ids_np.shape[0]}.")
    if len(response_hidden_embs) != int(semantic_ids_np.shape[0]):
        raise ValueError(
            "response_hidden_embs/semantic_ids size mismatch: "
            f"{len(response_hidden_embs)} vs {semantic_ids_np.shape[0]}."
        )

    k_layers = int(hidden_layer_count)
    if k_layers <= 0:
        raise ValueError(f"hidden_layer_count must be positive, got {hidden_layer_count}.")
    if int(H_np.shape[1]) % k_layers != 0:
        raise ValueError(
            "H hidden dimension is not divisible by hidden_layer_count: "
            f"h_dim={H_np.shape[1]}, hidden_layer_count={k_layers}."
        )

    hidden_size = int(H_np.shape[1] // k_layers)
    eps = 1e-12
    composed_hidden: List[Dict[int, np.ndarray]] = []

    for qid in range(int(semantic_ids_np.shape[0])):
        u_flat = np.asarray(H_np[qid], dtype=np.float32).reshape(-1)
        u_layerwise = u_flat.reshape(k_layers, hidden_size)
        row_map = response_hidden_embs[qid]
        qid_composed: Dict[int, np.ndarray] = {}
        for resp_id in range(int(semantic_ids_np.shape[1])):
            v_raw = row_map.get(int(resp_id))
            if v_raw is None:
                raise ValueError(
                    f"Missing response hidden vector for tbg_slt_hidden at qid={qid}, resp_id={resp_id}."
                )
            v_flat = np.asarray(v_raw, dtype=np.float32).reshape(-1)
            if int(v_flat.size) != int(H_np.shape[1]):
                raise ValueError(
                    "Response hidden size mismatch for tbg_slt_hidden: "
                    f"expected={H_np.shape[1]}, got={v_flat.size}."
                )
            v_layerwise = v_flat.reshape(k_layers, hidden_size)
            abs_diff = np.abs(u_layerwise - v_layerwise).reshape(-1)
            prod = (u_layerwise * v_layerwise).reshape(-1)
            dot = np.sum(u_layerwise * v_layerwise, axis=1)
            u_norm = np.linalg.norm(u_layerwise, axis=1)
            v_norm = np.linalg.norm(v_layerwise, axis=1)
            cos = dot / (u_norm * v_norm + eps)
            feat = np.concatenate(
                [u_flat, v_flat, abs_diff, prod, np.clip(cos, -1.0, 1.0).astype(np.float32)],
                axis=0,
            )
            qid_composed[int(resp_id)] = np.nan_to_num(
                feat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
        composed_hidden.append(qid_composed)
    return composed_hidden


def build_tbg_slt_concat_features(
    H: np.ndarray,
    question_slt_hidden: np.ndarray,
    response_embeddings: List[Dict[int, np.ndarray]],
    semantic_ids: np.ndarray,
) -> List[Dict[int, np.ndarray]]:
    H_np = np.asarray(H, dtype=np.float32)
    S_np = np.asarray(question_slt_hidden, dtype=np.float32)
    semantic_ids_np = np.asarray(semantic_ids, dtype=np.int64)
    if H_np.ndim != 2 or S_np.ndim != 2:
        raise ValueError(f"Expected 2D question features, got H={H_np.shape}, S={S_np.shape}.")
    if H_np.shape != S_np.shape:
        raise ValueError(f"TBG/SLT question feature shape mismatch: {H_np.shape} vs {S_np.shape}.")
    if int(H_np.shape[0]) != int(semantic_ids_np.shape[0]):
        raise ValueError(f"Question feature/semantic_ids mismatch: {H_np.shape[0]} vs {semantic_ids_np.shape[0]}.")
    if len(response_embeddings) != int(semantic_ids_np.shape[0]):
        raise ValueError(
            "response_embeddings/semantic_ids size mismatch: "
            f"{len(response_embeddings)} vs {semantic_ids_np.shape[0]}."
        )

    composed_hidden: List[Dict[int, np.ndarray]] = []
    for qid in range(int(semantic_ids_np.shape[0])):
        u_flat = np.asarray(H_np[qid], dtype=np.float32).reshape(-1)
        v_flat = np.asarray(S_np[qid], dtype=np.float32).reshape(-1)
        qid_composed: Dict[int, np.ndarray] = {}
        for resp_id in range(int(semantic_ids_np.shape[1])):
            e_raw = response_embeddings[qid].get(int(resp_id))
            if e_raw is None:
                raise ValueError(
                    f"Missing response embedding for tbg_slt_concat at qid={qid}, resp_id={resp_id}."
                )
            feat = np.concatenate([u_flat, v_flat, np.asarray(e_raw, dtype=np.float32).reshape(-1)], axis=0)
            qid_composed[int(resp_id)] = np.nan_to_num(
                feat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
        composed_hidden.append(qid_composed)
    return composed_hidden


def build_primary_hall_table(
    artifacts: LoadedSplitArtifacts,
    layer_range: str,
) -> PrimaryHallTable:
    if "slt" not in artifacts.X:
        raise ValueError(f"Primary hallucination probe requires X['slt'] in split={artifacts.split}.")
    selected_layers = resolve_layer_range_indices(artifacts.X["slt"], layer_range)
    features = flatten_selected_question_hidden(artifacts.X["slt"], selected_layers)
    hall_labels = np.asarray(1 - artifacts.y, dtype=np.int64).reshape(-1)
    if features.shape[0] != hall_labels.shape[0]:
        raise ValueError(
            f"Primary hall table feature/label mismatch for split={artifacts.split}: "
            f"{features.shape[0]} vs {hall_labels.shape[0]}."
        )
    return PrimaryHallTable(
        split=artifacts.split,
        qid=np.arange(features.shape[0], dtype=np.int64),
        features=features.astype(np.float32, copy=False),
        hall_labels=hall_labels,
        nonhall_labels=np.asarray(1 - hall_labels, dtype=np.float32),
    )


def build_question_rho_table(
    artifacts: LoadedSplitArtifacts,
    layer_range: str,
) -> QuestionRhoTable:
    if "tbg" not in artifacts.X:
        raise ValueError(f"Question-wise rho probe requires X['tbg'] in split={artifacts.split}.")
    selected_layers = resolve_layer_range_indices(artifacts.X["tbg"], layer_range)
    features = flatten_selected_question_hidden(artifacts.X["tbg"], selected_layers)
    if features.shape[0] != int(artifacts.semantic_ids.shape[0]):
        raise ValueError(
            f"Question-wise rho feature/question mismatch for split={artifacts.split}: "
            f"features={features.shape[0]}, semantic_ids={artifacts.semantic_ids.shape[0]}."
        )
    K = np.asarray(
        [len(np.unique(np.asarray(row, dtype=np.int64).reshape(-1))) for row in artifacts.semantic_ids],
        dtype=np.float32,
    )
    return QuestionRhoTable(
        split=artifacts.split,
        qid=np.arange(features.shape[0], dtype=np.int64),
        features=features.astype(np.float32, copy=False),
        K=K,
        question_count=int(features.shape[0]),
        uncertainty_labels=np.asarray(artifacts.uncertainty_labels, dtype=np.int64).reshape(-1),
        uncertainty_label_source=str(artifacts.uncertainty_label_source),
        primary_truth_labels=np.asarray(artifacts.y, dtype=np.int64).reshape(-1),
    )


def build_question_density_table(
    artifacts: LoadedSplitArtifacts,
    layer_range: str,
    density_probe_feature: str = "tbg",
    probe_target: str = "hallucination",
) -> QuestionDensityTable:
    feature_kind = normalize_density_probe_feature(density_probe_feature)
    if feature_kind not in artifacts.X:
        raise ValueError(
            f"Question-wise density probe requires X['{feature_kind}'] in split={artifacts.split}."
        )
    selected_layers = resolve_layer_range_indices(artifacts.X[feature_kind], layer_range)
    features = flatten_selected_question_hidden(artifacts.X[feature_kind], selected_layers)
    resolved_probe_target = normalize_probe_target(probe_target)
    if resolved_probe_target == "hallucination":
        probe_labels = np.asarray(1 - artifacts.y, dtype=np.int64).reshape(-1)
        probe_label_source = "1-y"
    else:
        ambig_label_path = os.path.join(artifacts.split_dir, "ambig_label.npy")
        if not os.path.exists(ambig_label_path):
            raise FileNotFoundError(
                "probe_target='ambiguity' requires ambig_label.npy for "
                f"split={artifacts.split}: {ambig_label_path}"
            )
        probe_labels = np.asarray(np.load(ambig_label_path), dtype=np.int64).reshape(-1)
        probe_label_source = "ambig_label.npy"
    if features.shape[0] != probe_labels.shape[0]:
        raise ValueError(
            f"Question-wise density feature/label mismatch for split={artifacts.split}: "
            f"features={features.shape[0]}, probe_labels={probe_labels.shape[0]}."
        )
    return QuestionDensityTable(
        split=artifacts.split,
        qid=np.arange(features.shape[0], dtype=np.int64),
        features=features.astype(np.float32, copy=False),
        probe_labels=probe_labels,
        probe_target=resolved_probe_target,
        probe_label_source=str(probe_label_source),
        question_count=int(features.shape[0]),
        uncertainty_labels=np.asarray(artifacts.uncertainty_labels, dtype=np.int64).reshape(-1),
        uncertainty_label_source=str(artifacts.uncertainty_label_source),
        primary_truth_labels=np.asarray(artifacts.y, dtype=np.int64).reshape(-1),
    )


def build_sampled_response_table(
    artifacts: LoadedSplitArtifacts,
    feature_type: str,
    layer_range: str,
) -> SampledResponseTable:
    feature_type_norm = normalize_feature_type(feature_type)
    feature_maps: List[Dict[int, np.ndarray]]

    if feature_type_norm == "tbg_concat":
        if "tbg" not in artifacts.X:
            raise ValueError(f"feature_type=tbg_concat requires X['tbg'] in split={artifacts.split}.")
        selected_layers = resolve_layer_range_indices(artifacts.X["tbg"], layer_range)
        H_tbg = flatten_selected_question_hidden(artifacts.X["tbg"], selected_layers)
        if artifacts.response_embeddings is None:
            raise ValueError("feature_type=tbg_concat requires response_embeddings artifacts.")
        validate_response_map_alignment(
            response_maps=artifacts.response_embeddings,
            semantic_ids=artifacts.semantic_ids,
            split=artifacts.split,
            artifact_name="response_embeddings",
        )
        feature_maps = []
        for qid in range(int(artifacts.semantic_ids.shape[0])):
            qid_map: Dict[int, np.ndarray] = {}
            for resp_id in range(int(artifacts.semantic_ids.shape[1])):
                emb = artifacts.response_embeddings[qid].get(int(resp_id))
                if emb is None:
                    raise ValueError(
                        f"Missing response embedding at split={artifacts.split}, qid={qid}, resp_id={resp_id}."
                    )
                qid_map[int(resp_id)] = np.concatenate(
                    [H_tbg[qid], np.asarray(emb, dtype=np.float32).reshape(-1)],
                    axis=0,
                ).astype(np.float32, copy=False)
            feature_maps.append(qid_map)
    elif feature_type_norm == "slt":
        if "slt" not in artifacts.X:
            raise ValueError(f"feature_type=slt requires X['slt'] in split={artifacts.split}.")
        selected_layers = resolve_layer_range_indices(artifacts.X["slt"], layer_range)
        if artifacts.response_slt_lastk is None:
            raise ValueError("feature_type=slt requires response_slt_lastk artifacts.")
        validate_response_map_alignment(
            response_maps=artifacts.response_slt_lastk,
            semantic_ids=artifacts.semantic_ids,
            split=artifacts.split,
            artifact_name="response_slt_lastk",
        )
        feature_maps = resolve_response_slt_selection(
            response_slt_all=artifacts.response_slt_lastk,
            X_slt=artifacts.X["slt"],
            selected_layers=selected_layers,
            split=artifacts.split,
        )
    elif feature_type_norm == "tbg_slt_hidden":
        if "tbg" not in artifacts.X:
            raise ValueError(f"feature_type=tbg_slt_hidden requires X['tbg'] in split={artifacts.split}.")
        if "slt" not in artifacts.X:
            raise ValueError(f"feature_type=tbg_slt_hidden requires X['slt'] in split={artifacts.split}.")
        selected_layers = resolve_layer_range_indices(artifacts.X["tbg"], layer_range)
        H_tbg = flatten_selected_question_hidden(artifacts.X["tbg"], selected_layers)
        if artifacts.response_slt_lastk is None:
            raise ValueError("feature_type=tbg_slt_hidden requires response_slt_lastk artifacts.")
        validate_response_map_alignment(
            response_maps=artifacts.response_slt_lastk,
            semantic_ids=artifacts.semantic_ids,
            split=artifacts.split,
            artifact_name="response_slt_lastk",
        )
        response_slt_selected = resolve_response_slt_selection(
            response_slt_all=artifacts.response_slt_lastk,
            X_slt=artifacts.X["slt"],
            selected_layers=selected_layers,
            split=artifacts.split,
        )
        feature_maps = build_tbg_slt_hidden_features(
            H=H_tbg,
            response_hidden_embs=response_slt_selected,
            semantic_ids=artifacts.semantic_ids,
            hidden_layer_count=int(len(selected_layers)),
        )
    elif feature_type_norm == "tbg_slt_concat":
        if "tbg" not in artifacts.X:
            raise ValueError(f"feature_type=tbg_slt_concat requires X['tbg'] in split={artifacts.split}.")
        if artifacts.response_embeddings is None:
            raise ValueError("feature_type=tbg_slt_concat requires response_embeddings artifacts.")
        if "slt" not in artifacts.X:
            raise ValueError("feature_type=tbg_slt_concat requires X['slt'].")
        selected_layers = resolve_layer_range_indices(artifacts.X["tbg"], layer_range)
        H_tbg = flatten_selected_question_hidden(artifacts.X["tbg"], selected_layers)
        validate_response_map_alignment(
            response_maps=artifacts.response_embeddings,
            semantic_ids=artifacts.semantic_ids,
            split=artifacts.split,
            artifact_name="response_embeddings",
        )
        H_slt = flatten_selected_question_hidden(artifacts.X["slt"], selected_layers)
        feature_maps = build_tbg_slt_concat_features(
            H=H_tbg,
            question_slt_hidden=H_slt,
            response_embeddings=artifacts.response_embeddings,
            semantic_ids=artifacts.semantic_ids,
        )
    else:
        raise ValueError(f"Unsupported feature_type={feature_type_norm}.")

    qids: List[int] = []
    resp_ids: List[int] = []
    cids: List[int] = []
    K_vals: List[int] = []
    features: List[np.ndarray] = []
    semantic_prob_targets: List[float] = []
    response_counts: List[int] = []

    for qid, ids_row in enumerate(artifacts.semantic_ids):
        ids_arr = np.asarray(ids_row, dtype=np.int64).reshape(-1)
        n_resp_q = int(ids_arr.shape[0])
        response_counts.append(n_resp_q)
        if n_resp_q == 0:
            continue
        cluster_counter = Counter(ids_arr.tolist())
        K_q = int(len(cluster_counter))
        row_map = feature_maps[qid]
        for resp_id in range(n_resp_q):
            feat = row_map.get(int(resp_id))
            if feat is None:
                raise ValueError(
                    f"Missing sampled feature at split={artifacts.split}, qid={qid}, resp_id={resp_id}."
                )
            cid = int(ids_arr[resp_id])
            qids.append(int(qid))
            resp_ids.append(int(resp_id))
            cids.append(cid)
            K_vals.append(K_q)
            features.append(np.asarray(feat, dtype=np.float32).reshape(-1))
            semantic_prob_targets.append(float(cluster_counter[cid]) / float(n_resp_q))

    if not features:
        raise ValueError(f"No sampled-response features were built for split={artifacts.split}.")

    feature_matrix = np.stack(features, axis=0).astype(np.float32, copy=False)
    return SampledResponseTable(
        split=artifacts.split,
        qid=np.asarray(qids, dtype=np.int64),
        resp_id=np.asarray(resp_ids, dtype=np.int64),
        cid=np.asarray(cids, dtype=np.int64),
        K=np.asarray(K_vals, dtype=np.float32),
        features=feature_matrix,
        semantic_prob_target=np.asarray(semantic_prob_targets, dtype=np.float32),
        response_count=np.asarray(response_counts, dtype=np.int64),
        question_count=int(artifacts.semantic_ids.shape[0]),
        uncertainty_labels=np.asarray(artifacts.uncertainty_labels, dtype=np.int64).reshape(-1),
        uncertainty_label_source=str(artifacts.uncertainty_label_source),
        primary_truth_labels=np.asarray(artifacts.y, dtype=np.int64).reshape(-1),
    )


def fit_primary_hallucination_probe(
    train_table: PrimaryHallTable,
    eval_tables: Dict[str, PrimaryHallTable],
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    unique_train_labels = np.unique(np.asarray(train_table.hall_labels, dtype=np.int64))
    clf: Optional[LogisticRegression]
    constant_hall_prob: Optional[float]
    if unique_train_labels.size < 2:
        clf = None
        constant_hall_prob = float(unique_train_labels[0]) if unique_train_labels.size == 1 else 0.0
    else:
        clf = LogisticRegression(
            max_iter=int(max_iter),
            solver="liblinear",
            random_state=int(random_state),
        )
        clf.fit(train_table.features, train_table.hall_labels)
        constant_hall_prob = None

    split_outputs: Dict[str, Dict[str, Any]] = {}
    for split, table in eval_tables.items():
        if clf is None:
            hall_prob = np.full(table.hall_labels.shape[0], float(constant_hall_prob), dtype=np.float32)
            classes = unique_train_labels.tolist()
        else:
            probs = clf.predict_proba(table.features)
            classes = clf.classes_.tolist()
            hall_col = int(classes.index(1)) if 1 in classes else int(np.argmax(classes))
            hall_prob = probs[:, hall_col].astype(np.float32, copy=False)
        nonhall_prob = (1.0 - hall_prob).astype(np.float32, copy=False)
        split_outputs[split] = {
            "hall_prob": hall_prob,
            "nonhall_prob": nonhall_prob,
            "loss": binary_ce_np(table.hall_labels.astype(np.float32), hall_prob),
            "metrics": safe_binary_metrics(table.hall_labels, hall_prob),
            "count": int(table.hall_labels.shape[0]),
        }

    return {
        "clf": clf,
        "constant_hall_prob": constant_hall_prob,
        "feature_dim": int(train_table.features.shape[1]),
        "max_iter": int(max_iter),
        "classes": unique_train_labels.tolist() if clf is None else clf.classes_.tolist(),
        "splits": split_outputs,
    }


def build_probe(
    input_dim: int,
    activation: str,
    n_hidden_layer: int,
    hidden_dim: int,
    device: torch.device,
) -> EPNScalarProbe:
    probe = EPNScalarProbe(
        input_dim=int(input_dim),
        activation=str(activation),
        n_hidden_layer=int(n_hidden_layer),
        hidden_dim=int(hidden_dim),
    )
    probe.to(device)
    return probe


def _batch_indices(n_items: int, batch_size: int, shuffle: bool, rng: np.random.Generator) -> Iterable[np.ndarray]:
    order = np.arange(int(n_items), dtype=np.int64)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, int(n_items), int(batch_size)):
        yield order[start : start + int(batch_size)]


def _predict_probe(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    out_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(X_arr.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(X_arr.shape[0]))
            xb = torch.from_numpy(X_arr[start:stop]).to(device)
            pred = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
            out_parts.append(pred)
    return np.concatenate(out_parts, axis=0) if out_parts else np.zeros(0, dtype=np.float32)


def train_semantic_probe(
    train_table: SampledResponseTable,
    valid_table: SampledResponseTable,
    n_hidden_layer: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    model = build_probe(
        input_dim=int(train_table.features.shape[1]),
        activation="sigmoid",
        n_hidden_layer=int(n_hidden_layer),
        hidden_dim=int(hidden_dim),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    best_state = None
    best_epoch = 0
    best_valid_loss = float("inf")
    history: List[Dict[str, float]] = []

    X_train = np.asarray(train_table.features, dtype=np.float32)
    y_train = np.asarray(train_table.semantic_prob_target, dtype=np.float32)
    X_valid = np.asarray(valid_table.features, dtype=np.float32)
    y_valid = np.asarray(valid_table.semantic_prob_target, dtype=np.float32)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for idx in _batch_indices(X_train.shape[0], batch_size, shuffle=True, rng=rng):
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = torch.clamp(model(xb), EPS, 1.0 - EPS)
            loss = F.binary_cross_entropy(pred, yb, reduction="mean")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.shape[0])
            total_count += int(idx.shape[0])

        train_loss = total_loss / float(max(1, total_count))
        valid_pred = np.clip(_predict_probe(model, X_valid, batch_size, device), EPS, 1.0 - EPS)
        valid_loss = binary_ce_np(y_valid, valid_pred)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Semantic probe training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "history": history,
        "input_dim": int(train_table.features.shape[1]),
        "n_hidden_layer": int(n_hidden_layer),
        "hidden_dim": int(hidden_dim),
    }


def predict_semantic_probe(
    model: torch.nn.Module,
    tables: Dict[str, SampledResponseTable],
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for split, table in tables.items():
        pred = _predict_probe(model, table.features, batch_size=batch_size, device=device)
        out[split] = np.clip(pred.astype(np.float32, copy=False), EPS, 1.0 - EPS)
    return out


def predict_semantic_probe_array(
    model: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    pred = _predict_probe(model, features, batch_size=batch_size, device=device)
    return np.clip(pred.astype(np.float32, copy=False), EPS, 1.0 - EPS)


def build_density_probe(
    input_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> BinaryMLPProbe:
    probe = BinaryMLPProbe(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
    )
    probe.to(device)
    return probe


def _predict_density_logits(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    out_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(X_arr.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(X_arr.shape[0]))
            xb = torch.from_numpy(X_arr[start:stop]).to(device)
            logits = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
            out_parts.append(logits)
    return np.concatenate(out_parts, axis=0) if out_parts else np.zeros(0, dtype=np.float32)


def _density_outputs_from_logits(logits: np.ndarray) -> Dict[str, np.ndarray]:
    logits_arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    hall_prob = sigmoid_np(logits_arr)
    rho = np.log1p(np.exp(-np.abs(logits_arr))) + np.maximum(logits_arr, 0.0)
    return {
        "logits": logits_arr.astype(np.float32, copy=False),
        "hall_prob": hall_prob.astype(np.float32, copy=False),
        "rho": rho.astype(np.float32, copy=False),
    }


def train_density_probe(
    train_table: QuestionDensityTable,
    valid_table: QuestionDensityTable,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    max_iter: int,
    tol: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    model = build_density_probe(
        input_dim=int(train_table.features.shape[1]),
        hidden_dim=int(hidden_dim),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    rng = np.random.default_rng(int(seed))
    prev_train_loss: Optional[float] = None
    converged = False
    final_epoch = 0
    history: List[Dict[str, float]] = []

    X_train = np.asarray(train_table.features, dtype=np.float32)
    y_train = np.asarray(train_table.probe_labels, dtype=np.float32)
    X_valid = np.asarray(valid_table.features, dtype=np.float32)
    y_valid = np.asarray(valid_table.probe_labels, dtype=np.float32)

    for epoch in range(1, int(max_iter) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for idx in _batch_indices(X_train.shape[0], batch_size, shuffle=True, rng=rng):
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(idx.shape[0])
            total_count += int(idx.shape[0])

        train_loss = total_loss / float(max(1, total_count))
        valid_logits = _predict_density_logits(model, X_valid, batch_size=batch_size, device=device)
        valid_prob = sigmoid_np(valid_logits)
        valid_loss = binary_ce_np(y_valid, valid_prob)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "valid_loss": valid_loss})
        final_epoch = int(epoch)
        if prev_train_loss is not None and abs(prev_train_loss - train_loss) <= float(tol):
            converged = True
            break
        prev_train_loss = float(train_loss)

    final_logits = _predict_density_logits(model, X_train, batch_size=batch_size, device=device)
    final_prob = sigmoid_np(final_logits)
    final_train_loss = binary_ce_np(y_train, final_prob)
    return {
        "model": model,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "input_dim": int(train_table.features.shape[1]),
        "hidden_dim": int(hidden_dim),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "n_iter": int(final_epoch),
        "converged": bool(converged),
        "history": history,
        "final_train_loss": float(final_train_loss),
        "final_valid_loss": float(history[-1]["valid_loss"]) if history else float("nan"),
    }


def predict_density_probe(
    model: torch.nn.Module,
    tables: Dict[str, QuestionDensityTable],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split, table in tables.items():
        out[split] = predict_density_probe_array(
            model=model,
            features=table.features,
            batch_size=batch_size,
            device=device,
        )
    return out


def predict_density_probe_array(
    model: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    logits = _predict_density_logits(model, features, batch_size=batch_size, device=device)
    return _density_outputs_from_logits(logits)


def p_epn_from_components(
    K: np.ndarray,
    rho: np.ndarray,
    semantic_prob: np.ndarray,
) -> np.ndarray:
    K_arr = np.asarray(K, dtype=np.float32).reshape(-1)
    rho_arr = np.asarray(rho, dtype=np.float32).reshape(-1)
    semantic_arr = np.asarray(semantic_prob, dtype=np.float32).reshape(-1)
    pred = (K_arr * rho_arr * semantic_arr + 1.0) / (K_arr * (1.0 + rho_arr) + EPS)
    return np.clip(pred.astype(np.float32, copy=False), EPS, 1.0 - EPS)


def _response_indices_by_qid(sampled_table: SampledResponseTable) -> List[np.ndarray]:
    return [
        np.where(np.asarray(sampled_table.qid, dtype=np.int64) == int(qid))[0].astype(np.int64, copy=False)
        for qid in range(int(sampled_table.question_count))
    ]


def train_rho_probe(
    train_table: QuestionRhoTable,
    valid_table: QuestionRhoTable,
    train_sampled_table: SampledResponseTable,
    valid_sampled_table: SampledResponseTable,
    semantic_prob_by_split: Dict[str, np.ndarray],
    hall_non_prob_by_split: Dict[str, np.ndarray],
    n_hidden_layer: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    # Legacy rho training objective disabled after the probe-2 redesign.
    #
    # The previous loss was:
    #   p_epn_q = torch.clamp(
    #       (K_q * rho_q * p_sem_q + 1.0) / (K_q * (1.0 + rho_q) + EPS),
    #       EPS,
    #       1.0 - EPS,
    #   )
    #   loss_terms.append(F.binary_cross_entropy(p_epn_q, target_q, reduction="mean"))
    #
    # The active standalone path now trains `train_density_probe(...)` instead.
    raise NotImplementedError("Legacy rho training is disabled. Use train_density_probe instead.")


def predict_rho_probe(
    model: torch.nn.Module,
    tables: Dict[str, QuestionRhoTable],
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for split, table in tables.items():
        pred = _predict_probe(model, table.features, batch_size=batch_size, device=device)
        out[split] = np.maximum(pred.astype(np.float32, copy=False), 0.0)
    return out


def compute_questionwise_rho_loss_and_response_predictions(
    question_table: QuestionRhoTable,
    sampled_table: SampledResponseTable,
    response_indices_by_qid: Optional[List[np.ndarray]],
    semantic_prob_pred: np.ndarray,
    hall_non_prob_by_q: np.ndarray,
    rho_pred_by_q: np.ndarray,
) -> Tuple[float, np.ndarray]:
    if int(question_table.question_count) != int(sampled_table.question_count):
        raise ValueError(
            "Question-count mismatch for question-wise rho loss: "
            f"question_table={question_table.question_count}, sampled_table={sampled_table.question_count}."
        )
    if response_indices_by_qid is None:
        response_indices_by_qid = _response_indices_by_qid(sampled_table)

    semantic_arr = np.asarray(semantic_prob_pred, dtype=np.float32).reshape(-1)
    rho_arr = np.asarray(rho_pred_by_q, dtype=np.float32).reshape(-1)
    nonhall_arr = np.asarray(hall_non_prob_by_q, dtype=np.float32).reshape(-1)
    if rho_arr.shape[0] != int(question_table.question_count):
        raise ValueError(
            f"Expected question-wise rho predictions with length={question_table.question_count}, got {rho_arr.shape[0]}."
        )
    if nonhall_arr.shape[0] != int(question_table.question_count):
        raise ValueError(
            "Question-wise non-hallucination target length mismatch: "
            f"expected={question_table.question_count}, got={nonhall_arr.shape[0]}."
        )

    p_epn_pred = np.zeros(int(sampled_table.features.shape[0]), dtype=np.float32)
    question_losses: List[float] = []
    for qid in range(int(question_table.question_count)):
        resp_idx = response_indices_by_qid[qid]
        if resp_idx.size == 0:
            continue
        rho_q = np.full(resp_idx.size, float(rho_arr[qid]), dtype=np.float32)
        p_epn_q = p_epn_from_components(
            K=np.asarray(sampled_table.K[resp_idx], dtype=np.float32),
            rho=rho_q,
            semantic_prob=np.asarray(semantic_arr[resp_idx], dtype=np.float32),
        )
        p_epn_pred[resp_idx] = p_epn_q
        target_q = np.full(resp_idx.size, float(nonhall_arr[qid]), dtype=np.float32)
        question_losses.append(binary_ce_np(target_q, p_epn_q))

    mean_loss = float(np.mean(question_losses)) if question_losses else float("nan")
    return mean_loss, p_epn_pred


def aggregate_cluster_semantic_prob(table: SampledResponseTable, semantic_prob_pred: np.ndarray) -> List[Dict[int, float]]:
    pred = np.asarray(semantic_prob_pred, dtype=np.float32).reshape(-1)
    result: List[Dict[int, float]] = []
    for qid in range(int(table.question_count)):
        mask_q = table.qid == qid
        qid_dict: Dict[int, float] = {}
        if np.any(mask_q):
            cid_arr = np.asarray(table.cid[mask_q], dtype=np.int64)
            pred_arr = pred[mask_q]
            for cid in sorted(np.unique(cid_arr).tolist()):
                cid_mask = cid_arr == int(cid)
                qid_dict[int(cid)] = float(np.mean(pred_arr[cid_mask]))
        result.append(qid_dict)
    return result


def aggregate_question_rho(table: SampledResponseTable, rho_pred: np.ndarray) -> np.ndarray:
    rho_arr = np.asarray(rho_pred, dtype=np.float32).reshape(-1)
    if rho_arr.shape[0] == int(table.question_count):
        return rho_arr.astype(np.float32, copy=False)
    if rho_arr.shape[0] != int(table.features.shape[0]):
        raise ValueError(
            "rho_pred length must match either question_count or sampled-response count: "
            f"question_count={table.question_count}, samples={table.features.shape[0]}, got={rho_arr.shape[0]}."
        )
    out = np.zeros(int(table.question_count), dtype=np.float32)
    for qid in range(int(table.question_count)):
        mask_q = table.qid == qid
        out[qid] = float(np.mean(rho_arr[mask_q])) if np.any(mask_q) else 0.0
    return out


def build_evidence_from_predictions(
    table: SampledResponseTable,
    semantic_prob_pred: np.ndarray,
    rho_pred: np.ndarray,
) -> Dict[str, Any]:
    semantic_by_cluster = aggregate_cluster_semantic_prob(table, semantic_prob_pred)
    rho_by_question = aggregate_question_rho(table, rho_pred)
    evidence_by_question: List[Dict[int, float]] = []
    vacuity_scores: List[float] = []
    dissonance_scores: List[float] = []
    for qid in range(int(table.question_count)):
        K_q = int(len(semantic_by_cluster[qid]))
        evidence_dict: Dict[int, float] = {}
        for cid, p_sem in semantic_by_cluster[qid].items():
            evidence_dict[int(cid)] = float(K_q * float(rho_by_question[qid]) * float(p_sem))
        evidence_by_question.append(evidence_dict)
        vacuity_scores.append(float(vacuity(evidence_dict)))
        dissonance_scores.append(float(dissonance(evidence_dict)))
    return {
        "semantic_prob_by_cluster": semantic_by_cluster,
        "rho_by_question": rho_by_question.astype(np.float32, copy=False),
        "evidence_by_question": evidence_by_question,
        "vacuity": np.asarray(vacuity_scores, dtype=np.float32),
        "dissonance": np.asarray(dissonance_scores, dtype=np.float32),
    }


def compute_split_metrics(
    split: str,
    hall_table: PrimaryHallTable,
    sampled_table: SampledResponseTable,
    hall_prob: np.ndarray,
    nonhall_prob: np.ndarray,
    semantic_prob_pred: np.ndarray,
    rho_pred: np.ndarray,
    p_epn_pred: np.ndarray,
    final_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    semantic_target = np.asarray(sampled_table.semantic_prob_target, dtype=np.float32)
    rho_target = np.asarray(nonhall_prob[sampled_table.qid], dtype=np.float32)
    uncertainty_labels = np.asarray(sampled_table.uncertainty_labels, dtype=np.int64)
    vacuity_scores = np.asarray(final_outputs["vacuity"], dtype=np.float32)
    dissonance_scores = np.asarray(final_outputs["dissonance"], dtype=np.float32)
    return {
        "split": split,
        "question_count": int(sampled_table.question_count),
        "sample_count": int(sampled_table.features.shape[0]),
        "uncertainty_label_source": str(sampled_table.uncertainty_label_source),
        "hall_probe": {
            "loss": binary_ce_np(hall_table.hall_labels.astype(np.float32), hall_prob),
            "metrics": safe_binary_metrics(hall_table.hall_labels, hall_prob),
        },
        "semantic_probe": {
            "loss": binary_ce_np(semantic_target, semantic_prob_pred),
        },
        "rho_probe": {
            "loss": binary_ce_np(rho_target, p_epn_pred),
        },
        "uncertainty_metrics": {
            "vacuity": safe_binary_metrics(uncertainty_labels, vacuity_scores),
            "dissonance": safe_binary_metrics(uncertainty_labels, dissonance_scores),
        },
        "mean_rho": float(np.mean(rho_pred)) if rho_pred.size > 0 else float("nan"),
        "mean_vacuity": float(np.mean(vacuity_scores)) if vacuity_scores.size > 0 else float("nan"),
        "mean_dissonance": float(np.mean(dissonance_scores)) if dissonance_scores.size > 0 else float("nan"),
    }


def compute_density_split_metrics(
    split: str,
    sampled_table: SampledResponseTable,
    density_table: QuestionDensityTable,
    semantic_prob_pred: np.ndarray,
    density_outputs: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    probe_prob = np.asarray(density_outputs["hall_prob"], dtype=np.float32).reshape(-1)
    rho = np.asarray(density_outputs["rho"], dtype=np.float32).reshape(-1)
    metrics = {
        "split": split,
        "question_count": int(density_table.question_count),
        "sample_count": int(sampled_table.features.shape[0]),
        "probe_target": str(density_table.probe_target),
        "probe_label_source": str(density_table.probe_label_source),
        "uncertainty_label_source": str(density_table.uncertainty_label_source),
        "semantic_probe": {
            "loss": binary_ce_np(
                np.asarray(sampled_table.semantic_prob_target, dtype=np.float32),
                np.asarray(semantic_prob_pred, dtype=np.float32),
            ),
        },
        "density_probe": {
            "loss": binary_ce_np(np.asarray(density_table.probe_labels, dtype=np.float32), probe_prob),
            "metrics": safe_binary_metrics(density_table.probe_labels, probe_prob),
        },
        "mean_probe_probability": float(np.mean(probe_prob)) if probe_prob.size > 0 else float("nan"),
        "mean_rho": float(np.mean(rho)) if rho.size > 0 else float("nan"),
    }
    if str(density_table.probe_target) == "hallucination":
        metrics["mean_hallucination_probability"] = float(metrics["mean_probe_probability"])
    else:
        metrics["mean_ambiguity_probability"] = float(metrics["mean_probe_probability"])
    return metrics


def save_training_split_predictions(
    path: str,
    sampled_table: SampledResponseTable,
    density_table: QuestionDensityTable,
    semantic_prob_pred: np.ndarray,
    density_outputs: Dict[str, np.ndarray],
) -> None:
    payload: Dict[str, Any] = {
        "qid": np.asarray(sampled_table.qid, dtype=np.int64),
        "resp_id": np.asarray(sampled_table.resp_id, dtype=np.int64),
        "cid": np.asarray(sampled_table.cid, dtype=np.int64),
        "K": np.asarray(sampled_table.K, dtype=np.float32),
        "semantic_prob_target": np.asarray(sampled_table.semantic_prob_target, dtype=np.float32),
        "semantic_prob_pred": np.asarray(semantic_prob_pred, dtype=np.float32),
        "density_qid": np.asarray(density_table.qid, dtype=np.int64),
        "density_probe_label": np.asarray(density_table.probe_labels, dtype=np.int64),
        "density_logits": np.asarray(density_outputs["logits"], dtype=np.float32),
        "density_probe_probability": np.asarray(density_outputs["hall_prob"], dtype=np.float32),
        "density_rho": np.asarray(density_outputs["rho"], dtype=np.float32),
        "probe_target": np.asarray([density_table.probe_target], dtype=object),
        "probe_label_source": np.asarray([density_table.probe_label_source], dtype=object),
        "uncertainty_labels": np.asarray(density_table.uncertainty_labels, dtype=np.int64),
        "primary_truth_labels": np.asarray(density_table.primary_truth_labels, dtype=np.int64),
        "uncertainty_label_source": np.asarray([density_table.uncertainty_label_source], dtype=object),
    }
    if str(density_table.probe_target) == "hallucination":
        payload["density_hall_label"] = np.asarray(density_table.probe_labels, dtype=np.int64)
        payload["density_hall_prob"] = np.asarray(density_outputs["hall_prob"], dtype=np.float32)
    else:
        payload["density_ambiguity_label"] = np.asarray(density_table.probe_labels, dtype=np.int64)
        payload["density_ambiguity_prob"] = np.asarray(density_outputs["hall_prob"], dtype=np.float32)
    np.savez_compressed(path, **payload)


def normalize_probability_vector(scores: np.ndarray) -> np.ndarray:
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if score_arr.size == 0:
        return score_arr.astype(np.float32, copy=False)
    score_arr = np.clip(score_arr, 0.0, None)
    total = float(np.sum(score_arr))
    if not np.isfinite(total) or total <= float(EPS):
        return np.full(score_arr.shape[0], 1.0 / float(score_arr.shape[0]), dtype=np.float32)
    return (score_arr / total).astype(np.float32, copy=False)


def build_response_evidence_dict(
    rho: float,
    semantic_prob_vector: np.ndarray,
    response_ids: Optional[Sequence[int]] = None,
) -> Dict[int, float]:
    p_hat = normalize_probability_vector(semantic_prob_vector)
    M = int(p_hat.shape[0])
    if response_ids is None:
        keys = list(range(M))
    else:
        keys = [int(resp_id) for resp_id in response_ids]
        if len(keys) != M:
            raise ValueError(
                "response_ids length must match semantic_prob_vector length: "
                f"ids={len(keys)}, probs={M}."
            )
    return {int(keys[m]): float(M * float(rho) * float(p_hat[m])) for m in range(M)}


def estimate_evidence_se_scores(
    evidence_by_question: Sequence[Dict[int, float]],
) -> np.ndarray:
    entropies: List[float] = []
    invalid_prob_flag = False
    for item in evidence_by_question:
        evidence = np.asarray(list((item or {}).values()), dtype=np.float32)
        total_evidence = float(np.sum(evidence))
        if total_evidence <= 0.0 or not np.isfinite(total_evidence):
            entropies.append(0.0)
            continue
        p = evidence / (total_evidence + float(EPS))
        mask = (p > 0.0) & np.isfinite(p)
        if not np.any(mask):
            entropies.append(0.0)
            continue
        invalid_prob_flag = invalid_prob_flag or bool(np.any(~mask))
        entropies.append(float(-(p[mask] * np.log(p[mask])).sum()))
    if invalid_prob_flag:
        print("[EPN Diagnostics] Non-finite or zero evidence probabilities were ignored.")
    return np.asarray(entropies, dtype=np.float32)


def build_cluster_embeddings_from_response(
    semantic_ids: np.ndarray,
    response_embeddings: Sequence[Dict[int, np.ndarray]],
) -> List[Dict[int, np.ndarray]]:
    semantic_ids_arr = np.asarray(semantic_ids, dtype=np.int64)
    if semantic_ids_arr.ndim != 2:
        raise ValueError(f"semantic_ids must be rank-2, got {semantic_ids_arr.shape}.")
    if len(response_embeddings) != int(semantic_ids_arr.shape[0]):
        raise ValueError(
            "response_embeddings/semantic_ids size mismatch: "
            f"{len(response_embeddings)} vs {semantic_ids_arr.shape[0]}."
        )

    cluster_embeddings: List[Dict[int, np.ndarray]] = []
    for qid in range(int(semantic_ids_arr.shape[0])):
        ids_row = np.asarray(semantic_ids_arr[qid], dtype=np.int64).reshape(-1)
        emb_row = response_embeddings[qid] or {}
        cluster_to_vecs: Dict[int, List[np.ndarray]] = {}
        for resp_id, emb in emb_row.items():
            resp_idx = int(resp_id)
            if resp_idx < 0 or resp_idx >= int(ids_row.shape[0]):
                continue
            cid = int(ids_row[resp_idx])
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.size == 0:
                continue
            cluster_to_vecs.setdefault(cid, []).append(vec)
        cluster_row: Dict[int, np.ndarray] = {}
        for cid, vecs in cluster_to_vecs.items():
            cluster_row[int(cid)] = np.stack(vecs, axis=0).mean(axis=0, dtype=np.float32)
        cluster_embeddings.append(cluster_row)
    return cluster_embeddings


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
    b_arr = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return None
    return float(np.dot(a_arr, b_arr) / denom)


def _mean_pairwise_cosine(
    embeddings: Sequence[np.ndarray],
    k: int = 2,
) -> Optional[float]:
    if len(embeddings) < int(k):
        return None
    sims: List[float] = []
    for i in range(int(k)):
        for j in range(i + 1, int(k)):
            sim = _cosine_sim(embeddings[i], embeddings[j])
            if sim is None:
                return None
            sims.append(float(sim))
    if not sims:
        return None
    return float(np.mean(sims))


def _closest_cluster(
    emb: Optional[np.ndarray],
    flat_clusters: Sequence[Tuple[int, int, np.ndarray, float]],
) -> Tuple[Optional[int], Optional[int], float]:
    if emb is None:
        return None, None, float("-inf")
    emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    emb_norm = float(np.linalg.norm(emb_arr))
    if emb_norm == 0.0:
        return None, None, float("-inf")
    best_qid = None
    best_cid = None
    best_sim = float("-inf")
    for qid, cid, centroid, centroid_norm in flat_clusters:
        denom = emb_norm * float(centroid_norm)
        if denom == 0.0:
            continue
        sim = float(np.dot(emb_arr, centroid) / denom)
        if sim > best_sim:
            best_qid = int(qid)
            best_cid = int(cid)
            best_sim = sim
    return best_qid, best_cid, best_sim


def compute_epn_test_diagnostics(
    artifacts: LoadedSplitArtifacts,
    inferred_response_embeddings: Sequence[Dict[int, np.ndarray]],
    evidence_by_question: Sequence[Dict[int, float]],
    sim_threshold: float = 0.9,
    evidence_threshold: float = 3.0,
    error_threshold: float = 0.5,
    diversity_threshold: float = 0.7,
    coverage_threshold: float = 0.9,
    coverage_min_size: int = 3,
) -> Dict[str, Any]:
    response_embeddings_ref = artifacts.response_embeddings
    if response_embeddings_ref is None:
        response_embeddings_path = os.path.join(
            artifacts.split_dir,
            f"response_embeddings_{int(artifacts.semantic_ids.shape[1])}.npz",
        )
        if not os.path.exists(response_embeddings_path):
            reason = f"missing reference response embeddings: {response_embeddings_path}"
            print(f"[EPN Diagnostics] Skipping: {reason}.")
            return {"available": False, "reason": reason}
        response_embeddings_ref = load_cluster_embeddings_npz(response_embeddings_path)

    baseline_se_path = os.path.join(artifacts.split_dir, "se.npy")
    if not os.path.exists(baseline_se_path):
        reason = f"missing baseline SE scores: {baseline_se_path}"
        print(f"[EPN Diagnostics] Skipping: {reason}.")
        return {"available": False, "reason": reason}

    baseline_se = np.asarray(np.load(baseline_se_path), dtype=np.float32).reshape(-1)
    reference_centroids = build_cluster_embeddings_from_response(
        semantic_ids=artifacts.semantic_ids,
        response_embeddings=response_embeddings_ref,
    )
    est_se = estimate_evidence_se_scores(evidence_by_question)

    question_count = int(
        min(
            len(inferred_response_embeddings),
            len(evidence_by_question),
            len(reference_centroids),
            int(np.asarray(artifacts.semantic_ids).shape[0]),
            int(baseline_se.shape[0]),
            int(est_se.shape[0]),
        )
    )
    if question_count <= 0:
        reason = "no aligned questions available for diagnostics"
        print(f"[EPN Diagnostics] Skipping: {reason}.")
        return {"available": False, "reason": reason}

    failure_qids: List[int] = []
    rel_low_count = 0
    irr_high_count = 0
    both_count = 0
    diversity_hits = 0
    diversity_total = 0
    diversity_sims: List[float] = []
    coverage_scores: List[float] = []
    coverage_question_count = 0
    coverage_cluster_total = 0
    alignment_hits = 0
    alignment_total = 0

    for qid in range(question_count):
        inferred_row = inferred_response_embeddings[qid] or {}
        evidence_row = evidence_by_question[qid] or {}
        resp_ids = sorted(int(resp_id) for resp_id in inferred_row.keys())
        resp_embs = [np.asarray(inferred_row[resp_id], dtype=np.float32).reshape(-1) for resp_id in resp_ids]
        resp_evidence = [evidence_row.get(int(resp_id)) for resp_id in resp_ids]

        centroids = reference_centroids[qid] or {}
        flat_clusters: List[Tuple[int, int, np.ndarray, float]] = []
        for cid, emb in centroids.items():
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                continue
            flat_clusters.append((int(qid), int(cid), vec, norm))

        err = abs(float(est_se[qid]) - float(baseline_se[qid]))
        if np.isfinite(err) and err >= float(error_threshold):
            failure_qids.append(int(qid))
            has_rel_low = False
            has_irr_high = False
            for emb, ev in zip(resp_embs, resp_evidence):
                if ev is None or not np.isfinite(ev):
                    continue
                _best_qid, _best_cid, sim = _closest_cluster(emb, flat_clusters)
                is_relevant = bool(sim >= float(sim_threshold))
                is_high = bool(float(ev) >= float(evidence_threshold))
                if is_relevant and not is_high:
                    has_rel_low = True
                if (not is_relevant) and is_high:
                    has_irr_high = True
            if has_rel_low:
                rel_low_count += 1
            if has_irr_high:
                irr_high_count += 1
            if has_rel_low and has_irr_high:
                both_count += 1

        if resp_embs:
            alignment_total += 1
            sims = [_closest_cluster(emb, flat_clusters)[2] for emb in resp_embs]
            if any(sim > float(sim_threshold) for sim in sims):
                alignment_hits += 1

        mean_sim = _mean_pairwise_cosine(resp_embs, k=2)
        if mean_sim is not None:
            diversity_total += 1
            diversity_sims.append(float(mean_sim))
            if float(mean_sim) <= float(diversity_threshold):
                diversity_hits += 1

        ids_row = np.asarray(artifacts.semantic_ids[qid], dtype=np.int64).reshape(-1)
        if ids_row.size > 0:
            vals, counts = np.unique(ids_row, return_counts=True)
            large_mask = counts >= int(coverage_min_size)
            if np.any(large_mask):
                covered = 0
                considered = 0
                for cid in vals[large_mask]:
                    centroid = centroids.get(int(cid))
                    if centroid is None:
                        continue
                    considered += 1
                    best_sim = None
                    for emb in resp_embs:
                        sim = _cosine_sim(emb, centroid)
                        if sim is None:
                            continue
                        if best_sim is None or sim > best_sim:
                            best_sim = sim
                    if best_sim is not None and best_sim >= float(coverage_threshold):
                        covered += 1
                if considered > 0:
                    coverage_question_count += 1
                    coverage_scores.append(float(covered) / float(considered))
                    coverage_cluster_total += int(considered)

    failure_count = len(failure_qids)
    failure_rate = float(failure_count) / float(question_count) if question_count else 0.0
    failure_rate_rel_low = float(rel_low_count) / float(failure_count) if failure_count else 0.0
    failure_rate_irr_high = float(irr_high_count) / float(failure_count) if failure_count else 0.0
    failure_rate_both = float(both_count) / float(failure_count) if failure_count else 0.0
    semantic_divergence = float(diversity_hits) / float(diversity_total) if diversity_total else 0.0
    alignment_precision = float(alignment_hits) / float(alignment_total) if alignment_total else 0.0
    large_cluster_coverage = float(np.mean(coverage_scores)) if coverage_scores else 0.0
    mean_pairwise_cosine = float(np.mean(diversity_sims)) if diversity_sims else 0.0

    print(
        "[EPN Diagnostics] "
        f"failure_rate={failure_rate:.3f} ({failure_count}/{question_count}) | "
        f"failure_rate_rel_low={failure_rate_rel_low:.3f} ({rel_low_count}/{max(failure_count, 1)}) | "
        f"failure_rate_irr_high={failure_rate_irr_high:.3f} ({irr_high_count}/{max(failure_count, 1)}) | "
        f"failure_rate_both={failure_rate_both:.3f} ({both_count}/{max(failure_count, 1)})"
    )
    print(
        "[EPN Diagnostics] "
        f"semantic_divergence={semantic_divergence:.3f} "
        f"(hits={diversity_hits}, total={diversity_total}, mean_pairwise_cosine={mean_pairwise_cosine:.3f}) | "
        f"large_cluster_coverage={large_cluster_coverage:.3f} "
        f"(questions={coverage_question_count}, clusters={coverage_cluster_total}) | "
        f"alignment_precision={alignment_precision:.3f} "
        f"({alignment_hits}/{alignment_total})"
    )

    return {
        "available": True,
        "question_count": int(question_count),
        "failure_qids": failure_qids,
        "failure_rate": {
            "value": float(failure_rate),
            "num": int(failure_count),
            "denom": int(question_count),
        },
        "failure_rate_rel_low": {
            "value": float(failure_rate_rel_low),
            "num": int(rel_low_count),
            "denom": int(failure_count),
        },
        "failure_rate_irr_high": {
            "value": float(failure_rate_irr_high),
            "num": int(irr_high_count),
            "denom": int(failure_count),
        },
        "failure_rate_both": {
            "value": float(failure_rate_both),
            "num": int(both_count),
            "denom": int(failure_count),
        },
        "semantic_divergence": {
            "value": float(semantic_divergence),
            "num": int(diversity_hits),
            "denom": int(diversity_total),
            "mean_pairwise_cosine": float(mean_pairwise_cosine),
        },
        "large_cluster_coverage": {
            "value": float(large_cluster_coverage),
            "question_count": int(coverage_question_count),
            "cluster_count": int(coverage_cluster_total),
        },
        "alignment_precision": {
            "value": float(alignment_precision),
            "num": int(alignment_hits),
            "denom": int(alignment_total),
        },
    }


def save_inference_predictions(
    path: str,
    responses: Sequence[Sequence[str]],
    semantic_raw_scores: Sequence[np.ndarray],
    semantic_prob_vectors: Sequence[np.ndarray],
    semantic_ids_itp: Sequence[Sequence[int]],
    hallucination_prob: np.ndarray,
    rho: np.ndarray,
    evidence_by_question: Sequence[Dict[int, float]],
    vacuity_scores: np.ndarray,
    dissonance_scores: np.ndarray,
    uncertainty_labels: np.ndarray,
    metric_eval_mask: np.ndarray,
    uncertainty_label_source: str,
    ood_refusal_eval: Dict[str, Any],
    hallucination_prob_by_response: Optional[Sequence[np.ndarray]] = None,
    rho_by_response: Optional[Sequence[np.ndarray]] = None,
    logits_by_response: Optional[Sequence[np.ndarray]] = None,
    response_sequence_probs: Optional[np.ndarray] = None,
    response_norm_sequence_probs: Optional[np.ndarray] = None,
    evidence_by_question_seq_prob: Optional[Sequence[Dict[int, float]]] = None,
    evidence_by_question_norm_seq_prob: Optional[Sequence[Dict[int, float]]] = None,
    vacuity_seq_prob: Optional[np.ndarray] = None,
    dissonance_seq_prob: Optional[np.ndarray] = None,
    vacuity_norm_seq_prob: Optional[np.ndarray] = None,
    dissonance_norm_seq_prob: Optional[np.ndarray] = None,
    probe_target: str = "hallucination",
    probe_label_source: str = "1-y",
) -> None:
    resolved_probe_target = normalize_probe_target(probe_target)
    payload: Dict[str, Any] = {
        "responses": np.asarray(list(responses), dtype=object),
        "semantic_raw_scores": np.asarray(list(semantic_raw_scores), dtype=object),
        "semantic_prob_vectors": np.asarray(list(semantic_prob_vectors), dtype=object),
        "semantic_ids_itp": np.asarray(list(semantic_ids_itp), dtype=object),
        "probe_target": np.asarray([resolved_probe_target], dtype=object),
        "probe_label_source": np.asarray([probe_label_source], dtype=object),
        "probe_probability": np.asarray(hallucination_prob, dtype=np.float32),
        "rho": np.asarray(rho, dtype=np.float32),
        "evidence_by_question": np.asarray(list(evidence_by_question), dtype=object),
        "vacuity": np.asarray(vacuity_scores, dtype=np.float32),
        "dissonance": np.asarray(dissonance_scores, dtype=np.float32),
        "uncertainty_labels": np.asarray(uncertainty_labels, dtype=np.int64),
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "uncertainty_label_source": np.asarray([uncertainty_label_source], dtype=object),
        "ood_refusal_eval": np.asarray([ood_refusal_eval], dtype=object),
    }
    if resolved_probe_target == "hallucination":
        payload["hallucination_prob"] = np.asarray(hallucination_prob, dtype=np.float32)
        if hallucination_prob_by_response is not None:
            payload["hallucination_prob_by_response"] = np.asarray(
                list(hallucination_prob_by_response),
                dtype=object,
            )
    else:
        payload["ambiguity_prob"] = np.asarray(hallucination_prob, dtype=np.float32)
        if hallucination_prob_by_response is not None:
            payload["ambiguity_prob_by_response"] = np.asarray(list(hallucination_prob_by_response), dtype=object)
    if hallucination_prob_by_response is not None:
        payload["probe_probability_by_response"] = np.asarray(list(hallucination_prob_by_response), dtype=object)
    if rho_by_response is not None:
        payload["rho_by_response"] = np.asarray(list(rho_by_response), dtype=object)
    if logits_by_response is not None:
        payload["logits_by_response"] = np.asarray(list(logits_by_response), dtype=object)
    if response_sequence_probs is not None:
        payload["response_sequence_probs"] = np.asarray(response_sequence_probs, dtype=np.float32)
    if response_norm_sequence_probs is not None:
        payload["response_norm_sequence_probs"] = np.asarray(response_norm_sequence_probs, dtype=np.float32)
    if evidence_by_question_seq_prob is not None:
        payload["evidence_by_question_seq_prob"] = np.asarray(list(evidence_by_question_seq_prob), dtype=object)
    if evidence_by_question_norm_seq_prob is not None:
        payload["evidence_by_question_norm_seq_prob"] = np.asarray(
            list(evidence_by_question_norm_seq_prob),
            dtype=object,
        )
    if vacuity_seq_prob is not None:
        payload["vacuity_seq_prob"] = np.asarray(vacuity_seq_prob, dtype=np.float32)
    if dissonance_seq_prob is not None:
        payload["dissonance_seq_prob"] = np.asarray(dissonance_seq_prob, dtype=np.float32)
    if vacuity_norm_seq_prob is not None:
        payload["vacuity_norm_seq_prob"] = np.asarray(vacuity_norm_seq_prob, dtype=np.float32)
    if dissonance_norm_seq_prob is not None:
        payload["dissonance_norm_seq_prob"] = np.asarray(dissonance_norm_seq_prob, dtype=np.float32)
    np.savez_compressed(path, **payload)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2)


def save_pickle(path: str, payload: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def save_torch_checkpoint(
    path: str,
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
) -> None:
    torch.save({"state_dict": state_dict, "metadata": metadata}, path)


def save_split_predictions(
    path: str,
    hall_table: PrimaryHallTable,
    sampled_table: SampledResponseTable,
    hall_prob: np.ndarray,
    nonhall_prob: np.ndarray,
    semantic_prob_pred: np.ndarray,
    rho_pred: np.ndarray,
    p_epn_pred: np.ndarray,
    final_outputs: Dict[str, Any],
) -> None:
    semantic_prob_by_cluster = np.asarray(final_outputs["semantic_prob_by_cluster"], dtype=object)
    evidence_by_question = np.asarray(final_outputs["evidence_by_question"], dtype=object)
    np.savez_compressed(
        path,
        primary_qid=np.asarray(hall_table.qid, dtype=np.int64),
        primary_hall_prob=np.asarray(hall_prob, dtype=np.float32),
        primary_nonhall_prob=np.asarray(nonhall_prob, dtype=np.float32),
        primary_hall_label=np.asarray(hall_table.hall_labels, dtype=np.int64),
        qid=np.asarray(sampled_table.qid, dtype=np.int64),
        resp_id=np.asarray(sampled_table.resp_id, dtype=np.int64),
        cid=np.asarray(sampled_table.cid, dtype=np.int64),
        K=np.asarray(sampled_table.K, dtype=np.float32),
        semantic_prob_target=np.asarray(sampled_table.semantic_prob_target, dtype=np.float32),
        semantic_prob_pred=np.asarray(semantic_prob_pred, dtype=np.float32),
        rho_pred=np.asarray(rho_pred, dtype=np.float32),
        rho_target_nonhall_prob=np.asarray(nonhall_prob[sampled_table.qid], dtype=np.float32),
        p_epn_pred=np.asarray(p_epn_pred, dtype=np.float32),
        question_rho=np.asarray(final_outputs["rho_by_question"], dtype=np.float32),
        vacuity=np.asarray(final_outputs["vacuity"], dtype=np.float32),
        dissonance=np.asarray(final_outputs["dissonance"], dtype=np.float32),
        uncertainty_labels=np.asarray(sampled_table.uncertainty_labels, dtype=np.int64),
        primary_truth_labels=np.asarray(sampled_table.primary_truth_labels, dtype=np.int64),
        semantic_prob_by_cluster=semantic_prob_by_cluster,
        evidence_by_question=evidence_by_question,
        uncertainty_label_source=np.asarray([sampled_table.uncertainty_label_source], dtype=object),
    )


# EUD evaluation helpers moved from eud.py for the EPN-D pipeline.
LLAMA_33_70B_SUFFIX = "llama-3.3-70b-instruct"
# DEFAULT_DISCOUNT_ZETA_GRID: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0)
DEFAULT_DISCOUNT_ZETA_GRID: Tuple[float, ...] = (0.1,)
HIST_FIG_DPI = 320
HIST_BINS = np.linspace(0.0, 1.0, 21)
HIST_COLORS = {
    "clear": "#4e79a7",
    "ambig": "#e15759",
    "ind": "#4e79a7",
    "ood": "#e15759",
    "truthful": "#59a14f",
    "hallucination": "#f28e2b",
}
OOD_ZETA_SELECTION_DATASET_NAMES = {"alcuna_trivia_qa", "temporalood_trivia_qa"}
AMBIG_ZETA_SELECTION_DATASET_NAMES = {
    "ambig_qa",
    "ambig_qa_disambig",
    "ambig_qa_filtered_ver3",
    "situated_qa",
    "situated_qa_disambig",
}
SUPPORTED_ZETA_SELECTION_DATASET_NAMES = OOD_ZETA_SELECTION_DATASET_NAMES | AMBIG_ZETA_SELECTION_DATASET_NAMES
PSEUDO_OOD_DATASET_MAP = {
    "alcuna_trivia_qa": "temporalood_trivia_qa",
    "temporalood_trivia_qa": "alcuna_trivia_qa",
}


@dataclass
class DiscountedSplitBundle:
    dataset_name: str
    split: str
    responses: List[List[str]]
    semantic_ids: List[np.ndarray]
    response_sequence_probs: np.ndarray
    response_norm_sequence_probs: np.ndarray
    labels: np.ndarray
    y: np.ndarray
    label_source: str
    response_slt_all: Optional[List[Dict[int, np.ndarray]]]
    response_slt_lastk_path: Optional[str]
    question_probe_features: Optional[np.ndarray] = None
    question_probe_feature_path: Optional[str] = None
    disambig_question_semantic_ids: Optional[List[np.ndarray]] = None
    load_dir: Optional[str] = None
    split_dir: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscountedBranchSpec:
    key: str
    prefix: str
    metric_stem: str
    label: str
    kind: str
    mode: str
    use_seq_prob: bool
    use_norm_seq_prob: bool
    payload_keys: Tuple[str, ...]


DISCOUNTED_BRANCH_SPECS: Tuple[DiscountedBranchSpec, ...] = (
    DiscountedBranchSpec(
        key="power",
        prefix="power_discount",
        metric_stem="power_discounted",
        label="EUD Power Discounted",
        kind="discounted",
        mode="power",
        use_seq_prob=False,
        use_norm_seq_prob=False,
        payload_keys=(
            "pred_evidence_discounted",
            "pred_ignorance_discounted",
            "response_reliability_scores",
            "total_ignorance",
            "dist_vacuity",
            "dist_dissonance",
        ),
    ),
    DiscountedBranchSpec(
        key="exp",
        prefix="exp_discount",
        metric_stem="exp_discounted",
        label="EUD Exp Discounted",
        kind="discounted",
        mode="exp",
        use_seq_prob=False,
        use_norm_seq_prob=False,
        payload_keys=(
            "pred_evidence_discounted",
            "pred_ignorance_discounted",
            "response_reliability_scores",
            "total_ignorance",
            "dist_vacuity",
            "dist_dissonance",
        ),
    ),
    DiscountedBranchSpec(
        key="mean_weighted",
        prefix="mean_weighted",
        metric_stem="mean_weighted",
        label="EUD Mean Weighted",
        kind="weighted",
        mode="mean_weighted",
        use_seq_prob=False,
        use_norm_seq_prob=False,
        payload_keys=("pred_evidence", "vacuity", "dissonance"),
    ),
    DiscountedBranchSpec(
        key="resp_wise_weighted",
        prefix="resp_wise_weighted",
        metric_stem="resp_wise_weighted",
        label="EUD Resp-Wise Weighted",
        kind="weighted",
        mode="resp_wise_weighted",
        use_seq_prob=False,
        use_norm_seq_prob=False,
        payload_keys=("pred_evidence", "vacuity", "dissonance"),
    ),
    DiscountedBranchSpec(
        key="power_seq_prob",
        prefix="power_discount_seq_prob",
        metric_stem="power_discounted_seq_prob",
        label="EUD Power Discounted Seq-Prob",
        kind="discounted",
        mode="power",
        use_seq_prob=True,
        use_norm_seq_prob=False,
        payload_keys=(
            "pred_evidence_discounted",
            "pred_ignorance_discounted",
            "response_reliability_scores",
            "total_ignorance",
            "dist_vacuity",
            "dist_dissonance",
        ),
    ),
    DiscountedBranchSpec(
        key="exp_seq_prob",
        prefix="exp_discount_seq_prob",
        metric_stem="exp_discounted_seq_prob",
        label="EUD Exp Discounted Seq-Prob",
        kind="discounted",
        mode="exp",
        use_seq_prob=True,
        use_norm_seq_prob=False,
        payload_keys=(
            "pred_evidence_discounted",
            "pred_ignorance_discounted",
            "response_reliability_scores",
            "total_ignorance",
            "dist_vacuity",
            "dist_dissonance",
        ),
    ),
    DiscountedBranchSpec(
        key="mean_weighted_seq_prob",
        prefix="mean_weighted_seq_prob",
        metric_stem="mean_weighted_seq_prob",
        label="EUD Mean Weighted Seq-Prob",
        kind="weighted",
        mode="mean_weighted",
        use_seq_prob=True,
        use_norm_seq_prob=False,
        payload_keys=("pred_evidence", "vacuity", "dissonance"),
    ),
    DiscountedBranchSpec(
        key="resp_wise_weighted_seq_prob",
        prefix="resp_wise_weighted_seq_prob",
        metric_stem="resp_wise_weighted_seq_prob",
        label="EUD Resp-Wise Weighted Seq-Prob",
        kind="weighted",
        mode="resp_wise_weighted",
        use_seq_prob=True,
        use_norm_seq_prob=False,
        payload_keys=("pred_evidence", "vacuity", "dissonance"),
    ),
    DiscountedBranchSpec(
        key="power_norm_seq_prob",
        prefix="power_discount_norm_seq_prob",
        metric_stem="power_discounted_norm_seq_prob",
        label="EUD Power Discounted Norm-Seq-Prob",
        kind="discounted",
        mode="power",
        use_seq_prob=False,
        use_norm_seq_prob=True,
        payload_keys=(
            "pred_evidence_discounted",
            "pred_ignorance_discounted",
            "response_reliability_scores",
            "total_ignorance",
            "dist_vacuity",
            "dist_dissonance",
        ),
    ),
    DiscountedBranchSpec(
        key="exp_norm_seq_prob",
        prefix="exp_discount_norm_seq_prob",
        metric_stem="exp_discounted_norm_seq_prob",
        label="EUD Exp Discounted Norm-Seq-Prob",
        kind="discounted",
        mode="exp",
        use_seq_prob=False,
        use_norm_seq_prob=True,
        payload_keys=(
            "pred_evidence_discounted",
            "pred_ignorance_discounted",
            "response_reliability_scores",
            "total_ignorance",
            "dist_vacuity",
            "dist_dissonance",
        ),
    ),
    DiscountedBranchSpec(
        key="mean_weighted_norm_seq_prob",
        prefix="mean_weighted_norm_seq_prob",
        metric_stem="mean_weighted_norm_seq_prob",
        label="EUD Mean Weighted Norm-Seq-Prob",
        kind="weighted",
        mode="mean_weighted",
        use_seq_prob=False,
        use_norm_seq_prob=True,
        payload_keys=("pred_evidence", "vacuity", "dissonance"),
    ),
    DiscountedBranchSpec(
        key="resp_wise_weighted_norm_seq_prob",
        prefix="resp_wise_weighted_norm_seq_prob",
        metric_stem="resp_wise_weighted_norm_seq_prob",
        label="EUD Resp-Wise Weighted Norm-Seq-Prob",
        kind="weighted",
        mode="resp_wise_weighted",
        use_seq_prob=False,
        use_norm_seq_prob=True,
        payload_keys=("pred_evidence", "vacuity", "dissonance"),
    ),
)


def _sequence_prob_from_length_normalized_logprob(
    length_normalized_sequence_logprob: np.ndarray,
) -> np.ndarray:
    return np.exp(np.asarray(length_normalized_sequence_logprob, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )


def _load_jsonl_rows(path: str) -> List[Dict[str, object]]:
    rows: List[Tuple[int, Dict[str, object]]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object rows in {path}, got {type(row).__name__} at line {line_no + 1}.")
            idx = int(row.get("i", line_no))
            rows.append((idx, row))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows]


def _load_jsonl_field(path: str, field: str) -> List[object]:
    return [row[field] for row in _load_jsonl_rows(path)]


def load_pred_json(path: str):
    return _load_jsonl_field(path, "pred")


def load_question_json(path: str):
    return _load_jsonl_field(path, "question")


def load_ref_json(path: str):
    return _load_jsonl_field(path, "ref")


def load_resp_json(path: str):
    return _load_jsonl_field(path, "responses")


def load_X_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].astype(np.float32, copy=False) for key in payload.files}


def normalize_zeta_grid(zeta_grid: Sequence[float]) -> List[float]:
    values = [float(z) for z in zeta_grid]
    if not values:
        raise ValueError("zeta_grid must contain at least one value.")
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float32))):
        raise ValueError("zeta_grid must contain only finite values.")
    return values


def get_estimated_se(pred_evidence_all):
    entropies = []
    flag = False
    for item in pred_evidence_all:
        evidence = np.asarray(list(item.values()), dtype=np.float32)
        total_evidence = float(np.sum(evidence))
        if total_evidence <= 0.0 or not np.isfinite(total_evidence):
            entropies.append(0.0)
            continue
        p = evidence / (total_evidence + 1e-9)
        mask = (p > 0) & np.isfinite(p)
        if not np.any(mask):
            entropies.append(0.0)
            continue
        flag = bool(flag or np.any(~mask))
        entropies.append(float(-(p[mask] * np.log(p[mask])).sum()))
    print(f"[Est SE] Probabiltiy 0.0 or nan flag: {flag}")
    return entropies


def _default_refusal_stats(total_count: int) -> Dict[str, object]:
    return {
        "applied": False,
        "accuracy": float("nan"),
        "excluded_ood_count": 0,
        "excluded_id_count": 0,
        "excluded_total_count": 0,
        "remaining_count": int(total_count),
        "total_count": int(total_count),
    }


def _is_disambig_question_dataset(dataset_name: Optional[str]) -> bool:
    return str(dataset_name or "").strip().lower().endswith("_disambig")


def _validate_response_sequence_probs_artifact(
    response_sequence_probs: np.ndarray,
    responses: Sequence[Sequence[str]],
    semantic_ids: Sequence[np.ndarray],
    source: str,
) -> np.ndarray:
    probs = np.asarray(response_sequence_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(
            f"Sampled-response sequence_prob in {source} must be rank-2, got shape {probs.shape}."
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"Sampled-response sequence_prob in {source} contains non-finite values.")
    if np.any(probs < 0.0):
        raise ValueError(f"Sampled-response sequence_prob in {source} contains negative values.")
    if probs.shape[0] != len(responses) or probs.shape[0] != len(semantic_ids):
        raise ValueError(
            "Sampled-response sequence_prob question count mismatch in "
            f"{source}: sequence_prob={probs.shape[0]}, responses={len(responses)}, "
            f"semantic_ids={len(semantic_ids)}."
        )
    for idx, (resp_row, sid_row, probs_row) in enumerate(zip(responses, semantic_ids, probs)):
        expected_n = len(resp_row)
        actual_sid = int(np.asarray(sid_row, dtype=np.int64).reshape(-1).shape[0])
        actual_probs = int(np.asarray(probs_row, dtype=np.float32).reshape(-1).shape[0])
        if actual_sid != expected_n or actual_probs != expected_n:
            raise ValueError(
                "Sampled-response sequence_prob / response alignment mismatch in "
                f"{source} at qid={idx}: responses={expected_n}, semantic_ids={actual_sid}, "
                f"sequence_prob={actual_probs}."
            )
    return probs.astype(np.float32, copy=False)


def _load_optional_jsonl_row_matrix(
    rows: Sequence[Dict[str, object]],
    field: str,
    *,
    dtype,
    source: str,
) -> Optional[List[np.ndarray]]:
    present_mask = [field in row and row[field] is not None for row in rows]
    if not any(present_mask):
        return None
    if not all(present_mask):
        missing_rows = [idx for idx, present in enumerate(present_mask) if not present]
        raise ValueError(
            f"Missing disambig-question field {field!r} in {source} rows {missing_rows[:10]}."
        )

    values: List[np.ndarray] = []
    for idx, row in enumerate(rows):
        try:
            arr = np.asarray(row[field], dtype=dtype).reshape(-1)
        except Exception as exc:
            raise ValueError(
                f"Unable to parse disambig-question field {field!r} at qid={idx} in {source}."
            ) from exc
        values.append(arr)
    return values


def _stack_jsonl_row_matrix(
    rows: Sequence[np.ndarray],
    *,
    field: str,
    source: str,
    dtype,
) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=dtype)
    widths = [int(np.asarray(row).reshape(-1).shape[0]) for row in rows]
    if len(set(widths)) != 1:
        raise ValueError(
            f"Disambig-question field {field!r} has inconsistent row widths in {source}: {widths[:10]}."
        )
    return np.stack([np.asarray(row, dtype=dtype).reshape(-1) for row in rows], axis=0).astype(dtype, copy=False)


def _validate_disambig_question_semantic_ids_artifact(
    disambig_question_semantic_ids: Sequence[np.ndarray],
    responses: Sequence[Sequence[str]],
    source: str,
) -> List[np.ndarray]:
    if len(disambig_question_semantic_ids) != len(responses):
        raise ValueError(
            "Disambig-question semantic_ids question count mismatch in "
            f"{source}: semantic_ids={len(disambig_question_semantic_ids)}, responses={len(responses)}."
        )

    validated_rows: List[np.ndarray] = []
    for idx, (resp_row, sid_row) in enumerate(zip(responses, disambig_question_semantic_ids)):
        ids_row = np.asarray(sid_row, dtype=np.int64).reshape(-1)
        if np.any(ids_row < 0):
            raise ValueError(
                f"Disambig-question semantic_ids must be non-negative in {source} at qid={idx}."
            )
        if ids_row.shape[0] != len(resp_row):
            raise ValueError(
                "Disambig-question semantic_ids / response alignment mismatch in "
                f"{source} at qid={idx}: responses={len(resp_row)}, semantic_ids={ids_row.shape[0]}."
            )
        validated_rows.append(ids_row)
    return validated_rows


def _validate_disambig_question_sequence_probs_artifact(
    disambig_question_sequence_probs: np.ndarray,
    responses: Sequence[Sequence[str]],
    disambig_question_semantic_ids: Sequence[np.ndarray],
    source: str,
    field_label: str,
) -> np.ndarray:
    probs = np.asarray(disambig_question_sequence_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(
            f"Disambig-question {field_label} in {source} must be rank-2, got shape {probs.shape}."
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"Disambig-question {field_label} in {source} contains non-finite values.")
    if np.any(probs < 0.0):
        raise ValueError(f"Disambig-question {field_label} in {source} contains negative values.")
    if probs.shape[0] != len(responses) or probs.shape[0] != len(disambig_question_semantic_ids):
        raise ValueError(
            "Disambig-question sequence_prob question count mismatch in "
            f"{source}: {field_label}={probs.shape[0]}, responses={len(responses)}, "
            f"semantic_ids={len(disambig_question_semantic_ids)}."
        )
    for idx, (resp_row, sid_row, probs_row) in enumerate(zip(responses, disambig_question_semantic_ids, probs)):
        expected_n = len(resp_row)
        actual_sid = int(np.asarray(sid_row, dtype=np.int64).reshape(-1).shape[0])
        actual_probs = int(np.asarray(probs_row, dtype=np.float32).reshape(-1).shape[0])
        if actual_sid != expected_n or actual_probs != expected_n:
            raise ValueError(
                "Disambig-question sequence_prob / response alignment mismatch in "
                f"{source} at qid={idx}: responses={expected_n}, semantic_ids={actual_sid}, "
                f"{field_label}={actual_probs}."
            )
    return probs.astype(np.float32, copy=False)


def _build_weighted_evidence_dicts(
    semantic_ids: Sequence[np.ndarray],
    weights: np.ndarray,
    *,
    source: str,
) -> List[Dict[int, float]]:
    weight_arr = np.asarray(weights, dtype=np.float32)
    if weight_arr.ndim != 2:
        raise ValueError(f"Weighted evidence requires rank-2 weights in {source}, got {weight_arr.shape}.")
    if weight_arr.shape[0] != len(semantic_ids):
        raise ValueError(
            f"Weighted evidence question count mismatch in {source}: weights={weight_arr.shape[0]}, "
            f"semantic_ids={len(semantic_ids)}."
        )

    evidence_all: List[Dict[int, float]] = []
    for qid, (ids_row, weight_row) in enumerate(zip(semantic_ids, weight_arr)):
        ids = np.asarray(ids_row, dtype=np.int64).reshape(-1)
        weights_q = np.asarray(weight_row, dtype=np.float32).reshape(-1)
        if ids.shape[0] != weights_q.shape[0]:
            raise ValueError(
                f"Weighted evidence row mismatch in {source} at qid={qid}: "
                f"semantic_ids={ids.shape[0]}, weights={weights_q.shape[0]}."
            )
        evidence_dict: Dict[int, float] = {}
        for cluster_id, weight in zip(ids.tolist(), weights_q.tolist()):
            cluster_key = int(cluster_id)
            evidence_dict[cluster_key] = evidence_dict.get(cluster_key, 0.0) + float(weight)
        evidence_all.append(evidence_dict)
    return evidence_all


def _zeta_matches(left: float, right: float, atol: float = 1e-8) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=float(atol)))


def _validate_zeta_selection_args(args) -> None:
    if not bool(getattr(args, "zeta_selection", False)):
        return

    dataset_name = str(args.dataset_name or "").strip().lower()
    pseudo_dataset_name = getattr(args, "pseudo_OOD_dataset_name", None)
    pseudo_dataset_name = None if pseudo_dataset_name is None else str(pseudo_dataset_name).strip().lower()

    if dataset_name not in SUPPORTED_ZETA_SELECTION_DATASET_NAMES:
        raise ValueError(
            "zeta_selection=True is supported only for "
            f"{sorted(SUPPORTED_ZETA_SELECTION_DATASET_NAMES)}, got dataset_name={args.dataset_name!r}."
        )

    if dataset_name in OOD_ZETA_SELECTION_DATASET_NAMES:
        expected_pseudo = PSEUDO_OOD_DATASET_MAP[dataset_name]
        if pseudo_dataset_name != expected_pseudo:
            raise ValueError(
                f"{args.dataset_name} requires pseudo_OOD_dataset_name={expected_pseudo!r} "
                f"when zeta_selection=True, got {getattr(args, 'pseudo_OOD_dataset_name', None)!r}."
            )
    elif pseudo_dataset_name not in {None, ""}:
        raise ValueError(
            f"pseudo_OOD_dataset_name is unsupported for ambiguity datasets like {args.dataset_name!r}."
        )


def _is_llama_33_70b_model_name(model_name: str) -> bool:
    norm = str(model_name or "").strip().lower().rstrip("/")
    if not norm:
        return False
    basename = norm.split("/")[-1]
    return basename.endswith(LLAMA_33_70B_SUFFIX) or norm.endswith(LLAMA_33_70B_SUFFIX)


def _ensure_bitsandbytes_available() -> None:
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "8-bit quantized loading requires bitsandbytes in the active environment."
        )


def _load_semantic_ids(path: str) -> List[np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    rows: List[np.ndarray] = []
    for row in raw:
        arr = np.asarray(row, dtype=np.int64).reshape(-1)
        if np.any(arr < 0):
            raise ValueError(f"semantic_ids must be non-negative. Found negative ids in {path}.")
        rows.append(arr)
    return rows


def _resolve_dataset_labels(
    split_dir: str,
    dataset_name: str,
    y: np.ndarray,
) -> Tuple[np.ndarray, str, Optional[str]]:
    for filename in ("ambig_label.npy", "ood_label.npy"):
        label_path = os.path.join(split_dir, filename)
        if os.path.exists(label_path):
            return np.asarray(np.load(label_path), dtype=np.int64).reshape(-1), filename, label_path

    return np.asarray(1 - y, dtype=np.int64).reshape(-1), "1-y", None


def _safe_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    split: str,
    metric_name: str,
) -> Tuple[float, float]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    finite_mask = np.isfinite(scores_arr)
    labels_arr = labels_arr[finite_mask]
    scores_arr = scores_arr[finite_mask]
    if labels_arr.size == 0:
        print(f"[{metric_name}] Skipping AUROC/AUPR for {split} split (no finite scores).")
        return float("nan"), float("nan")
    unique_labels = np.unique(labels_arr)
    if unique_labels.size < 2:
        print(
            f"[{metric_name}] Skipping AUROC/AUPR for {split} split "
            f"(only one class present: {unique_labels.tolist()})."
        )
        return float("nan"), float("nan")
    return float(roc_auc_score(labels_arr, scores_arr)), float(average_precision_score(labels_arr, scores_arr))


def _count_unique_semantic_clusters(ids_row) -> int:
    row = np.asarray(ids_row, dtype=np.int64).reshape(-1)
    if row.size == 0:
        return 0
    return int(np.unique(row).size)


def _resolve_semantic_cluster_counts(semantic_ids) -> np.ndarray:
    return np.asarray([_count_unique_semantic_clusters(row) for row in semantic_ids], dtype=np.int64)


def _compute_no_clear_err_auroc_aupr(
    labels,
    y,
    scores,
    metric_eval_mask,
    split: str,
    method_name: str,
) -> Tuple[float, float]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    n = int(min(labels_arr.size, y_arr.size, scores_arr.size, mask_arr.size))
    if n <= 0:
        return float("nan"), float("nan")
    labels_arr = labels_arr[:n]
    y_arr = y_arr[:n]
    scores_arr = scores_arr[:n]
    mask_arr = mask_arr[:n]
    hallucination_arr = np.asarray(y_arr == 0, dtype=np.int64)
    final_mask = np.logical_and(mask_arr, ~(np.logical_and(labels_arr == 0, hallucination_arr == 1)))
    return _safe_binary_metrics(labels_arr[final_mask], scores_arr[final_mask], split, f"{method_name} no_clear_err")


def _compute_no_clear_large_n_cluster_auroc_aupr(
    labels,
    semantic_ids,
    scores,
    metric_eval_mask,
    split: str,
    method_name: str,
    large_cluster_threshold: int = 5,
) -> Tuple[float, float]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    n_cluster = _resolve_semantic_cluster_counts(semantic_ids)
    n = int(min(labels_arr.size, scores_arr.size, mask_arr.size, n_cluster.size))
    if n <= 0:
        return float("nan"), float("nan")
    labels_arr = labels_arr[:n]
    scores_arr = scores_arr[:n]
    mask_arr = mask_arr[:n]
    n_cluster = n_cluster[:n]
    final_mask = np.logical_and(mask_arr, ~(np.logical_and(labels_arr == 0, n_cluster >= int(large_cluster_threshold))))
    return _safe_binary_metrics(
        labels_arr[final_mask],
        scores_arr[final_mask],
        split,
        f"{method_name} no_clear_large_n_cluster",
    )


def _build_gt_evidence_dicts(semantic_ids: Sequence[np.ndarray]) -> List[Dict[int, float]]:
    evidence_all: List[Dict[int, float]] = []
    for row in semantic_ids:
        counts = Counter(np.asarray(row, dtype=np.int64).reshape(-1).tolist())
        evidence_all.append({int(cid): float(cnt) for cid, cnt in counts.items()})
    return evidence_all


def _scale_evidence_dict(
    evidence_dict: Dict[int, float],
    scale: float,
) -> Dict[int, float]:
    scale_value = float(scale)
    return {
        int(cluster_id): float(value) * scale_value
        for cluster_id, value in evidence_dict.items()
    }


def _compute_disambig_question_theils_u(
    response_semantic_ids: Sequence[np.ndarray],
    disambig_question_semantic_ids: Sequence[np.ndarray],
    *,
    source: str,
) -> np.ndarray:
    if len(response_semantic_ids) != len(disambig_question_semantic_ids):
        raise ValueError(
            "Theil's U question count mismatch in "
            f"{source}: response_semantic_ids={len(response_semantic_ids)}, "
            f"disambig_question_semantic_ids={len(disambig_question_semantic_ids)}."
        )

    scores = np.zeros(len(response_semantic_ids), dtype=np.float32)
    for qid, (response_ids_row, disambig_ids_row) in enumerate(
        zip(response_semantic_ids, disambig_question_semantic_ids)
    ):
        response_ids = np.asarray(response_ids_row, dtype=np.int64).reshape(-1)
        disambig_ids = np.asarray(disambig_ids_row, dtype=np.int64).reshape(-1)
        if response_ids.shape[0] != disambig_ids.shape[0]:
            raise ValueError(
                "Theil's U row mismatch in "
                f"{source} at qid={qid}: response_semantic_ids={response_ids.shape[0]}, "
                f"disambig_question_semantic_ids={disambig_ids.shape[0]}."
            )
        if response_ids.size == 0:
            scores[qid] = 0.0
            continue
        score = float(
            theils_u(
                target_ids=response_ids,
                conditioning_ids=disambig_ids,
            )
        )
        if not np.isfinite(score):
            raise ValueError(f"Non-finite Theil's U in {source} at qid={qid}.")
        scores[qid] = float(np.clip(score, 0.0, 1.0))
    return scores


def _evaluate_evidence_dict_metrics(
    *,
    split: str,
    labels: np.ndarray,
    metric_eval_mask: np.ndarray,
    evidence_all: Sequence[Dict[int, float]],
    metric_name: str,
) -> Dict[str, object]:
    vacuity_scores = np.asarray(
        [float(vacuity(item)) if item else 0.0 for item in evidence_all],
        dtype=np.float32,
    )
    dissonance_scores = np.asarray(
        [float(dissonance(item)) if item else 0.0 for item in evidence_all],
        dtype=np.float32,
    )
    label_arr = np.asarray(labels, dtype=np.int64)
    eval_labels = label_arr[np.asarray(metric_eval_mask, dtype=bool)]
    eval_vacuity = vacuity_scores[np.asarray(metric_eval_mask, dtype=bool)]
    eval_dissonance = dissonance_scores[np.asarray(metric_eval_mask, dtype=bool)]
    auroc_vac, aupr_vac = _safe_binary_metrics(eval_labels, eval_vacuity, split, f"{metric_name} vacuity")
    auroc_dis, aupr_dis = _safe_binary_metrics(eval_labels, eval_dissonance, split, f"{metric_name} dissonance")
    return {
        "evidence": list(evidence_all),
        "vacuity": vacuity_scores,
        "dissonance": dissonance_scores,
        "AUROC": {"vacuity": auroc_vac, "dissonance": auroc_dis},
        "AUPR": {"vacuity": aupr_vac, "dissonance": aupr_dis},
    }


def _evaluate_scalar_score_metrics(
    *,
    split: str,
    labels: np.ndarray,
    metric_eval_mask: np.ndarray,
    scores: np.ndarray,
    metric_name: str,
) -> Dict[str, object]:
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    if not (score_arr.size == label_arr.size == mask_arr.size):
        raise ValueError(
            f"{metric_name} length mismatch: "
            f"scores={score_arr.size}, labels={label_arr.size}, mask={mask_arr.size}."
        )
    eval_labels = label_arr[mask_arr]
    eval_scores = score_arr[mask_arr]
    auroc, aupr = _safe_binary_metrics(eval_labels, eval_scores, split, metric_name)
    return {
        "scores": score_arr,
        "AUROC": auroc,
        "AUPR": aupr,
    }


def _empty_disambig_question_variant_result(n_questions: int) -> Dict[str, object]:
    nan_scores = np.full(int(n_questions), np.nan, dtype=np.float32)
    return {
        "evidence": [],
        "vacuity": nan_scores.copy(),
        "dissonance": nan_scores.copy(),
        "AUROC": {"vacuity": float("nan"), "dissonance": float("nan")},
        "AUPR": {"vacuity": float("nan"), "dissonance": float("nan")},
    }


def _empty_scalar_score_result(n_questions: int) -> Dict[str, object]:
    return {
        "scores": np.full(int(n_questions), np.nan, dtype=np.float32),
        "AUROC": float("nan"),
        "AUPR": float("nan"),
    }


def _sampled_res_y_path(load_dir: str, split: str) -> str:
    return os.path.join(load_dir, split, "sampled_res_y.npy")


def _primary_hallucination_labels(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    if np.any((y_arr < 0) | (y_arr > 1)):
        raise ValueError(f"Expected y in {{0, 1}} for hallucination labels, got values {np.unique(y_arr).tolist()}.")
    return np.asarray(1 - y_arr, dtype=np.int64)


def _load_sampled_res_y_split(
    load_dir: str,
    split: str,
    n_resp: int,
) -> Tuple[np.ndarray, str]:
    path = _sampled_res_y_path(load_dir, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing sampled-response hallucination labels: {path}. "
            "Regenerate semantic clusters to create sampled_res_y.npy."
        )
    sampled_res_y = np.asarray(np.load(path), dtype=np.int64)
    if sampled_res_y.ndim != 2 or int(sampled_res_y.shape[1]) != int(n_resp):
        raise ValueError(
            f"sampled_res_y shape mismatch for split={split}: "
            f"expected second dim={int(n_resp)}, got {tuple(sampled_res_y.shape)}."
        )
    return sampled_res_y, path


def _load_response_slt_lastk_split(
    load_dir: str,
    split: str,
    n_resp: int,
) -> Tuple[List[Dict[int, np.ndarray]], str]:
    path = _response_slt_lastk_path(load_dir, split, n_resp)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing response-level SLT artifact: {path}. "
            "Regenerate semantic clusters to create response_slt_lastk artifacts."
        )
    return load_cluster_embeddings_npz(path), path


def _infer_response_feature_dim(
    response_slt_all: Sequence[Dict[int, np.ndarray]],
    split: str,
) -> int:
    for qid, row in enumerate(response_slt_all):
        for resp_id, vec in row.items():
            dim = int(np.asarray(vec, dtype=np.float32).reshape(-1).shape[0])
            if dim <= 0:
                raise ValueError(
                    f"response_slt_lastk contains empty vectors at split={split}, qid={qid}, resp_id={resp_id}."
                )
            return dim
    raise ValueError(f"No response-level SLT vectors found for split={split}.")


def _flatten_primary_hidden_states(
    X_hidden: np.ndarray,
    last_k: int,
    split: str,
    feature_name: str,
) -> np.ndarray:
    X_arr = np.asarray(X_hidden, dtype=np.float32)
    if X_arr.ndim != 3:
        raise ValueError(
            f"Expected X['{feature_name}'] with shape [n_layers, n_questions, hidden_dim] for split={split}, "
            f"got {X_arr.shape}."
        )
    n_layers = int(X_arr.shape[0])
    k = int(last_k)
    if k <= 0:
        raise ValueError("hallucination_probe_last_k must be positive.")
    if k > n_layers:
        raise ValueError(
            f"hallucination_probe_last_k={k} exceeds available {feature_name} layers ({n_layers}) for split={split}."
        )
    X_input = X_arr[n_layers - k :, :, :]
    X_flat = np.transpose(X_input, (1, 0, 2)).reshape(X_input.shape[1], -1)
    X_flat = _normalize_flat_features(X_flat)
    return X_flat.astype(np.float32, copy=False)


def _normalize_flat_features(X_flat: np.ndarray) -> np.ndarray:
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(X_flat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X_flat / norms


def _validate_response_slt_artifacts(
    response_slt_all: Sequence[Dict[int, np.ndarray]],
    responses: Sequence[Sequence[str]],
    semantic_ids: Sequence[np.ndarray],
    expected_dim: int,
    split: str,
) -> None:
    if not (len(response_slt_all) == len(responses) == len(semantic_ids)):
        raise ValueError(
            f"Response-level SLT size mismatch for split={split}: "
            f"response_slt={len(response_slt_all)}, responses={len(responses)}, semantic_ids={len(semantic_ids)}."
        )

    for qid, (slt_row, resp_row, sid_row) in enumerate(zip(response_slt_all, responses, semantic_ids)):
        ids_row = np.asarray(sid_row, dtype=np.int64).reshape(-1)
        n_resp = len(resp_row)
        if ids_row.shape[0] != n_resp:
            raise ValueError(
                f"Response/semantic_ids mismatch at qid={qid} for split={split}: "
                f"responses={n_resp}, semantic_ids={ids_row.shape[0]}."
            )
        expected_keys = list(range(n_resp))
        actual_keys = sorted(int(key) for key in slt_row.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"response_slt_lastk response-id mismatch at qid={qid} for split={split}: "
                f"expected keys {expected_keys[:10]}{'...' if len(expected_keys) > 10 else ''}, "
                f"got {actual_keys[:10]}{'...' if len(actual_keys) > 10 else ''}."
            )
        for resp_id in expected_keys:
            vec = np.asarray(slt_row[int(resp_id)], dtype=np.float32).reshape(-1)
            if vec.shape[0] != int(expected_dim):
                raise ValueError(
                    f"response_slt_lastk dim mismatch at split={split}, qid={qid}, resp_id={resp_id}: "
                    f"expected={int(expected_dim)}, got={vec.shape[0]}."
                )


def _validate_question_probe_features(
    question_probe_features: Optional[np.ndarray],
    expected_dim: int,
    question_count: int,
    split: str,
    probe_feature: str,
) -> np.ndarray:
    if question_probe_features is None:
        raise ValueError(f"Missing question-level {probe_feature} probe features for split={split}.")
    X_arr = np.asarray(question_probe_features, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError(
            f"Question-level {probe_feature} probe features for split={split} must be rank-2, got {X_arr.shape}."
        )
    if int(X_arr.shape[0]) != int(question_count):
        raise ValueError(
            f"Question-level {probe_feature} probe feature count mismatch for split={split}: "
            f"features={int(X_arr.shape[0])}, questions={int(question_count)}."
        )
    if int(X_arr.shape[1]) != int(expected_dim):
        raise ValueError(
            f"Question-level {probe_feature} probe feature dim mismatch for split={split}: "
            f"expected={int(expected_dim)}, got={int(X_arr.shape[1])}."
        )
    if not np.all(np.isfinite(X_arr)):
        raise ValueError(
            f"Question-level {probe_feature} probe features contain non-finite values for split={split}."
        )
    return X_arr.astype(np.float32, copy=False)


def _load_response_probe_split(
    args,
    split: str,
    last_k: int,
) -> Dict[str, object]:
    linear_data = _load_linear_probe_split(args, split)
    if "slt" not in linear_data["X"]:
        raise ValueError(f"Hallucination SLT probe requires X['slt'] in split={split}.")

    responses_path = os.path.join(linear_data["load_dir"], split, "se_resp_res.jsonl")
    semantic_ids_path = os.path.join(linear_data["load_dir"], split, f"semantic_ids_{args.n_resp}.npy")
    if not os.path.exists(responses_path) or not os.path.exists(semantic_ids_path):
        missing = [path for path in (responses_path, semantic_ids_path) if not os.path.exists(path)]
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    responses = load_resp_json(responses_path)
    semantic_ids = _load_semantic_ids(semantic_ids_path)
    response_slt_all, response_slt_path = _load_response_slt_lastk_split(
        linear_data["load_dir"],
        split,
        int(args.n_resp),
    )
    sampled_res_y, sampled_res_y_path = _load_sampled_res_y_split(
        linear_data["load_dir"],
        split,
        int(args.n_resp),
    )

    if sampled_res_y.shape[0] != len(responses):
        raise ValueError(
            f"sampled_res_y question count mismatch for split={split}: "
            f"labels={sampled_res_y.shape[0]}, responses={len(responses)}."
        )

    X_slt = np.asarray(linear_data["X"]["slt"], dtype=np.float32)
    if X_slt.ndim != 3:
        raise ValueError(
            f"Expected X['slt'] with shape [n_layers, n_questions, hidden_dim] for split={split}, got {X_slt.shape}."
        )
    n_layers = int(X_slt.shape[0])
    if int(last_k) <= 0:
        raise ValueError("hallucination_probe_last_k must be positive.")
    if int(last_k) > n_layers:
        raise ValueError(
            f"hallucination_probe_last_k={int(last_k)} exceeds available SLT layers ({n_layers}) for split={split}."
        )
    expected_dim = int(X_slt.shape[2]) * int(last_k)
    actual_dim = _infer_response_feature_dim(response_slt_all, split)
    if actual_dim != expected_dim:
        raise ValueError(
            f"Response-level SLT feature dim mismatch for split={split}: "
            f"expected={expected_dim}, got={actual_dim}. "
            "Regenerate response_slt_lastk artifacts with matching slt_last_k."
        )

    _validate_response_slt_artifacts(
        response_slt_all=response_slt_all,
        responses=responses,
        semantic_ids=semantic_ids,
        expected_dim=expected_dim,
        split=split,
    )

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    question_ids: List[int] = []
    response_counts: List[int] = []
    for qid, (resp_row, slt_row) in enumerate(zip(responses, response_slt_all)):
        n_resp_q = len(resp_row)
        response_counts.append(int(n_resp_q))
        if n_resp_q > int(sampled_res_y.shape[1]):
            raise ValueError(
                f"sampled_res_y response count mismatch for split={split}, qid={qid}: "
                f"responses={n_resp_q}, labels_per_question={sampled_res_y.shape[1]}."
            )
        for resp_id in range(n_resp_q):
            X_rows.append(np.asarray(slt_row[int(resp_id)], dtype=np.float32).reshape(-1))
            y_rows.append(int(1 - int(sampled_res_y[qid, resp_id])))
            question_ids.append(int(qid))

    if X_rows:
        X_flat = np.vstack(X_rows).astype(np.float32, copy=False)
        X_flat = _normalize_flat_features(X_flat).astype(np.float32, copy=False)
    else:
        X_flat = np.zeros((0, expected_dim), dtype=np.float32)

    return {
        "X_flat": X_flat,
        "hallucination_labels": np.asarray(y_rows, dtype=np.int64),
        "question_ids": np.asarray(question_ids, dtype=np.int64),
        "question_count": int(len(responses)),
        "response_counts": np.asarray(response_counts, dtype=np.int64),
        "feature_dim": int(expected_dim),
        "split": split,
        "load_dir": linear_data["load_dir"],
        "paths": {
            "response_slt_lastk": response_slt_path,
            "sampled_res_y": sampled_res_y_path,
            "responses": responses_path,
            "semantic_ids": semantic_ids_path,
            "X": linear_data["paths"]["X"],
        },
    }


def _resolve_probe_feature(args) -> str:
    probe_feature = str(getattr(args, "probe_feature", "slt")).strip().lower()
    if probe_feature not in {"slt", "tbg"}:
        raise ValueError(f"probe_feature must be 'slt' or 'tbg', got {probe_feature!r}.")
    return probe_feature


def _normalize_probe_target(probe_target: str) -> str:
    norm = str(probe_target or "hallucination").strip().lower()
    if norm not in {"hallucination", "ambiguity"}:
        raise ValueError(
            f"probe_target must be 'hallucination' or 'ambiguity', got {probe_target!r}."
        )
    return norm


def _probe_display_name(probe_target: str, *, capitalize: bool = False) -> str:
    base = "hallucination" if _normalize_probe_target(probe_target) == "hallucination" else "ambiguity"
    return base.title() if capitalize else base


def _probe_metric_prefix(probe_target: str) -> str:
    return f"{_normalize_probe_target(probe_target)}_probe"


def _resolve_question_probe_labels(
    split_dir: str,
    y: np.ndarray,
    probe_target: str,
) -> Tuple[np.ndarray, str, Optional[str]]:
    resolved_probe_target = _normalize_probe_target(probe_target)
    if resolved_probe_target == "hallucination":
        return _primary_hallucination_labels(y), "1-y", None

    ambig_path = os.path.join(split_dir, "ambig_label.npy")
    if not os.path.exists(ambig_path):
        raise FileNotFoundError(
            "probe_target='ambiguity' requires ambig_label.npy for "
            f"split_dir={split_dir}: {ambig_path}"
        )
    probe_labels = np.asarray(np.load(ambig_path), dtype=np.int64).reshape(-1)
    return probe_labels, "ambig_label.npy", ambig_path


def _validate_probe_target_artifacts(args) -> None:
    if _normalize_probe_target(getattr(args, "probe_target", "hallucination")) != "ambiguity":
        return

    model_id = args.model_name.split("/")[-1]
    load_dir = _artifact_load_dir(model_id, args.dataset_name, args.n_samples, args.judge_type)
    required_splits = sorted({"train", "valid", str(args.eval_split)})
    missing = []
    for split in required_splits:
        ambig_path = os.path.join(load_dir, split, "ambig_label.npy")
        if not os.path.exists(ambig_path):
            missing.append(f"{split}: {ambig_path}")
    if missing:
        raise FileNotFoundError(
            "probe_target='ambiguity' requires ambig_label.npy for all required splits. Missing: "
            + ", ".join(missing)
        )


def _load_question_probe_split(
    args,
    split: str,
    last_k: int,
    probe_feature: str,
    dataset_name: Optional[str] = None,
) -> Dict[str, object]:
    linear_data = _load_linear_probe_split(args, split, dataset_name=dataset_name)
    feature_key = str(probe_feature).strip().lower()
    probe_target = _normalize_probe_target(getattr(args, "probe_target", "hallucination"))
    if feature_key not in linear_data["X"]:
        raise ValueError(
            f"{_probe_display_name(probe_target, capitalize=True)} probe with probe_feature={feature_key!r} "
            f"requires X['{feature_key}'] in split={split}."
        )

    X_flat = _flatten_primary_hidden_states(linear_data["X"][feature_key], last_k, split, feature_key)
    split_dir = os.path.join(linear_data["load_dir"], split)
    probe_labels, probe_label_source, probe_label_path = _resolve_question_probe_labels(
        split_dir=split_dir,
        y=linear_data["y"],
        probe_target=probe_target,
    )
    if X_flat.shape[0] != probe_labels.shape[0]:
        raise ValueError(
            f"Question-level {feature_key} probe input mismatch for split={split}: "
            f"features={X_flat.shape[0]}, labels={probe_labels.shape[0]}."
        )

    return {
        "X_flat": X_flat,
        "probe_labels": probe_labels,
        "probe_target": str(probe_target),
        "probe_label_source": str(probe_label_source),
        "question_ids": np.arange(X_flat.shape[0], dtype=np.int64),
        "question_count": int(X_flat.shape[0]),
        "feature_dim": int(X_flat.shape[1]),
        "split": split,
        "load_dir": linear_data["load_dir"],
        "paths": {
            "X": linear_data["paths"]["X"],
            "y": linear_data["paths"]["y"],
            "uncertainty_labels": linear_data["paths"]["uncertainty_labels"],
            "probe_labels": probe_label_path if probe_label_path is not None else linear_data["paths"]["y"],
        },
        "probe_feature": feature_key,
    }


def _resolve_uncertainty_display_labels(label_source: str) -> Tuple[str, str]:
    if str(label_source) == "ambig_label.npy":
        return "clear", "ambig"
    if str(label_source) == "ood_label.npy":
        return "ind", "ood"
    return "truthful", "hallucination"


def _hist_group_values(
    values: np.ndarray,
    labels: np.ndarray,
    metric_eval_mask: np.ndarray,
    label_source: str,
) -> List[Tuple[str, np.ndarray]]:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    n = int(min(values_arr.shape[0], labels_arr.shape[0], mask_arr.shape[0]))
    values_arr = values_arr[:n]
    labels_arr = labels_arr[:n]
    mask_arr = mask_arr[:n]
    neg_label, pos_label = _resolve_uncertainty_display_labels(label_source)
    grouped: List[Tuple[str, np.ndarray]] = []
    for label_value, label_name in ((0, neg_label), (1, pos_label)):
        group_mask = np.logical_and(mask_arr, labels_arr == int(label_value))
        group_values = values_arr[group_mask]
        group_values = group_values[np.isfinite(group_values)]
        grouped.append((label_name, group_values.astype(float, copy=False)))
    return grouped


def _flatten_response_hallucination_prob_by_label(
    response_hallucination_prob: Sequence[Sequence[float]],
    labels: np.ndarray,
    metric_eval_mask: np.ndarray,
    label_source: str,
) -> List[Tuple[str, np.ndarray]]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    n = int(min(len(response_hallucination_prob), labels_arr.shape[0], mask_arr.shape[0]))
    neg_label, pos_label = _resolve_uncertainty_display_labels(label_source)
    grouped_rows: Dict[int, List[np.ndarray]] = {0: [], 1: []}
    for qid in range(n):
        if not bool(mask_arr[qid]):
            continue
        arr = np.asarray(response_hallucination_prob[qid], dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        label_value = int(labels_arr[qid])
        if label_value in grouped_rows:
            grouped_rows[label_value].append(arr.astype(float, copy=False))
    return [
        (
            neg_label,
            np.concatenate(grouped_rows[0]).astype(float, copy=False)
            if grouped_rows[0]
            else np.zeros(0, dtype=float),
        ),
        (
            pos_label,
            np.concatenate(grouped_rows[1]).astype(float, copy=False)
            if grouped_rows[1]
            else np.zeros(0, dtype=float),
        ),
    ]


def _plot_histogram_figure(
    grouped_values: Sequence[Tuple[str, np.ndarray]],
    title: str,
    xlabel: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True, dpi=HIST_FIG_DPI)
    for name, values in grouped_values:
        values_arr = np.asarray(values, dtype=float).reshape(-1)
        if values_arr.size == 0:
            continue
        ax.hist(
            values_arr,
            bins=HIST_BINS,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.5,
            color=HIST_COLORS.get(str(name), "#4e79a7"),
            label=f"{name} (n={values_arr.size})",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    return fig


def _log_wandb_image(key: str, fig, step: int = 0) -> None:
    if wandb.run is None:
        plt.close(fig)
        return
    image = wandb.Image(fig)
    wandb.run.summary[key] = image
    wandb.log({key: image}, step=int(step), commit=False)
    plt.close(fig)


def _log_discounted_hallucination_histograms(
    split: str,
    artifacts: SplitArtifacts,
    discounted_results: Dict[str, object],
    step: int = 0,
) -> None:
    labels = np.asarray(artifacts.labels, dtype=np.int64).reshape(-1)
    metric_eval_mask = np.asarray(discounted_results.get("metric_eval_mask"), dtype=bool).reshape(-1)
    if metric_eval_mask.shape[0] != labels.shape[0]:
        raise ValueError(
            "Histogram mask mismatch: "
            f"mask={metric_eval_mask.shape[0]}, labels={labels.shape[0]}."
        )
    probe_target = _normalize_probe_target(discounted_results.get("probe_target", "hallucination"))
    label_pair = "/".join(_resolve_uncertainty_display_labels(artifacts.label_source))
    hist_specs = (
        [
            (
                "response_hallucination_prob",
                "hist_question_mean_hallu_prob",
                "hist_response_hallu_prob",
                f"{split}: mean sampled hallucination prob by {label_pair}",
                "mean predicted hallucination probability",
                f"{split}: sampled response hallucination prob by {label_pair}",
                "predicted hallucination probability",
            )
        ]
        if probe_target == "hallucination"
        else [
            (
                "response_ambiguity_prob",
                "hist_question_mean_ambiguity_prob",
                "hist_response_ambiguity_prob",
                f"{split}: mean sampled ambiguity prob by {label_pair}",
                "mean predicted ambiguity probability",
                f"{split}: sampled response ambiguity prob by {label_pair}",
                "predicted ambiguity probability",
            ),
            (
                "response_discount_penalty_prob",
                "hist_question_mean_clear_penalty_prob",
                "hist_response_clear_penalty_prob",
                f"{split}: mean sampled clear-prob penalty by {label_pair}",
                "mean discount penalty probability",
                f"{split}: sampled response clear-prob penalty by {label_pair}",
                "discount penalty probability",
            ),
        ]
    )
    logged_any = False
    for (
        payload_key,
        question_wandb_key,
        response_wandb_key,
        question_title,
        question_xlabel,
        response_title,
        response_xlabel,
    ) in hist_specs:
        response_prob = discounted_results.get(payload_key)
        if response_prob is None:
            continue
        if len(response_prob) != labels.shape[0]:
            raise ValueError(
                "Histogram input mismatch: "
                f"{payload_key}={len(response_prob)}, labels={labels.shape[0]}."
            )

        question_mean_prob = np.full(labels.shape[0], np.nan, dtype=float)
        for qid, probs in enumerate(response_prob):
            arr = np.asarray(probs, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                question_mean_prob[qid] = float(np.mean(arr))

        grouped_question_values = _hist_group_values(
            question_mean_prob,
            labels,
            metric_eval_mask,
            artifacts.label_source,
        )
        question_fig = _plot_histogram_figure(
            grouped_question_values,
            title=question_title,
            xlabel=question_xlabel,
        )
        _log_wandb_image(f"{split}/{question_wandb_key}", question_fig, step=step)

        grouped_response_values = _flatten_response_hallucination_prob_by_label(
            response_prob,
            labels,
            metric_eval_mask,
            artifacts.label_source,
        )
        response_fig = _plot_histogram_figure(
            grouped_response_values,
            title=response_title,
            xlabel=response_xlabel,
        )
        _log_wandb_image(f"{split}/{response_wandb_key}", response_fig, step=step)
        logged_any = True

    if not logged_any:
        print("[Histogram] Skipping probe-probability histograms because response scores are unavailable.")


def _load_split_artifacts_for(
    args,
    dataset_name: str,
    split: str,
) -> SplitArtifacts:
    model_id = args.model_name.split("/")[-1]
    load_dir = _artifact_load_dir(model_id, dataset_name, args.n_samples, args.judge_type)
    split_dir = os.path.join(load_dir, split)
    paths = _split_artifact_paths(load_dir, split, int(args.n_resp))
    missing = [
        path
        for key, path in paths.items()
        if key in {"responses", "semantic_ids", "sampled_response_sequence_scores", "se", "y"}
        and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    response_rows = _load_jsonl_rows(paths["responses"])
    responses = [row["responses"] for row in response_rows]
    semantic_ids = _load_semantic_ids(paths["semantic_ids"])
    sequence_scores = load_sequence_scores_npz(paths["sampled_response_sequence_scores"])
    response_sequence_probs = _validate_response_sequence_probs_artifact(
        sequence_scores["sequence_prob"],
        responses,
        semantic_ids,
        paths["sampled_response_sequence_scores"],
    )
    response_norm_sequence_probs = _validate_response_sequence_probs_artifact(
        _sequence_prob_from_length_normalized_logprob(
            sequence_scores["length_normalized_sequence_logprob"]
        ),
        responses,
        semantic_ids,
        paths["sampled_response_sequence_scores"],
    )
    se_scores = np.asarray(np.load(paths["se"]), dtype=np.float32).reshape(-1)
    y = np.asarray(np.load(paths["y"]), dtype=np.int64).reshape(-1)
    labels, label_source, label_path = _resolve_dataset_labels(split_dir, dataset_name, y)

    disambig_question_semantic_ids = None
    disambig_question_sequence_probs = None
    disambig_question_norm_sequence_probs = None
    disambig_question_metrics_available = False
    disambig_question_metrics_skip_reason = None
    if _is_disambig_question_dataset(dataset_name):
        disambig_question_semantic_ids_path = paths["disambig_question_semantic_ids"]
        row_disambig_semantic_ids = _load_optional_jsonl_row_matrix(
            response_rows,
            "disambig_question_semantic_ids",
            dtype=np.int64,
            source=paths["responses"],
        )
        row_disambig_sequence_probs = _load_optional_jsonl_row_matrix(
            response_rows,
            "disambig_question_sequence_prob",
            dtype=np.float32,
            source=paths["responses"],
        )
        row_disambig_norm_logprobs = _load_optional_jsonl_row_matrix(
            response_rows,
            "disambig_question_length_normalized_sequence_logprob",
            dtype=np.float32,
            source=paths["responses"],
        )

        if disambig_question_semantic_ids_path is not None and os.path.exists(disambig_question_semantic_ids_path):
            disambig_question_semantic_ids = _load_semantic_ids(disambig_question_semantic_ids_path)
            disambig_question_semantic_ids_source = disambig_question_semantic_ids_path
        elif row_disambig_semantic_ids is not None:
            disambig_question_semantic_ids = [
                np.asarray(row, dtype=np.int64).reshape(-1) for row in row_disambig_semantic_ids
            ]
            disambig_question_semantic_ids_source = f"{paths['responses']}:disambig_question_semantic_ids"
        else:
            disambig_question_semantic_ids_source = (
                disambig_question_semantic_ids_path
                if disambig_question_semantic_ids_path is not None
                else paths["responses"]
            )

        component_presence = {
            "semantic_ids": disambig_question_semantic_ids is not None,
            "sequence_prob": row_disambig_sequence_probs is not None,
            "norm_sequence_prob": row_disambig_norm_logprobs is not None,
        }
        if not any(component_presence.values()):
            disambig_question_metrics_skip_reason = (
                f"No saved disambig-question semantic ids or sequence scores found in {split_dir}."
            )
            disambig_question_semantic_ids = None
        else:
            missing_components = [name for name, present in component_presence.items() if not present]
            if missing_components:
                raise ValueError(
                    "Disambig-question artifacts are incomplete in "
                    f"{split_dir}: missing {missing_components}."
                )
            assert disambig_question_semantic_ids is not None
            assert row_disambig_sequence_probs is not None
            assert row_disambig_norm_logprobs is not None
            disambig_question_semantic_ids = _validate_disambig_question_semantic_ids_artifact(
                disambig_question_semantic_ids,
                responses,
                disambig_question_semantic_ids_source,
            )
            disambig_question_sequence_probs = _validate_disambig_question_sequence_probs_artifact(
                _stack_jsonl_row_matrix(
                    row_disambig_sequence_probs,
                    field="disambig_question_sequence_prob",
                    source=paths["responses"],
                    dtype=np.float32,
                ),
                responses,
                disambig_question_semantic_ids,
                f"{paths['responses']}:disambig_question_sequence_prob",
                "sequence_prob",
            )
            disambig_question_norm_sequence_probs = _validate_disambig_question_sequence_probs_artifact(
                _sequence_prob_from_length_normalized_logprob(
                    _stack_jsonl_row_matrix(
                        row_disambig_norm_logprobs,
                        field="disambig_question_length_normalized_sequence_logprob",
                        source=paths["responses"],
                        dtype=np.float32,
                    )
                ),
                responses,
                disambig_question_semantic_ids,
                f"{paths['responses']}:disambig_question_length_normalized_sequence_logprob",
                "norm_sequence_prob",
            )
            disambig_question_metrics_available = True

    n_questions = len(responses)
    lengths = {
        "responses": n_questions,
        "semantic_ids": len(semantic_ids),
        "sequence_prob": int(response_sequence_probs.shape[0]),
        "norm_sequence_prob": int(response_norm_sequence_probs.shape[0]),
        "se": int(se_scores.shape[0]),
        "y": int(y.shape[0]),
        "labels": int(labels.shape[0]),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Split artifact size mismatch for {split_dir}: {lengths}")

    for idx, (resp_row, sid_row) in enumerate(zip(responses, semantic_ids)):
        if len(resp_row) != int(sid_row.shape[0]):
            raise ValueError(
                f"Response/semantic_ids mismatch at qid={idx} in {split_dir}: "
                f"responses={len(resp_row)}, semantic_ids={int(sid_row.shape[0])}"
            )

    paths["labels"] = label_path
    return SplitArtifacts(
        load_dir=load_dir,
        split_dir=split_dir,
        split=split,
        responses=responses,
        semantic_ids=semantic_ids,
        response_sequence_probs=response_sequence_probs,
        response_norm_sequence_probs=response_norm_sequence_probs,
        se_scores=se_scores,
        y=y,
        labels=labels,
        label_source=label_source,
        paths=paths,
        disambig_question_semantic_ids=disambig_question_semantic_ids,
        disambig_question_sequence_probs=disambig_question_sequence_probs,
        disambig_question_norm_sequence_probs=disambig_question_norm_sequence_probs,
        disambig_question_metrics_available=disambig_question_metrics_available,
        disambig_question_metrics_skip_reason=disambig_question_metrics_skip_reason,
    )


def _load_split_artifacts(args) -> SplitArtifacts:
    return _load_split_artifacts_for(args, args.dataset_name, args.eval_split)


def _metric_dict_for_history(split: str, method_name: str, results: Dict[str, object]) -> Dict[str, float]:
    if method_name == "verbalized" and (
        "AUROC_epistemic" in results or "AUROC_aleatoric" in results
    ):
        metrics = {
            f"{split}/auroc_epistemic_verbalized": float(results.get("AUROC_epistemic", float("nan"))),
            f"{split}/aupr_epistemic_verbalized": float(results.get("AUPR_epistemic", float("nan"))),
            f"{split}/auroc_epistemic_verbalized_no_clear_err": float(
                results.get("AUROC_epistemic_no_clear_err", float("nan"))
            ),
            f"{split}/aupr_epistemic_verbalized_no_clear_err": float(
                results.get("AUPR_epistemic_no_clear_err", float("nan"))
            ),
            f"{split}/auroc_epistemic_verbalized_no_clear_large_n_cluster": float(
                results.get("AUROC_epistemic_no_clear_large_n_cluster", float("nan"))
            ),
            f"{split}/aupr_epistemic_verbalized_no_clear_large_n_cluster": float(
                results.get("AUPR_epistemic_no_clear_large_n_cluster", float("nan"))
            ),
            f"{split}/auroc_aleatoric_verbalized": float(results.get("AUROC_aleatoric", float("nan"))),
            f"{split}/aupr_aleatoric_verbalized": float(results.get("AUPR_aleatoric", float("nan"))),
            f"{split}/auroc_aleatoric_verbalized_no_clear_err": float(
                results.get("AUROC_aleatoric_no_clear_err", float("nan"))
            ),
            f"{split}/aupr_aleatoric_verbalized_no_clear_err": float(
                results.get("AUPR_aleatoric_no_clear_err", float("nan"))
            ),
            f"{split}/auroc_aleatoric_verbalized_no_clear_large_n_cluster": float(
                results.get("AUROC_aleatoric_no_clear_large_n_cluster", float("nan"))
            ),
            f"{split}/aupr_aleatoric_verbalized_no_clear_large_n_cluster": float(
                results.get("AUPR_aleatoric_no_clear_large_n_cluster", float("nan"))
            ),
        }
        refusal_stats = results.get("ood_refusal_eval")
        if isinstance(refusal_stats, dict) and refusal_stats.get("applied", False):
            metrics.update(
                {
                    f"{split}/refusal_accuracy_{method_name}": float(refusal_stats.get("accuracy", float("nan"))),
                    f"{split}/excluded_ood_count_{method_name}": int(refusal_stats.get("excluded_ood_count", 0)),
                    f"{split}/excluded_id_count_{method_name}": int(refusal_stats.get("excluded_id_count", 0)),
                    f"{split}/excluded_total_count_{method_name}": int(refusal_stats.get("excluded_total_count", 0)),
                    f"{split}/remaining_count_{method_name}": int(refusal_stats.get("remaining_count", 0)),
                    f"{split}/total_count_{method_name}": int(refusal_stats.get("total_count", 0)),
                }
            )
        return metrics

    metrics = {
        f"{split}/baseline_auroc_{method_name}": float(results["AUROC"]),
        f"{split}/baseline_aupr_{method_name}": float(results["AUPR"]),
        f"{split}/baseline_auroc_{method_name}_no_clear_err": float(results.get("AUROC_no_clear_err", float("nan"))),
        f"{split}/baseline_aupr_{method_name}_no_clear_err": float(results.get("AUPR_no_clear_err", float("nan"))),
        f"{split}/baseline_auroc_{method_name}_no_clear_large_n_cluster": float(
            results.get("AUROC_no_clear_large_n_cluster", float("nan"))
        ),
        f"{split}/baseline_aupr_{method_name}_no_clear_large_n_cluster": float(
            results.get("AUPR_no_clear_large_n_cluster", float("nan"))
        ),
    }
    refusal_stats = results.get("ood_refusal_eval")
    if isinstance(refusal_stats, dict) and refusal_stats.get("applied", False):
        metrics.update(
            {
                f"{split}/refusal_accuracy_{method_name}": float(refusal_stats.get("accuracy", float("nan"))),
                f"{split}/excluded_ood_count_{method_name}": int(refusal_stats.get("excluded_ood_count", 0)),
                f"{split}/excluded_id_count_{method_name}": int(refusal_stats.get("excluded_id_count", 0)),
                f"{split}/excluded_total_count_{method_name}": int(refusal_stats.get("excluded_total_count", 0)),
                f"{split}/remaining_count_{method_name}": int(refusal_stats.get("remaining_count", 0)),
                f"{split}/total_count_{method_name}": int(refusal_stats.get("total_count", 0)),
            }
        )
    return metrics


def _artifact_baseline_metrics(split: str, baseline_scores: Dict[str, object]) -> Dict[str, float]:
    metrics = {
        f"{split}/baseline_auroc_se": float(baseline_scores["AUROC"]["se"]),
        f"{split}/baseline_auroc_cnt_se": float(baseline_scores["AUROC"]["cluster_cnt_se"]),
        f"{split}/baseline_aupr_se": float(baseline_scores["AUPR"]["se"]),
        f"{split}/baseline_aupr_cnt_se": float(baseline_scores["AUPR"]["cluster_cnt_se"]),
        f"{split}/baseline_auroc_se_no_clear_err": float(
            baseline_scores.get("AUROC_no_clear_err", {}).get("se", float("nan"))
        ),
        f"{split}/baseline_aupr_se_no_clear_err": float(
            baseline_scores.get("AUPR_no_clear_err", {}).get("se", float("nan"))
        ),
        f"{split}/baseline_auroc_se_no_clear_large_n_cluster": float(
            baseline_scores.get("AUROC_no_clear_large_n_cluster", {}).get("se", float("nan"))
        ),
        f"{split}/baseline_aupr_se_no_clear_large_n_cluster": float(
            baseline_scores.get("AUPR_no_clear_large_n_cluster", {}).get("se", float("nan"))
        ),
    }
    refusal_stats = baseline_scores.get("ood_refusal_eval")
    if isinstance(refusal_stats, dict) and refusal_stats.get("applied", False):
        metrics.update(
            {
                f"{split}/refusal_accuracy": float(refusal_stats.get("accuracy", float("nan"))),
                f"{split}/excluded_ood_count": int(refusal_stats.get("excluded_ood_count", 0)),
                f"{split}/excluded_id_count": int(refusal_stats.get("excluded_id_count", 0)),
                f"{split}/excluded_total_count": int(refusal_stats.get("excluded_total_count", 0)),
                f"{split}/remaining_count": int(refusal_stats.get("remaining_count", 0)),
                f"{split}/total_count": int(refusal_stats.get("total_count", 0)),
            }
        )
    return metrics


def _main_summary_metrics(
    split: str,
    results: Dict[str, object],
) -> Dict[str, float]:
    metrics = {
        f"{split}/mean_n_clusters": float(np.mean([len(row) for row in results["semantic_ids"]])),
        f"{split}/auroc_vacuity": float(results["AUROC"]["vacuity"]),
        f"{split}/auroc_dissonance": float(results["AUROC"]["dissonance"]),
        f"{split}/aupr_vacuity": float(results["AUPR"]["vacuity"]),
        f"{split}/aupr_dissonance": float(results["AUPR"]["dissonance"]),
    }
    refusal_stats = results.get("ood_refusal_eval")
    if isinstance(refusal_stats, dict) and refusal_stats.get("applied", False):
        metrics.update(
            {
                f"{split}/refusal_accuracy": float(refusal_stats.get("accuracy", float("nan"))),
                f"{split}/excluded_ood_count": int(refusal_stats.get("excluded_ood_count", 0)),
                f"{split}/excluded_id_count": int(refusal_stats.get("excluded_id_count", 0)),
                f"{split}/excluded_total_count": int(refusal_stats.get("excluded_total_count", 0)),
                f"{split}/remaining_count": int(refusal_stats.get("remaining_count", 0)),
                f"{split}/total_count": int(refusal_stats.get("total_count", 0)),
            }
        )
    return metrics


def _disambig_question_summary_metrics(
    split: str,
    results: Dict[str, object],
) -> Dict[str, float]:
    count_results = results["count"]
    seq_prob_results = results["seq_prob"]
    norm_seq_prob_results = results["norm_seq_prob"]
    metrics = {
        f"{split}/auroc_disambig_question_vacuity": float(count_results["AUROC"]["vacuity"]),
        f"{split}/auroc_disambig_question_dissonance": float(count_results["AUROC"]["dissonance"]),
        f"{split}/aupr_disambig_question_vacuity": float(count_results["AUPR"]["vacuity"]),
        f"{split}/aupr_disambig_question_dissonance": float(count_results["AUPR"]["dissonance"]),
        f"{split}/auroc_disambig_question_seq_prob_vacuity": float(seq_prob_results["AUROC"]["vacuity"]),
        f"{split}/auroc_disambig_question_seq_prob_dissonance": float(seq_prob_results["AUROC"]["dissonance"]),
        f"{split}/aupr_disambig_question_seq_prob_vacuity": float(seq_prob_results["AUPR"]["vacuity"]),
        f"{split}/aupr_disambig_question_seq_prob_dissonance": float(seq_prob_results["AUPR"]["dissonance"]),
        f"{split}/auroc_disambig_question_norm_seq_prob_vacuity": float(
            norm_seq_prob_results["AUROC"]["vacuity"]
        ),
        f"{split}/auroc_disambig_question_norm_seq_prob_dissonance": float(
            norm_seq_prob_results["AUROC"]["dissonance"]
        ),
        f"{split}/aupr_disambig_question_norm_seq_prob_vacuity": float(
            norm_seq_prob_results["AUPR"]["vacuity"]
        ),
        f"{split}/aupr_disambig_question_norm_seq_prob_dissonance": float(
            norm_seq_prob_results["AUPR"]["dissonance"]
        ),
    }
    if bool(results.get("available")) or bool(results.get("skip_reason")):
        theils_u_seq_prob_results = results["theils_u_seq_prob"]
        theils_u_norm_seq_prob_results = results["theils_u_norm_seq_prob"]
        theils_u_resp_norm_seq_prob_dissonance_results = results["theils_u_resp_norm_seq_prob_dissonance"]
        metrics.update(
            {
                f"{split}/auroc_theils_u_seq_prob_vacuity": float(theils_u_seq_prob_results["AUROC"]["vacuity"]),
                f"{split}/auroc_theils_u_seq_prob_dissonance": float(
                    theils_u_seq_prob_results["AUROC"]["dissonance"]
                ),
                f"{split}/aupr_theils_u_seq_prob_vacuity": float(theils_u_seq_prob_results["AUPR"]["vacuity"]),
                f"{split}/aupr_theils_u_seq_prob_dissonance": float(
                    theils_u_seq_prob_results["AUPR"]["dissonance"]
                ),
                f"{split}/auroc_theils_u_norm_seq_prob_vacuity": float(
                    theils_u_norm_seq_prob_results["AUROC"]["vacuity"]
                ),
                f"{split}/auroc_theils_u_norm_seq_prob_dissonance": float(
                    theils_u_norm_seq_prob_results["AUROC"]["dissonance"]
                ),
                f"{split}/aupr_theils_u_norm_seq_prob_vacuity": float(
                    theils_u_norm_seq_prob_results["AUPR"]["vacuity"]
                ),
                f"{split}/aupr_theils_u_norm_seq_prob_dissonance": float(
                    theils_u_norm_seq_prob_results["AUPR"]["dissonance"]
                ),
                f"{split}/auroc_theils_u_resp_norm_seq_prob_dissonance": float(
                    theils_u_resp_norm_seq_prob_dissonance_results["AUROC"]
                ),
                f"{split}/aupr_theils_u_resp_norm_seq_prob_dissonance": float(
                    theils_u_resp_norm_seq_prob_dissonance_results["AUPR"]
                ),
            }
        )
    return metrics


def _hallucination_probe_summary_metrics(
    split: str,
    probe_results: Dict[str, object],
) -> Dict[str, float]:
    probe_target = _normalize_probe_target(probe_results.get("probe_target", "hallucination"))
    metric_prefix = _probe_metric_prefix(probe_target)
    metrics = {
        f"{split}/{metric_prefix}_auroc": float(probe_results["AUROC"]),
        f"{split}/{metric_prefix}_aupr": float(probe_results["AUPR"]),
        f"{split}/probe_ece": float(probe_results.get("probe_ece", float("nan"))),
    }
    if str(split) != "valid":
        metrics["valid/probe_ece"] = float(probe_results.get("valid_probe_ece", float("nan")))
    if bool(probe_results.get("calibration_enabled", False)):
        metrics["valid/calibrated_probe_ece"] = float(probe_results.get("valid_calibrated_probe_ece", float("nan")))
    return metrics


def _discounted_summary_metrics(
    split: str,
    discounted_results: Dict[str, object],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for spec in DISCOUNTED_BRANCH_SPECS:
        branch_results = discounted_results[spec.key]
        metrics.update(
            {
                f"{split}/best_{spec.prefix}_zeta": float(branch_results["best_zeta"]),
                f"{split}/best_{spec.prefix}_selection_auroc": float(branch_results["best_selection_auroc"]),
                f"{split}/best_{spec.prefix}_selection_aupr": float(branch_results["best_selection_aupr"]),
                f"{split}/auroc_{spec.metric_stem}_vacuity": float(branch_results["AUROC"]["vacuity"]),
                f"{split}/auroc_{spec.metric_stem}_dissonance": float(branch_results["AUROC"]["dissonance"]),
                f"{split}/aupr_{spec.metric_stem}_vacuity": float(branch_results["AUPR"]["vacuity"]),
                f"{split}/aupr_{spec.metric_stem}_dissonance": float(branch_results["AUPR"]["dissonance"]),
            }
        )
        metrics.update(
            {
                f"{split}/{spec.prefix}_selected_zeta": float(branch_results["selected_zeta"]),
                f"{split}/{spec.prefix}_selection_auroc": float(branch_results["selection_auroc"]),
                f"{split}/{spec.prefix}_selection_aupr": float(branch_results["selection_aupr"]),
                f"{split}/{spec.prefix}_eval_oracle_best_zeta": float(branch_results["eval_oracle_best_zeta"]),
                f"{split}/{spec.prefix}_eval_oracle_best_auroc": float(branch_results["eval_oracle_best_auroc"]),
                f"{split}/{spec.prefix}_eval_oracle_best_aupr": float(branch_results["eval_oracle_best_aupr"]),
                f"{split}/{spec.prefix}_selected_matches_eval_best": float(
                    bool(branch_results["selected_matches_eval_best"])
                ),
            }
        )
    if bool(discounted_results.get("zeta_selection_enabled", False)):
        metrics.update(
            {
                f"{split}/pseudo_ood_replaced_count": float(discounted_results.get("pseudo_ood_replaced_count", 0)),
                f"{split}/pseudo_ood_positive_count": float(discounted_results.get("pseudo_ood_positive_count", 0)),
            }
        )
    exp_theils_resp = discounted_results.get("exp_discounted_theils_u_resp_norm_seq_prob_dissonance")
    if isinstance(exp_theils_resp, dict):
        metrics.update(
            {
                f"{split}/best_exp_discounted_theils_u_resp_norm_seq_prob_dissonance_zeta": float(
                    exp_theils_resp["best_zeta"]
                ),
                f"{split}/auroc_exp_discounted_theils_u_resp_norm_seq_prob_dissonance": float(
                    exp_theils_resp["AUROC"]
                ),
                f"{split}/aupr_exp_discounted_theils_u_resp_norm_seq_prob_dissonance": float(
                    exp_theils_resp["AUPR"]
                ),
                f"{split}/exp_discounted_theils_u_resp_norm_seq_prob_dissonance_selected_zeta": float(
                    exp_theils_resp["selected_zeta"]
                ),
                f"{split}/exp_discounted_theils_u_resp_norm_seq_prob_dissonance_selection_auroc": float(
                    exp_theils_resp["selection_auroc"]
                ),
                f"{split}/exp_discounted_theils_u_resp_norm_seq_prob_dissonance_selection_aupr": float(
                    exp_theils_resp["selection_aupr"]
                ),
                f"{split}/exp_discounted_theils_u_resp_norm_seq_prob_dissonance_eval_oracle_best_zeta": float(
                    exp_theils_resp["eval_oracle_best_zeta"]
                ),
                f"{split}/exp_discounted_theils_u_resp_norm_seq_prob_dissonance_eval_oracle_best_auroc": float(
                    exp_theils_resp["eval_oracle_best_auroc"]
                ),
                f"{split}/exp_discounted_theils_u_resp_norm_seq_prob_dissonance_eval_oracle_best_aupr": float(
                    exp_theils_resp["eval_oracle_best_aupr"]
                ),
                f"{split}/exp_discounted_theils_u_resp_norm_seq_prob_dissonance_selected_matches_eval_best": float(
                    bool(exp_theils_resp["selected_matches_eval_best"])
                ),
            }
        )
    return metrics


def _write_discounted_wandb_summary_metadata(discounted_results: Dict[str, object]) -> None:
    if wandb.run is None:
        return

    wandb.run.summary["zeta_selection_enabled"] = bool(discounted_results.get("zeta_selection_enabled", False))
    wandb.run.summary["zeta_selection_split"] = str(discounted_results.get("selection_split"))
    wandb.run.summary["zeta_selection_dataset"] = str(discounted_results.get("selection_dataset_name"))
    wandb.run.summary["zeta_selection_label_source"] = str(discounted_results.get("selection_label_source"))
    pseudo_dataset_name = discounted_results.get("pseudo_ood_dataset_name")
    if pseudo_dataset_name is not None:
        wandb.run.summary["pseudo_ood_dataset_name"] = str(pseudo_dataset_name)
    wandb.run.summary["pseudo_ood_replaced_count"] = int(discounted_results.get("pseudo_ood_replaced_count", 0))
    wandb.run.summary["pseudo_ood_positive_count"] = int(discounted_results.get("pseudo_ood_positive_count", 0))

    for spec in DISCOUNTED_BRANCH_SPECS:
        branch_results = discounted_results[spec.key]
        wandb.run.summary[f"{spec.prefix}_selected_zeta"] = float(branch_results["selected_zeta"])
        wandb.run.summary[f"{spec.prefix}_selection_split"] = str(branch_results["selection_split"])
        wandb.run.summary[f"{spec.prefix}_selection_dataset"] = str(branch_results["selection_dataset_name"])
        wandb.run.summary[f"{spec.prefix}_selection_label_source"] = str(branch_results["selection_label_source"])
        wandb.run.summary[f"{spec.prefix}_selection_auroc"] = float(branch_results["selection_auroc"])
        wandb.run.summary[f"{spec.prefix}_selection_aupr"] = float(branch_results["selection_aupr"])
        wandb.run.summary[f"{spec.prefix}_eval_oracle_best_zeta"] = float(branch_results["eval_oracle_best_zeta"])
        wandb.run.summary[f"{spec.prefix}_eval_oracle_best_auroc"] = float(branch_results["eval_oracle_best_auroc"])
        wandb.run.summary[f"{spec.prefix}_eval_oracle_best_aupr"] = float(branch_results["eval_oracle_best_aupr"])
        wandb.run.summary[f"{spec.prefix}_selected_matches_eval_best"] = bool(
            branch_results["selected_matches_eval_best"]
        )


def _log_metrics(metrics: Dict[str, float], step: int = 0) -> None:
    if wandb.run is None:
        return
    wandb.run.summary.update(metrics)
    history_metrics = dict(metrics)
    history_metrics["epoch"] = int(step)
    wandb.log(history_metrics, step=int(step), commit=False)


def _print_discounted_grid_summary(
    branch_label: str,
    grid_label: str,
    grid_results: Sequence[Dict[str, object]],
) -> None:
    for row in grid_results:
        print(
            f"[{branch_label}][{grid_label}][zeta={float(row['zeta']):.2f}] "
            f"AUROC vacuity={float(row['AUROC']['vacuity']):.4f}, "
            f"dissonance={float(row['AUROC']['dissonance']):.4f}; "
            f"AUPR vacuity={float(row['AUPR']['vacuity']):.4f}, "
            f"dissonance={float(row['AUPR']['dissonance']):.4f}"
        )


def _print_discounted_branch_diagnostics(
    branch_label: str,
    branch_results: Dict[str, object],
) -> None:
    print(
        f"[{branch_label}] selected_zeta={float(branch_results['selected_zeta']):.2f} "
        f"selection_split={str(branch_results['selection_split'])} "
        f"selection_auroc={float(branch_results['selection_auroc']):.4f} "
        f"selection_aupr={float(branch_results['selection_aupr']):.4f} "
        f"eval_oracle_best_zeta={float(branch_results['eval_oracle_best_zeta']):.2f} "
        f"eval_oracle_best_auroc={float(branch_results['eval_oracle_best_auroc']):.4f} "
        f"eval_oracle_best_aupr={float(branch_results['eval_oracle_best_aupr']):.4f} "
        f"selected_matches_eval_best={bool(branch_results['selected_matches_eval_best'])}"
    )


def _fit_validation_temperature(
    args,
    clf,
    last_k: int,
    probe_feature: str,
    fit_temperature: bool = True,
) -> Dict[str, object]:
    probe_target = _normalize_probe_target(getattr(args, "probe_target", "hallucination"))
    probe_display_name = f"{_probe_display_name(probe_target, capitalize=True)} Probe"
    valid_data = _load_question_probe_split(args, "valid", last_k, probe_feature)
    valid_labels = np.asarray(valid_data["probe_labels"], dtype=np.int64)
    valid_mask, _ = compute_ood_refusal_filter(
        labels=valid_labels,
        dataset_name=args.dataset_name,
        split="valid",
        remove_refusal=bool(args.remove_refusal),
        load_dir=valid_data["load_dir"],
    )
    valid_mask_arr = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid_mask_arr.shape[0] != int(valid_data["question_count"]):
        raise ValueError(
            "Validation calibration mask length mismatch: "
            f"mask={valid_mask_arr.shape[0]}, question_count={int(valid_data['question_count'])}."
        )

    valid_logits = clf.predict_logits(np.asarray(valid_data["X_flat"], dtype=np.float32))
    valid_scores_raw = positive_probability_from_logits(valid_logits)
    valid_probe_ece, valid_probe_calibration = compute_binary_ece(
        labels=valid_labels,
        scores=valid_scores_raw,
        metric_eval_mask=valid_mask_arr,
        split="valid",
        metric_name=probe_display_name,
    )
    fit_payload = None
    temperature = 1.0
    available = False
    if fit_temperature:
        fit_payload = fit_temperature_scaler(
            logits=valid_logits,
            labels=valid_labels,
            metric_eval_mask=valid_mask_arr,
        )
        temperature = float(fit_payload.get("temperature", 1.0))
        available = bool(fit_payload.get("available", False))
    if available:
        valid_scores = positive_probability_from_logits(valid_logits, temperature=temperature)
        valid_ece, valid_calibration = compute_binary_ece(
            labels=valid_labels,
            scores=valid_scores,
            metric_eval_mask=valid_mask_arr,
            split="valid",
            metric_name=f"{probe_display_name} Calibrated",
        )
    else:
        valid_ece = float("nan")
        valid_calibration = None

    return {
        "enabled": True,
        "available": available,
        "temperature": temperature,
        "source_split": "valid",
        "source_probe_feature": str(probe_feature),
        "probe_target": str(probe_target),
        "fit": fit_payload,
        "metric_eval_mask": valid_mask_arr,
        "valid_probe_ece": float(valid_probe_ece),
        "valid_probe_calibration": valid_probe_calibration,
        "valid_calibrated_probe_ece": float(valid_ece),
        "valid_calibration": valid_calibration,
    }


def _load_causal_lm(model_name: str):
    resolved_model_name = _resolve_hf_model_name(model_name)
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if _needs_8bit_quantized_loading(model_name):
        _ensure_bitsandbytes_available()
        print(f"[int8] Loading {resolved_model_name} with device_map=auto.")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    elif _is_gemma_4_e4b_model_name(model_name):
        if Gemma4ForCausalLM is None:
            raise ImportError("Gemma4ForCausalLM is unavailable in the installed transformers package.")
        model = Gemma4ForCausalLM.from_pretrained(
            resolved_model_name,
            device_map="auto",
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map="auto",
            torch_dtype=dtype,
        )
    return model, tokenizer


def _load_baseline_data(args, artifacts: SplitArtifacts, ptrue) -> Dict[str, object]:
    preds_path = artifacts.paths["preds"]
    questions_path = artifacts.paths["questions"]
    references_path = artifacts.paths["references"]
    missing = [
        path
        for path in [preds_path, questions_path, references_path]
        if path is None or not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing baseline input files: " + ", ".join(missing))

    questions = load_question_json(questions_path)
    references = load_ref_json(references_path)
    preds = load_pred_json(preds_path)
    contexts = ptrue._load_contexts(questions) if ptrue._uses_context() else None
    if not (len(questions) == len(references) == len(preds) == len(artifacts.labels)):
        raise ValueError(
            "Baseline artifact size mismatch: "
            f"questions={len(questions)}, references={len(references)}, "
            f"preds={len(preds)}, labels={len(artifacts.labels)}"
        )

    ambig_path = os.path.join(artifacts.split_dir, "ambig_label.npy")
    ood_path = os.path.join(artifacts.split_dir, "ood_label.npy")
    uncertainty_labels_by_kind = {
        "aleatoric": np.asarray(np.load(ambig_path), dtype=np.int64) if os.path.exists(ambig_path) else None,
        "epistemic": np.asarray(np.load(ood_path), dtype=np.int64) if os.path.exists(ood_path) else None,
    }

    return {
        "se_resp": artifacts.responses,
        "questions": questions,
        "references": references,
        "preds": preds,
        "contexts": contexts,
        "y": artifacts.y,
        "semantic_ids": np.asarray(artifacts.semantic_ids, dtype=object),
        "hallucination_labels": artifacts.labels,
        "uncertainty_label_kind": (
            "ambig" if artifacts.label_source == "ambig_label.npy"
            else "ood" if artifacts.label_source == "ood_label.npy"
            else "fallback"
        ),
        "uncertainty_labels_by_kind": uncertainty_labels_by_kind,
        "split": artifacts.split,
        "load_dir": artifacts.load_dir,
        "paths": dict(artifacts.paths),
    }


def _load_linear_probe_split(args, split: str, dataset_name: Optional[str] = None) -> Dict[str, object]:
    model_id = args.model_name.split("/")[-1]
    dataset_name = str(dataset_name or args.dataset_name)
    load_dir = _artifact_load_dir(model_id, dataset_name, args.n_samples, args.judge_type)
    split_dir = os.path.join(load_dir, split)
    X_path = os.path.join(split_dir, "X.npz")
    y_path = os.path.join(split_dir, "y.npy")
    semantic_ids_path = os.path.join(split_dir, f"semantic_ids_{args.n_resp}.npy")
    preds_path = os.path.join(split_dir, "preds.jsonl")

    missing = [path for path in [X_path, y_path, semantic_ids_path] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    X = load_X_npz(X_path)
    y = np.asarray(np.load(y_path), dtype=np.int64).reshape(-1)
    hallucination_labels, label_source, label_path = _resolve_dataset_labels(split_dir, args.dataset_name, y)
    semantic_ids = np.load(semantic_ids_path, allow_pickle=True)
    preds = load_pred_json(preds_path) if os.path.exists(preds_path) else None
    return {
        "X": X,
        "y": y,
        "semantic_ids": semantic_ids,
        "hallucination_labels": hallucination_labels,
        "preds": preds,
        "paths": {
            "X": X_path,
            "y": y_path,
            "semantic_ids": semantic_ids_path,
            "uncertainty_labels": label_path,
        },
        "split": split,
        "load_dir": load_dir,
        "label_source": label_source,
    }


def _validate_discounted_bundle_lengths(bundle: DiscountedSplitBundle) -> None:
    lengths = {
        "responses": len(bundle.responses),
        "semantic_ids": len(bundle.semantic_ids),
        "sequence_prob": int(np.asarray(bundle.response_sequence_probs, dtype=np.float32).shape[0]),
        "norm_sequence_prob": int(np.asarray(bundle.response_norm_sequence_probs, dtype=np.float32).shape[0]),
        "labels": int(np.asarray(bundle.labels, dtype=np.int64).reshape(-1).shape[0]),
        "y": int(np.asarray(bundle.y, dtype=np.int64).reshape(-1).shape[0]),
    }
    if bundle.response_slt_all is not None:
        lengths["response_slt_all"] = len(bundle.response_slt_all)
    if bundle.question_probe_features is not None:
        question_probe_arr = np.asarray(bundle.question_probe_features)
        lengths["question_probe_features"] = int(question_probe_arr.shape[0]) if question_probe_arr.ndim > 0 else 0
    if bundle.disambig_question_semantic_ids is not None:
        lengths["disambig_question_semantic_ids"] = len(bundle.disambig_question_semantic_ids)
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Discounted split bundle size mismatch for dataset={bundle.dataset_name}, split={bundle.split}: {lengths}"
        )
    _validate_response_sequence_probs_artifact(
        bundle.response_sequence_probs,
        bundle.responses,
        bundle.semantic_ids,
        f"{bundle.dataset_name}:{bundle.split}",
    )
    _validate_response_sequence_probs_artifact(
        bundle.response_norm_sequence_probs,
        bundle.responses,
        bundle.semantic_ids,
        f"{bundle.dataset_name}:{bundle.split}:norm_seq_prob",
    )


def _load_discounted_bundle_probe_artifacts(
    args,
    dataset_name: str,
    split: str,
) -> Tuple[Optional[List[Dict[int, np.ndarray]]], Optional[str], Optional[np.ndarray], Optional[str]]:
    probe_feature = _resolve_probe_feature(args)
    if probe_feature == "slt":
        response_slt_all, response_slt_lastk_path = _load_response_slt_lastk_split(
            _artifact_load_dir(args.model_name.split("/")[-1], dataset_name, args.n_samples, args.judge_type),
            split,
            int(args.n_resp),
        )
        return response_slt_all, response_slt_lastk_path, None, None

    last_k = int(getattr(args, "hallucination_probe_last_k", 5))
    probe_data = _load_question_probe_split(
        args,
        split,
        last_k,
        probe_feature,
        dataset_name=dataset_name,
    )
    return None, None, np.asarray(probe_data["X_flat"], dtype=np.float32), probe_data["paths"]["X"]


def _load_discounted_split_bundle(
    args,
    dataset_name: str,
    split: str,
) -> DiscountedSplitBundle:
    artifacts = _load_split_artifacts_for(args, dataset_name, split)
    response_slt_all, response_slt_lastk_path, question_probe_features, question_probe_feature_path = (
        _load_discounted_bundle_probe_artifacts(
            args,
            dataset_name,
            split,
        )
    )
    bundle = DiscountedSplitBundle(
        dataset_name=str(dataset_name),
        split=str(split),
        responses=artifacts.responses,
        semantic_ids=artifacts.semantic_ids,
        response_sequence_probs=np.asarray(artifacts.response_sequence_probs, dtype=np.float32).astype(
            np.float32, copy=False
        ),
        response_norm_sequence_probs=np.asarray(artifacts.response_norm_sequence_probs, dtype=np.float32).astype(
            np.float32, copy=False
        ),
        labels=np.asarray(artifacts.labels, dtype=np.int64),
        y=np.asarray(artifacts.y, dtype=np.int64),
        label_source=str(artifacts.label_source),
        response_slt_all=response_slt_all,
        response_slt_lastk_path=response_slt_lastk_path,
        question_probe_features=question_probe_features,
        question_probe_feature_path=question_probe_feature_path,
        disambig_question_semantic_ids=artifacts.disambig_question_semantic_ids,
        load_dir=artifacts.load_dir,
        split_dir=artifacts.split_dir,
        metadata={
            "base_split": str(split),
            "artifact_dataset_name": str(dataset_name),
        },
    )
    _validate_discounted_bundle_lengths(bundle)
    return bundle


def _build_discounted_bundle_from_artifacts(
    args,
    artifacts: SplitArtifacts,
) -> DiscountedSplitBundle:
    response_slt_all, response_slt_lastk_path, question_probe_features, question_probe_feature_path = (
        _load_discounted_bundle_probe_artifacts(
            args,
            str(args.dataset_name),
            artifacts.split,
        )
    )
    bundle = DiscountedSplitBundle(
        dataset_name=str(args.dataset_name),
        split=str(artifacts.split),
        responses=artifacts.responses,
        semantic_ids=artifacts.semantic_ids,
        response_sequence_probs=np.asarray(artifacts.response_sequence_probs, dtype=np.float32).astype(
            np.float32, copy=False
        ),
        response_norm_sequence_probs=np.asarray(artifacts.response_norm_sequence_probs, dtype=np.float32).astype(
            np.float32, copy=False
        ),
        labels=np.asarray(artifacts.labels, dtype=np.int64),
        y=np.asarray(artifacts.y, dtype=np.int64),
        label_source=str(artifacts.label_source),
        response_slt_all=response_slt_all,
        response_slt_lastk_path=response_slt_lastk_path,
        question_probe_features=question_probe_features,
        question_probe_feature_path=question_probe_feature_path,
        disambig_question_semantic_ids=artifacts.disambig_question_semantic_ids,
        load_dir=artifacts.load_dir,
        split_dir=artifacts.split_dir,
        metadata={
            "base_split": str(artifacts.split),
            "artifact_dataset_name": str(args.dataset_name),
        },
    )
    _validate_discounted_bundle_lengths(bundle)
    return bundle


def _build_pseudo_ood_selection_bundle(args) -> DiscountedSplitBundle:
    dataset_name = str(args.dataset_name)
    pseudo_dataset_name = str(args.pseudo_OOD_dataset_name)
    base_bundle = _load_discounted_split_bundle(args, dataset_name, "valid")
    pseudo_test_bundle = _load_discounted_split_bundle(args, pseudo_dataset_name, "test")

    if str(base_bundle.label_source) != "ood_label.npy":
        raise ValueError(
            f"Expected current validation bundle for {dataset_name} to use ood_label.npy, got {base_bundle.label_source!r}."
        )
    if str(pseudo_test_bundle.label_source) != "ood_label.npy":
        raise ValueError(
            f"Expected pseudo-OOD test bundle for {pseudo_dataset_name} to use ood_label.npy, "
            f"got {pseudo_test_bundle.label_source!r}."
        )

    pseudo_ood_idx = np.flatnonzero(np.asarray(pseudo_test_bundle.labels, dtype=np.int64) == 1)
    if pseudo_ood_idx.size == 0:
        raise ValueError(f"Pseudo-OOD source {pseudo_dataset_name} test split contains no OOD rows.")

    base_count = len(base_bundle.responses)
    replace_count = int(pseudo_ood_idx.size)
    if base_count != 2 * replace_count:
        raise ValueError(
            "Pseudo-OOD validation replacement requires the current valid split to be exactly twice "
            f"the pseudo-OOD count, got valid_count={base_count}, pseudo_ood_count={replace_count}."
        )

    rng = np.random.default_rng(42)
    replace_idx = np.sort(rng.choice(base_count, size=replace_count, replace=False).astype(np.int64))

    mixed_responses = list(base_bundle.responses)
    mixed_semantic_ids = list(base_bundle.semantic_ids)
    mixed_disambig_question_semantic_ids = (
        None
        if base_bundle.disambig_question_semantic_ids is None
        else list(base_bundle.disambig_question_semantic_ids)
    )
    mixed_response_slt_all = None if base_bundle.response_slt_all is None else list(base_bundle.response_slt_all)
    mixed_question_probe_features = (
        None
        if base_bundle.question_probe_features is None
        else np.asarray(base_bundle.question_probe_features, dtype=np.float32).copy()
    )
    mixed_response_sequence_probs = np.asarray(base_bundle.response_sequence_probs, dtype=np.float32).copy()
    mixed_response_norm_sequence_probs = np.asarray(base_bundle.response_norm_sequence_probs, dtype=np.float32).copy()
    mixed_labels = np.asarray(base_bundle.labels, dtype=np.int64).copy()
    mixed_y = np.asarray(base_bundle.y, dtype=np.int64).copy()

    for dest_idx, src_idx in zip(replace_idx.tolist(), pseudo_ood_idx.tolist()):
        mixed_responses[dest_idx] = pseudo_test_bundle.responses[src_idx]
        mixed_semantic_ids[dest_idx] = pseudo_test_bundle.semantic_ids[src_idx]
        if mixed_disambig_question_semantic_ids is not None:
            if pseudo_test_bundle.disambig_question_semantic_ids is None:
                raise ValueError("Pseudo-OOD selection bundle is missing disambig-question semantic ids.")
            mixed_disambig_question_semantic_ids[dest_idx] = pseudo_test_bundle.disambig_question_semantic_ids[src_idx]
        if mixed_response_slt_all is not None:
            if pseudo_test_bundle.response_slt_all is None:
                raise ValueError("Pseudo-OOD selection bundle is missing response-level SLT artifacts.")
            mixed_response_slt_all[dest_idx] = pseudo_test_bundle.response_slt_all[src_idx]
        if mixed_question_probe_features is not None:
            if pseudo_test_bundle.question_probe_features is None:
                raise ValueError("Pseudo-OOD selection bundle is missing question-level probe features.")
            mixed_question_probe_features[dest_idx] = pseudo_test_bundle.question_probe_features[src_idx]
        mixed_response_sequence_probs[dest_idx] = pseudo_test_bundle.response_sequence_probs[src_idx]
        mixed_response_norm_sequence_probs[dest_idx] = pseudo_test_bundle.response_norm_sequence_probs[src_idx]
        mixed_labels[dest_idx] = int(pseudo_test_bundle.labels[src_idx])
        mixed_y[dest_idx] = int(pseudo_test_bundle.y[src_idx])

    selection_bundle = DiscountedSplitBundle(
        dataset_name=str(dataset_name),
        split="valid_pseudo_ood",
        responses=mixed_responses,
        semantic_ids=mixed_semantic_ids,
        response_sequence_probs=mixed_response_sequence_probs.astype(np.float32, copy=False),
        response_norm_sequence_probs=mixed_response_norm_sequence_probs.astype(np.float32, copy=False),
        labels=mixed_labels,
        y=mixed_y,
        label_source=str(base_bundle.label_source),
        response_slt_all=mixed_response_slt_all,
        response_slt_lastk_path=base_bundle.response_slt_lastk_path,
        question_probe_features=(
            None
            if mixed_question_probe_features is None
            else mixed_question_probe_features.astype(np.float32, copy=False)
        ),
        question_probe_feature_path=base_bundle.question_probe_feature_path,
        disambig_question_semantic_ids=mixed_disambig_question_semantic_ids,
        load_dir=base_bundle.load_dir,
        split_dir=base_bundle.split_dir,
        metadata={
            "base_split": "valid",
            "pseudo_ood_dataset_name": str(pseudo_dataset_name),
            "pseudo_ood_source_split": "test",
            "pseudo_ood_replaced_count": int(replace_count),
            "pseudo_ood_positive_count": int(np.sum(mixed_labels == 1)),
            "replacement_indices": replace_idx.astype(int).tolist(),
        },
    )
    _validate_discounted_bundle_lengths(selection_bundle)
    if int(np.sum(selection_bundle.labels == 1)) != replace_count:
        raise ValueError(
            "Pseudo-OOD validation replacement must yield half positive labels, "
            f"got positive_count={int(np.sum(selection_bundle.labels == 1))}, expected={replace_count}."
        )
    return selection_bundle


def _evaluate_artifact_baselines(
    artifacts: SplitArtifacts,
    gt_evidence_all: List[Dict[int, float]],
    metric_eval_mask: np.ndarray,
    ood_refusal_eval: Dict[str, object],
) -> Dict[str, object]:
    labels = np.asarray(artifacts.labels, dtype=np.int64)
    eval_labels = labels[metric_eval_mask]
    cluster_cnt_se_scores = np.asarray(get_estimated_se(gt_evidence_all), dtype=np.float32)
    eval_se = np.asarray(artifacts.se_scores, dtype=np.float32)[metric_eval_mask]
    eval_cluster_cnt_se = cluster_cnt_se_scores[metric_eval_mask]
    auroc_se, aupr_se = _safe_binary_metrics(eval_labels, eval_se, artifacts.split, "Baseline se")
    auroc_cnt, aupr_cnt = _safe_binary_metrics(
        eval_labels,
        eval_cluster_cnt_se,
        artifacts.split,
        "Baseline cluster_cnt_se",
    )
    auroc_no_clear_err_se, aupr_no_clear_err_se = _compute_no_clear_err_auroc_aupr(
        labels=labels,
        y=artifacts.y,
        scores=np.asarray(artifacts.se_scores, dtype=np.float32),
        metric_eval_mask=metric_eval_mask,
        split=artifacts.split,
        method_name="EUD baseline se",
    )
    auroc_no_clear_large_cluster_se, aupr_no_clear_large_cluster_se = _compute_no_clear_large_n_cluster_auroc_aupr(
        labels=labels,
        semantic_ids=artifacts.semantic_ids,
        scores=np.asarray(artifacts.se_scores, dtype=np.float32),
        metric_eval_mask=metric_eval_mask,
        split=artifacts.split,
        method_name="EUD baseline se",
    )
    return {
        "AUROC": {"se": auroc_se, "cluster_cnt_se": auroc_cnt},
        "AUPR": {"se": aupr_se, "cluster_cnt_se": aupr_cnt},
        "AUROC_no_clear_err": {"se": auroc_no_clear_err_se},
        "AUPR_no_clear_err": {"se": aupr_no_clear_err_se},
        "AUROC_no_clear_large_n_cluster": {"se": auroc_no_clear_large_cluster_se},
        "AUPR_no_clear_large_n_cluster": {"se": aupr_no_clear_large_cluster_se},
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "ood_refusal_eval": ood_refusal_eval,
        "cluster_cnt_se_scores": cluster_cnt_se_scores,
    }


def _evaluate_eud(
    artifacts: SplitArtifacts,
    gt_evidence_all: List[Dict[int, float]],
    metric_eval_mask: np.ndarray,
    ood_refusal_eval: Dict[str, object],
) -> Dict[str, object]:
    vacuity_scores = np.asarray([vacuity(item) for item in gt_evidence_all], dtype=np.float32)
    dissonance_scores = np.asarray([dissonance(item) for item in gt_evidence_all], dtype=np.float32)
    labels = np.asarray(artifacts.labels, dtype=np.int64)
    eval_labels = labels[metric_eval_mask]
    eval_vacuity = vacuity_scores[metric_eval_mask]
    eval_dissonance = dissonance_scores[metric_eval_mask]
    auroc_vac, aupr_vac = _safe_binary_metrics(eval_labels, eval_vacuity, artifacts.split, "EUD vacuity")
    auroc_dis, aupr_dis = _safe_binary_metrics(eval_labels, eval_dissonance, artifacts.split, "EUD dissonance")
    return {
        "semantic_ids": artifacts.semantic_ids,
        "vacuity": vacuity_scores,
        "dissonance": dissonance_scores,
        "uncertainty_labels": labels,
        "y": artifacts.y,
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "ood_refusal_eval": ood_refusal_eval,
        "AUROC": {"vacuity": auroc_vac, "dissonance": auroc_dis},
        "AUPR": {"vacuity": aupr_vac, "dissonance": aupr_dis},
    }


def _evaluate_disambig_question_eud(
    artifacts: SplitArtifacts,
    metric_eval_mask: np.ndarray,
    ood_refusal_eval: Dict[str, object],
) -> Dict[str, object]:
    n_questions = len(artifacts.responses)
    empty_result = {
        "available": False,
        "skip_reason": artifacts.disambig_question_metrics_skip_reason,
        "semantic_ids": artifacts.disambig_question_semantic_ids,
        "uncertainty_labels": np.asarray(artifacts.labels, dtype=np.int64),
        "y": artifacts.y,
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "ood_refusal_eval": ood_refusal_eval,
        "count": _empty_disambig_question_variant_result(n_questions),
        "seq_prob": _empty_disambig_question_variant_result(n_questions),
        "norm_seq_prob": _empty_disambig_question_variant_result(n_questions),
        "theils_u_seq_prob": _empty_disambig_question_variant_result(n_questions),
        "theils_u_norm_seq_prob": _empty_disambig_question_variant_result(n_questions),
        "theils_u_resp_norm_seq_prob_dissonance": _empty_scalar_score_result(n_questions),
    }
    if not bool(artifacts.disambig_question_metrics_available):
        return empty_result

    if artifacts.disambig_question_semantic_ids is None:
        empty_result["skip_reason"] = "Disambig-question semantic ids are unavailable."
        return empty_result
    if artifacts.disambig_question_sequence_probs is None:
        empty_result["skip_reason"] = "Disambig-question sequence probabilities are unavailable."
        return empty_result
    if artifacts.disambig_question_norm_sequence_probs is None:
        empty_result["skip_reason"] = "Disambig-question normalized sequence probabilities are unavailable."
        return empty_result

    count_evidence = _build_gt_evidence_dicts(artifacts.disambig_question_semantic_ids)
    seq_prob_evidence = _build_weighted_evidence_dicts(
        artifacts.disambig_question_semantic_ids,
        artifacts.disambig_question_sequence_probs,
        source=f"{artifacts.split_dir}:disambig_question_sequence_prob",
    )
    norm_seq_prob_evidence = _build_weighted_evidence_dicts(
        artifacts.disambig_question_semantic_ids,
        artifacts.disambig_question_norm_sequence_probs,
        source=f"{artifacts.split_dir}:disambig_question_norm_sequence_prob",
    )
    response_norm_seq_prob_evidence = _build_weighted_evidence_dicts(
        artifacts.semantic_ids,
        artifacts.response_norm_sequence_probs,
        source=f"{artifacts.split_dir}:response_norm_sequence_prob",
    )
    theils_u_scores = _compute_disambig_question_theils_u(
        artifacts.semantic_ids,
        artifacts.disambig_question_semantic_ids,
        source=f"{artifacts.split_dir}:disambig_question_theils_u",
    )
    theils_u_seq_prob_evidence = [
        _scale_evidence_dict(evidence_dict, theils_u_score)
        for evidence_dict, theils_u_score in zip(seq_prob_evidence, theils_u_scores.tolist())
    ]
    theils_u_norm_seq_prob_evidence = [
        _scale_evidence_dict(evidence_dict, theils_u_score)
        for evidence_dict, theils_u_score in zip(norm_seq_prob_evidence, theils_u_scores.tolist())
    ]
    theils_u_resp_norm_seq_prob_evidence = [
        _scale_evidence_dict(evidence_dict, theils_u_score)
        for evidence_dict, theils_u_score in zip(response_norm_seq_prob_evidence, theils_u_scores.tolist())
    ]
    theils_u_resp_norm_seq_prob_dissonance = np.asarray(
        [float(dissonance(item)) if item else 0.0 for item in theils_u_resp_norm_seq_prob_evidence],
        dtype=np.float32,
    )
    labels = np.asarray(artifacts.labels, dtype=np.int64)
    return {
        "available": True,
        "skip_reason": None,
        "semantic_ids": artifacts.disambig_question_semantic_ids,
        "theils_u": theils_u_scores,
        "uncertainty_labels": labels,
        "y": artifacts.y,
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "ood_refusal_eval": ood_refusal_eval,
        "count": _evaluate_evidence_dict_metrics(
            split=artifacts.split,
            labels=labels,
            metric_eval_mask=metric_eval_mask,
            evidence_all=count_evidence,
            metric_name="EUD disambig-question",
        ),
        "seq_prob": _evaluate_evidence_dict_metrics(
            split=artifacts.split,
            labels=labels,
            metric_eval_mask=metric_eval_mask,
            evidence_all=seq_prob_evidence,
            metric_name="EUD disambig-question seq_prob",
        ),
        "norm_seq_prob": _evaluate_evidence_dict_metrics(
            split=artifacts.split,
            labels=labels,
            metric_eval_mask=metric_eval_mask,
            evidence_all=norm_seq_prob_evidence,
            metric_name="EUD disambig-question norm_seq_prob",
        ),
        "theils_u_seq_prob": _evaluate_evidence_dict_metrics(
            split=artifacts.split,
            labels=labels,
            metric_eval_mask=metric_eval_mask,
            evidence_all=theils_u_seq_prob_evidence,
            metric_name="EUD disambig-question Theil's-U seq_prob",
        ),
        "theils_u_norm_seq_prob": _evaluate_evidence_dict_metrics(
            split=artifacts.split,
            labels=labels,
            metric_eval_mask=metric_eval_mask,
            evidence_all=theils_u_norm_seq_prob_evidence,
            metric_name="EUD disambig-question Theil's-U norm_seq_prob",
        ),
        "theils_u_resp_norm_seq_prob_dissonance": _evaluate_scalar_score_metrics(
            split=artifacts.split,
            labels=labels,
            metric_eval_mask=metric_eval_mask,
            scores=theils_u_resp_norm_seq_prob_dissonance,
            metric_name="EUD response Theil's-U norm_seq_prob dissonance",
        ),
    }


def _evaluate_hallucination_slt_probe(
    args,
    artifacts: SplitArtifacts,
    metric_eval_mask: np.ndarray,
) -> Dict[str, object]:
    last_k = int(getattr(args, "hallucination_probe_last_k", 5))
    probe_type = str(getattr(args, "probe_type", "lr")).strip().lower()
    probe_feature = _resolve_probe_feature(args)
    probe_target = _normalize_probe_target(getattr(args, "probe_target", "hallucination"))
    probe_display_name = f"{_probe_display_name(probe_target, capitalize=True)} Probe"
    train_data = _load_question_probe_split(args, "train", last_k, probe_feature)
    eval_data = _load_question_probe_split(args, artifacts.split, last_k, probe_feature)

    X_train_flat = np.asarray(train_data["X_flat"], dtype=np.float32)
    X_eval_flat = np.asarray(eval_data["X_flat"], dtype=np.float32)

    if X_train_flat.shape[1] != X_eval_flat.shape[1]:
        raise ValueError(
            f"Question-level {probe_feature} probe feature dim mismatch between train and eval: "
            f"train={X_train_flat.shape[1]}, eval={X_eval_flat.shape[1]}."
        )

    y_train_probe = np.asarray(train_data["probe_labels"], dtype=np.int64)
    y_eval_probe = np.asarray(eval_data["probe_labels"], dtype=np.int64)
    if X_train_flat.shape[0] != y_train_probe.shape[0]:
        raise ValueError(
            f"Train question-level {probe_feature} probe input mismatch: "
            f"features={X_train_flat.shape[0]}, labels={y_train_probe.shape[0]}."
        )
    if X_eval_flat.shape[0] != y_eval_probe.shape[0]:
        raise ValueError(
            f"Eval question-level {probe_feature} probe input mismatch: "
            f"features={X_eval_flat.shape[0]}, labels={y_eval_probe.shape[0]}."
        )

    question_metric_mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    eval_question_ids = np.asarray(eval_data["question_ids"], dtype=np.int64).reshape(-1)
    if question_metric_mask_arr.shape[0] != int(eval_data["question_count"]):
        raise ValueError(
            f"Eval question-mask length mismatch for {probe_display_name.lower()}: "
            f"mask={question_metric_mask_arr.shape[0]}, question_count={int(eval_data['question_count'])}."
        )
    metric_mask_arr = question_metric_mask_arr[eval_question_ids] if eval_question_ids.size > 0 else np.zeros(0, dtype=bool)
    calibration_enabled = bool(getattr(args, "calibrate", False))
    calibration_available = False
    temperature = 1.0
    temperature_fit = None
    valid_calibrated_probe_ece = float("nan")
    valid_probe_ece = float("nan")
    valid_calibration = None
    valid_probe_calibration = None

    if np.unique(y_train_probe).size < 2:
        clf = None
        eval_logits = np.full(X_eval_flat.shape[0], np.nan, dtype=float)
        raw_scores = np.full(X_eval_flat.shape[0], np.nan, dtype=float)
        scores = np.full(X_eval_flat.shape[0], np.nan, dtype=float)
        probe_converged = False
        probe_n_iter = 0
        print(
            f"[{probe_display_name}] Skipping question-level {probe_feature} probe fit because train probe labels "
            f"contain one class: {np.unique(y_train_probe).tolist()}."
        )
        auroc = float("nan")
        aupr = float("nan")
    else:
        clf = build_eud_probe(
            probe_type=probe_type,
            input_dim=int(X_train_flat.shape[1]),
            random_state=42,
        )
        clf.fit(X_train_flat, y_train_probe)
        eval_logits = clf.predict_logits(X_eval_flat)
        raw_scores = positive_probability_from_logits(eval_logits)
        calibration_payload = _fit_validation_temperature(
            args,
            clf,
            last_k,
            probe_feature=probe_feature,
            fit_temperature=calibration_enabled,
        )
        calibration_available = bool(calibration_payload["available"])
        temperature = float(calibration_payload["temperature"])
        temperature_fit = calibration_payload["fit"]
        valid_probe_ece = float(calibration_payload["valid_probe_ece"])
        valid_calibrated_probe_ece = float(calibration_payload["valid_calibrated_probe_ece"])
        valid_probe_calibration = calibration_payload["valid_probe_calibration"]
        valid_calibration = calibration_payload["valid_calibration"]
        scores = positive_probability_from_logits(eval_logits, temperature=temperature)
        probe_converged = bool(getattr(clf, "converged_", False))
        probe_n_iter = int(getattr(clf, "n_iter_", 0))
        eval_labels = y_eval_probe[metric_mask_arr]
        eval_scores = scores[metric_mask_arr]
        auroc, aupr = _safe_binary_metrics(
            eval_labels,
            eval_scores,
            artifacts.split,
            probe_display_name,
        )

    probe_ece, calibration = compute_binary_ece(
        labels=y_eval_probe,
        scores=scores,
        metric_eval_mask=metric_mask_arr,
        split=artifacts.split,
        metric_name=probe_display_name,
    )

    return {
        "clf": clf,
        "scores": scores.astype(float, copy=False),
        "raw_scores": raw_scores.astype(float, copy=False),
        "logits": eval_logits.astype(float, copy=False),
        "train_labels": y_train_probe,
        "eval_labels": y_eval_probe,
        "metric_eval_mask": metric_mask_arr,
        "AUROC": auroc,
        "AUPR": aupr,
        "probe_ece": float(probe_ece),
        "calibration": calibration,
        "calibration_enabled": bool(calibration_enabled),
        "calibration_available": bool(calibration_available),
        "temperature": float(temperature),
        "temperature_fit": temperature_fit,
        "valid_probe_ece": float(valid_probe_ece),
        "valid_probe_calibration": valid_probe_calibration,
        "valid_calibrated_probe_ece": float(valid_calibrated_probe_ece),
        "valid_calibration": valid_calibration,
        "last_k": last_k,
        "feature_dim": int(train_data["feature_dim"]),
        "train_count": int(y_train_probe.shape[0]),
        "eval_count": int(y_eval_probe.shape[0]),
        "probe_target": str(probe_target),
        "probe_label_source": str(train_data["probe_label_source"]),
        "probe_feature": probe_feature,
        "probe_type": probe_type,
        "probe_converged": bool(probe_converged),
        "probe_n_iter": int(probe_n_iter),
        "train_paths": dict(train_data["paths"]),
        "eval_paths": dict(eval_data["paths"]),
    }


def _discount_selection_spec(label_source: str) -> Tuple[str, str]:
    if str(label_source) == "ambig_label.npy":
        return "ambiguity", "dissonance"
    return "ood", "vacuity"


def _build_discounted_evidence_dict(
    ids_row: np.ndarray,
    reliability_q: np.ndarray,
) -> Dict[int, float]:
    evidence_dict: Dict[int, float] = {}
    for cid in np.unique(ids_row):
        resp_indices = np.where(ids_row == int(cid))[0]
        evidence_dict[int(cid)] = float(np.sum(reliability_q[resp_indices]))
    return evidence_dict


def _build_discounted_ignorance_dict(
    ids_row: np.ndarray,
    reliability_q: np.ndarray,
) -> Dict[int, float]:
    ignorance_dict: Dict[int, float] = {}
    for cid in np.unique(ids_row):
        resp_indices = np.where(ids_row == int(cid))[0]
        ignorance_dict[int(cid)] = float(np.sum(1.0 - reliability_q[resp_indices]))
    return ignorance_dict


def _vacuity_with_ignorance(
    evidence_dict: Dict[int, float],
    ignorance_dict: Dict[int, float],
    total_responses: float,
) -> float:
    # Discounted vacuity now follows the standard evidence definition.
    _ = ignorance_dict, total_responses
    return float(vacuity(evidence_dict)) if evidence_dict else 0.0


def _discounted_dissonance(
    evidence_dict: Dict[int, float],
    total_responses: float,
) -> float:
    _ = total_responses
    return float(dissonance(evidence_dict)) if evidence_dict else 0.0


def _discount_reliability_values(
    mode: str,
    penalty_probs_q: np.ndarray,
    zeta: float,
) -> np.ndarray:
    penalty_probs_arr = np.asarray(penalty_probs_q, dtype=np.float32).reshape(-1)
    if mode == "power":
        return np.power(1.0 - penalty_probs_arr, float(zeta)).astype(np.float32, copy=False)
    if mode == "exp":
        return np.exp(-float(zeta) * penalty_probs_arr).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported discounted branch mode: {mode!r}.")


def _build_mean_weighted_evidence_dict(
    ids_row: np.ndarray,
    penalty_probs_q: np.ndarray,
    zeta: float,
    sequence_probs_q: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    evidence_dict: Dict[int, float] = {}
    for cid in np.unique(ids_row):
        resp_indices = np.where(ids_row == int(cid))[0]
        cluster_evidence = (
            float(np.sum(sequence_probs_q[resp_indices], dtype=np.float64))
            if sequence_probs_q is not None
            else float(resp_indices.size)
        )
        cluster_penalty_prob = float(np.mean(penalty_probs_q[resp_indices]))
        evidence_dict[int(cid)] = float(np.exp(-float(zeta) * cluster_penalty_prob) * cluster_evidence)
    return evidence_dict


def _build_resp_wise_weighted_evidence_dict(
    ids_row: np.ndarray,
    penalty_probs_q: np.ndarray,
    zeta: float,
    sequence_probs_q: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    evidence_dict: Dict[int, float] = {}
    for cid in np.unique(ids_row):
        resp_indices = np.where(ids_row == int(cid))[0]
        response_penalty_prob = penalty_probs_q[resp_indices]
        response_weights = np.exp(-float(zeta) * response_penalty_prob)
        if sequence_probs_q is not None:
            response_weights = response_weights * sequence_probs_q[resp_indices]
        evidence_dict[int(cid)] = float(np.sum(response_weights, dtype=np.float64))
    return evidence_dict


def _select_best_grid_result(
    grid_results: Sequence[Dict[str, object]],
    selection_family: str,
) -> Dict[str, object]:
    if not grid_results:
        raise ValueError("zeta grid search produced no candidate results.")

    best_idx: Optional[int] = None
    best_score = -np.inf
    for idx, row in enumerate(grid_results):
        score = float(row["AUROC"][selection_family])
        if np.isfinite(score) and (best_idx is None or score > best_score):
            best_idx = idx
            best_score = score

    if best_idx is None:
        for idx, row in enumerate(grid_results):
            if float(row["zeta"]) == 1.0:
                return row
        return grid_results[0]
    return grid_results[int(best_idx)]


def _empty_discounted_branch_result(
    n_questions: int,
    zeta_grid: Sequence[float],
) -> Dict[str, object]:
    nan_scores = np.full(int(n_questions), np.nan, dtype=np.float32)
    return {
        "pred_evidence_discounted": None,
        "pred_ignorance_discounted": None,
        "response_reliability_scores": None,
        "total_ignorance": nan_scores.copy(),
        "dist_vacuity": nan_scores.copy(),
        "dist_dissonance": nan_scores.copy(),
        "AUROC": {"vacuity": float("nan"), "dissonance": float("nan")},
        "AUPR": {"vacuity": float("nan"), "dissonance": float("nan")},
        "grid_results": [],
        "zeta_grid": [float(z) for z in zeta_grid],
        "best_zeta": 1.0,
        "best_selection_auroc": float("nan"),
        "best_selection_aupr": float("nan"),
    }


def _empty_weighted_branch_result(
    n_questions: int,
    zeta_grid: Sequence[float],
) -> Dict[str, object]:
    nan_scores = np.full(int(n_questions), np.nan, dtype=np.float32)
    return {
        "pred_evidence": None,
        "vacuity": nan_scores.copy(),
        "dissonance": nan_scores.copy(),
        "AUROC": {"vacuity": float("nan"), "dissonance": float("nan")},
        "AUPR": {"vacuity": float("nan"), "dissonance": float("nan")},
        "grid_results": [],
        "zeta_grid": [float(z) for z in zeta_grid],
        "best_zeta": 1.0,
        "best_selection_auroc": float("nan"),
        "best_selection_aupr": float("nan"),
    }


def _empty_exp_discounted_theils_u_resp_result(
    n_questions: int,
    zeta_grid: Sequence[float],
    *,
    selection_split: str,
    selection_dataset_name: str,
    selection_label_source: str,
    skip_reason: Optional[str] = None,
) -> Dict[str, object]:
    result = _empty_scalar_score_result(n_questions)
    result.update(
        {
            "grid_results": [],
            "selection_grid_results": [],
            "eval_grid_results": [],
            "zeta_grid": [float(z) for z in zeta_grid],
            "best_zeta": 1.0,
            "selected_zeta": 1.0,
            "selection_auroc": float("nan"),
            "selection_aupr": float("nan"),
            "best_selection_auroc": float("nan"),
            "best_selection_aupr": float("nan"),
            "selection_split": str(selection_split),
            "selection_dataset_name": str(selection_dataset_name),
            "selection_label_source": str(selection_label_source),
            "eval_oracle_best_zeta": 1.0,
            "eval_oracle_best_auroc": float("nan"),
            "eval_oracle_best_aupr": float("nan"),
            "selected_matches_eval_best": True,
            "skip_reason": "" if skip_reason is None else str(skip_reason),
        }
    )
    return result


def _select_best_scalar_grid_result(grid_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not grid_results:
        raise ValueError("Scalar zeta grid search produced no candidate results.")

    best_idx: Optional[int] = None
    best_score = -np.inf
    for idx, row in enumerate(grid_results):
        score = float(row["AUROC"])
        if np.isfinite(score) and (best_idx is None or score > best_score):
            best_idx = idx
            best_score = score

    if best_idx is None:
        for row in grid_results:
            if float(row["zeta"]) == 1.0:
                return row
        return grid_results[0]
    return grid_results[int(best_idx)]


def _summarize_scalar_grid_results(grid_results: Sequence[Dict[str, object]]) -> List[Dict[str, float]]:
    return [
        {
            "zeta": float(row["zeta"]),
            "AUROC": float(row["AUROC"]),
            "AUPR": float(row["AUPR"]),
        }
        for row in grid_results
    ]


def _find_scalar_grid_result_by_zeta(
    grid_results: Sequence[Dict[str, object]],
    zeta: float,
) -> Dict[str, object]:
    for row in grid_results:
        if _zeta_matches(float(row["zeta"]), float(zeta)):
            return row
    raise ValueError(f"Unable to find scalar grid result for zeta={float(zeta):.6f}.")


def _build_selected_scalar_result(
    *,
    selection_results: Dict[str, object],
    eval_results: Dict[str, object],
    selection_split: str,
    selection_dataset_name: str,
    selection_label_source: str,
) -> Dict[str, object]:
    selected_zeta = float(selection_results["best_zeta"])
    selected_eval_row = _find_scalar_grid_result_by_zeta(eval_results["grid_results"], selected_zeta)
    eval_oracle_best_zeta = float(eval_results["best_zeta"])
    return {
        "scores": np.asarray(selected_eval_row["scores"], dtype=np.float32),
        "AUROC": float(selected_eval_row["AUROC"]),
        "AUPR": float(selected_eval_row["AUPR"]),
        "grid_results": _summarize_scalar_grid_results(eval_results["grid_results"]),
        "selection_grid_results": _summarize_scalar_grid_results(selection_results["grid_results"]),
        "eval_grid_results": _summarize_scalar_grid_results(eval_results["grid_results"]),
        "zeta_grid": [float(z) for z in eval_results["zeta_grid"]],
        "best_zeta": float(selected_zeta),
        "selected_zeta": float(selected_zeta),
        "selection_auroc": float(selection_results["best_selection_auroc"]),
        "selection_aupr": float(selection_results["best_selection_aupr"]),
        "best_selection_auroc": float(selection_results["best_selection_auroc"]),
        "best_selection_aupr": float(selection_results["best_selection_aupr"]),
        "selection_split": str(selection_split),
        "selection_dataset_name": str(selection_dataset_name),
        "selection_label_source": str(selection_label_source),
        "eval_oracle_best_zeta": float(eval_oracle_best_zeta),
        "eval_oracle_best_auroc": float(eval_results["best_selection_auroc"]),
        "eval_oracle_best_aupr": float(eval_results["best_selection_aupr"]),
        "selected_matches_eval_best": bool(_zeta_matches(selected_zeta, eval_oracle_best_zeta)),
        "skip_reason": "",
    }


def _evaluate_exp_discounted_theils_u_resp_norm_seq_prob_dissonance_grid(
    *,
    split_inputs: Dict[str, object],
    zeta_grid: Sequence[float],
) -> Dict[str, object]:
    ids_rows_all = split_inputs["ids_rows_all"]
    penalty_probs_all = split_inputs["penalty_probs_all"]
    norm_sequence_probs_all = split_inputs["norm_sequence_probs_all"]
    disambig_ids_all = split_inputs.get("disambig_question_semantic_ids_all")
    if disambig_ids_all is None:
        raise ValueError("Missing disambig-question semantic ids for exp-discounted Theil's-U response metric.")

    n_questions = len(ids_rows_all)
    theils_u_scores = _compute_disambig_question_theils_u(
        ids_rows_all,
        disambig_ids_all,
        source=f"{split_inputs['metric_split_name']}:exp_discounted_theils_u_resp_norm_seq_prob_dissonance",
    )
    eval_labels = np.asarray(split_inputs["eval_labels"], dtype=np.int64).reshape(-1)
    metric_mask_arr = np.asarray(split_inputs["metric_eval_mask"], dtype=bool).reshape(-1)
    if metric_mask_arr.shape[0] != n_questions:
        raise ValueError(
            "Exp-discounted Theil's-U response metric mask mismatch: "
            f"mask={metric_mask_arr.shape[0]}, questions={n_questions}."
        )

    grid_results: List[Dict[str, object]] = []
    for zeta in zeta_grid:
        scores = np.zeros(n_questions, dtype=np.float32)
        for qid, (ids_row, penalty_probs_q, norm_sequence_probs_q, theils_u_score) in enumerate(
            zip(ids_rows_all, penalty_probs_all, norm_sequence_probs_all, theils_u_scores.tolist())
        ):
            ids = np.asarray(ids_row, dtype=np.int64).reshape(-1)
            penalties = np.asarray(penalty_probs_q, dtype=np.float32).reshape(-1)
            norm_probs = np.asarray(norm_sequence_probs_q, dtype=np.float32).reshape(-1)
            if ids.size == 0:
                scores[qid] = 0.0
                continue
            if not (ids.shape[0] == penalties.shape[0] == norm_probs.shape[0]):
                raise ValueError(
                    "Exp-discounted Theil's-U response metric row mismatch at "
                    f"{split_inputs['metric_split_name']}, qid={qid}: "
                    f"ids={ids.shape[0]}, penalties={penalties.shape[0]}, norm_probs={norm_probs.shape[0]}."
                )
            if not np.all(np.isfinite(penalties)) or not np.all(np.isfinite(norm_probs)):
                raise ValueError(
                    "Non-finite exp-discounted Theil's-U response metric inputs at "
                    f"{split_inputs['metric_split_name']}, qid={qid}."
                )
            response_weights = (
                np.exp(-float(zeta) * penalties).astype(np.float32, copy=False)
                * norm_probs.astype(np.float32, copy=False)
                * float(theils_u_score)
            )
            evidence_dict = _build_discounted_evidence_dict(ids, response_weights)
            scores[qid] = float(dissonance(evidence_dict)) if evidence_dict else 0.0

        auroc, aupr = _safe_binary_metrics(
            eval_labels,
            scores[metric_mask_arr],
            str(split_inputs["metric_split_name"]),
            f"EUD exp-discounted Theil's-U response norm-seq-prob dissonance zeta={float(zeta):.2f}",
        )
        grid_results.append(
            {
                "zeta": float(zeta),
                "scores": scores,
                "AUROC": auroc,
                "AUPR": aupr,
            }
        )

    best_result = _select_best_scalar_grid_result(grid_results)
    return {
        "scores": np.asarray(best_result["scores"], dtype=np.float32),
        "AUROC": float(best_result["AUROC"]),
        "AUPR": float(best_result["AUPR"]),
        "grid_results": grid_results,
        "zeta_grid": [float(z) for z in zeta_grid],
        "best_zeta": float(best_result["zeta"]),
        "best_selection_auroc": float(best_result["AUROC"]),
        "best_selection_aupr": float(best_result["AUPR"]),
    }


def _evaluate_discounted_branch(
    *,
    branch_name: str,
    branch_mode: str,
    split: str,
    selection_family: str,
    zeta_grid: Sequence[float],
    ids_rows_all: Sequence[np.ndarray],
    penalty_probs_all: Sequence[np.ndarray],
    sequence_probs_all: Optional[Sequence[np.ndarray]],
    norm_sequence_probs_all: Optional[Sequence[np.ndarray]],
    eval_labels: np.ndarray,
    metric_mask_arr: np.ndarray,
    use_seq_prob: bool,
    use_norm_seq_prob: bool,
) -> Dict[str, object]:
    n_questions = len(ids_rows_all)
    grid_results: List[Dict[str, object]] = []
    if bool(use_seq_prob) and bool(use_norm_seq_prob):
        raise ValueError(f"{branch_name} cannot enable both raw and normalized sequence probability weighting.")
    for zeta in zeta_grid:
        pred_evidence_discounted: List[Dict[int, float]] = []
        pred_ignorance_discounted: List[Dict[int, float]] = []
        response_reliability_scores: List[List[float]] = []
        total_ignorance = np.zeros(n_questions, dtype=np.float32)
        dist_vacuity = np.zeros(n_questions, dtype=np.float32)
        dist_dissonance = np.zeros(n_questions, dtype=np.float32)

        for qid, (ids_row, penalty_probs_q) in enumerate(zip(ids_rows_all, penalty_probs_all)):
            if penalty_probs_q.size == 0:
                pred_evidence_discounted.append({})
                pred_ignorance_discounted.append({})
                response_reliability_scores.append([])
                total_ignorance[qid] = 0.0
                dist_vacuity[qid] = 0.0
                dist_dissonance[qid] = 0.0
                continue

            base_reliability_q = _discount_reliability_values(branch_mode, penalty_probs_q, float(zeta)).reshape(-1)
            if base_reliability_q.shape[0] != penalty_probs_q.shape[0]:
                raise ValueError(
                    f"{branch_name} discounted reliability shape mismatch at split={split}, qid={qid}: "
                    f"expected={penalty_probs_q.shape[0]}, got={base_reliability_q.shape[0]}."
                )
            if not np.all(np.isfinite(base_reliability_q)):
                raise ValueError(
                    f"Non-finite {branch_name} discounted reliabilities at split={split}, qid={qid}, "
                    f"zeta={float(zeta):.4f}."
                )
            if use_seq_prob or use_norm_seq_prob:
                sequence_probs_source = sequence_probs_all if use_seq_prob else norm_sequence_probs_all
                seq_prob_label = "sequence_probs_all" if use_seq_prob else "norm_sequence_probs_all"
                if sequence_probs_source is None:
                    raise ValueError(
                        f"{branch_name} requires {seq_prob_label} when sequence-probability weighting is enabled."
                    )
                sequence_probs_q = np.asarray(sequence_probs_source[qid], dtype=np.float32).reshape(-1)
                if sequence_probs_q.shape[0] != penalty_probs_q.shape[0]:
                    raise ValueError(
                        f"{branch_name} seq_prob shape mismatch at split={split}, qid={qid}: "
                        f"expected={penalty_probs_q.shape[0]}, got={sequence_probs_q.shape[0]}."
                    )
                if not np.all(np.isfinite(sequence_probs_q)):
                    raise ValueError(
                        f"Non-finite {branch_name} seq_prob values at split={split}, qid={qid}."
                    )
                reliability_q = (base_reliability_q * sequence_probs_q).astype(np.float32, copy=False)
            else:
                reliability_q = base_reliability_q.astype(np.float32, copy=False)
            if reliability_q.shape[0] != penalty_probs_q.shape[0]:
                raise ValueError(
                    f"{branch_name} discounted reliability shape mismatch at split={split}, qid={qid}: "
                    f"expected={penalty_probs_q.shape[0]}, got={reliability_q.shape[0]}."
                )
            if not np.all(np.isfinite(reliability_q)):
                raise ValueError(
                    f"Non-finite {branch_name} discounted reliabilities at split={split}, qid={qid}, "
                    f"zeta={float(zeta):.4f}."
                )

            evidence_dict = _build_discounted_evidence_dict(ids_row, reliability_q)
            ignorance_dict = _build_discounted_ignorance_dict(ids_row, reliability_q)
            pred_evidence_discounted.append(evidence_dict)
            pred_ignorance_discounted.append(ignorance_dict)
            response_reliability_scores.append(reliability_q.astype(float).tolist())
            total_ignorance[qid] = (
                float(np.sum(list(ignorance_dict.values()), dtype=np.float64)) if ignorance_dict else 0.0
            )
            dist_vacuity[qid] = _vacuity_with_ignorance(
                evidence_dict,
                ignorance_dict,
                float(penalty_probs_q.size),
            )
            dist_dissonance[qid] = (
                _discounted_dissonance(
                    evidence_dict,
                    float(penalty_probs_q.size),
                )
                if evidence_dict
                else 0.0
            )

        auroc_vac, aupr_vac = _safe_binary_metrics(
            eval_labels,
            dist_vacuity[metric_mask_arr],
            split,
            f"EUD {branch_name} discounted vacuity zeta={float(zeta):.2f}",
        )
        auroc_dis, aupr_dis = _safe_binary_metrics(
            eval_labels,
            dist_dissonance[metric_mask_arr],
            split,
            f"EUD {branch_name} discounted dissonance zeta={float(zeta):.2f}",
        )
        grid_results.append(
            {
                "zeta": float(zeta),
                "pred_evidence_discounted": pred_evidence_discounted,
                "pred_ignorance_discounted": pred_ignorance_discounted,
                "response_reliability_scores": response_reliability_scores,
                "total_ignorance": total_ignorance,
                "dist_vacuity": dist_vacuity,
                "dist_dissonance": dist_dissonance,
                "AUROC": {"vacuity": auroc_vac, "dissonance": auroc_dis},
                "AUPR": {"vacuity": aupr_vac, "dissonance": aupr_dis},
            }
        )

    best_result = _select_best_grid_result(grid_results, selection_family)
    return {
        "pred_evidence_discounted": best_result["pred_evidence_discounted"],
        "pred_ignorance_discounted": best_result["pred_ignorance_discounted"],
        "response_reliability_scores": best_result["response_reliability_scores"],
        "total_ignorance": best_result["total_ignorance"],
        "dist_vacuity": best_result["dist_vacuity"],
        "dist_dissonance": best_result["dist_dissonance"],
        "AUROC": best_result["AUROC"],
        "AUPR": best_result["AUPR"],
        "grid_results": grid_results,
        "zeta_grid": [float(z) for z in zeta_grid],
        "best_zeta": float(best_result["zeta"]),
        "best_selection_auroc": float(best_result["AUROC"][selection_family]),
        "best_selection_aupr": float(best_result["AUPR"][selection_family]),
    }


def _evaluate_weighted_branch(
    *,
    branch_name: str,
    split: str,
    selection_family: str,
    zeta_grid: Sequence[float],
    ids_rows_all: Sequence[np.ndarray],
    penalty_probs_all: Sequence[np.ndarray],
    sequence_probs_all: Optional[Sequence[np.ndarray]],
    norm_sequence_probs_all: Optional[Sequence[np.ndarray]],
    eval_labels: np.ndarray,
    metric_mask_arr: np.ndarray,
    evidence_builder_fn: Callable[[np.ndarray, np.ndarray, float, Optional[np.ndarray]], Dict[int, float]],
    use_seq_prob: bool,
    use_norm_seq_prob: bool,
) -> Dict[str, object]:
    n_questions = len(ids_rows_all)
    grid_results: List[Dict[str, object]] = []
    if bool(use_seq_prob) and bool(use_norm_seq_prob):
        raise ValueError(f"{branch_name} cannot enable both raw and normalized sequence probability weighting.")
    for zeta in zeta_grid:
        pred_evidence: List[Dict[int, float]] = []
        vacuity_scores = np.zeros(n_questions, dtype=np.float32)
        dissonance_scores = np.zeros(n_questions, dtype=np.float32)

        for qid, (ids_row, penalty_probs_q) in enumerate(zip(ids_rows_all, penalty_probs_all)):
            if penalty_probs_q.size == 0:
                pred_evidence.append({})
                vacuity_scores[qid] = 0.0
                dissonance_scores[qid] = 0.0
                continue

            sequence_probs_q = None
            if use_seq_prob or use_norm_seq_prob:
                sequence_probs_source = sequence_probs_all if use_seq_prob else norm_sequence_probs_all
                seq_prob_label = "sequence_probs_all" if use_seq_prob else "norm_sequence_probs_all"
                if sequence_probs_source is None:
                    raise ValueError(
                        f"{branch_name} requires {seq_prob_label} when sequence-probability weighting is enabled."
                    )
                sequence_probs_q = np.asarray(sequence_probs_source[qid], dtype=np.float32).reshape(-1)
                if sequence_probs_q.shape[0] != penalty_probs_q.shape[0]:
                    raise ValueError(
                        f"{branch_name} seq_prob shape mismatch at split={split}, qid={qid}: "
                        f"expected={penalty_probs_q.shape[0]}, got={sequence_probs_q.shape[0]}."
                    )
                if not np.all(np.isfinite(sequence_probs_q)):
                    raise ValueError(
                        f"Non-finite {branch_name} seq_prob values at split={split}, qid={qid}."
                    )
            evidence_dict = evidence_builder_fn(ids_row, penalty_probs_q, float(zeta), sequence_probs_q)
            if not all(np.isfinite(float(val)) for val in evidence_dict.values()):
                raise ValueError(
                    f"Non-finite {branch_name} weighted evidence at split={split}, qid={qid}, "
                    f"zeta={float(zeta):.4f}."
                )
            pred_evidence.append(evidence_dict)
            vacuity_scores[qid] = float(vacuity(evidence_dict)) if evidence_dict else 0.0
            dissonance_scores[qid] = float(dissonance(evidence_dict)) if evidence_dict else 0.0

        auroc_vac, aupr_vac = _safe_binary_metrics(
            eval_labels,
            vacuity_scores[metric_mask_arr],
            split,
            f"EUD {branch_name} weighted vacuity zeta={float(zeta):.2f}",
        )
        auroc_dis, aupr_dis = _safe_binary_metrics(
            eval_labels,
            dissonance_scores[metric_mask_arr],
            split,
            f"EUD {branch_name} weighted dissonance zeta={float(zeta):.2f}",
        )
        grid_results.append(
            {
                "zeta": float(zeta),
                "pred_evidence": pred_evidence,
                "vacuity": vacuity_scores,
                "dissonance": dissonance_scores,
                "AUROC": {"vacuity": auroc_vac, "dissonance": auroc_dis},
                "AUPR": {"vacuity": aupr_vac, "dissonance": aupr_dis},
            }
        )

    best_result = _select_best_grid_result(grid_results, selection_family)
    return {
        "pred_evidence": best_result["pred_evidence"],
        "vacuity": best_result["vacuity"],
        "dissonance": best_result["dissonance"],
        "AUROC": best_result["AUROC"],
        "AUPR": best_result["AUPR"],
        "grid_results": grid_results,
        "zeta_grid": [float(z) for z in zeta_grid],
        "best_zeta": float(best_result["zeta"]),
        "best_selection_auroc": float(best_result["AUROC"][selection_family]),
        "best_selection_aupr": float(best_result["AUPR"][selection_family]),
    }


def _summarize_grid_results(grid_results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "zeta": float(row["zeta"]),
            "AUROC": {
                "vacuity": float(row["AUROC"]["vacuity"]),
                "dissonance": float(row["AUROC"]["dissonance"]),
            },
            "AUPR": {
                "vacuity": float(row["AUPR"]["vacuity"]),
                "dissonance": float(row["AUPR"]["dissonance"]),
            },
        }
        for row in grid_results
    ]


def _find_grid_result_by_zeta(
    grid_results: Sequence[Dict[str, object]],
    zeta: float,
) -> Dict[str, object]:
    for row in grid_results:
        if _zeta_matches(float(row["zeta"]), float(zeta)):
            return row
    raise ValueError(f"Unable to find grid result for zeta={float(zeta):.6f}.")


def _copy_binary_metric_dict(metric_dict: Dict[str, object]) -> Dict[str, float]:
    return {
        "vacuity": float(metric_dict["vacuity"]),
        "dissonance": float(metric_dict["dissonance"]),
    }


def _prepare_discounted_split_bundle(
    bundle: DiscountedSplitBundle,
    probe_results: Dict[str, object],
    metric_eval_mask: Optional[np.ndarray] = None,
    ood_refusal_eval: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    _validate_discounted_bundle_lengths(bundle)
    metric_mask_arr = (
        np.ones(len(bundle.responses), dtype=bool)
        if metric_eval_mask is None
        else np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    )
    if metric_mask_arr.shape[0] != len(bundle.responses):
        raise ValueError(
            "Discounted EUD metric mask mismatch: "
            f"dataset={bundle.dataset_name}, split={bundle.split}, "
            f"mask={metric_mask_arr.shape[0]}, questions={len(bundle.responses)}."
        )
    if ood_refusal_eval is None:
        ood_refusal_eval = _default_refusal_stats(len(bundle.responses))

    expected_dim = int(probe_results["feature_dim"])
    metric_split_name = f"{bundle.dataset_name}:{bundle.split}"
    probe_feature = str(probe_results.get("probe_feature", "slt")).strip().lower()
    if probe_feature == "slt":
        if bundle.response_slt_all is None or bundle.response_slt_lastk_path is None:
            raise ValueError(
                f"probe_feature='slt' requires response-level SLT artifacts for {metric_split_name}."
            )
        _validate_response_slt_artifacts(
            response_slt_all=bundle.response_slt_all,
            responses=bundle.responses,
            semantic_ids=bundle.semantic_ids,
            expected_dim=expected_dim,
            split=metric_split_name,
        )
        question_probe_features = None
    elif probe_feature == "tbg":
        question_probe_features = _validate_question_probe_features(
            question_probe_features=bundle.question_probe_features,
            expected_dim=expected_dim,
            question_count=len(bundle.responses),
            split=metric_split_name,
            probe_feature=probe_feature,
        )
    else:
        raise ValueError(f"Unsupported probe_feature in discounted evaluation: {probe_feature!r}.")

    probe_target = _normalize_probe_target(probe_results.get("probe_target", "hallucination"))
    response_probe_probability: List[List[float]] = []
    response_discount_penalty_prob: List[List[float]] = []
    penalty_probs_all: List[np.ndarray] = []
    ids_rows_all: List[np.ndarray] = []
    sequence_probs_all: List[np.ndarray] = []
    norm_sequence_probs_all: List[np.ndarray] = []

    clf = probe_results["clf"]
    temperature = float(probe_results.get("temperature", 1.0))
    for qid, (resp_row, sid_row, seq_prob_row, norm_seq_prob_row) in enumerate(
        zip(
            bundle.responses,
            bundle.semantic_ids,
            bundle.response_sequence_probs,
            bundle.response_norm_sequence_probs,
        )
    ):
        ids_row = np.asarray(sid_row, dtype=np.int64).reshape(-1)
        sequence_probs_q = np.asarray(seq_prob_row, dtype=np.float32).reshape(-1)
        norm_sequence_probs_q = np.asarray(norm_seq_prob_row, dtype=np.float32).reshape(-1)
        n_resp = len(resp_row)
        ids_rows_all.append(ids_row)
        if n_resp == 0:
            response_probe_probability.append([])
            response_discount_penalty_prob.append([])
            penalty_probs_all.append(np.zeros(0, dtype=np.float32))
            sequence_probs_all.append(np.zeros(0, dtype=np.float32))
            norm_sequence_probs_all.append(np.zeros(0, dtype=np.float32))
            continue
        if ids_row.shape[0] != n_resp:
            raise ValueError(
                "Discounted EUD response/semantic_ids mismatch at "
                f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}: "
                f"responses={n_resp}, semantic_ids={ids_row.shape[0]}."
            )
        if sequence_probs_q.shape[0] != n_resp:
            raise ValueError(
                "Discounted EUD response/sequence_prob mismatch at "
                f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}: "
                f"responses={n_resp}, sequence_prob={sequence_probs_q.shape[0]}."
            )
        if norm_sequence_probs_q.shape[0] != n_resp:
            raise ValueError(
                "Discounted EUD response/norm_sequence_prob mismatch at "
                f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}: "
                f"responses={n_resp}, norm_sequence_prob={norm_sequence_probs_q.shape[0]}."
            )
        if not np.all(np.isfinite(sequence_probs_q)):
            raise ValueError(
                "Non-finite response sequence probabilities at "
                f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}."
            )
        if not np.all(np.isfinite(norm_sequence_probs_q)):
            raise ValueError(
                "Non-finite normalized response sequence probabilities at "
                f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}."
            )
        if np.any(sequence_probs_q < 0.0):
            raise ValueError(
                "Negative response sequence probabilities at "
                f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}."
            )
        if np.any(norm_sequence_probs_q < 0.0):
            raise ValueError(
                "Negative normalized response sequence probabilities at "
                f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}."
            )

        if probe_feature == "slt":
            assert bundle.response_slt_all is not None
            slt_row = bundle.response_slt_all[qid]
            slt_matrix = np.vstack(
                [np.asarray(slt_row[int(resp_id)], dtype=np.float32).reshape(-1) for resp_id in range(n_resp)]
            )
            slt_matrix = _normalize_flat_features(slt_matrix).astype(np.float32, copy=False)
            logits_q = clf.predict_logits(slt_matrix)
            probe_probs_q = positive_probability_from_logits(logits_q, temperature=temperature).astype(float, copy=False)
            if probe_probs_q.shape[0] != n_resp:
                raise ValueError(
                    f"{_probe_display_name(probe_target, capitalize=True)}-probe response count mismatch at "
                    f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}: "
                    f"expected={n_resp}, got={probe_probs_q.shape[0]}."
                )
            if not np.all(np.isfinite(probe_probs_q)):
                raise ValueError(
                    f"Non-finite {_probe_display_name(probe_target)} probabilities at "
                    f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid} "
                    f"from {bundle.response_slt_lastk_path}."
                )
            probe_probs_q = np.clip(probe_probs_q.astype(np.float32, copy=False), 0.0, 1.0)
        else:
            assert question_probe_features is not None
            logits_q = clf.predict_logits(question_probe_features[qid : qid + 1])
            probe_prob_scalar = positive_probability_from_logits(logits_q, temperature=temperature).astype(
                float,
                copy=False,
            )
            if probe_prob_scalar.shape[0] != 1:
                raise ValueError(
                    f"{_probe_display_name(probe_target, capitalize=True)}-probe question score count mismatch at "
                    f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}: "
                    f"expected=1, got={probe_prob_scalar.shape[0]}."
                )
            if not np.all(np.isfinite(probe_prob_scalar)):
                raise ValueError(
                    f"Non-finite question-level {_probe_display_name(probe_target)} probability at "
                    f"dataset={bundle.dataset_name}, split={bundle.split}, qid={qid}."
                )
            probe_prob_value = float(np.clip(probe_prob_scalar[0], 0.0, 1.0))
            probe_probs_q = np.full(n_resp, probe_prob_value, dtype=np.float32)
        penalty_probs_q = (
            probe_probs_q
            if probe_target == "hallucination"
            else np.clip(1.0 - probe_probs_q, 0.0, 1.0).astype(np.float32, copy=False)
        )
        penalty_probs_all.append(penalty_probs_q)
        sequence_probs_all.append(sequence_probs_q.astype(np.float32, copy=False))
        norm_sequence_probs_all.append(norm_sequence_probs_q.astype(np.float32, copy=False))
        response_probe_probability.append(probe_probs_q.astype(float).tolist())
        response_discount_penalty_prob.append(penalty_probs_q.astype(float).tolist())

    labels = np.asarray(bundle.labels, dtype=np.int64)
    disambig_question_semantic_ids_all = None
    if bundle.disambig_question_semantic_ids is not None:
        disambig_question_semantic_ids_all = [
            np.asarray(row, dtype=np.int64).reshape(-1)
            for row in bundle.disambig_question_semantic_ids
        ]
        _compute_disambig_question_theils_u(
            bundle.semantic_ids,
            disambig_question_semantic_ids_all,
            source=f"{metric_split_name}:discounted_bundle_disambig_question_alignment",
        )
    return {
        "probe_target": str(probe_target),
        "bundle": bundle,
        "metric_split_name": metric_split_name,
        "metric_eval_mask": metric_mask_arr,
        "ood_refusal_eval": ood_refusal_eval,
        "response_probe_probability": response_probe_probability,
        "response_discount_penalty_prob": response_discount_penalty_prob,
        "penalty_probs_all": penalty_probs_all,
        "ids_rows_all": ids_rows_all,
        "disambig_question_semantic_ids_all": disambig_question_semantic_ids_all,
        "sequence_probs_all": sequence_probs_all,
        "norm_sequence_probs_all": norm_sequence_probs_all,
        "eval_labels": labels[metric_mask_arr],
        "response_slt_lastk_path": None if bundle.response_slt_lastk_path is None else str(bundle.response_slt_lastk_path),
    }


def _resolve_discounted_selection_bundle(
    args,
    eval_bundle: DiscountedSplitBundle,
    eval_metric_mask: np.ndarray,
    eval_ood_refusal_eval: Dict[str, object],
) -> Tuple[DiscountedSplitBundle, np.ndarray, Dict[str, object], Dict[str, object]]:
    if not bool(getattr(args, "zeta_selection", False)):
        return (
            eval_bundle,
            np.asarray(eval_metric_mask, dtype=bool).reshape(-1),
            dict(eval_ood_refusal_eval),
            {
                "zeta_selection_enabled": False,
                "selection_split": str(eval_bundle.split),
                "selection_dataset_name": str(eval_bundle.dataset_name),
                "selection_label_source": str(eval_bundle.label_source),
                "pseudo_ood_dataset_name": None,
                "pseudo_ood_replaced_count": 0,
                "pseudo_ood_positive_count": 0,
            },
        )

    if str(args.dataset_name) in OOD_ZETA_SELECTION_DATASET_NAMES:
        selection_bundle = _build_pseudo_ood_selection_bundle(args)
    else:
        selection_bundle = _load_discounted_split_bundle(args, str(args.dataset_name), "valid")

    selection_metric_mask = np.ones(len(selection_bundle.responses), dtype=bool)
    selection_metadata = {
        "zeta_selection_enabled": True,
        "selection_split": str(selection_bundle.split),
        "selection_dataset_name": str(selection_bundle.dataset_name),
        "selection_label_source": str(selection_bundle.label_source),
        "pseudo_ood_dataset_name": selection_bundle.metadata.get("pseudo_ood_dataset_name"),
        "pseudo_ood_replaced_count": int(selection_bundle.metadata.get("pseudo_ood_replaced_count", 0)),
        "pseudo_ood_positive_count": int(selection_bundle.metadata.get("pseudo_ood_positive_count", 0)),
    }
    return (
        selection_bundle,
        selection_metric_mask,
        _default_refusal_stats(len(selection_bundle.responses)),
        selection_metadata,
    )


def _build_selected_branch_result(
    *,
    selection_results: Dict[str, object],
    eval_results: Dict[str, object],
    payload_keys: Sequence[str],
    selection_split: str,
    selection_dataset_name: str,
    selection_label_source: str,
) -> Dict[str, object]:
    selected_zeta = float(selection_results["best_zeta"])
    selected_eval_row = _find_grid_result_by_zeta(eval_results["grid_results"], selected_zeta)
    eval_oracle_best_zeta = float(eval_results["best_zeta"])

    branch_result = {key: selected_eval_row[key] for key in payload_keys}
    branch_result.update(
        {
            "AUROC": _copy_binary_metric_dict(selected_eval_row["AUROC"]),
            "AUPR": _copy_binary_metric_dict(selected_eval_row["AUPR"]),
            "grid_results": _summarize_grid_results(eval_results["grid_results"]),
            "selection_grid_results": _summarize_grid_results(selection_results["grid_results"]),
            "eval_grid_results": _summarize_grid_results(eval_results["grid_results"]),
            "zeta_grid": [float(z) for z in eval_results["zeta_grid"]],
            "best_zeta": float(selected_zeta),
            "selected_zeta": float(selected_zeta),
            "selection_auroc": float(selection_results["best_selection_auroc"]),
            "selection_aupr": float(selection_results["best_selection_aupr"]),
            "best_selection_auroc": float(selection_results["best_selection_auroc"]),
            "best_selection_aupr": float(selection_results["best_selection_aupr"]),
            "selection_split": str(selection_split),
            "selection_dataset_name": str(selection_dataset_name),
            "selection_label_source": str(selection_label_source),
            "eval_oracle_best_zeta": float(eval_oracle_best_zeta),
            "eval_oracle_best_auroc": float(eval_results["best_selection_auroc"]),
            "eval_oracle_best_aupr": float(eval_results["best_selection_aupr"]),
            "selected_matches_eval_best": bool(_zeta_matches(selected_zeta, eval_oracle_best_zeta)),
        }
    )
    return branch_result


def _decorate_empty_branch_result(
    branch_result: Dict[str, object],
    *,
    selection_split: str,
    selection_dataset_name: str,
    selection_label_source: str,
) -> Dict[str, object]:
    selected_zeta = float(branch_result["best_zeta"])
    branch_result["selected_zeta"] = float(selected_zeta)
    branch_result["selection_auroc"] = float(branch_result["best_selection_auroc"])
    branch_result["selection_aupr"] = float(branch_result["best_selection_aupr"])
    branch_result["selection_split"] = str(selection_split)
    branch_result["selection_dataset_name"] = str(selection_dataset_name)
    branch_result["selection_label_source"] = str(selection_label_source)
    branch_result["selection_grid_results"] = _summarize_grid_results(branch_result["grid_results"])
    branch_result["eval_grid_results"] = _summarize_grid_results(branch_result["grid_results"])
    branch_result["eval_oracle_best_zeta"] = float(selected_zeta)
    branch_result["eval_oracle_best_auroc"] = float(branch_result["best_selection_auroc"])
    branch_result["eval_oracle_best_aupr"] = float(branch_result["best_selection_aupr"])
    branch_result["selected_matches_eval_best"] = bool(_zeta_matches(selected_zeta, branch_result["eval_oracle_best_zeta"]))
    return branch_result


def _empty_branch_result_for_spec(
    spec: DiscountedBranchSpec,
    *,
    n_questions: int,
    zeta_grid: Sequence[float],
    selection_split: str,
    selection_dataset_name: str,
    selection_label_source: str,
) -> Dict[str, object]:
    base_result = (
        _empty_discounted_branch_result(n_questions, zeta_grid)
        if spec.kind == "discounted"
        else _empty_weighted_branch_result(n_questions, zeta_grid)
    )
    return _decorate_empty_branch_result(
        base_result,
        selection_split=selection_split,
        selection_dataset_name=selection_dataset_name,
        selection_label_source=selection_label_source,
    )


def _evaluate_discounted_branch_spec(
    spec: DiscountedBranchSpec,
    *,
    split_inputs: Dict[str, object],
    selection_family: str,
    zeta_grid: Sequence[float],
) -> Dict[str, object]:
    if spec.kind == "discounted":
        return _evaluate_discounted_branch(
            branch_name=spec.key,
            branch_mode=spec.mode,
            split=split_inputs["metric_split_name"],
            selection_family=selection_family,
            zeta_grid=zeta_grid,
            ids_rows_all=split_inputs["ids_rows_all"],
            penalty_probs_all=split_inputs["penalty_probs_all"],
            sequence_probs_all=split_inputs["sequence_probs_all"],
            norm_sequence_probs_all=split_inputs["norm_sequence_probs_all"],
            eval_labels=split_inputs["eval_labels"],
            metric_mask_arr=split_inputs["metric_eval_mask"],
            use_seq_prob=bool(spec.use_seq_prob),
            use_norm_seq_prob=bool(spec.use_norm_seq_prob),
        )
    evidence_builder_fn = {
        "mean_weighted": _build_mean_weighted_evidence_dict,
        "resp_wise_weighted": _build_resp_wise_weighted_evidence_dict,
    }.get(spec.mode)
    if evidence_builder_fn is None:
        raise ValueError(f"Unsupported weighted branch mode: {spec.mode!r}.")
    return _evaluate_weighted_branch(
        branch_name=spec.key,
        split=split_inputs["metric_split_name"],
        selection_family=selection_family,
        zeta_grid=zeta_grid,
        ids_rows_all=split_inputs["ids_rows_all"],
        penalty_probs_all=split_inputs["penalty_probs_all"],
        sequence_probs_all=split_inputs["sequence_probs_all"],
        norm_sequence_probs_all=split_inputs["norm_sequence_probs_all"],
        eval_labels=split_inputs["eval_labels"],
        metric_mask_arr=split_inputs["metric_eval_mask"],
        evidence_builder_fn=evidence_builder_fn,
        use_seq_prob=bool(spec.use_seq_prob),
        use_norm_seq_prob=bool(spec.use_norm_seq_prob),
    )


def _build_discounted_eud_payload(
    *,
    artifacts: SplitArtifacts,
    metric_eval_mask: np.ndarray,
    ood_refusal_eval: Dict[str, object],
    probe_target: str,
    probe_label_source: str,
    response_probe_probability: Optional[List[List[float]]],
    response_discount_penalty_prob: Optional[List[List[float]]],
    response_slt_lastk_path: Optional[str],
    zeta_grid: Sequence[float],
    selection_task: str,
    selection_family: str,
    selection_metadata: Dict[str, object],
    branch_results: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "probe_target": str(probe_target),
        "probe_label_source": str(probe_label_source),
        "uncertainty_labels": np.asarray(artifacts.labels, dtype=np.int64),
        "y": artifacts.y,
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool).reshape(-1),
        "ood_refusal_eval": ood_refusal_eval,
        "zeta_grid": [float(z) for z in zeta_grid],
        "selection_task": selection_task,
        "selection_family": selection_family,
        "zeta_selection_enabled": bool(selection_metadata["zeta_selection_enabled"]),
        "selection_split": str(selection_metadata["selection_split"]),
        "selection_dataset_name": str(selection_metadata["selection_dataset_name"]),
        "selection_label_source": str(selection_metadata["selection_label_source"]),
        "pseudo_ood_dataset_name": selection_metadata.get("pseudo_ood_dataset_name"),
        "pseudo_ood_replaced_count": int(selection_metadata.get("pseudo_ood_replaced_count", 0)),
        "pseudo_ood_positive_count": int(selection_metadata.get("pseudo_ood_positive_count", 0)),
    }
    if str(probe_target) == "hallucination":
        payload["response_hallucination_prob"] = response_probe_probability
    else:
        payload["response_ambiguity_prob"] = response_probe_probability
        payload["response_discount_penalty_prob"] = response_discount_penalty_prob
    if response_slt_lastk_path:
        payload["response_slt_lastk_path"] = str(response_slt_lastk_path)
    payload.update(branch_results)
    return payload


def _evaluate_discounted_eud(
    args,
    artifacts: SplitArtifacts,
    metric_eval_mask: np.ndarray,
    ood_refusal_eval: Dict[str, object],
    probe_results: Dict[str, object],
) -> Dict[str, object]:
    n_questions = len(artifacts.responses)
    selection_task, selection_family = _discount_selection_spec(artifacts.label_source)
    zeta_grid = normalize_zeta_grid(getattr(args, "zeta_grid", DEFAULT_DISCOUNT_ZETA_GRID))
    probe_target = _normalize_probe_target(probe_results.get("probe_target", "hallucination"))
    probe_label_source = str(probe_results.get("probe_label_source", "1-y"))
    eval_bundle = _build_discounted_bundle_from_artifacts(args, artifacts)
    selection_bundle, selection_metric_mask, selection_refusal_eval, selection_metadata = _resolve_discounted_selection_bundle(
        args,
        eval_bundle,
        metric_eval_mask,
        ood_refusal_eval,
    )
    clf = probe_results.get("clf")
    if clf is None:
        print(
            "[EUD Discounted/Weighted Branches] Skipping probe-derived metrics "
            f"because the {_probe_display_name(probe_target)} probe was not fit."
        )
        empty_branch_results = {
            spec.key: _empty_branch_result_for_spec(
                spec,
                n_questions=n_questions,
                zeta_grid=zeta_grid,
                selection_split=str(selection_metadata["selection_split"]),
                selection_dataset_name=str(selection_metadata["selection_dataset_name"]),
                selection_label_source=str(selection_metadata["selection_label_source"]),
            )
            for spec in DISCOUNTED_BRANCH_SPECS
        }
        empty_branch_results["exp_discounted_theils_u_resp_norm_seq_prob_dissonance"] = (
            _empty_exp_discounted_theils_u_resp_result(
                n_questions,
                zeta_grid,
                selection_split=str(selection_metadata["selection_split"]),
                selection_dataset_name=str(selection_metadata["selection_dataset_name"]),
                selection_label_source=str(selection_metadata["selection_label_source"]),
                skip_reason="Probe was not fit.",
            )
        )
        return _build_discounted_eud_payload(
            artifacts=artifacts,
            metric_eval_mask=metric_eval_mask,
            ood_refusal_eval=ood_refusal_eval,
            probe_target=probe_target,
            probe_label_source=probe_label_source,
            response_probe_probability=None,
            response_discount_penalty_prob=None,
            response_slt_lastk_path=None,
            zeta_grid=zeta_grid,
            selection_task=selection_task,
            selection_family=selection_family,
            selection_metadata=selection_metadata,
            branch_results=empty_branch_results,
        )

    eval_inputs = _prepare_discounted_split_bundle(
        eval_bundle,
        probe_results,
        metric_eval_mask=metric_eval_mask,
        ood_refusal_eval=ood_refusal_eval,
    )
    if selection_bundle is eval_bundle and np.array_equal(selection_metric_mask, np.asarray(metric_eval_mask, dtype=bool).reshape(-1)):
        selection_inputs = eval_inputs
    else:
        selection_inputs = _prepare_discounted_split_bundle(
            selection_bundle,
            probe_results,
            metric_eval_mask=selection_metric_mask,
            ood_refusal_eval=selection_refusal_eval,
        )

    branch_results: Dict[str, Dict[str, object]] = {}
    for spec in DISCOUNTED_BRANCH_SPECS:
        selection_results = _evaluate_discounted_branch_spec(
            spec,
            split_inputs=selection_inputs,
            selection_family=selection_family,
            zeta_grid=zeta_grid,
        )
        eval_results = (
            selection_results
            if selection_inputs is eval_inputs
            else _evaluate_discounted_branch_spec(
                spec,
                split_inputs=eval_inputs,
                selection_family=selection_family,
                zeta_grid=zeta_grid,
            )
        )
        branch_results[spec.key] = _build_selected_branch_result(
            selection_results=selection_results,
            eval_results=eval_results,
            payload_keys=spec.payload_keys,
            selection_split=str(selection_metadata["selection_split"]),
            selection_dataset_name=str(selection_metadata["selection_dataset_name"]),
            selection_label_source=str(selection_metadata["selection_label_source"]),
        )

    if probe_target != "hallucination":
        branch_results["exp_discounted_theils_u_resp_norm_seq_prob_dissonance"] = (
            _empty_exp_discounted_theils_u_resp_result(
                n_questions,
                zeta_grid,
                selection_split=str(selection_metadata["selection_split"]),
                selection_dataset_name=str(selection_metadata["selection_dataset_name"]),
                selection_label_source=str(selection_metadata["selection_label_source"]),
                skip_reason="Requires probe_target='hallucination' so exp(-zeta * P(hallu)) is available.",
            )
        )
    elif eval_inputs.get("disambig_question_semantic_ids_all") is None:
        branch_results["exp_discounted_theils_u_resp_norm_seq_prob_dissonance"] = (
            _empty_exp_discounted_theils_u_resp_result(
                n_questions,
                zeta_grid,
                selection_split=str(selection_metadata["selection_split"]),
                selection_dataset_name=str(selection_metadata["selection_dataset_name"]),
                selection_label_source=str(selection_metadata["selection_label_source"]),
                skip_reason="Disambig-question semantic ids are unavailable.",
            )
        )
    elif selection_inputs.get("disambig_question_semantic_ids_all") is None:
        branch_results["exp_discounted_theils_u_resp_norm_seq_prob_dissonance"] = (
            _empty_exp_discounted_theils_u_resp_result(
                n_questions,
                zeta_grid,
                selection_split=str(selection_metadata["selection_split"]),
                selection_dataset_name=str(selection_metadata["selection_dataset_name"]),
                selection_label_source=str(selection_metadata["selection_label_source"]),
                skip_reason="Selection split disambig-question semantic ids are unavailable.",
            )
        )
    else:
        selection_exp_results = _evaluate_exp_discounted_theils_u_resp_norm_seq_prob_dissonance_grid(
            split_inputs=selection_inputs,
            zeta_grid=zeta_grid,
        )
        eval_exp_results = (
            selection_exp_results
            if selection_inputs is eval_inputs
            else _evaluate_exp_discounted_theils_u_resp_norm_seq_prob_dissonance_grid(
                split_inputs=eval_inputs,
                zeta_grid=zeta_grid,
            )
        )
        branch_results["exp_discounted_theils_u_resp_norm_seq_prob_dissonance"] = _build_selected_scalar_result(
            selection_results=selection_exp_results,
            eval_results=eval_exp_results,
            selection_split=str(selection_metadata["selection_split"]),
            selection_dataset_name=str(selection_metadata["selection_dataset_name"]),
            selection_label_source=str(selection_metadata["selection_label_source"]),
        )

    return _build_discounted_eud_payload(
        artifacts=artifacts,
        metric_eval_mask=eval_inputs["metric_eval_mask"],
        ood_refusal_eval=eval_inputs["ood_refusal_eval"],
        probe_target=probe_target,
        probe_label_source=probe_label_source,
        response_probe_probability=eval_inputs["response_probe_probability"],
        response_discount_penalty_prob=eval_inputs["response_discount_penalty_prob"],
        response_slt_lastk_path=eval_inputs["response_slt_lastk_path"],
        zeta_grid=zeta_grid,
        selection_task=selection_task,
        selection_family=selection_family,
        selection_metadata=selection_metadata,
        branch_results=branch_results,
    )


def _run_optional_baselines(args, artifacts: SplitArtifacts, step: int = 0) -> None:
    try:
        from baseline import LinearProbe, PTrue, SelfCheckGPT, Verbalized, _log_baseline_summary
    except Exception as exc:
        raise RuntimeError(
            "run_baseline=True requires the baseline evaluation stack and its dependencies "
            "(notably sklearn) in the active environment."
        ) from exc

    class _EUDLinearProbe(LinearProbe):
        def _load_hidden_and_labels(self, split: str) -> Dict[str, object]:
            return _load_linear_probe_split(self.args, split)

    print("[Baseline] Loading causal LM for PTrue.")
    model, tokenizer = _load_causal_lm(args.model_name)

    ptrue = PTrue(args, model=model, tokenizer=tokenizer)
    data = _load_baseline_data(args, artifacts, ptrue)
    ptrue.data = data
    ptrue_results = ptrue.evaluate(data["questions"], data["preds"], data["hallucination_labels"])
    print(f"[{artifacts.split} PTrue] AUROC: {ptrue_results['AUROC']:.4f}, AUPR: {ptrue_results['AUPR']:.4f}")
    _log_baseline_summary(artifacts.split, "ptrue", ptrue_results)
    _log_metrics(_metric_dict_for_history(artifacts.split, "ptrue", ptrue_results), step=step)

    if data.get("uncertainty_label_kind") in {"ambig", "ood", "fallback"}:
        verbalized = Verbalized(args, model=model, tokenizer=tokenizer)
        verbalized.data = data
        verbalized_results = verbalized.evaluate(data["questions"], data["preds"], data["hallucination_labels"])
        print(
            f"[{artifacts.split} Verbalized] "
            f"Epistemic AUROC: {verbalized_results['AUROC_epistemic']:.4f}, "
            f"AUPR: {verbalized_results['AUPR_epistemic']:.4f}; "
            f"Aleatoric AUROC: {verbalized_results['AUROC_aleatoric']:.4f}, "
            f"AUPR: {verbalized_results['AUPR_aleatoric']:.4f}"
        )
        _log_baseline_summary(artifacts.split, "verbalized", verbalized_results)
        _log_metrics(_metric_dict_for_history(artifacts.split, "verbalized", verbalized_results), step=step)

    selfcheck = SelfCheckGPT(args)
    selfcheck.data = data
    selfcheck_results = selfcheck.evaluate(data["questions"], data["preds"], data["hallucination_labels"])
    print(
        f"[{artifacts.split} SelfCheckGPT] AUROC: {selfcheck_results['AUROC']:.4f}, "
        f"AUPR: {selfcheck_results['AUPR']:.4f}"
    )
    _log_baseline_summary(artifacts.split, "selfcheckgpt", selfcheck_results)
    _log_metrics(_metric_dict_for_history(artifacts.split, "selfcheckgpt", selfcheck_results), step=step)

    linear_probe = _EUDLinearProbe(args)
    linear_probe_results = linear_probe.evaluate(eval_split=artifacts.split)
    print(
        f"[{artifacts.split} LinearProbe] AUROC: {linear_probe_results['AUROC']:.4f}, "
        f"AUPR: {linear_probe_results['AUPR']:.4f}"
    )
    _log_baseline_summary(artifacts.split, "linear_probe", linear_probe_results)
    _log_metrics(_metric_dict_for_history(artifacts.split, "linear_probe", linear_probe_results), step=step)
