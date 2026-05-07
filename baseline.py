import argparse
import json
import os
import re
import warnings
from distutils.util import strtobool as _bool
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import Gemma4ForCausalLM
except ImportError:
    Gemma4ForCausalLM = None

from feature_collector import _ensure_bitsandbytes_available, _instruction_tuned_system_prompt, \
    _needs_8bit_quantized_loading, _resolve_hf_model_name, _is_gemma_4_e4b_model_name
from helper_functions import (
    count_or_frac,
    compute_ood_refusal_filter,
    load_pred_json,
    load_question_json,
    load_ref_json,
    load_resp_json,
    load_X_npz,
)

WORKING_DIR = "./"


def _resolve_model_name(*, tokenizer=None, model=None) -> str:
    tokenizer_name = getattr(tokenizer, "name_or_path", "")
    if tokenizer_name:
        return str(tokenizer_name)
    return str(getattr(model, "name_or_path", ""))


def _messages_with_optional_system(content: str, model_name: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    system_prompt = _instruction_tuned_system_prompt(model_name)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _prepend_optional_system_prompt(prompt: str, model_name: str) -> str:
    system_prompt = _instruction_tuned_system_prompt(model_name)
    if system_prompt:
        return f"{system_prompt}\n{prompt}"
    return prompt


def _chat_template_model_inputs(tokenizer, messages, device: torch.device, **template_kwargs) -> Dict[str, torch.Tensor]:
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **template_kwargs,
    )
    if isinstance(model_inputs, torch.Tensor):
        input_ids = model_inputs.to(device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }

    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(device)

    if hasattr(model_inputs, "items"):
        normalized_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        input_ids = normalized_inputs.get("input_ids")
        if input_ids is None:
            raise KeyError("Chat template inputs missing input_ids.")
        if "attention_mask" not in normalized_inputs:
            normalized_inputs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
        return normalized_inputs

    raise TypeError(f"Unsupported chat template output type: {type(model_inputs)!r}")


def _safe_binary_auroc_aupr(labels, scores, dataset_name: str, split: str, method_name: str):
    labels_arr = np.asarray(labels, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    unique_labels = np.unique(labels_arr)
    if unique_labels.size < 2:
        if dataset_name in {"alcuna_trivia_qa", "repli_qa_trivia_qa"} and split == "train":
            print(f"[{method_name}] Skipping AUROC/AUPR for {dataset_name} train split (only one OOD class present).")
        else:
            print(f"[{method_name}] Skipping AUROC/AUPR for {split} split (only one class present: {unique_labels.tolist()}).")
        return float("nan"), float("nan")
    return float(roc_auc_score(labels_arr, scores_arr)), float(average_precision_score(labels_arr, scores_arr))


def _compute_no_clear_err_auroc_aupr(
    ambig_labels,
    y,
    scores,
    metric_eval_mask,
    dataset_name: str,
    split: str,
    method_name: str,
) -> Tuple[float, float]:
    ambig_arr = np.asarray(ambig_labels, dtype=np.int64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)

    n = int(min(ambig_arr.size, y_arr.size, scores_arr.size))
    if n <= 0:
        return float("nan"), float("nan")
    ambig_arr = ambig_arr[:n]
    y_arr = y_arr[:n]
    scores_arr = scores_arr[:n]
    if mask_arr.size != n:
        mask_arr = np.ones(n, dtype=bool)
    else:
        mask_arr = mask_arr[:n]

    hallucination_arr = np.asarray(y_arr == 0, dtype=np.int64)
    exclusion_mask = np.logical_and(ambig_arr == 0, hallucination_arr == 1)
    final_mask = np.logical_and(mask_arr, ~exclusion_mask)
    labels_eval = ambig_arr[final_mask]
    scores_eval = scores_arr[final_mask]

    return _safe_binary_auroc_aupr(
        labels_eval,
        scores_eval,
        dataset_name=dataset_name,
        split=split,
        method_name=f"{method_name} no_clear_err",
    )


def _count_unique_semantic_clusters(ids_row) -> int:
    if ids_row is None:
        return 0
    row = np.asarray(ids_row, dtype=np.int64).reshape(-1)
    if row.size == 0:
        return 0
    return int(np.unique(row).size)


def _resolve_semantic_cluster_counts(
    semantic_ids,
    expected_size: Optional[int] = None,
) -> np.ndarray:
    target_size = int(expected_size) if expected_size is not None else None
    if semantic_ids is None:
        return np.zeros(0, dtype=np.int64)

    if isinstance(semantic_ids, dict):
        rows_by_idx: Dict[int, object] = {}
        for key, row in semantic_ids.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if idx < 0:
                continue
            rows_by_idx[idx] = row

        if not rows_by_idx:
            return np.zeros(0, dtype=np.int64)

        if target_size is None:
            ordered_idx = sorted(rows_by_idx.keys())
            return np.asarray(
                [_count_unique_semantic_clusters(rows_by_idx[idx]) for idx in ordered_idx],
                dtype=np.int64,
            )

        if target_size <= 0:
            return np.zeros(0, dtype=np.int64)

        if all(idx in rows_by_idx for idx in range(target_size)):
            return np.asarray(
                [_count_unique_semantic_clusters(rows_by_idx[idx]) for idx in range(target_size)],
                dtype=np.int64,
            )

        ordered_idx = sorted(rows_by_idx.keys())[:target_size]
        return np.asarray(
            [_count_unique_semantic_clusters(rows_by_idx[idx]) for idx in ordered_idx],
            dtype=np.int64,
        )

    try:
        n_rows = len(semantic_ids)
    except TypeError:
        n_rows = 0
    if n_rows <= 0:
        return np.zeros(0, dtype=np.int64)
    if target_size is None:
        target_size = int(n_rows)
    if target_size <= 0:
        return np.zeros(0, dtype=np.int64)

    n_common = int(min(target_size, n_rows))
    if n_common <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(
        [_count_unique_semantic_clusters(semantic_ids[idx]) for idx in range(n_common)],
        dtype=np.int64,
    )


def _compute_no_clear_large_n_cluster_auroc_aupr(
    ambig_labels,
    semantic_ids,
    scores,
    metric_eval_mask,
    dataset_name: str,
    split: str,
    method_name: str,
    large_cluster_threshold: int = 5,
) -> Tuple[float, float]:
    ambig_arr = np.asarray(ambig_labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    n_cluster = _resolve_semantic_cluster_counts(
        semantic_ids=semantic_ids,
        expected_size=ambig_arr.size,
    )

    if ambig_arr.size == 0:
        return float("nan"), float("nan")
    if (
        scores_arr.size != ambig_arr.size
        or mask_arr.size != ambig_arr.size
        or n_cluster.size != ambig_arr.size
    ):
        print(
            f"[{method_name} no_clear_large_n_cluster] Skipping AUROC/AUPR for {split} split "
            f"(length mismatch: labels={ambig_arr.size}, scores={scores_arr.size}, "
            f"mask={mask_arr.size}, semantic_ids={n_cluster.size})."
        )
        return float("nan"), float("nan")

    exclusion_mask = np.logical_and(
        ambig_arr == 0,
        n_cluster >= int(large_cluster_threshold),
    )
    final_mask = np.logical_and(mask_arr, ~exclusion_mask)
    labels_eval = ambig_arr[final_mask]
    scores_eval = scores_arr[final_mask]

    return _safe_binary_auroc_aupr(
        labels_eval,
        scores_eval,
        dataset_name=dataset_name,
        split=split,
        method_name=f"{method_name} no_clear_large_n_cluster",
    )


def _parse_true_false_response(text: object) -> Optional[bool]:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None
    match = re.search(r"\b(true|false)\b", normalized)
    if match is None:
        return None
    return match.group(1) == "true"


_CONFIDENCE_NUMBER_PATTERN = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(%)?")


def _normalize_confidence_value(
    value_text: object,
    *,
    has_percent_sign: bool = False,
    context: str = "",
) -> Optional[float]:
    try:
        value = float(str(value_text))
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    lower_context = str(context or "").lower()
    is_percentage = bool(has_percent_sign) or "percent" in lower_context or value > 1.0
    if is_percentage:
        value /= 100.0

    return float(np.clip(value, 0.0, 1.0))


def _parse_confidence_response(text: object) -> Optional[float]:
    normalized = str(text or "").strip()
    if not normalized:
        return None

    matches = list(_CONFIDENCE_NUMBER_PATTERN.finditer(normalized))
    if not matches:
        return None
    match = matches[-1]
    return _normalize_confidence_value(
        match.group(1),
        has_percent_sign=bool(match.group(2)),
        context=match.group(0),
    )


def _parse_dual_confidence_response(text: object) -> Tuple[Optional[float], Optional[float]]:
    normalized = str(text or "").strip()
    if not normalized:
        return None, None

    # Prefer separated segments and take the last numeric value in each segment
    # so list markers like "1. 80%, 2. 20%" do not become false positives.
    segments = [segment.strip() for segment in re.split(r"[\n,;]+", normalized) if segment.strip()]
    segment_values: List[float] = []
    for segment in segments:
        value = _parse_confidence_response(segment)
        if value is None:
            continue
        segment_values.append(value)
        if len(segment_values) >= 2:
            return segment_values[0], segment_values[1]

    percent_matches = [match for match in _CONFIDENCE_NUMBER_PATTERN.finditer(normalized) if match.group(2)]
    if len(percent_matches) >= 2:
        return (
            _normalize_confidence_value(
                percent_matches[0].group(1),
                has_percent_sign=True,
                context=percent_matches[0].group(0),
            ),
            _normalize_confidence_value(
                percent_matches[1].group(1),
                has_percent_sign=True,
                context=percent_matches[1].group(0),
            ),
        )

    cleaned = re.sub(r"(^|[\s,;])([12])\s*[\):]\s*", r"\1", normalized)
    cleaned = re.sub(r"(^|[\s,;])([12])\.(?=\s)", r"\1", cleaned)
    fallback_matches = list(_CONFIDENCE_NUMBER_PATTERN.finditer(cleaned))
    if len(fallback_matches) >= 2:
        return (
            _normalize_confidence_value(
                fallback_matches[0].group(1),
                has_percent_sign=bool(fallback_matches[0].group(2)),
                context=fallback_matches[0].group(0),
            ),
            _normalize_confidence_value(
                fallback_matches[1].group(1),
                has_percent_sign=bool(fallback_matches[1].group(2)),
                context=fallback_matches[1].group(0),
            ),
        )

    return _parse_confidence_response(normalized), None


class Baseline:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.data: Optional[Dict[str, object]] = None

    def load_data(self, split: Optional[str] = None) -> Dict[str, object]:
        split_name = split or self.args.eval_split
        model_id = self.args.model_name.split("/")[-1]
        load_dir = os.path.join(
            WORKING_DIR,
            "input_hidden_states",
            f"{model_id}_{self.args.dataset_name}_{self.args.n_samples}_{self.args.judge_type}",
        )
        split_dir = os.path.join(load_dir, split_name)

        loss_type = (getattr(self.args, "loss_type", "") or "").lower()
        response_level_mode = loss_type in {"response-wise", "ranking"}
        se_resp_name = "se_resp_res.jsonl" if response_level_mode else "se_resp.jsonl"
        se_resp_path = os.path.join(split_dir, se_resp_name)
        questions_path = os.path.join(split_dir, "questions.jsonl")
        references_path = os.path.join(split_dir, "references.jsonl")
        y_path = os.path.join(split_dir, "y.npy")
        semantic_ids_path = os.path.join(split_dir, f"semantic_ids_{self.args.n_resp}.npy")

        preds_path = self._find_preds_path(split_dir)
        missing = self._collect_missing_paths(
            [se_resp_path, questions_path, references_path, y_path, semantic_ids_path],
            preds_path,
        )
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(f"Missing required files: {missing_list}")

        se_resp = load_resp_json(se_resp_path)
        questions = load_question_json(questions_path)
        references = load_ref_json(references_path)
        preds = load_pred_json(preds_path)
        contexts = self._load_contexts(questions) if self._uses_context() else None

        y = np.load(y_path)
        semantic_ids = np.load(semantic_ids_path)
        hallucination_labels, uncertainty_path, uncertainty_label_kind = self._resolve_uncertainty_labels(split_dir, y)
        uncertainty_labels_by_kind = self._load_optional_uncertainty_labels(split_dir, y)

        probe_eval = None
        probe_eval_path = None
        # if self.args.wandb_run_id:
        #     probe_eval_path = self._find_probe_eval_path(
        #         model_id,
        #         self.args.dataset_name,
        #         self.args.n_samples,
        #         self.args.judge_type,
        #         self.args.wandb_run_id,
        #     )
        #     if probe_eval_path is None:
        #         output_dir = self._get_probe_output_dir(
        #             model_id,
        #             self.args.dataset_name,
        #             self.args.n_samples,
        #             self.args.judge_type,
        #         )
        #         candidates = [
        #             os.path.join(output_dir, f"probe_eval_{self.args.wandb_run_id}.json"),
        #             os.path.join(output_dir, f"probe_eval_reg_{self.args.wandb_run_id}.json"),
        #         ]
        #         raise FileNotFoundError(
        #             "Probe eval file not found. Tried: " + ", ".join(candidates)
        #         )
        #     with open(probe_eval_path, "r", encoding="utf-8") as f:
        #         probe_eval = json.load(f)

        return self._set_data(
            se_resp=se_resp,
            questions=questions,
            references=references,
            preds=preds,
            contexts=contexts,
            y=y,
            semantic_ids=semantic_ids,
            hallucination_labels=hallucination_labels,
            uncertainty_label_kind=uncertainty_label_kind,
            uncertainty_labels_by_kind=uncertainty_labels_by_kind,
            probe_eval=probe_eval,
            paths={
                "se_resp": se_resp_path,
                "questions": questions_path,
                "references": references_path,
                "preds": preds_path,
                "y": y_path,
                "semantic_ids": semantic_ids_path,
                "uncertainty_labels": uncertainty_path,
                "probe_eval": probe_eval_path,
            },
            split=split_name,
            load_dir=load_dir,
        )

    def evaluate(self, questions, preds, hallucination_labels):
        raise NotImplementedError("Baseline.evaluate must be implemented by subclasses.")

    def _set_data(self, **items: object) -> Dict[str, object]:
        self.data = items
        return items

    def _apply_refusal_filter(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        split_name: str,
        preds: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        scores_arr = np.asarray(scores, dtype=float).reshape(-1)
        if labels_arr.shape[0] != scores_arr.shape[0]:
            raise ValueError(
                "labels/scores length mismatch for refusal filtering: "
                f"{labels_arr.shape[0]} vs {scores_arr.shape[0]}"
            )
        load_dir = None
        if isinstance(self.data, dict):
            load_dir = self.data.get("load_dir")
        metric_eval_mask, ood_refusal_eval = compute_ood_refusal_filter(
            labels=labels_arr,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            remove_refusal=bool(getattr(self.args, "remove_refusal", False)),
            load_dir=load_dir,
            preds=preds,
        )
        eval_labels = labels_arr[metric_eval_mask]
        eval_scores = scores_arr[metric_eval_mask]
        return eval_labels, eval_scores, metric_eval_mask, ood_refusal_eval

    @staticmethod
    def _collect_missing_paths(
        required_paths: list,
        preds_path: Optional[str],
    ) -> list:
        missing = [path for path in required_paths if not os.path.exists(path)]
        if preds_path is None:
            missing.append("<missing preds.jsonl>")
        return missing

    @staticmethod
    def _resolve_uncertainty_labels(split_dir: str, y: np.ndarray) -> Tuple[np.ndarray, Optional[str], str]:
        ambig_path = os.path.join(split_dir, "ambig_label.npy")
        if os.path.exists(ambig_path):
            return np.asarray(np.load(ambig_path), dtype=np.int64), ambig_path, "ambig"

        ood_path = os.path.join(split_dir, "ood_label.npy")
        if os.path.exists(ood_path):
            return np.asarray(np.load(ood_path), dtype=np.int64), ood_path, "ood"

        return np.asarray(1.0 - y, dtype=np.int64), None, "fallback"

    @staticmethod
    def _load_optional_uncertainty_labels(split_dir: str, y: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        ambig_path = os.path.join(split_dir, "ambig_label.npy")
        ood_path = os.path.join(split_dir, "ood_label.npy")

        aleatoric_labels = None
        if os.path.exists(ambig_path):
            aleatoric_labels = np.asarray(np.load(ambig_path), dtype=np.int64)

        if os.path.exists(ood_path):
            epistemic_labels = np.asarray(np.load(ood_path), dtype=np.int64)
        elif aleatoric_labels is None:
            epistemic_labels = np.asarray(1.0 - y, dtype=np.int64)
        else:
            epistemic_labels = None

        if epistemic_labels is None and aleatoric_labels is not None:
            epistemic_labels = np.asarray(aleatoric_labels, dtype=np.int64)
        if aleatoric_labels is None and epistemic_labels is not None:
            aleatoric_labels = np.asarray(epistemic_labels, dtype=np.int64)

        return {
            "aleatoric": aleatoric_labels,
            "epistemic": epistemic_labels,
        }

    @staticmethod
    def _find_preds_path(split_dir: str) -> Optional[str]:
        path = os.path.join(split_dir, "preds.jsonl")
        if os.path.exists(path):
            return path
        return None

    def _uses_context(self) -> bool:
        dataset_name = self.args.dataset_name.lower()
        return "boolq" in dataset_name or "squad_split" in dataset_name

    def _load_contexts(self, questions: List[str]) -> Optional[List[str]]:
        dataset_name = self.args.dataset_name.lower()
        if "boolq" in dataset_name:
            dataset_id = "google/boolq"
            context_field = "passage"
            splits = ("train", "validation")
        elif "squad_split" in dataset_name:
            dataset_id = "squad_v2"
            context_field = "context"
            splits = ("train",)
        else:
            return None

        try:
            from datasets import load_dataset, concatenate_datasets
        except Exception:
            return None

        try:
            raw = load_dataset(dataset_id)
        except Exception:
            return None

        datasets = [raw[split] for split in splits if split in raw]
        if not datasets:
            return None
        full = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

        needed = set(questions)
        context_by_question: Dict[str, str] = {}
        for ex in full:
            question = ex.get("question")
            if question in needed:
                context_by_question[question] = ex.get(context_field, "")
                needed.discard(question)
                if not needed:
                    break

        missing_questions = [q for q in questions if not context_by_question.get(q)]
        if missing_questions:
            unique_missing = list(dict.fromkeys(missing_questions))
            sample = ", ".join(repr(q) for q in unique_missing[:3])
            warnings.warn(
                "Context lookup missed for "
                f"{len(missing_questions)} question(s) "
                f"({len(unique_missing)} unique). "
                f"Example(s): {sample}"
            )

        return [context_by_question.get(q, "") for q in questions]

    @staticmethod
    def _get_probe_output_dir(
        model_id: str,
        dataset_name: str,
        n_samples: int,
        judge_type: str,
    ) -> str:
        return os.path.join(
            WORKING_DIR,
            "outputs",
            f"{model_id}_{dataset_name}_{n_samples}_{judge_type}",
        )

    @classmethod
    def _find_probe_eval_path(
        cls,
        model_id: str,
        dataset_name: str,
        n_samples: int,
        judge_type: str,
        wandb_run_id: str,
    ) -> Optional[str]:
        output_dir = cls._get_probe_output_dir(
            model_id, dataset_name, n_samples, judge_type
        )
        candidates = [
            os.path.join(output_dir, f"probe_eval_{wandb_run_id}.json"),
            os.path.join(output_dir, f"probe_eval_reg_{wandb_run_id}.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None


class SelfCheckGPT(Baseline):
    # potsawee/deberta-v3-large-mnli
    MODEL_NAME = "potsawee/deberta-v3-large-mnli"

    @classmethod
    def instantiate_nli_model(cls, device: Optional[str] = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(cls.MODEL_NAME, use_safetensors=True)
        model.to(device)
        model.eval()
        return model, tokenizer, device

    @staticmethod
    def _resolve_label_indices(model):
        label2id = getattr(model.config, "label2id", None) or {}
        label2id = {str(k).lower(): int(v) for k, v in label2id.items()}
        contradiction_idx = label2id.get("contradiction")
        entailment_idx = label2id.get("entailment")

        if contradiction_idx is None or entailment_idx is None:
            id2label = getattr(model.config, "id2label", None) or {}
            for k, v in id2label.items():
                name = str(v).lower()
                if name == "contradiction":
                    contradiction_idx = int(k)
                elif name == "entailment":
                    entailment_idx = int(k)

        if contradiction_idx is None:
            contradiction_idx = 0
        if entailment_idx is None:
            if getattr(model.config, "num_labels", 0) > 2:
                entailment_idx = 2 if contradiction_idx != 2 else 1
            else:
                entailment_idx = 1 if contradiction_idx == 0 else 0
        return contradiction_idx, entailment_idx

    @staticmethod
    def _contradiction_probs(logits, contradiction_idx, entailment_idx):
        num_labels = logits.shape[-1]
        if num_labels == 2:
            probs = torch.softmax(logits, dim=-1)
            return probs[:, -1]
        idx = [contradiction_idx, entailment_idx]
        if min(idx) < 0 or max(idx) >= num_labels or contradiction_idx == entailment_idx:
            idx = [0, 2 if num_labels > 2 else 1]
        selected_logits = logits[:, idx]
        probs = torch.softmax(selected_logits, dim=-1)
        return probs[:, 0]

    def evaluate(self, questions, preds, hallucination_labels):
        assert len(questions) == len(preds) == len(hallucination_labels), \
            "Length of questions, preds, and hallucination_labels must be equal."
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer, self.device = self.instantiate_nli_model()
        else:
            self.device = self.model.device

        if self.data is None or "se_resp" not in self.data:
            raise ValueError("SelfCheckGPT.evaluate requires se_resp data; call load_data first.")
        se_resp = self.data["se_resp"]
        if len(se_resp) != len(preds):
            if len(se_resp) < len(preds):
                raise ValueError("se_resp length is smaller than preds; cannot align.")
            se_resp = se_resp[:len(preds)]

        contradiction_idx, entailment_idx = self._resolve_label_indices(self.model)

        self.model.eval()
        scores = []
        with torch.inference_mode():
            for pred, samples in tqdm(zip(preds, se_resp), total=len(preds)):
                if not samples:
                    scores.append(0.0)
                    continue
                pairs = [(sample, pred) for sample in samples]
                inputs = self.tokenizer(
                    pairs,
                    add_special_tokens=True,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=True,
                    return_attention_mask=True,
                )
                inputs = inputs.to(self.device)
                logits = self.model(**inputs).logits
                contradiction_probs = self._contradiction_probs(
                    logits, contradiction_idx, entailment_idx
                )
                scores.append(float(contradiction_probs.mean().item()))

        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(hallucination_labels, dtype=np.int64)
        split_name = (self.data or {}).get("split", getattr(self.args, "eval_split", "test"))
        eval_labels, eval_scores, metric_eval_mask, ood_refusal_eval = self._apply_refusal_filter(
            labels=labels_arr,
            scores=scores_arr,
            split_name=split_name,
            preds=list(preds),
        )
        auroc, aupr = _safe_binary_auroc_aupr(
            eval_labels,
            eval_scores,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="SelfCheckGPT",
        )
        y_arr = np.asarray((self.data or {}).get("y", []), dtype=np.int64)
        auroc_no_clear_err, aupr_no_clear_err = _compute_no_clear_err_auroc_aupr(
            ambig_labels=labels_arr,
            y=y_arr,
            scores=scores_arr,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="SelfCheckGPT",
        )
        auroc_no_clear_large_n_cluster, aupr_no_clear_large_n_cluster = _compute_no_clear_large_n_cluster_auroc_aupr(
            ambig_labels=labels_arr,
            semantic_ids=(self.data or {}).get("semantic_ids"),
            scores=scores_arr,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="SelfCheckGPT",
        )

        return {
            "scores": scores_arr,
            "AUROC": auroc,
            "AUPR": aupr,
            "AUROC_no_clear_err": auroc_no_clear_err,
            "AUPR_no_clear_err": aupr_no_clear_err,
            "AUROC_no_clear_large_n_cluster": auroc_no_clear_large_n_cluster,
            "AUPR_no_clear_large_n_cluster": aupr_no_clear_large_n_cluster,
            "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
            "ood_refusal_eval": ood_refusal_eval,
        }


class PTrue(Baseline):
    @staticmethod
    def candidate_token_ids(tokenizer, symbol: str) -> List[int]:
        """Return tokenizer ids that can emit the given symbol as a single token."""
        candidates = [symbol, f" {symbol}", f"\n{symbol}", f"\n {symbol}"]
        token_ids: List[int] = []
        for text in candidates:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
        return list(dict.fromkeys(token_ids))

    def evaluate(self, questions, preds, hallucination_labels):

        assert len(questions) == len(preds) == len(hallucination_labels), \
            "Length of questions, preds, and hallucination_labels must be equal."
        if self.model is None or self.tokenizer is None:
            raise ValueError("PTrue.evaluate requires self.model and self.tokenizer to be set.")

        true_token_ids = []
        for symbol in ("True", "true"):
            true_token_ids.extend(self.candidate_token_ids(self.tokenizer, symbol))
        true_token_ids = list(dict.fromkeys(true_token_ids))
        if not true_token_ids:
            raise ValueError("Tokenizer does not represent True/true or A as a single token.")

        use_context = self._uses_context()
        contexts = None
        if use_context and self.data is not None:
            contexts = self.data.get("contexts")
        if use_context and contexts is None:
            use_context = False
        if use_context and len(contexts) != len(questions):
            raise ValueError("contexts length must match questions for context-aware PTrue.")

        model_name = _resolve_model_name(tokenizer=self.tokenizer, model=self.model)
        self.model.eval()
        scores = []
        responses = []
        with torch.inference_mode():
            if use_context:
                iterator = zip(questions, preds, contexts)
            else:
                iterator = ((q, p, None) for q, p in zip(questions, preds))
            for question, pred, context in tqdm(iterator, total=len(preds)):
                if use_context:
                    content = f"""
                    Context:\n\n{context}\n\nQuestion:\n\n{question}\n\nProposed Answer: {pred}
                    
                    Is the proposed answer:\n\tTrue or\n\tFalse?\nThe proposed answer is: 
                    """
                else:
                    content = f"""
                    Question:\n\n{question}\n\nProposed Answer: {pred}
                    
                    Is the proposed answer:\n\tTrue or\n\tFalse?\nThe proposed answer is: 
                    """
                messages = _messages_with_optional_system(content, model_name)
                model_inputs = _chat_template_model_inputs(
                    self.tokenizer,
                    messages,
                    self.model.device,
                    max_new_tokens=1,
                )
                outputs = self.model(**model_inputs)
                next_token_logits = outputs.logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                pred_token_id = int(torch.argmax(next_token_logits, dim=-1)[0].item())
                responses.append(self.tokenizer.decode([pred_token_id], skip_special_tokens=True))
                prob_true = probs[0, true_token_ids].sum().item()
                scores.append(-prob_true)

        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(hallucination_labels, dtype=np.int64)
        split_name = (self.data or {}).get("split", getattr(self.args, "eval_split", "test"))
        eval_labels, eval_scores, metric_eval_mask, ood_refusal_eval = self._apply_refusal_filter(
            labels=labels_arr,
            scores=scores_arr,
            split_name=split_name,
            preds=list(preds),
        )
        auroc, aupr = _safe_binary_auroc_aupr(
            eval_labels,
            eval_scores,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="PTrue",
        )
        y_arr = np.asarray((self.data or {}).get("y", []), dtype=np.int64)
        auroc_no_clear_err, aupr_no_clear_err = _compute_no_clear_err_auroc_aupr(
            ambig_labels=labels_arr,
            y=y_arr,
            scores=scores_arr,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="PTrue",
        )
        auroc_no_clear_large_n_cluster, aupr_no_clear_large_n_cluster = _compute_no_clear_large_n_cluster_auroc_aupr(
            ambig_labels=labels_arr,
            semantic_ids=(self.data or {}).get("semantic_ids"),
            scores=scores_arr,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="PTrue",
        )

        return {
            "scores": scores_arr,
            "responses": responses,
            "AUROC": auroc,
            "AUPR": aupr,
            "AUROC_no_clear_err": auroc_no_clear_err,
            "AUPR_no_clear_err": aupr_no_clear_err,
            "AUROC_no_clear_large_n_cluster": auroc_no_clear_large_n_cluster,
            "AUPR_no_clear_large_n_cluster": aupr_no_clear_large_n_cluster,
            "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
            "ood_refusal_eval": ood_refusal_eval,
        }


class Verbalized(Baseline):
    # OOD_PROMPT_TEMPLATE = (
    #     "Question: {question}\n\n"
    #     "Do you have prior knowledge to answer this question? "
    #     "Respond only with a percentage (0-100\%) of confidence that you have the knowledge. "
    # )
    # AMBIG_PROMPT_TEMPLATE = (
    #     "Question: {question}\n\n"
    #     "Do you find this question ambiguous and can be interpreted in multiple ways? "
    #     "Respond only with a percentage (0-100\%) of confidence that the question is ambiguous. "
    # )
    AIO_TEMPLATE = (
        "Question: {question}\n\n"
        "We are assessing the multidimensional uncertainty of a question. "
        "Please respond with two confidence percentages (0-100\%) for the following questions:\n"
        "1. Do you have prior knowledge to answer this question?\n"
        "2. Do you find this question ambiguous and can be interpreted in multiple ways?\n"
        "You should respond with two percentage numbers (0-100\%) separated by a comma, use the order: "
        "[confidence that you have the knowledge]\%, [confidence that the question is ambiguous]\%"
    )

    @staticmethod
    def _resolve_model_input_device(model) -> torch.device:
        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for mapped_device in hf_device_map.values():
                if isinstance(mapped_device, torch.device):
                    return mapped_device
                if isinstance(mapped_device, int):
                    return torch.device(f"cuda:{int(mapped_device)}")
                if isinstance(mapped_device, str):
                    text = mapped_device.strip().lower()
                    if text.startswith("cuda") or text == "cpu":
                        return torch.device(text)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def _prompt_template_for_task(cls, task_kind: str) -> str:
        if task_kind in {"ood", "ambig", "fallback"}:
            return cls.AIO_TEMPLATE
        raise ValueError(f"Unsupported verbalized task kind: {task_kind!r}")

    @staticmethod
    def _score_from_confidence(confidence: Optional[float], task_kind: str) -> float:
        if confidence is None or not np.isfinite(confidence):
            confidence_value = 0.0
        else:
            confidence_value = float(np.clip(confidence, 0.0, 1.0))
        if task_kind == "ood":
            return 1.0 - confidence_value
        if task_kind == "ambig":
            return confidence_value
        raise ValueError(f"Unsupported verbalized task kind: {task_kind!r}")

    @staticmethod
    def _empty_refusal_stats(total_count: int) -> Dict[str, object]:
        return {
            "applied": False,
            "accuracy": float("nan"),
            "excluded_ood_count": 0,
            "excluded_id_count": 0,
            "excluded_total_count": 0,
            "remaining_count": int(total_count),
            "total_count": int(total_count),
        }

    @staticmethod
    def _slice_optional_labels(labels: Optional[np.ndarray], size: int) -> Optional[np.ndarray]:
        if labels is None:
            return None
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        return labels_arr[:size]

    def _evaluate_family_scores(
        self,
        *,
        family_name: str,
        labels: Optional[np.ndarray],
        scores: np.ndarray,
        preds: Optional[List[str]],
        split_name: str,
    ) -> Dict[str, object]:
        scores_arr = np.asarray(scores, dtype=float).reshape(-1)
        total_count = int(scores_arr.shape[0])
        if labels is None:
            return {
                "AUROC": float("nan"),
                "AUPR": float("nan"),
                "AUROC_no_clear_err": float("nan"),
                "AUPR_no_clear_err": float("nan"),
                "AUROC_no_clear_large_n_cluster": float("nan"),
                "AUPR_no_clear_large_n_cluster": float("nan"),
                "metric_eval_mask": np.ones(total_count, dtype=bool),
                "ood_refusal_eval": self._empty_refusal_stats(total_count),
            }

        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.shape[0] != scores_arr.shape[0]:
            raise ValueError(
                f"Verbalized {family_name} labels/scores length mismatch: "
                f"{labels_arr.shape[0]} vs {scores_arr.shape[0]}"
            )

        eval_labels, eval_scores, metric_eval_mask, ood_refusal_eval = self._apply_refusal_filter(
            labels=labels_arr,
            scores=scores_arr,
            split_name=split_name,
            preds=preds,
        )
        auroc, aupr = _safe_binary_auroc_aupr(
            eval_labels,
            eval_scores,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name=f"Verbalized {family_name}",
        )

        y_arr = np.asarray((self.data or {}).get("y", []), dtype=np.int64).reshape(-1)
        y_arr = y_arr[: scores_arr.shape[0]]
        semantic_ids = (self.data or {}).get("semantic_ids")
        if semantic_ids is not None:
            try:
                semantic_ids = semantic_ids[: scores_arr.shape[0]]
            except TypeError:
                pass
        auroc_no_clear_err, aupr_no_clear_err = _compute_no_clear_err_auroc_aupr(
            ambig_labels=labels_arr,
            y=y_arr,
            scores=scores_arr,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name=f"Verbalized {family_name}",
        )
        auroc_no_clear_large_n_cluster, aupr_no_clear_large_n_cluster = _compute_no_clear_large_n_cluster_auroc_aupr(
            ambig_labels=labels_arr,
            semantic_ids=semantic_ids,
            scores=scores_arr,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name=f"Verbalized {family_name}",
        )

        return {
            "AUROC": auroc,
            "AUPR": aupr,
            "AUROC_no_clear_err": auroc_no_clear_err,
            "AUPR_no_clear_err": aupr_no_clear_err,
            "AUROC_no_clear_large_n_cluster": auroc_no_clear_large_n_cluster,
            "AUPR_no_clear_large_n_cluster": aupr_no_clear_large_n_cluster,
            "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
            "ood_refusal_eval": ood_refusal_eval,
        }

    def _tokenize_prompt(self, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
        model_name = _resolve_model_name(tokenizer=self.tokenizer, model=self.model)
        messages = _messages_with_optional_system(prompt, model_name)
        if getattr(self.tokenizer, "chat_template", None):
            return _chat_template_model_inputs(self.tokenizer, messages, device)

        encoded_prompt = _prepend_optional_system_prompt(prompt, model_name)
        encoded = self.tokenizer(encoded_prompt, return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items()}

    def evaluate(self, questions, preds, hallucination_labels):
        assert len(questions) == len(hallucination_labels), \
            "Length of questions and hallucination_labels must be equal."
        if self.model is None or self.tokenizer is None:
            raise ValueError("Verbalized.evaluate requires self.model and self.tokenizer to be set.")

        task_kind = str((self.data or {}).get("uncertainty_label_kind", "fallback"))
        prompt_template = self._prompt_template_for_task(task_kind)
        device = self._resolve_model_input_device(self.model)
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if pad_token_id is None:
            pad_token_id = eos_token_id

        self.model.eval()
        responses = []
        parsed_confidence_pairs = []
        with torch.inference_mode():
            for question in tqdm(questions, total=len(questions)):
                prompt = prompt_template.format(question=str(question))
                model_inputs = self._tokenize_prompt(prompt, device=device)
                input_len = int(model_inputs["input_ids"].shape[-1])
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                generated_tokens = generated[:, input_len:]
                response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
                responses.append(response)
                knowledge_confidence, ambiguity_confidence = _parse_dual_confidence_response(response)
                parsed_confidence_pairs.append(
                    (
                        0.0 if knowledge_confidence is None else float(knowledge_confidence),
                        0.0 if ambiguity_confidence is None else float(ambiguity_confidence),
                    )
                )

        if parsed_confidence_pairs:
            parsed_confidences_arr = np.asarray(parsed_confidence_pairs, dtype=float)
        else:
            parsed_confidences_arr = np.zeros((0, 2), dtype=float)

        knowledge_confidences_arr = parsed_confidences_arr[:, 0]
        ambiguity_confidences_arr = parsed_confidences_arr[:, 1]
        epistemic_scores_arr = 1.0 - knowledge_confidences_arr
        aleatoric_scores_arr = ambiguity_confidences_arr

        labels_by_kind = (self.data or {}).get("uncertainty_labels_by_kind") or {}
        epistemic_labels = self._slice_optional_labels(labels_by_kind.get("epistemic"), len(questions))
        aleatoric_labels = self._slice_optional_labels(labels_by_kind.get("aleatoric"), len(questions))
        default_labels = np.asarray(hallucination_labels, dtype=np.int64).reshape(-1)[: len(questions)]
        if epistemic_labels is None and task_kind in {"ood", "fallback"}:
            epistemic_labels = default_labels
        if aleatoric_labels is None and task_kind == "ambig":
            aleatoric_labels = default_labels
        if epistemic_labels is None and aleatoric_labels is not None:
            epistemic_labels = np.asarray(aleatoric_labels, dtype=np.int64).reshape(-1)[: len(questions)]
        if aleatoric_labels is None and epistemic_labels is not None:
            aleatoric_labels = np.asarray(epistemic_labels, dtype=np.int64).reshape(-1)[: len(questions)]
        split_name = (self.data or {}).get("split", getattr(self.args, "eval_split", "test"))
        preds_eval = list(preds) if preds is not None else None

        epistemic_metrics = self._evaluate_family_scores(
            family_name="epistemic",
            labels=epistemic_labels,
            scores=epistemic_scores_arr,
            preds=preds_eval,
            split_name=split_name,
        )
        aleatoric_metrics = self._evaluate_family_scores(
            family_name="aleatoric",
            labels=aleatoric_labels,
            scores=aleatoric_scores_arr,
            preds=preds_eval,
            split_name=split_name,
        )

        if task_kind == "ambig":
            active_family_name = "aleatoric"
            active_scores = aleatoric_scores_arr
            active_confidences = ambiguity_confidences_arr
            active_metrics = aleatoric_metrics
        else:
            active_family_name = "epistemic"
            active_scores = epistemic_scores_arr
            active_confidences = knowledge_confidences_arr
            active_metrics = epistemic_metrics

        return {
            "scores": active_scores,
            "confidence_scores": active_confidences,
            "responses": responses,
            "parsed_confidences": parsed_confidences_arr,
            "parsed_responses": parsed_confidences_arr,
            "parsed_confidence_pairs": parsed_confidences_arr,
            "knowledge_confidences": knowledge_confidences_arr,
            "ambiguity_confidences": ambiguity_confidences_arr,
            "epistemic_scores": epistemic_scores_arr,
            "aleatoric_scores": aleatoric_scores_arr,
            "active_family": active_family_name,
            "AUROC": active_metrics["AUROC"],
            "AUPR": active_metrics["AUPR"],
            "AUROC_no_clear_err": active_metrics["AUROC_no_clear_err"],
            "AUPR_no_clear_err": active_metrics["AUPR_no_clear_err"],
            "AUROC_no_clear_large_n_cluster": active_metrics["AUROC_no_clear_large_n_cluster"],
            "AUPR_no_clear_large_n_cluster": active_metrics["AUPR_no_clear_large_n_cluster"],
            "AUROC_epistemic": epistemic_metrics["AUROC"],
            "AUPR_epistemic": epistemic_metrics["AUPR"],
            "AUROC_epistemic_no_clear_err": epistemic_metrics["AUROC_no_clear_err"],
            "AUPR_epistemic_no_clear_err": epistemic_metrics["AUPR_no_clear_err"],
            "AUROC_epistemic_no_clear_large_n_cluster": epistemic_metrics["AUROC_no_clear_large_n_cluster"],
            "AUPR_epistemic_no_clear_large_n_cluster": epistemic_metrics["AUPR_no_clear_large_n_cluster"],
            "AUROC_aleatoric": aleatoric_metrics["AUROC"],
            "AUPR_aleatoric": aleatoric_metrics["AUPR"],
            "AUROC_aleatoric_no_clear_err": aleatoric_metrics["AUROC_no_clear_err"],
            "AUPR_aleatoric_no_clear_err": aleatoric_metrics["AUPR_no_clear_err"],
            "AUROC_aleatoric_no_clear_large_n_cluster": aleatoric_metrics["AUROC_no_clear_large_n_cluster"],
            "AUPR_aleatoric_no_clear_large_n_cluster": aleatoric_metrics["AUPR_no_clear_large_n_cluster"],
            "metric_eval_mask": np.asarray(active_metrics["metric_eval_mask"], dtype=bool),
            "metric_eval_mask_epistemic": np.asarray(epistemic_metrics["metric_eval_mask"], dtype=bool),
            "metric_eval_mask_aleatoric": np.asarray(aleatoric_metrics["metric_eval_mask"], dtype=bool),
            "ood_refusal_eval": active_metrics["ood_refusal_eval"],
            "ood_refusal_eval_epistemic": epistemic_metrics["ood_refusal_eval"],
            "ood_refusal_eval_aleatoric": aleatoric_metrics["ood_refusal_eval"],
        }


class LinearProbe(Baseline):
    def __init__(self, args, model=None, tokenizer=None):
        super().__init__(args, model=model, tokenizer=tokenizer)
        self.last_k = getattr(args, "linear_probe_last_k", 5)

    @staticmethod
    def _flatten_hidden_states(X_tbg: np.ndarray, last_k: int) -> np.ndarray:
        if X_tbg.ndim != 3:
            raise ValueError("Expected hidden states with shape [n_layers, n_questions, hidden_dim].")
        n_layers = X_tbg.shape[0]
        k = min(int(last_k), n_layers)
        if k <= 0:
            raise ValueError("linear_probe_last_k must be positive.")
        layer_range = np.arange(n_layers - k, n_layers)
        X_input = X_tbg[layer_range, :, :]
        X_flat = np.transpose(X_input, (1, 0, 2)).reshape(X_input.shape[1], -1)
        X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(X_flat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (X_flat / norms).astype(np.float32, copy=False)

    def _load_hidden_and_labels(self, split: str) -> Dict[str, object]:
        model_id = self.args.model_name.split("/")[-1]
        load_dir = os.path.join(
            WORKING_DIR,
            "input_hidden_states",
            f"{model_id}_{self.args.dataset_name}_{self.args.n_samples}_{self.args.judge_type}",
        )
        split_dir = os.path.join(load_dir, split)

        X_path = os.path.join(split_dir, "X.npz")
        y_path = os.path.join(split_dir, "y.npy")
        semantic_ids_path = os.path.join(split_dir, f"semantic_ids_{self.args.n_resp}.npy")
        missing = [path for path in [X_path, y_path, semantic_ids_path] if not os.path.exists(path)]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(f"Missing required files: {missing_list}")

        X = load_X_npz(X_path)
        y = np.load(y_path)
        semantic_ids = np.load(semantic_ids_path)
        hallucination_labels, uncertainty_path, uncertainty_label_kind = self._resolve_uncertainty_labels(split_dir, y)
        preds_path = self._find_preds_path(split_dir)
        preds = load_pred_json(preds_path) if preds_path else None

        return {
            "X": X,
            "y": y,
            "semantic_ids": semantic_ids,
            "hallucination_labels": hallucination_labels,
            "uncertainty_label_kind": uncertainty_label_kind,
            "preds": preds,
            "paths": {
                "X": X_path,
                "y": y_path,
                "semantic_ids": semantic_ids_path,
                "uncertainty_labels": uncertainty_path,
                "preds": preds_path,
            },
            "split": split,
            "load_dir": load_dir,
        }

    def evaluate(self, questions=None, preds=None, hallucination_labels=None, eval_split: str = "test"):
        train_data = self._load_hidden_and_labels("train")
        eval_data = self._load_hidden_and_labels(eval_split)

        X_train = train_data["X"]
        X_eval = eval_data["X"]
        if "tbg" not in X_train or "tbg" not in X_eval:
            raise ValueError("LinearProbe expects 'tbg' hidden states in X.npz.")

        X_train_flat = self._flatten_hidden_states(X_train["tbg"], self.last_k)
        X_eval_flat = self._flatten_hidden_states(X_eval["tbg"], self.last_k)

        y_train = np.asarray(train_data["hallucination_labels"], dtype=np.int64)
        y_eval = np.asarray(eval_data["hallucination_labels"], dtype=np.int64)
        y_eval_raw = np.asarray(eval_data["y"], dtype=np.int64)
        semantic_ids_eval = np.asarray(eval_data["semantic_ids"])
        preds_eval = list(eval_data.get("preds") or [])

        if getattr(self.args, "use_subset", False):
            n_subset_train = min(self.args.n_infer_subset, len(y_train), X_train_flat.shape[0])
            n_subset_eval = min(self.args.n_infer_subset, len(y_eval), X_eval_flat.shape[0])
            X_train_flat = X_train_flat[:n_subset_train]
            y_train = y_train[:n_subset_train]
            X_eval_flat = X_eval_flat[:n_subset_eval]
            y_eval = y_eval[:n_subset_eval]
            y_eval_raw = y_eval_raw[:n_subset_eval]
            semantic_ids_eval = semantic_ids_eval[:n_subset_eval]
            if preds_eval:
                preds_eval = preds_eval[:n_subset_eval]

        if np.unique(y_train).size < 2:
            split_name = eval_split
            if getattr(self.args, "dataset_name", "") in {"alcuna_trivia_qa", "repli_qa_trivia_qa"}:
                ds_name = getattr(self.args, "dataset_name", "")
                print(
                    f"[LinearProbe] Skipping training/evaluation for {split_name} split "
                    f"because {ds_name} train labels contain only one class."
                )
            else:
                print(
                    f"[LinearProbe] Skipping training/evaluation for {split_name} split "
                    f"because train labels contain one class: {np.unique(y_train).tolist()}."
                )
            return {
                "scores": np.full(len(y_eval), np.nan, dtype=float),
                "AUROC": float("nan"),
                "AUPR": float("nan"),
                "AUROC_no_clear_err": float("nan"),
                "AUPR_no_clear_err": float("nan"),
                "AUROC_no_clear_large_n_cluster": float("nan"),
                "AUPR_no_clear_large_n_cluster": float("nan"),
                "metric_eval_mask": np.ones(len(y_eval), dtype=bool),
                "ood_refusal_eval": {
                    "applied": False,
                    "accuracy": float("nan"),
                    "excluded_ood_count": 0,
                    "excluded_id_count": 0,
                    "excluded_total_count": 0,
                    "remaining_count": int(len(y_eval)),
                    "total_count": int(len(y_eval)),
                },
            }

        clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
        clf.fit(X_train_flat, y_train)

        probas = clf.predict_proba(X_eval_flat)
        pos_idx = int(np.where(clf.classes_ == 1)[0][0]) if probas.shape[1] > 1 else 0
        scores = probas[:, pos_idx].astype(float)
        split_name = str(eval_data.get("split", eval_split))
        metric_eval_mask, ood_refusal_eval = compute_ood_refusal_filter(
            labels=y_eval,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            remove_refusal=bool(getattr(self.args, "remove_refusal", False)),
            load_dir=str(eval_data.get("load_dir", "")),
            preds=preds_eval if preds_eval else None,
        )
        eval_labels = y_eval[metric_eval_mask]
        eval_scores = scores[metric_eval_mask]

        auroc, aupr = _safe_binary_auroc_aupr(
            eval_labels,
            eval_scores,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=eval_split,
            method_name="LinearProbe",
        )
        auroc_no_clear_err, aupr_no_clear_err = _compute_no_clear_err_auroc_aupr(
            ambig_labels=y_eval,
            y=y_eval_raw,
            scores=scores,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="LinearProbe",
        )
        auroc_no_clear_large_n_cluster, aupr_no_clear_large_n_cluster = _compute_no_clear_large_n_cluster_auroc_aupr(
            ambig_labels=y_eval,
            semantic_ids=semantic_ids_eval,
            scores=scores,
            metric_eval_mask=metric_eval_mask,
            dataset_name=getattr(self.args, "dataset_name", ""),
            split=split_name,
            method_name="LinearProbe",
        )

        return {
            "scores": scores,
            "AUROC": auroc,
            "AUPR": aupr,
            "AUROC_no_clear_err": auroc_no_clear_err,
            "AUPR_no_clear_err": aupr_no_clear_err,
            "AUROC_no_clear_large_n_cluster": auroc_no_clear_large_n_cluster,
            "AUPR_no_clear_large_n_cluster": aupr_no_clear_large_n_cluster,
            "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
            "ood_refusal_eval": ood_refusal_eval,
        }


def _log_baseline_summary(split: str, method_name: str, results: Dict[str, object]) -> None:
    if wandb.run is None:
        return
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
            metrics.update({
                f"{split}/refusal_accuracy_{method_name}": float(refusal_stats.get("accuracy", float("nan"))),
                f"{split}/excluded_ood_count_{method_name}": int(refusal_stats.get("excluded_ood_count", 0)),
                f"{split}/excluded_id_count_{method_name}": int(refusal_stats.get("excluded_id_count", 0)),
                f"{split}/excluded_total_count_{method_name}": int(refusal_stats.get("excluded_total_count", 0)),
                f"{split}/remaining_count_{method_name}": int(refusal_stats.get("remaining_count", 0)),
                f"{split}/total_count_{method_name}": int(refusal_stats.get("total_count", 0)),
            })
        wandb.run.summary.update(metrics)
        return

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
        metrics.update({
            f"{split}/refusal_accuracy_{method_name}": float(refusal_stats.get("accuracy", float("nan"))),
            f"{split}/excluded_ood_count_{method_name}": int(refusal_stats.get("excluded_ood_count", 0)),
            f"{split}/excluded_id_count_{method_name}": int(refusal_stats.get("excluded_id_count", 0)),
            f"{split}/excluded_total_count_{method_name}": int(refusal_stats.get("excluded_total_count", 0)),
            f"{split}/remaining_count_{method_name}": int(refusal_stats.get("remaining_count", 0)),
            f"{split}/total_count_{method_name}": int(refusal_stats.get("total_count", 0)),
        })
    wandb.run.summary.update(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EUP Baseline Evaluation")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str,
                        default="trivia_qa_debug")
    parser.add_argument("--judge_type", type=str, default="rouge")
    parser.add_argument("--loss_type", type=str, default="response-wise")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--valid_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--test_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--n_resp", type=int, default=10)

    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Existing W&B run ID to resume when logging_original_run is true.")

    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--ckpt_step", type=int, default=1)

    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--use_subset", type=_bool, default=False)
    parser.add_argument("--n_infer_subset", type=int, default=100)
    parser.add_argument("--logging_original_run", type=_bool, default=False)
    parser.add_argument("--linear_probe_last_k", type=int, default=5)
    parser.add_argument("--remove_refusal", type=_bool, default=False)

    args = parser.parse_args()
    log_to_original_run = bool(args.wandb_run_id and args.logging_original_run)
    if log_to_original_run:
        wandb.init(project="EPN-Baselines", id=args.wandb_run_id, resume="must")

    resolved_model_name = _resolve_hf_model_name(args.model_name)
    needs_8bit = _needs_8bit_quantized_loading(args.model_name)
    dtype = torch.float16 if "llama-2-7b" in args.model_name.lower() else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        device_map="auto",
        torch_dtype=dtype,
        padding_side="left",
        verbose=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if needs_8bit:
        _ensure_bitsandbytes_available()
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"[int8] Loading {resolved_model_name} with device_map=auto.")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    elif _is_gemma_4_e4b_model_name(args.model_name):
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
    ptrue_baseline = PTrue(args, model=model, tokenizer=tokenizer)
    data = ptrue_baseline.load_data()
    questions = data["questions"]
    preds = data["preds"]
    hallucination_labels = data["hallucination_labels"]
    if args.use_subset:
        n_subset = min(
            args.n_infer_subset,
            len(questions),
            len(preds),
            len(hallucination_labels),
        )
        questions = questions[:n_subset]
        preds = preds[:n_subset]
        hallucination_labels = hallucination_labels[:n_subset]

    results = ptrue_baseline.evaluate(questions, preds, hallucination_labels)
    data["ptrue"] = results
    print(f"[PTrue] AUROC: {results['AUROC']:.4f}, AUPR: {results['AUPR']:.4f}")
    _log_baseline_summary(args.eval_split, "ptrue", results)

    verbalized_baseline = Verbalized(args, model=model, tokenizer=tokenizer)
    verbalized_baseline.data = data
    verbalized_results = verbalized_baseline.evaluate(
        questions,
        preds,
        hallucination_labels,
    )
    data["verbalized"] = verbalized_results
    print(
        f"[Verbalized] "
        f"Epistemic AUROC: {verbalized_results['AUROC_epistemic']:.4f}, "
        f"AUPR: {verbalized_results['AUPR_epistemic']:.4f}; "
        f"Aleatoric AUROC: {verbalized_results['AUROC_aleatoric']:.4f}, "
        f"AUPR: {verbalized_results['AUPR_aleatoric']:.4f}"
    )
    _log_baseline_summary(args.eval_split, "verbalized", verbalized_results)

    selfcheck_baseline = SelfCheckGPT(args)
    selfcheck_baseline.data = data
    selfcheck_results = selfcheck_baseline.evaluate(
        questions,
        preds,
        hallucination_labels,
    )
    data["selfcheckgpt"] = selfcheck_results
    print(
        f"[SelfCheckGPT] AUROC: {selfcheck_results['AUROC']:.4f}, "
        f"AUPR: {selfcheck_results['AUPR']:.4f}"
    )
    _log_baseline_summary(args.eval_split, "selfcheckgpt", selfcheck_results)

    linear_probe = LinearProbe(args)
    linear_probe_results = linear_probe.evaluate()
    data["linear_probe"] = linear_probe_results
    print(
        f"[LinearProbe] AUROC: {linear_probe_results['AUROC']:.4f}, "
        f"AUPR: {linear_probe_results['AUPR']:.4f}"
    )
    _log_baseline_summary(args.eval_split, "linear_probe", linear_probe_results)


    pass
