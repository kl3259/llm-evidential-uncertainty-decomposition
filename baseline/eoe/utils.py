from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


THRESHOLDS = np.arange(1, 100, dtype=np.float32) / 100.0


def set_all_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32, copy=False)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = torch.clamp(mask.sum(dim=1), min=1e-12)
    return masked.sum(dim=1) / denom


@dataclass
class TextEmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    device: Optional[str] = None


class TextEmbedder:
    def __init__(self, config: TextEmbedderConfig):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self._backend = "transformers"
        self._st_model = None
        self._tokenizer = None
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(config.model_name, device=device)
            self._backend = "sentence_transformers"
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self._model = AutoModel.from_pretrained(config.model_name)
            self._model.to(self.device)
            self._model.eval()

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        bs = int(batch_size or self.config.batch_size)
        if self._backend == "sentence_transformers":
            arr = self._st_model.encode(
                list(texts),
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
            return l2_normalize(arr) if self.config.normalize else arr

        vectors: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), bs):
                batch = list(texts[start : start + bs])
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        arr = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(arr) if self.config.normalize else arr


def resolve_binary_mask(labels: np.ndarray) -> np.ndarray:
    label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    return np.logical_or(label_arr == 0, label_arr == 1)


def compute_score_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if (
        average_precision_score is None
        or f1_score is None
        or precision_score is None
        or recall_score is None
        or roc_auc_score is None
    ):
        raise ModuleNotFoundError(
            "scikit-learn is required to compute AUROC/AUPR metrics."
        )
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if labels_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError(
            f"labels/scores shape mismatch: {labels_arr.shape[0]} vs {scores_arr.shape[0]}"
        )
    unique = np.unique(labels_arr)
    if labels_arr.size == 0 or unique.size < 2:
        return {
            "auroc": float("nan"),
            "aupr": float("nan"),
            "best_f1": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_threshold": float("nan"),
            "positive_mean": float("nan"),
            "negative_mean": float("nan"),
        }

    auroc = float(roc_auc_score(labels_arr, scores_arr))
    aupr = float(average_precision_score(labels_arr, scores_arr))
    pos_scores = scores_arr[labels_arr == 1]
    neg_scores = scores_arr[labels_arr == 0]
    pos_mean = float(np.mean(pos_scores)) if pos_scores.size > 0 else float("nan")
    neg_mean = float(np.mean(neg_scores)) if neg_scores.size > 0 else float("nan")

    best_idx = 0
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    for idx, threshold in enumerate(THRESHOLDS):
        preds = (scores_arr > threshold).astype(np.int64)
        f1_val = f1_score(labels_arr, preds, zero_division=0)
        if f1_val > best_f1:
            best_f1 = float(f1_val)
            best_idx = int(idx)
            best_precision = float(precision_score(labels_arr, preds, zero_division=0))
            best_recall = float(recall_score(labels_arr, preds, zero_division=0))

    return {
        "auroc": auroc,
        "aupr": aupr,
        "best_f1": float(best_f1),
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_threshold": float(THRESHOLDS[best_idx]),
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
    }


def build_summary_metrics(
    *,
    dataset_name: str,
    model_name: str,
    baseline_method: str,
    n_samples: int,
    eval_split: str,
    label_resolved: np.ndarray,
    au_trace: np.ndarray,
    eu_trace: np.ndarray,
    total_trace: np.ndarray,
) -> Dict[str, float]:
    labels = np.asarray(label_resolved, dtype=np.int64).reshape(-1)
    valid_mask = resolve_binary_mask(labels)
    valid_labels = labels[valid_mask]
    au_valid = np.asarray(au_trace, dtype=np.float32).reshape(-1)[valid_mask]
    eu_valid = np.asarray(eu_trace, dtype=np.float32).reshape(-1)[valid_mask]
    tu_valid = np.asarray(total_trace, dtype=np.float32).reshape(-1)[valid_mask]

    au_metrics = compute_score_metrics(valid_labels, au_valid)
    eu_metrics = compute_score_metrics(valid_labels, eu_valid)
    tu_metrics = compute_score_metrics(valid_labels, tu_valid)

    summary: Dict[str, float] = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "baseline_method": baseline_method,
        "n_samples": int(n_samples),
        "eval_split": eval_split,
        "num_examples": int(labels.shape[0]),
        "num_valid_binary_labels": int(valid_mask.sum()),
        "num_positive": int((valid_labels == 1).sum()),
        "num_negative": int((valid_labels == 0).sum()),
        "aleatoric_auroc": au_metrics["auroc"],
        "aleatoric_aupr": au_metrics["aupr"],
        "epistemic_auroc": eu_metrics["auroc"],
        "epistemic_aupr": eu_metrics["aupr"],
        "total_auroc": tu_metrics["auroc"],
        "total_aupr": tu_metrics["aupr"],
    }
    for key, value in au_metrics.items():
        summary[f"aleatoric_{key}"] = value
    for key, value in eu_metrics.items():
        summary[f"epistemic_{key}"] = value
    for key, value in tu_metrics.items():
        summary[f"total_{key}"] = value
    return summary


def _resolve_wandb_run_url(run) -> Optional[str]:
    run_url = getattr(run, "url", None)
    if run_url:
        return run_url
    get_url = getattr(run, "get_url", None)
    if callable(get_url):
        try:
            run_url = get_url()
        except Exception:
            run_url = None
    if run_url:
        return run_url
    run_path = getattr(run, "path", None)
    if isinstance(run_path, (list, tuple)) and len(run_path) == 3:
        entity, project, run_id = run_path
        if entity and project and run_id:
            return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    return None


def log_summary_to_wandb(
    *,
    summary_metrics: Dict[str, float],
    project: str,
    entity: Optional[str],
    run_name: str,
    config: Dict[str, object],
) -> Optional[str]:
    if wandb is None:
        raise ImportError("wandb is not installed but --wandb_log=True was requested.")
    run = wandb.run
    if run is None:
        init_kwargs = {"project": project, "name": run_name, "config": config}
        if entity:
            init_kwargs["entity"] = entity
        run = wandb.init(**init_kwargs)
    run_url = _resolve_wandb_run_url(run)
    wandb.run.summary.update(summary_metrics)
    wandb.log(summary_metrics)
    run.finish()
    return run_url
