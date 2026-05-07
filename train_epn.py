from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from distutils.util import strtobool as _bool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import wandb

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

try:
    from transformers import Gemma4ForCausalLM
except ImportError:
    Gemma4ForCausalLM = None

from epn_loss import AugmentedResponseUCEforEPNC, PseudoOODRegularizedEPNCLoss, ResponseUCEforEPNC, UCEforEPN
from epn_helper import (
    DEFAULT_DISCOUNT_ZETA_GRID,
    DISCOUNTED_BRANCH_SPECS,
    DiscountedSplitBundle,
    _artifact_baseline_metrics,
    _artifact_load_dir,
    _build_discounted_eud_payload,
    _build_gt_evidence_dicts,
    _build_selected_branch_result,
    _build_selected_scalar_result,
    _default_refusal_stats,
    _disambig_question_summary_metrics,
    _discount_selection_spec,
    _discounted_summary_metrics,
    _empty_branch_result_for_spec,
    _empty_exp_discounted_theils_u_resp_result,
    _evaluate_artifact_baselines,
    _evaluate_disambig_question_eud,
    _evaluate_discounted_branch_spec,
    _evaluate_discounted_eud,
    _evaluate_eud,
    _evaluate_exp_discounted_theils_u_resp_norm_seq_prob_dissonance_grid,
    _ensure_bitsandbytes_available,
    _load_split_artifacts_for,
    _load_baseline_data,
    _load_linear_probe_split,
    _log_metrics,
    _main_summary_metrics,
    _metric_dict_for_history,
    _prepare_discounted_split_bundle,
    _run_optional_baselines,
    normalize_zeta_grid,
    compute_ood_refusal_filter,
    count_or_frac,
)
from eud_helper import build_probe, positive_probability_from_logits
from feature_collector import _is_gemma_4_e4b_model_name, _needs_8bit_quantized_loading, _resolve_hf_model_name
from helper_functions import load_cluster_embeddings_npz
from uncertainty import dissonance, vacuity


WORKING_DIR = "./"
EPN_SUPPORTED_SEMANTIC_PROB_SOURCES = ("norm_seq_prob",)
EPN_SEMANTIC_PROB_SOURCE = "norm_seq_prob"
EPND_AUGMENTED_LOSS_NAME = "AugmentedResponseUCEforEPNC"
EPND_RESPONSE_LOSS_NAME = "ResponseUCEforEPNC"
EPND_UCE_LOSS_NAME = "UCEforEPN"
EPND_HALLUCINATION_BCE_LOSS_NAME = "HallucinationBCEWithLogitsLoss"
DENSITY_LOSS_ALIASES = {
    "uce": EPND_UCE_LOSS_NAME,
    EPND_UCE_LOSS_NAME.lower(): EPND_UCE_LOSS_NAME,
    "bce": EPND_HALLUCINATION_BCE_LOSS_NAME,
    "bcewithlogitsloss": EPND_HALLUCINATION_BCE_LOSS_NAME,
    EPND_HALLUCINATION_BCE_LOSS_NAME.lower(): EPND_HALLUCINATION_BCE_LOSS_NAME,
    EPND_AUGMENTED_LOSS_NAME.lower(): EPND_AUGMENTED_LOSS_NAME,
    EPND_RESPONSE_LOSS_NAME.lower(): EPND_RESPONSE_LOSS_NAME,
}
DENSITY_LOSS_CHOICES = (
    EPND_AUGMENTED_LOSS_NAME,
    EPND_RESPONSE_LOSS_NAME,
    EPND_UCE_LOSS_NAME,
    EPND_HALLUCINATION_BCE_LOSS_NAME,
    "BCEWithLogitsLoss",
    "BCE",
    "bce",
)
RHO_ACTIVATION = "sigmoid"
EPN_D_DEFAULT_ZETA_GRID = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0)
SEMANTIC_SUPERVISION_TARGET_CHOICES = (
    "raw",
    "norm",
    "norm_seq_prob",
    "raw_norm_seq_prob",
)
AUX_RELIABILITY_EPS = 1e-6
PSEUDO_OOD_DATASET_MAP = {
    "alcuna_trivia_qa": "temporalood",
    "temporalood_trivia_qa": "alcuna",
}
PSEUDO_OOD_FIXED_N_SAMPLES = 2000
PSEUDO_OOD_EXPECTED_SPLIT_COUNTS = {
    "train": 1600,
    "valid": 200,
    "test": 200,
}


@dataclass
class SplitBundle:
    split: str
    artifacts: Any
    X: Dict[str, np.ndarray]
    response_slt_all: Optional[List[Dict[int, np.ndarray]]]
    response_slt_path: Optional[str]
    sampled_res_y: Optional[np.ndarray]
    sampled_res_y_path: Optional[str]


@dataclass
class PseudoOODSplitBundle:
    split: str
    dataset_name: str
    load_dir: str
    split_dir: str
    X: Optional[Dict[str, np.ndarray]]
    labels: np.ndarray
    y: np.ndarray


@dataclass
class AuxiliaryProbeResult:
    clf: Any
    constant_prob: Optional[float]
    feature_dim: int
    probe_target: str
    probe_feature: str
    probe_type: str
    probe_label_source: str
    train_count: int
    eval_count: int
    AUROC: float
    AUPR: float
    scores: np.ndarray
    eval_labels: np.ndarray
    probe_converged: bool
    probe_n_iter: int
    temperature: float = 1.0


@dataclass
class EPNTable:
    split: str
    semantic_prob_source: str
    semantic_supervision_target: str
    question_count: int
    cluster_ids: np.ndarray
    semantic_probs: np.ndarray
    semantic_supports: np.ndarray
    semantic_support_totals: np.ndarray
    cluster_targets: np.ndarray
    augmented_targets: Optional[np.ndarray]
    cluster_mask: np.ndarray
    K: np.ndarray
    question_features: Optional[np.ndarray]
    response_features: Optional[List[np.ndarray]]
    response_supports: Optional[List[np.ndarray]]
    response_aux_probs: Optional[List[np.ndarray]]
    response_cluster_index: Optional[List[np.ndarray]]
    labels: np.ndarray
    y: np.ndarray
    label_source: str
    metric_eval_mask: np.ndarray
    ood_refusal_eval: Dict[str, object]
    semantic_support_zero_count: int = 0
    semantic_prob_zero_denominator_count: int = 0
    pseudo_ood_mask: Optional[np.ndarray] = None
    uce_targets: Optional[np.ndarray] = None


class ScalarProbe(torch.nn.Module):
    def __init__(self, probe_type: str, input_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        kind = str(probe_type).strip().lower()
        if kind not in {"lr", "mlp"}:
            raise ValueError(f"Unsupported probe type: {probe_type!r}.")
        self.probe_type = kind
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        if kind == "lr":
            self.net = torch.nn.Linear(self.input_dim, 1)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def seed_everything(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)


def load_X_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].astype(np.float32, copy=False) for key in payload.files}


def normalize_feature_kind(feature: str) -> str:
    norm = str(feature or "tbg").strip().lower()
    if norm not in {"tbg", "slt"}:
        raise ValueError(f"feature must be 'tbg' or 'slt', got {feature!r}.")
    return norm


def normalize_probe_target(target: str) -> str:
    norm = str(target or "hallucination").strip().lower()
    if norm not in {"hallucination", "ambiguity"}:
        raise ValueError(f"probe target must be 'hallucination' or 'ambiguity', got {target!r}.")
    return norm


def normalize_epn_semantic_prob_source(source: str) -> str:
    norm = str(source or EPN_SEMANTIC_PROB_SOURCE).strip().lower().replace("-", "_")
    aliases = {
        "norm_sequence_prob": "norm_seq_prob",
        "normalized_sequence_prob": "norm_seq_prob",
        "norm_seq_prob": "norm_seq_prob",
    }
    norm = aliases.get(norm, norm)
    if norm not in EPN_SUPPORTED_SEMANTIC_PROB_SOURCES:
        raise ValueError(
            f"Unsupported EPN semantic probability source {source!r}; "
            f"expected one of {list(EPN_SUPPORTED_SEMANTIC_PROB_SOURCES)}."
        )
    return norm


def normalize_semantic_supervision_target(target: str) -> str:
    norm = str(target or "norm").strip().lower().replace("-", "_")
    aliases = {
        "raw": "raw",
        "mean": "raw",
        "norm": "norm",
        "normalized": "norm",
        "norm_seq_prob": "norm_seq_prob",
        "norm_sequence_prob": "norm_seq_prob",
        "normalized_sequence_prob": "norm_seq_prob",
        "raw_norm_seq_prob": "raw_norm_seq_prob",
        "raw_norm_sequence_prob": "raw_norm_seq_prob",
        "raw_normalized_sequence_prob": "raw_norm_seq_prob",
    }
    norm = aliases.get(norm, norm)
    if norm not in SEMANTIC_SUPERVISION_TARGET_CHOICES:
        raise ValueError(
            f"Unsupported semantic supervision target {target!r}; "
            f"expected one of {list(SEMANTIC_SUPERVISION_TARGET_CHOICES)}."
        )
    return norm


def normalize_density_activation(activation: str) -> str:
    norm = str(activation or RHO_ACTIVATION).strip().lower()
    if norm != RHO_ACTIVATION:
        print(
            f"[EPN-D] Ignoring --activation={activation!r}; rho activation is forced to sigmoid.",
            flush=True,
        )
    return RHO_ACTIVATION


def density_rho_from_logits(logits: torch.Tensor, activation: str = RHO_ACTIVATION) -> torch.Tensor:
    normalize_density_activation(activation)
    return torch.sigmoid(logits)


class HallucinationBCEWithLogitsLoss(torch.nn.Module):
    """Masked BCE for the EPN-D hallucination-probe ablation.

    The positive class is hallucination, so ``sigmoid(logit)`` is P(hall).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        self.reduction = str(reduction)

    def forward(
        self,
        response_logits: torch.Tensor,
        response_hallucination_targets: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        if response_logits.dim() != 2:
            raise ValueError(
                "HallucinationBCEWithLogitsLoss expects response_logits with shape "
                f"[B, R], got {tuple(response_logits.shape)}."
            )
        if response_hallucination_targets.shape != response_logits.shape:
            raise ValueError(
                "Hallucination BCE target shape mismatch: "
                f"targets={tuple(response_hallucination_targets.shape)}, "
                f"logits={tuple(response_logits.shape)}."
            )
        if response_mask.shape != response_logits.shape:
            raise ValueError(
                "Hallucination BCE mask shape mismatch: "
                f"mask={tuple(response_mask.shape)}, logits={tuple(response_logits.shape)}."
            )

        dtype = response_logits.dtype
        targets = torch.clamp(response_hallucination_targets.to(dtype=dtype), min=0.0, max=1.0)
        mask = response_mask.to(dtype=dtype)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            response_logits,
            targets,
            reduction="none",
        ) * mask
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.sum() / torch.clamp(mask.sum(), min=1.0)


def normalize_density_loss_name(loss_name: str) -> str:
    raw = str(loss_name or EPND_AUGMENTED_LOSS_NAME).strip()
    norm = DENSITY_LOSS_ALIASES.get(raw.lower())
    if norm is None:
        raise ValueError(
            f"Unsupported density loss {loss_name!r}; expected one of {list(DENSITY_LOSS_CHOICES)}."
        )
    return norm


def is_hallucination_bce_loss_name(loss_name: str) -> bool:
    return normalize_density_loss_name(loss_name) == EPND_HALLUCINATION_BCE_LOSS_NAME


def is_hallucination_bce_loss(criterion: torch.nn.Module) -> bool:
    base = getattr(criterion, "base_loss", criterion)
    return isinstance(base, HallucinationBCEWithLogitsLoss)


def is_uce_for_epn_loss(criterion: torch.nn.Module) -> bool:
    base = getattr(criterion, "base_loss", criterion)
    return isinstance(base, UCEforEPN)


def build_density_loss(
    loss_name: str,
    lambda_coef: float,
    semantic_prob_source: str = EPN_SEMANTIC_PROB_SOURCE,
    use_pseudo_ood: bool = False,
) -> torch.nn.Module:
    normalize_epn_semantic_prob_source(semantic_prob_source)
    loss_norm = normalize_density_loss_name(loss_name)
    base_loss: Optional[torch.nn.Module] = None
    if loss_norm == EPND_AUGMENTED_LOSS_NAME:
        base_loss = AugmentedResponseUCEforEPNC(reduction="mean")
    elif loss_norm == EPND_RESPONSE_LOSS_NAME:
        base_loss = ResponseUCEforEPNC(reduction="mean")
    elif loss_norm == EPND_UCE_LOSS_NAME:
        base_loss = UCEforEPN(reduction="mean")
    elif loss_norm == EPND_HALLUCINATION_BCE_LOSS_NAME:
        if bool(use_pseudo_ood):
            raise ValueError(
                "HallucinationBCEWithLogitsLoss requires use_pseudo_OOD=False; "
                "pseudo-OOD rows would otherwise be treated as hallucination-positive labels."
            )
        return HallucinationBCEWithLogitsLoss(reduction="mean")
    if base_loss is None:
        raise ValueError(
            f"Unsupported density loss {loss_name!r}; expected one of {list(DENSITY_LOSS_CHOICES)}."
        )
    if bool(use_pseudo_ood):
        return PseudoOODRegularizedEPNCLoss(
            base_loss=base_loss,
            lambda_coef=float(lambda_coef),
        )
    return base_loss


def normalize_layer_range(layer_range: str) -> str:
    norm = str(layer_range or "last_5").strip().lower()
    if norm not in {"last_1", "last_5", "all"}:
        raise ValueError(f"Unsupported layer_range={layer_range!r}.")
    return norm


def selected_layer_indices(X_hidden: np.ndarray, layer_range: str) -> np.ndarray:
    X_arr = np.asarray(X_hidden, dtype=np.float32)
    if X_arr.ndim != 3:
        raise ValueError(f"Expected hidden states with shape [L, N, D], got {X_arr.shape}.")
    n_layers = int(X_arr.shape[0])
    mode = normalize_layer_range(layer_range)
    if mode == "all":
        return np.arange(n_layers, dtype=np.int64)
    if mode == "last_1":
        return np.arange(max(0, n_layers - 1), n_layers, dtype=np.int64)
    k = min(5, n_layers)
    return np.arange(n_layers - k, n_layers, dtype=np.int64)


def layer_range_to_last_k(X_hidden: np.ndarray, layer_range: str) -> int:
    return int(selected_layer_indices(X_hidden, layer_range).shape[0])


def normalize_flat_features(X_flat: np.ndarray) -> np.ndarray:
    arr = np.asarray(X_flat, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


def flatten_question_features(
    X_hidden: np.ndarray,
    layer_range: str,
    *,
    split: str,
    feature: str,
) -> np.ndarray:
    X_arr = np.asarray(X_hidden, dtype=np.float32)
    idx = selected_layer_indices(X_arr, layer_range)
    X_sel = X_arr[idx, :, :]
    X_flat = np.transpose(X_sel, (1, 0, 2)).reshape(X_sel.shape[1], -1)
    if not np.all(np.isfinite(X_flat)):
        X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    if X_flat.shape[0] == 0:
        raise ValueError(f"No question features built for split={split}, feature={feature}.")
    return normalize_flat_features(X_flat)


def response_slt_path(load_dir: str, split: str, n_resp: int) -> str:
    return os.path.join(load_dir, split, f"response_slt_lastk_{int(n_resp)}.npz")


def sampled_res_y_path(load_dir: str, split: str) -> str:
    return os.path.join(load_dir, split, "sampled_res_y.npy")


def load_split_bundle(args, split: str, need_response_slt: bool) -> SplitBundle:
    artifacts = _load_split_artifacts_for(args, str(args.dataset_name), split)
    X_path = os.path.join(artifacts.split_dir, "X.npz")
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Missing X artifact: {X_path}")
    X = load_X_npz(X_path)

    response_slt_all = None
    response_slt_file = None
    if need_response_slt:
        response_slt_file = response_slt_path(artifacts.load_dir, split, int(args.n_resp))
        if not os.path.exists(response_slt_file):
            raise FileNotFoundError(f"Missing response-level SLT artifact: {response_slt_file}")
        response_slt_all = load_cluster_embeddings_npz(response_slt_file)

    sampled_res_y = None
    sampled_file = sampled_res_y_path(artifacts.load_dir, split)
    if os.path.exists(sampled_file):
        sampled_res_y = np.asarray(np.load(sampled_file), dtype=np.int64)

    return SplitBundle(
        split=split,
        artifacts=artifacts,
        X=X,
        response_slt_all=response_slt_all,
        response_slt_path=response_slt_file,
        sampled_res_y=sampled_res_y,
        sampled_res_y_path=sampled_file if os.path.exists(sampled_file) else None,
    )


def normalize_pseudo_ood_dataset_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    norm = str(name).strip().lower()
    return norm if norm else None


def pseudo_ood_enabled_for_args(args) -> bool:
    dataset_name = str(getattr(args, "dataset_name", "") or "").strip().lower()
    return bool(getattr(args, "use_pseudo_OOD", False)) and dataset_name in PSEUDO_OOD_DATASET_MAP


def validate_pseudo_ood_args(args) -> None:
    args.use_pseudo_OOD = bool(getattr(args, "use_pseudo_OOD", False))
    args.pseudo_OOD_dataset_name = normalize_pseudo_ood_dataset_name(
        getattr(args, "pseudo_OOD_dataset_name", None)
    )
    dataset_name = str(getattr(args, "dataset_name", "") or "").strip().lower()
    if not args.use_pseudo_OOD or dataset_name not in PSEUDO_OOD_DATASET_MAP:
        return
    expected = PSEUDO_OOD_DATASET_MAP[dataset_name]
    if args.pseudo_OOD_dataset_name != expected:
        raise ValueError(
            f"dataset_name={args.dataset_name!r} requires "
            f"pseudo_OOD_dataset_name={expected!r} when use_pseudo_OOD=True; "
            f"got {args.pseudo_OOD_dataset_name!r}."
        )


def pseudo_ood_artifact_load_dir(args) -> str:
    model_id = str(args.model_name).rstrip("/").split("/")[-1]
    pseudo_dataset_name = normalize_pseudo_ood_dataset_name(args.pseudo_OOD_dataset_name)
    if pseudo_dataset_name is None:
        raise ValueError("pseudo_OOD_dataset_name is required for pseudo-OOD artifact loading.")
    return _artifact_load_dir(
        model_id,
        pseudo_dataset_name,
        PSEUDO_OOD_FIXED_N_SAMPLES,
        str(args.judge_type),
    )


def load_pseudo_ood_split_bundle(
    args,
    split: str,
    *,
    load_features: bool,
) -> PseudoOODSplitBundle:
    split_name = str(split)
    if split_name not in PSEUDO_OOD_EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unsupported pseudo-OOD split={split!r}.")
    expected_count = int(PSEUDO_OOD_EXPECTED_SPLIT_COUNTS[split_name])
    load_dir = pseudo_ood_artifact_load_dir(args)
    split_dir = os.path.join(load_dir, split_name)
    X_path = os.path.join(split_dir, "X.npz")
    label_path = os.path.join(split_dir, "ood_label.npy")
    y_path = os.path.join(split_dir, "y.npy")
    missing = [path for path in (X_path, label_path, y_path) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing pseudo-OOD files: " + ", ".join(missing))

    labels = np.asarray(np.load(label_path), dtype=np.int64).reshape(-1)
    y = np.asarray(np.load(y_path), dtype=np.int64).reshape(-1)
    if labels.shape[0] != expected_count:
        raise ValueError(
            f"Pseudo-OOD {split_name} label count mismatch at {label_path}: "
            f"got {labels.shape[0]}, expected {expected_count}."
        )
    if y.shape[0] != expected_count:
        raise ValueError(
            f"Pseudo-OOD {split_name} y count mismatch at {y_path}: "
            f"got {y.shape[0]}, expected {expected_count}."
        )
    if not np.all(labels == 1):
        raise ValueError(
            f"Pseudo-OOD {split_name} labels must all be OOD-positive in {label_path}; "
            f"found labels={np.unique(labels).tolist()}."
        )

    X = None
    if load_features:
        X = load_X_npz(X_path)
        for key, value in X.items():
            arr = np.asarray(value)
            if arr.ndim < 2 or int(arr.shape[1]) != expected_count:
                raise ValueError(
                    f"Pseudo-OOD X['{key}'] question count mismatch at {X_path}: "
                    f"shape={arr.shape}, expected second dim {expected_count}."
                )

    return PseudoOODSplitBundle(
        split=split_name,
        dataset_name=str(args.pseudo_OOD_dataset_name),
        load_dir=load_dir,
        split_dir=split_dir,
        X=X,
        labels=labels,
        y=y,
    )


def load_pseudo_ood_bundles(args) -> Dict[str, PseudoOODSplitBundle]:
    if not pseudo_ood_enabled_for_args(args):
        return {}
    bundles = {
        "train": load_pseudo_ood_split_bundle(args, "train", load_features=True),
        "valid": load_pseudo_ood_split_bundle(args, "valid", load_features=True),
        "test": load_pseudo_ood_split_bundle(args, "test", load_features=False),
    }
    return bundles


def response_slt_matrices(bundle: SplitBundle, layer_range: str) -> List[np.ndarray]:
    if bundle.response_slt_all is None:
        raise ValueError(f"Missing response SLT features for split={bundle.split}.")
    if "slt" not in bundle.X:
        raise ValueError(f"Missing X['slt'] for split={bundle.split}.")

    X_slt = np.asarray(bundle.X["slt"], dtype=np.float32)
    hidden_size = int(X_slt.shape[2])
    total_layers = int(X_slt.shape[0])
    requested = selected_layer_indices(X_slt, layer_range)

    matrices: List[np.ndarray] = []
    for qid, (resp_row, slt_row) in enumerate(zip(bundle.artifacts.responses, bundle.response_slt_all)):
        n_resp = len(resp_row)
        row_vectors: List[np.ndarray] = []
        for resp_id in range(n_resp):
            if int(resp_id) not in slt_row:
                raise ValueError(
                    f"Missing response SLT vector at split={bundle.split}, qid={qid}, resp_id={resp_id}."
                )
            vec = np.asarray(slt_row[int(resp_id)], dtype=np.float32).reshape(-1)
            if vec.size % hidden_size != 0:
                raise ValueError(
                    f"response_slt_lastk dim is not divisible by hidden size at split={bundle.split}, "
                    f"qid={qid}, resp_id={resp_id}: dim={vec.size}, hidden_size={hidden_size}."
                )
            stored_layers = int(vec.size // hidden_size)
            if stored_layers == total_layers:
                local_idx = requested
            else:
                start_global = total_layers - stored_layers
                if np.any(requested < start_global):
                    raise ValueError(
                        "Requested SLT layers are unavailable in response_slt_lastk artifacts: "
                        f"split={bundle.split}, qid={qid}, requested={requested.tolist()}, "
                        f"available_global={list(range(start_global, total_layers))}."
                    )
                local_idx = requested - start_global
            row_vectors.append(vec.reshape(stored_layers, hidden_size)[local_idx, :].reshape(-1))
        if row_vectors:
            matrices.append(normalize_flat_features(np.vstack(row_vectors)))
        else:
            matrices.append(np.zeros((0, int(requested.shape[0]) * hidden_size), dtype=np.float32))
    return matrices


def load_question_probe_labels(bundle: SplitBundle, probe_target: str) -> Tuple[np.ndarray, str, Optional[str]]:
    target = normalize_probe_target(probe_target)
    y = np.asarray(bundle.artifacts.y, dtype=np.int64).reshape(-1)
    if target == "hallucination":
        if np.any((y < 0) | (y > 1)):
            raise ValueError(f"Expected y in {{0,1}} for split={bundle.split}, got {np.unique(y).tolist()}.")
        return np.asarray(1 - y, dtype=np.int64), "1-y", os.path.join(bundle.artifacts.split_dir, "y.npy")

    path = os.path.join(bundle.artifacts.split_dir, "ambig_label.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"aux_probe='ambiguity' requires {path}")
    return np.asarray(np.load(path), dtype=np.int64).reshape(-1), "ambig_label.npy", path


def validate_ambiguity_artifacts(args, splits: Sequence[str]) -> None:
    if normalize_probe_target(args.aux_probe) != "ambiguity":
        return
    model_id = str(args.model_name).rstrip("/").split("/")[-1]
    load_dir = _artifact_load_dir(model_id, str(args.dataset_name), int(args.n_samples), str(args.judge_type))
    missing = []
    for split in splits:
        path = os.path.join(load_dir, split, "ambig_label.npy")
        if not os.path.exists(path):
            missing.append(f"{split}: {path}")
    if missing:
        raise FileNotFoundError("aux_probe='ambiguity' requires ambig_label.npy. Missing: " + ", ".join(missing))


def build_aux_training_data(
    args,
    bundle: SplitBundle,
    response_slt_by_split: Dict[str, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, str]:
    aux_target = normalize_probe_target(args.aux_probe)
    aux_feature = normalize_feature_kind(args.aux_probe_feature)
    labels_q, label_source, _ = load_question_probe_labels(bundle, aux_target)
    if aux_feature not in bundle.X:
        raise ValueError(f"aux_probe_feature={aux_feature!r} requires X['{aux_feature}'] for split={bundle.split}.")
    X = flatten_question_features(bundle.X[aux_feature], args.layer_range, split=bundle.split, feature=aux_feature)
    if X.shape[0] != labels_q.shape[0]:
        raise ValueError(f"Aux question feature/label mismatch for split={bundle.split}.")
    return X, labels_q.astype(np.int64, copy=False), label_source


def fit_auxiliary_probe(
    args,
    train_bundle: SplitBundle,
    eval_bundle: SplitBundle,
    response_slt_by_split: Dict[str, List[np.ndarray]],
    metric_eval_mask: np.ndarray,
) -> AuxiliaryProbeResult:
    X_train, y_train, label_source = build_aux_training_data(args, train_bundle, response_slt_by_split)
    X_eval, y_eval, _ = build_aux_training_data(args, eval_bundle, response_slt_by_split)
    if X_train.shape[1] != X_eval.shape[1]:
        raise ValueError(f"Aux feature dimension mismatch: train={X_train.shape[1]}, eval={X_eval.shape[1]}.")

    unique = np.unique(y_train)
    if unique.size < 2:
        clf = None
        constant_prob = float(unique[0]) if unique.size == 1 else 0.0
        scores = np.full(y_eval.shape[0], constant_prob, dtype=np.float64)
        converged = False
        n_iter = 0
    else:
        clf = build_probe(
            probe_type=str(args.aux_probe_type),
            input_dim=int(X_train.shape[1]),
            random_state=int(args.seed),
        )
        clf.fit(X_train, y_train)
        logits = clf.predict_logits(X_eval)
        scores = positive_probability_from_logits(logits)
        constant_prob = None
        converged = bool(getattr(clf, "converged_", False))
        n_iter = int(getattr(clf, "n_iter_", 0))

    mask = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != y_eval.shape[0]:
        mask = np.ones(y_eval.shape[0], dtype=bool)
    auroc, aupr = safe_binary_metrics(y_eval[mask], scores[mask])
    return AuxiliaryProbeResult(
        clf=clf,
        constant_prob=constant_prob,
        feature_dim=int(X_train.shape[1]),
        probe_target=normalize_probe_target(args.aux_probe),
        probe_feature=normalize_feature_kind(args.aux_probe_feature),
        probe_type=str(args.aux_probe_type),
        probe_label_source=str(label_source),
        train_count=int(y_train.shape[0]),
        eval_count=int(y_eval.shape[0]),
        AUROC=float(auroc),
        AUPR=float(aupr),
        scores=np.asarray(scores, dtype=np.float64),
        eval_labels=np.asarray(y_eval, dtype=np.int64),
        probe_converged=bool(converged),
        probe_n_iter=int(n_iter),
    )


def _predict_aux_probs(clf: Any, constant_prob: Optional[float], X: np.ndarray) -> np.ndarray:
    if constant_prob is not None or clf is None:
        return np.full(X.shape[0], float(0.0 if constant_prob is None else constant_prob), dtype=np.float64)
    logits = clf.predict_logits(X)
    return np.asarray(positive_probability_from_logits(logits), dtype=np.float64)


def predict_aux_reliability_by_question(
    args,
    aux_result: AuxiliaryProbeResult,
    bundle: SplitBundle,
    response_slt_by_split: Dict[str, List[np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    aux_feature = normalize_feature_kind(args.aux_probe_feature)
    prob_by_question: List[np.ndarray] = []
    reliability_by_question: List[np.ndarray] = []

    if aux_feature == "slt":
        if bundle.split not in response_slt_by_split:
            raise ValueError(
                f"aux_probe_feature='slt' requires response-level SLT features for split={bundle.split}."
            )
        response_matrices = response_slt_by_split[bundle.split]
        if len(response_matrices) != len(bundle.artifacts.responses):
            raise ValueError(
                f"Response-level SLT question count mismatch for split={bundle.split}: "
                f"features={len(response_matrices)}, responses={len(bundle.artifacts.responses)}."
            )
        for qid, (resp_row, X_resp) in enumerate(zip(bundle.artifacts.responses, response_matrices)):
            n_resp = len(resp_row)
            X_arr = np.asarray(X_resp, dtype=np.float32)
            if X_arr.ndim != 2:
                raise ValueError(
                    f"Response-level SLT features must be rank-2 at split={bundle.split}, "
                    f"qid={qid}, got {X_arr.shape}."
                )
            if X_arr.shape[0] != n_resp:
                raise ValueError(
                    f"Response-level SLT response count mismatch at split={bundle.split}, qid={qid}: "
                    f"features={X_arr.shape[0]}, responses={n_resp}."
                )
            if n_resp > 0 and X_arr.shape[1] != int(aux_result.feature_dim):
                raise ValueError(
                    f"Response-level SLT feature dim mismatch at split={bundle.split}, qid={qid}: "
                    f"features={X_arr.shape[1]}, aux_probe_dim={int(aux_result.feature_dim)}. "
                    "Use a layer_range compatible with response_slt_lastk artifacts."
                )
            prob_vec = _predict_aux_probs(aux_result.clf, aux_result.constant_prob, X_arr)
            prob_vec = np.clip(prob_vec, 0.0, 1.0)
            rel_vec = np.clip(1.0 - prob_vec, AUX_RELIABILITY_EPS, 1.0)
            prob_by_question.append(prob_vec.astype(np.float32, copy=False))
            reliability_by_question.append(rel_vec.astype(np.float32, copy=False))
        return prob_by_question, reliability_by_question

    if aux_feature not in bundle.X:
        raise ValueError(f"aux_probe_feature={aux_feature!r} requires X['{aux_feature}'] for split={bundle.split}.")
    X_q = flatten_question_features(bundle.X[aux_feature], args.layer_range, split=bundle.split, feature=aux_feature)
    probs_q = _predict_aux_probs(aux_result.clf, aux_result.constant_prob, X_q)
    for qid, resp_row in enumerate(bundle.artifacts.responses):
        n_resp = len(resp_row)
        prob_vec = np.full(n_resp, float(probs_q[qid]), dtype=np.float64)
        prob_vec = np.clip(prob_vec, 0.0, 1.0)
        rel_vec = np.clip(1.0 - prob_vec, AUX_RELIABILITY_EPS, 1.0)
        prob_by_question.append(prob_vec.astype(np.float32, copy=False))
        reliability_by_question.append(rel_vec.astype(np.float32, copy=False))
    return prob_by_question, reliability_by_question


def response_norm_sequence_prob_weights(
    bundle: SplitBundle,
    qid: int,
    source: str,
) -> np.ndarray:
    attr = "response_norm_sequence_probs"
    if not hasattr(bundle.artifacts, attr):
        raise AttributeError(f"Split artifacts for split={bundle.split} do not expose {attr}.")
    weights = np.asarray(getattr(bundle.artifacts, attr)[int(qid)], dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            f"Non-finite {source} response probabilities at split={bundle.split}, qid={int(qid)}."
        )
    if np.any(weights < 0.0):
        raise ValueError(
            f"Negative {source} response probabilities at split={bundle.split}, qid={int(qid)}."
        )
    return weights


def response_weights_for_semantic_prob_source(
    bundle: SplitBundle,
    qid: int,
    semantic_prob_source: str,
) -> Optional[np.ndarray]:
    source = normalize_epn_semantic_prob_source(semantic_prob_source)
    if source == "count":
        return None
    return response_norm_sequence_prob_weights(bundle, qid, source)


def semantic_supervision_target_weights(
    bundle: SplitBundle,
    qid: int,
    target_mode: str,
) -> Optional[np.ndarray]:
    mode = normalize_semantic_supervision_target(target_mode)
    if mode not in {"norm_seq_prob", "raw_norm_seq_prob"}:
        return None
    return response_norm_sequence_prob_weights(
        bundle,
        qid,
        f"semantic_supervision_target={mode}",
    )


def build_epn_table(
    args,
    bundle: SplitBundle,
    aux_prob_by_question: Sequence[np.ndarray],
    reliability_by_question: Sequence[np.ndarray],
    response_slt_by_split: Dict[str, List[np.ndarray]],
    semantic_prob_source: str = EPN_SEMANTIC_PROB_SOURCE,
) -> EPNTable:
    prob_source = normalize_epn_semantic_prob_source(semantic_prob_source)
    target_mode = normalize_semantic_supervision_target(
        getattr(args, "semantic_supervision_target", "norm")
    )
    primary_y = np.asarray(bundle.artifacts.y, dtype=np.int64).reshape(-1)
    if primary_y.shape[0] != len(bundle.artifacts.semantic_ids):
        raise ValueError(
            f"Primary y/semantic_ids mismatch at split={bundle.split}: "
            f"y={primary_y.shape[0]}, semantic_ids={len(bundle.artifacts.semantic_ids)}."
        )
    if np.any((primary_y < 0) | (primary_y > 1)):
        raise ValueError(f"Expected primary y in {{0,1}} for split={bundle.split}.")
    cluster_rows: List[np.ndarray] = []
    prob_rows: List[np.ndarray] = []
    support_rows: List[np.ndarray] = []
    support_total_rows: List[float] = []
    target_rows: List[np.ndarray] = []
    uce_target_rows: List[np.ndarray] = []
    unsupported_target_rows: List[float] = []
    response_support_rows: List[np.ndarray] = []
    response_aux_prob_rows: List[np.ndarray] = []
    response_cluster_rows: List[np.ndarray] = []
    K_vals: List[int] = []
    zero_support_count = 0

    for qid, ids_row in enumerate(bundle.artifacts.semantic_ids):
        ids = np.asarray(ids_row, dtype=np.int64).reshape(-1)
        aux_prob = np.asarray(aux_prob_by_question[qid], dtype=np.float32).reshape(-1)
        rel = np.asarray(reliability_by_question[qid], dtype=np.float32).reshape(-1)
        response_weights = response_weights_for_semantic_prob_source(bundle, qid, prob_source)
        target_weights = semantic_supervision_target_weights(bundle, qid, target_mode)
        if aux_prob.shape[0] != ids.shape[0]:
            raise ValueError(
                f"semantic_ids/aux_prob mismatch at split={bundle.split}, qid={qid}: "
                f"ids={ids.shape[0]}, aux_prob={aux_prob.shape[0]}."
            )
        if ids.shape[0] != rel.shape[0]:
            raise ValueError(
                f"semantic_ids/reliability mismatch at split={bundle.split}, qid={qid}: "
                f"ids={ids.shape[0]}, reliability={rel.shape[0]}."
            )
        if not np.all(np.isfinite(aux_prob)):
            raise ValueError(f"Non-finite aux hallucination probabilities at split={bundle.split}, qid={qid}.")
        aux_prob = np.clip(aux_prob, 0.0, 1.0).astype(np.float32, copy=False)
        if not np.all(np.isfinite(rel)):
            raise ValueError(f"Non-finite reliability values at split={bundle.split}, qid={qid}.")
        if response_weights is not None and response_weights.shape[0] != ids.shape[0]:
            raise ValueError(
                f"semantic_ids/{prob_source} mismatch at split={bundle.split}, qid={qid}: "
                f"ids={ids.shape[0]}, {prob_source}={response_weights.shape[0]}."
            )
        if target_weights is not None and target_weights.shape[0] != ids.shape[0]:
            raise ValueError(
                f"semantic_ids/semantic_supervision_target={target_mode} mismatch at "
                f"split={bundle.split}, qid={qid}: ids={ids.shape[0]}, "
                f"target_weights={target_weights.shape[0]}."
            )
        if ids.size == 0:
            cluster_rows.append(np.zeros(0, dtype=np.int64))
            prob_rows.append(np.zeros(0, dtype=np.float32))
            support_rows.append(np.zeros(0, dtype=np.float32))
            support_total_rows.append(0.0)
            target_rows.append(np.zeros(0, dtype=np.float32))
            uce_target_rows.append(np.zeros(0, dtype=np.float32))
            unsupported_target_rows.append(0.0)
            response_support_rows.append(np.zeros(0, dtype=np.float32))
            response_aux_prob_rows.append(np.zeros(0, dtype=np.float32))
            response_cluster_rows.append(np.zeros(0, dtype=np.int64))
            K_vals.append(0)
            continue
        unique_ids = np.asarray(sorted(np.unique(ids).tolist()), dtype=np.int64)
        cluster_to_local = {int(cid): idx for idx, cid in enumerate(unique_ids.tolist())}
        response_cluster_index = np.asarray(
            [cluster_to_local[int(cid)] for cid in ids.tolist()],
            dtype=np.int64,
        )
        probs = np.zeros(unique_ids.shape[0], dtype=np.float32)
        supports = np.zeros(unique_ids.shape[0], dtype=np.float32)
        targets = np.zeros(unique_ids.shape[0], dtype=np.float32)
        uce_targets = np.zeros(unique_ids.shape[0], dtype=np.float32)
        unsupported_target = 0.0
        support_total = float(unique_ids.shape[0])
        if ids.shape[0] > 0:
            response_support = np.full(
                ids.shape[0],
                float(unique_ids.shape[0]) / float(ids.shape[0]),
                dtype=np.float32,
            )
        else:
            response_support = np.zeros(0, dtype=np.float32)
        if response_weights is not None:
            support_total = float(np.sum(response_weights, dtype=np.float64))
            if support_total <= 0.0:
                zero_support_count += 1
            response_support = response_weights.astype(np.float32, copy=False)
        target_weight_vec = (
            np.ones(ids.shape[0], dtype=np.float32)
            if target_weights is None
            else np.asarray(target_weights, dtype=np.float32).reshape(-1)
        )
        unsupported_target = float(np.sum(target_weight_vec * aux_prob, dtype=np.float64))
        for idx, cid in enumerate(unique_ids.tolist()):
            mask = ids == int(cid)
            probs[idx] = float(np.mean(mask.astype(np.float32)))
            if response_weights is None:
                supports[idx] = float(unique_ids.shape[0]) * probs[idx]
            else:
                supports[idx] = float(np.sum(response_weights[mask], dtype=np.float64))
            targets[idx] = (
                float(np.sum(target_weight_vec[mask] * rel[mask], dtype=np.float64))
                if np.any(mask)
                else 0.0
            )
            if int(primary_y[qid]) == 1 and np.any(mask):
                uce_targets[idx] = float(np.sum(target_weight_vec[mask], dtype=np.float64))
        if target_mode in {"norm", "norm_seq_prob"}:
            target_total = float(np.sum(targets, dtype=np.float64) + unsupported_target)
            if target_total <= 0.0 or not np.isfinite(target_total):
                raise ValueError(
                    "Augmented supervision target normalization denominator is zero or non-finite "
                    f"at split={bundle.split}, qid={qid}, mode={target_mode}."
                )
            targets = (targets / target_total).astype(np.float32, copy=False)
            unsupported_target = float(unsupported_target / target_total)
        if int(primary_y[qid]) == 1:
            uce_total = float(np.sum(uce_targets, dtype=np.float64))
            if uce_total <= 0.0 or not np.isfinite(uce_total):
                raise ValueError(
                    "UCEforEPN target normalization denominator is zero or non-finite "
                    f"at split={bundle.split}, qid={qid}, mode={target_mode}."
                )
            uce_targets = (uce_targets / uce_total).astype(np.float32, copy=False)
        cluster_rows.append(unique_ids)
        prob_rows.append(probs)
        support_rows.append(supports)
        support_total_rows.append(support_total)
        target_rows.append(targets)
        uce_target_rows.append(uce_targets)
        unsupported_target_rows.append(float(unsupported_target))
        response_support_rows.append(response_support.astype(np.float32, copy=False))
        response_aux_prob_rows.append(aux_prob.astype(np.float32, copy=False))
        response_cluster_rows.append(response_cluster_index)
        K_vals.append(int(unique_ids.shape[0]))

    max_clusters = max(1, max(K_vals) if K_vals else 1)
    n_questions = len(cluster_rows)
    cluster_ids = np.full((n_questions, max_clusters), -1, dtype=np.int64)
    semantic_probs = np.zeros((n_questions, max_clusters), dtype=np.float32)
    semantic_supports = np.zeros((n_questions, max_clusters), dtype=np.float32)
    semantic_support_totals = np.asarray(support_total_rows, dtype=np.float32).reshape(-1)
    cluster_targets = np.zeros((n_questions, max_clusters), dtype=np.float32)
    augmented_targets = np.zeros((n_questions, max_clusters + 1), dtype=np.float32)
    uce_targets = np.zeros((n_questions, max_clusters + 1), dtype=np.float32)
    cluster_mask = np.zeros((n_questions, max_clusters), dtype=bool)
    for qid, (ids, probs, supports, targets, uce_target, unsupported_target) in enumerate(
        zip(cluster_rows, prob_rows, support_rows, target_rows, uce_target_rows, unsupported_target_rows)
    ):
        width = int(ids.shape[0])
        if width == 0:
            augmented_targets[qid, max_clusters] = float(unsupported_target)
            continue
        cluster_ids[qid, :width] = ids
        semantic_probs[qid, :width] = probs
        semantic_supports[qid, :width] = supports
        cluster_targets[qid, :width] = targets
        augmented_targets[qid, :width] = targets
        uce_targets[qid, :width] = uce_target
        augmented_targets[qid, max_clusters] = float(unsupported_target)
        cluster_mask[qid, :width] = True

    density_feature = normalize_feature_kind(args.density_probe_feature)
    question_features = None
    response_features = None
    if density_feature == "slt":
        if bundle.split not in response_slt_by_split:
            raise ValueError(
                f"density_probe_feature='slt' requires response-level SLT features for split={bundle.split}."
            )
        response_features = response_slt_by_split[bundle.split]
        if len(response_features) != n_questions:
            raise ValueError(
                f"Response-level density feature count mismatch for split={bundle.split}: "
                f"features={len(response_features)}, questions={n_questions}."
            )
        for qid, (features_q, support_q) in enumerate(zip(response_features, response_support_rows)):
            arr = np.asarray(features_q, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != support_q.shape[0]:
                raise ValueError(
                    f"Response-level density feature shape mismatch at split={bundle.split}, qid={qid}: "
                    f"features={arr.shape}, response_supports={support_q.shape}."
                )
    else:
        if density_feature not in bundle.X:
            raise ValueError(
                f"density_probe_feature={density_feature!r} requires X['{density_feature}'] for split={bundle.split}."
            )
        question_features = flatten_question_features(
            bundle.X[density_feature],
            args.layer_range,
            split=bundle.split,
            feature=density_feature,
        )

    metric_eval_mask, ood_refusal_eval = compute_ood_refusal_filter(
        labels=bundle.artifacts.labels,
        dataset_name=str(args.dataset_name),
        split=bundle.split,
        remove_refusal=bool(args.remove_refusal),
        load_dir=bundle.artifacts.load_dir,
    )

    return EPNTable(
        split=bundle.split,
        semantic_prob_source=prob_source,
        semantic_supervision_target=target_mode,
        question_count=n_questions,
        cluster_ids=cluster_ids,
        semantic_probs=semantic_probs,
        semantic_supports=semantic_supports,
        semantic_support_totals=semantic_support_totals,
        cluster_targets=cluster_targets,
        augmented_targets=augmented_targets,
        uce_targets=uce_targets,
        cluster_mask=cluster_mask,
        K=np.asarray(K_vals, dtype=np.float32),
        question_features=question_features,
        response_features=response_features,
        response_supports=response_support_rows,
        response_aux_probs=response_aux_prob_rows,
        response_cluster_index=response_cluster_rows,
        labels=np.asarray(bundle.artifacts.labels, dtype=np.int64).reshape(-1),
        y=np.asarray(bundle.artifacts.y, dtype=np.int64).reshape(-1),
        label_source=str(bundle.artifacts.label_source),
        metric_eval_mask=np.asarray(metric_eval_mask, dtype=bool).reshape(-1),
        ood_refusal_eval=ood_refusal_eval,
        semantic_support_zero_count=int(zero_support_count),
        semantic_prob_zero_denominator_count=int(zero_support_count),
        pseudo_ood_mask=np.zeros(n_questions, dtype=bool),
    )


def _pad_table_columns(matrix: np.ndarray, width: int, fill_value: float) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 table matrix, got shape {arr.shape}.")
    if int(arr.shape[1]) == int(width):
        return arr.copy()
    if int(arr.shape[1]) > int(width):
        raise ValueError(f"Cannot pad matrix from width {arr.shape[1]} down to {width}.")
    padded = np.full((arr.shape[0], int(width)), fill_value, dtype=arr.dtype)
    padded[:, : arr.shape[1]] = arr
    return padded


def _pad_augmented_targets(matrix: Optional[np.ndarray], semantic_width: int) -> np.ndarray:
    if matrix is None:
        return np.zeros((0, int(semantic_width) + 1), dtype=np.float32)
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Expected augmented targets with shape [N, C+1], got {arr.shape}.")
    old_width = int(arr.shape[1]) - 1
    if old_width == int(semantic_width):
        return arr.copy()
    if old_width > int(semantic_width):
        raise ValueError(f"Cannot pad augmented targets from width {old_width} down to {semantic_width}.")
    padded = np.zeros((arr.shape[0], int(semantic_width) + 1), dtype=np.float32)
    padded[:, :old_width] = arr[:, :old_width]
    padded[:, int(semantic_width)] = arr[:, old_width]
    return padded


def append_pseudo_ood_rows(
    args,
    table: EPNTable,
    pseudo_bundle: PseudoOODSplitBundle,
) -> EPNTable:
    if pseudo_bundle.X is None:
        raise ValueError(f"Pseudo-OOD split={pseudo_bundle.split} was loaded without features.")
    density_feature = normalize_feature_kind(args.density_probe_feature)
    if density_feature not in pseudo_bundle.X:
        raise ValueError(
            f"Pseudo-OOD density_probe_feature={density_feature!r} requires "
            f"X['{density_feature}'] in {pseudo_bundle.split_dir}/X.npz."
        )
    pseudo_features = flatten_question_features(
        pseudo_bundle.X[density_feature],
        args.layer_range,
        split=f"{pseudo_bundle.split}_pseudo_ood",
        feature=density_feature,
    )
    if table.question_features is not None:
        if pseudo_features.shape[1] != table.question_features.shape[1]:
            raise ValueError(
                "Pseudo-OOD feature dimension mismatch: "
                f"base={table.question_features.shape[1]}, pseudo={pseudo_features.shape[1]}."
            )
    else:
        if table.response_features is None:
            raise ValueError("Pseudo-OOD EPN-D response-level training requires response density features.")
        feature_dim = density_input_dim(table)
        if pseudo_features.shape[1] != feature_dim:
            raise ValueError(
                "Pseudo-OOD response-level feature dimension mismatch: "
                f"base={feature_dim}, pseudo={pseudo_features.shape[1]}."
            )

    n_pseudo = int(pseudo_features.shape[0])
    n_clusters = max(1, int(args.n_resp))
    width = max(int(table.cluster_ids.shape[1]), n_clusters)

    base_cluster_ids = _pad_table_columns(table.cluster_ids, width, -1)
    base_semantic_probs = _pad_table_columns(table.semantic_probs, width, 0.0)
    base_semantic_supports = _pad_table_columns(table.semantic_supports, width, 0.0)
    base_cluster_targets = _pad_table_columns(table.cluster_targets, width, 0.0)
    base_augmented_targets = _pad_augmented_targets(table.augmented_targets, width)
    base_uce_targets = (
        np.zeros((table.question_count, width + 1), dtype=np.float32)
        if table.uce_targets is None
        else _pad_augmented_targets(table.uce_targets, width)
    )
    base_cluster_mask = _pad_table_columns(table.cluster_mask, width, False)

    pseudo_cluster_ids = np.full((n_pseudo, width), -1, dtype=np.int64)
    pseudo_semantic_probs = np.zeros((n_pseudo, width), dtype=np.float32)
    pseudo_semantic_supports = np.zeros((n_pseudo, width), dtype=np.float32)
    pseudo_cluster_targets = np.zeros((n_pseudo, width), dtype=np.float32)
    pseudo_augmented_targets = np.zeros((n_pseudo, width + 1), dtype=np.float32)
    pseudo_uce_targets = np.zeros((n_pseudo, width + 1), dtype=np.float32)
    pseudo_cluster_mask = np.zeros((n_pseudo, width), dtype=bool)
    pseudo_cluster_ids[:, :n_clusters] = np.arange(n_clusters, dtype=np.int64)
    pseudo_semantic_probs[:, :n_clusters] = 1.0 / float(n_clusters)
    pseudo_semantic_supports[:, :n_clusters] = 1.0
    pseudo_cluster_mask[:, :n_clusters] = True
    pseudo_response_supports = [
        np.ones(n_clusters, dtype=np.float32)
        for _ in range(n_pseudo)
    ]
    pseudo_response_cluster_index = [
        np.arange(n_clusters, dtype=np.int64)
        for _ in range(n_pseudo)
    ]
    pseudo_response_aux_probs = [
        np.ones(n_clusters, dtype=np.float32)
        for _ in range(n_pseudo)
    ]

    if table.question_features is None:
        assert table.response_features is not None
        pseudo_response_features = [
            np.broadcast_to(
                pseudo_features[row_idx : row_idx + 1],
                (n_clusters, pseudo_features.shape[1]),
            ).copy()
            for row_idx in range(n_pseudo)
        ]
        question_features = None
        response_features = list(table.response_features) + pseudo_response_features
    else:
        question_features = np.concatenate([table.question_features, pseudo_features], axis=0)
        response_features = table.response_features

    table_pseudo_mask = (
        np.zeros(table.question_count, dtype=bool)
        if table.pseudo_ood_mask is None
        else np.asarray(table.pseudo_ood_mask, dtype=bool).reshape(-1)
    )
    if table_pseudo_mask.shape[0] != table.question_count:
        raise ValueError(
            f"Pseudo-OOD mask length mismatch for split={table.split}: "
            f"mask={table_pseudo_mask.shape[0]}, question_count={table.question_count}."
        )

    return EPNTable(
        split=table.split,
        semantic_prob_source=table.semantic_prob_source,
        semantic_supervision_target=table.semantic_supervision_target,
        question_count=int(table.question_count) + n_pseudo,
        cluster_ids=np.concatenate([base_cluster_ids, pseudo_cluster_ids], axis=0),
        semantic_probs=np.concatenate([base_semantic_probs, pseudo_semantic_probs], axis=0),
        semantic_supports=np.concatenate([base_semantic_supports, pseudo_semantic_supports], axis=0),
        semantic_support_totals=np.concatenate(
            [
                np.asarray(table.semantic_support_totals, dtype=np.float32).reshape(-1),
                np.full(n_pseudo, float(n_clusters), dtype=np.float32),
            ],
            axis=0,
        ),
        cluster_targets=np.concatenate([base_cluster_targets, pseudo_cluster_targets], axis=0),
        augmented_targets=np.concatenate([base_augmented_targets, pseudo_augmented_targets], axis=0),
        uce_targets=np.concatenate([base_uce_targets, pseudo_uce_targets], axis=0),
        cluster_mask=np.concatenate([base_cluster_mask, pseudo_cluster_mask], axis=0),
        K=np.concatenate(
            [
                np.asarray(table.K, dtype=np.float32).reshape(-1),
                np.full(n_pseudo, float(n_clusters), dtype=np.float32),
            ],
            axis=0,
        ),
        question_features=question_features,
        response_features=response_features,
        response_supports=list(table.response_supports or []) + pseudo_response_supports,
        response_aux_probs=list(table.response_aux_probs or []) + pseudo_response_aux_probs,
        response_cluster_index=list(table.response_cluster_index or []) + pseudo_response_cluster_index,
        labels=np.concatenate(
            [
                np.asarray(table.labels, dtype=np.int64).reshape(-1),
                np.asarray(pseudo_bundle.labels, dtype=np.int64).reshape(-1),
            ],
            axis=0,
        ),
        y=np.concatenate(
            [
                np.asarray(table.y, dtype=np.int64).reshape(-1),
                np.asarray(pseudo_bundle.y, dtype=np.int64).reshape(-1),
            ],
            axis=0,
        ),
        label_source=f"{table.label_source}+pseudo_ood:{pseudo_bundle.dataset_name}",
        metric_eval_mask=np.concatenate(
            [
                np.asarray(table.metric_eval_mask, dtype=bool).reshape(-1),
                np.zeros(n_pseudo, dtype=bool),
            ],
            axis=0,
        ),
        ood_refusal_eval=dict(table.ood_refusal_eval),
        semantic_support_zero_count=int(table.semantic_support_zero_count),
        semantic_prob_zero_denominator_count=int(table.semantic_prob_zero_denominator_count),
        pseudo_ood_mask=np.concatenate([table_pseudo_mask, np.ones(n_pseudo, dtype=bool)], axis=0),
    )


def safe_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores_arr)
    labels_arr = labels_arr[finite]
    scores_arr = scores_arr[finite]
    if labels_arr.size == 0 or np.unique(labels_arr).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels_arr, scores_arr)), float(average_precision_score(labels_arr, scores_arr))


def subjective_logic_projected_prob_from_evidence(evidence: np.ndarray) -> np.ndarray:
    evidence_arr = np.clip(np.asarray(evidence, dtype=np.float64).reshape(-1), 0.0, None)
    k_count = int(evidence_arr.shape[0])
    if k_count <= 0:
        return np.zeros(0, dtype=np.float32)

    evidence_total = float(np.sum(evidence_arr, dtype=np.float64))
    denom = evidence_total + float(k_count)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros(k_count, dtype=np.float32)

    belief = evidence_arr / denom
    uncertainty = float(k_count) / denom
    base_rate = 1.0 / float(k_count)
    projected = belief + base_rate * uncertainty
    return projected.astype(np.float32, copy=False)


def categorical_entropy(probs: np.ndarray) -> float:
    prob_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    prob_arr = prob_arr[np.isfinite(prob_arr) & (prob_arr > 0.0)]
    if prob_arr.size == 0:
        return 0.0
    return float(-np.sum(prob_arr * np.log(prob_arr), dtype=np.float64))


def density_input_dim(table: EPNTable) -> int:
    if table.question_features is not None:
        return int(table.question_features.shape[1])
    if table.response_features is None:
        raise ValueError("Density table has no features.")
    for mat in table.response_features:
        if mat.shape[0] > 0:
            return int(mat.shape[1])
    raise ValueError(f"No response density features found for split={table.split}.")


def density_logits_for_indices(
    model: torch.nn.Module,
    table: EPNTable,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if table.question_features is not None:
        xb = torch.from_numpy(table.question_features[idx_arr]).to(device)
        return model(xb)

    if table.response_features is None:
        raise ValueError("Missing response features for density probe.")
    mats = [table.response_features[int(i)] for i in idx_arr.tolist()]
    max_resp = max(1, max(int(mat.shape[0]) for mat in mats))
    feat_dim = density_input_dim(table)
    batch = np.zeros((idx_arr.shape[0], max_resp, feat_dim), dtype=np.float32)
    mask = np.zeros((idx_arr.shape[0], max_resp), dtype=np.float32)
    for row_idx, mat in enumerate(mats):
        n_resp = int(mat.shape[0])
        if n_resp == 0:
            continue
        batch[row_idx, :n_resp, :] = mat
        mask[row_idx, :n_resp] = 1.0
    xb = torch.from_numpy(batch.reshape(-1, feat_dim)).to(device)
    logits = model(xb).reshape(batch.shape[0], max_resp)
    return logits


def response_density_tensors_for_indices(
    table: EPNTable,
    indices: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if table.response_supports is None or table.response_aux_probs is None or table.response_cluster_index is None:
        raise ValueError("EPN-D requires response supports, aux probabilities, and cluster indices.")
    idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    support_rows = [np.asarray(table.response_supports[int(i)], dtype=np.float32).reshape(-1) for i in idx_arr.tolist()]
    aux_rows = [np.asarray(table.response_aux_probs[int(i)], dtype=np.float32).reshape(-1) for i in idx_arr.tolist()]
    cluster_rows = [np.asarray(table.response_cluster_index[int(i)], dtype=np.int64).reshape(-1) for i in idx_arr.tolist()]
    max_resp = max(1, max(int(row.shape[0]) for row in support_rows))
    supports = np.zeros((idx_arr.shape[0], max_resp), dtype=np.float32)
    aux_probs = np.zeros((idx_arr.shape[0], max_resp), dtype=np.float32)
    cluster_index = np.zeros((idx_arr.shape[0], max_resp), dtype=np.int64)
    response_mask = np.zeros((idx_arr.shape[0], max_resp), dtype=np.float32)
    for row_idx, (support_row, aux_row, cluster_row) in enumerate(zip(support_rows, aux_rows, cluster_rows)):
        if support_row.shape[0] != cluster_row.shape[0]:
            raise ValueError(
                f"Response support/cluster-index mismatch for split={table.split}, "
                f"qid={int(idx_arr[row_idx])}: supports={support_row.shape[0]}, clusters={cluster_row.shape[0]}."
            )
        if aux_row.shape[0] != support_row.shape[0]:
            raise ValueError(
                f"Response aux/support mismatch for split={table.split}, "
                f"qid={int(idx_arr[row_idx])}: aux={aux_row.shape[0]}, supports={support_row.shape[0]}."
            )
        n_resp = int(support_row.shape[0])
        if n_resp == 0:
            continue
        supports[row_idx, :n_resp] = support_row
        aux_probs[row_idx, :n_resp] = aux_row
        cluster_index[row_idx, :n_resp] = cluster_row
        response_mask[row_idx, :n_resp] = 1.0
    return (
        torch.from_numpy(supports).to(device),
        torch.from_numpy(aux_probs).to(device),
        torch.from_numpy(cluster_index).to(device),
        torch.from_numpy(response_mask).to(device),
    )


def rho_logits_by_response_from_logits(
    logits: torch.Tensor,
    response_supports: torch.Tensor,
) -> torch.Tensor:
    if logits.dim() == 1:
        return logits.reshape(-1, 1).expand_as(response_supports)
    if logits.dim() == 2:
        if logits.shape != response_supports.shape:
            raise ValueError(
                "Response support/rho-logit shape mismatch: "
                f"supports={tuple(response_supports.shape)}, logits={tuple(logits.shape)}."
            )
        return logits
    raise ValueError(f"EPN-D rho logits must have shape [B] or [B, R], got {tuple(logits.shape)}.")


def q_rel_logits_from_rho_logits(
    rho_logits: torch.Tensor,
    zeta: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = density_rho_from_logits(rho_logits)
    q_rel = torch.exp(-float(zeta) * rho)
    q_rel_clamped = torch.clamp(q_rel, min=float(eps), max=1.0 - float(eps))
    return torch.logit(q_rel_clamped), rho, q_rel


def response_cluster_evidence(
    rho: torch.Tensor,
    response_supports: torch.Tensor,
    zeta: float,
    response_cluster_index: torch.Tensor,
    response_mask: torch.Tensor,
    cluster_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rho.dim() == 1:
        rho_by_response = rho.reshape(-1, 1).expand_as(response_supports)
    elif rho.dim() == 2:
        rho_by_response = rho
    else:
        raise ValueError(f"EPN-D rho must have shape [B] or [B, R], got {tuple(rho.shape)}.")
    if response_supports.shape != rho_by_response.shape:
        raise ValueError(
            "Response support/rho shape mismatch: "
            f"supports={tuple(response_supports.shape)}, rho={tuple(rho_by_response.shape)}."
        )
    if response_cluster_index.shape != rho_by_response.shape:
        raise ValueError(
            "Response cluster-index/rho shape mismatch: "
            f"indices={tuple(response_cluster_index.shape)}, rho={tuple(rho_by_response.shape)}."
        )
    if response_mask.shape != rho_by_response.shape:
        raise ValueError(
            "Response mask/rho shape mismatch: "
            f"mask={tuple(response_mask.shape)}, rho={tuple(rho_by_response.shape)}."
        )
    if cluster_mask.dim() != 2 or cluster_mask.shape[0] != rho_by_response.shape[0]:
        raise ValueError(
            "Cluster mask/response rho batch mismatch: "
            f"cluster_mask={tuple(cluster_mask.shape)}, rho={tuple(rho_by_response.shape)}."
        )
    dtype = rho_by_response.dtype
    response_mask_t = response_mask.to(dtype=dtype)
    rho_prob = torch.clamp(rho_by_response.to(dtype=dtype), min=0.0, max=1.0)
    q_rel_by_response = torch.exp(-float(zeta) * rho_prob) * response_mask_t
    retained_by_response = (
        torch.clamp(response_supports.to(dtype=dtype), min=0.0)
        * q_rel_by_response
    )
    evidence = rho_by_response.new_zeros((rho_by_response.shape[0], cluster_mask.shape[1]))
    scatter_index = torch.clamp(response_cluster_index.to(dtype=torch.long), min=0)
    evidence.scatter_add_(1, scatter_index, retained_by_response)
    evidence = evidence * cluster_mask.to(dtype=dtype)
    return evidence, q_rel_by_response, rho_prob * response_mask_t, retained_by_response


def density_loss_input_for_indices(
    table: EPNTable,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    source = normalize_epn_semantic_prob_source(table.semantic_prob_source)
    matrix = table.semantic_supports if source == "norm_seq_prob" else table.semantic_probs
    return torch.from_numpy(matrix[np.asarray(indices, dtype=np.int64)]).to(device)


def pseudo_ood_mask_for_indices(
    table: EPNTable,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    idx = np.asarray(indices, dtype=np.int64)
    if table.pseudo_ood_mask is None:
        mask = np.zeros(idx.shape[0], dtype=bool)
    else:
        mask_arr = np.asarray(table.pseudo_ood_mask, dtype=bool).reshape(-1)
        if mask_arr.shape[0] != int(table.question_count):
            raise ValueError(
                f"Pseudo-OOD mask length mismatch for split={table.split}: "
                f"mask={mask_arr.shape[0]}, question_count={table.question_count}."
            )
        mask = mask_arr[idx]
    return torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)


def response_evidence_uce_per_question(
    evidence: torch.Tensor,
    targets: torch.Tensor,
    cluster_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if evidence.dim() != 2:
        raise ValueError(f"Response evidence UCE expects evidence with shape [B, C], got {tuple(evidence.shape)}.")
    if targets.shape != evidence.shape:
        raise ValueError(
            "Response evidence UCE target shape mismatch: "
            f"targets={tuple(targets.shape)}, evidence={tuple(evidence.shape)}."
        )
    if cluster_mask.shape != evidence.shape:
        raise ValueError(
            "Response evidence UCE mask shape mismatch: "
            f"mask={tuple(cluster_mask.shape)}, evidence={tuple(evidence.shape)}."
        )
    dtype = evidence.dtype
    mask = cluster_mask.to(dtype=dtype)
    evidence_pos = torch.clamp(evidence.to(dtype=dtype), min=0.0) * mask
    y_hat = targets.to(dtype=dtype)
    k_count = torch.clamp(mask.sum(dim=1), min=1.0)
    alpha_0 = k_count + evidence_pos.sum(dim=1)
    alpha_k = 1.0 + evidence_pos
    terms = y_hat * (
        torch.digamma(torch.clamp(alpha_0, min=float(eps))).unsqueeze(1)
        - torch.digamma(torch.clamp(alpha_k, min=float(eps)))
    )
    return torch.sum(terms * mask, dim=1)


def active_dirichlet_entropy_per_question(
    alpha: torch.Tensor,
    cluster_mask: torch.Tensor,
) -> torch.Tensor:
    if alpha.shape != cluster_mask.shape:
        raise ValueError(
            "Dirichlet entropy mask shape mismatch: "
            f"alpha={tuple(alpha.shape)}, mask={tuple(cluster_mask.shape)}."
        )
    mask = cluster_mask.to(dtype=torch.bool)
    entropy_vals: List[torch.Tensor] = []
    for row_alpha, row_mask in zip(alpha, mask):
        active_alpha = row_alpha[row_mask]
        if active_alpha.numel() == 0:
            entropy_vals.append(row_alpha.new_zeros(()))
        else:
            entropy_vals.append(torch.distributions.Dirichlet(active_alpha).entropy())
    return torch.stack(entropy_vals, dim=0)


def density_regularization_kind(criterion: torch.nn.Module) -> str:
    base = getattr(criterion, "base_loss", criterion)
    name = base.__class__.__name__
    if name.endswith("_Mag"):
        return "mag"
    if name.endswith("_KL"):
        return "kl"
    return "none"


def density_regularization_lambda(criterion: torch.nn.Module) -> float:
    base = getattr(criterion, "base_loss", criterion)
    if hasattr(base, "lambda_coef"):
        return float(base.lambda_coef)
    if hasattr(criterion, "lambda_coef"):
        return float(criterion.lambda_coef)
    return 0.0


def density_target_array_for_loss(criterion: torch.nn.Module, table: EPNTable) -> np.ndarray:
    target_matrix = table.uce_targets if is_uce_for_epn_loss(criterion) else table.augmented_targets
    target_name = "uce_targets" if is_uce_for_epn_loss(criterion) else "augmented_targets"
    if target_matrix is None:
        raise ValueError(f"EPN-D density training requires {target_name} for split={table.split}.")
    arr = np.asarray(target_matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != int(table.question_count):
        raise ValueError(
            f"EPN-D {target_name} shape mismatch for split={table.split}: "
            f"targets={arr.shape}, question_count={table.question_count}."
        )
    return arr


def density_targets_for_indices(
    criterion: torch.nn.Module,
    table: EPNTable,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    return torch.from_numpy(density_target_array_for_loss(criterion, table)[idx_arr]).to(device)


def response_density_batch_loss(
    criterion: torch.nn.Module,
    table: EPNTable,
    indices: np.ndarray,
    rho_logits: torch.Tensor,
    zeta: float,
    targets: Optional[torch.Tensor],
    cluster_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    response_supports, response_hall_probs, response_cluster_index, response_mask = response_density_tensors_for_indices(
        table,
        indices,
        device,
    )
    rho_logits_by_response = rho_logits_by_response_from_logits(rho_logits, response_supports)
    if is_hallucination_bce_loss(criterion):
        return criterion(rho_logits_by_response, response_hall_probs, response_mask)

    if targets is None:
        targets = density_targets_for_indices(criterion, table, indices, device)
    q_rel_logits, _, _ = q_rel_logits_from_rho_logits(rho_logits_by_response, float(zeta))
    pseudo_mask = (
        pseudo_ood_mask_for_indices(table, indices, device)
        if bool(getattr(criterion, "supports_pseudo_ood", False))
        else None
    )
    return (
        criterion(
            q_rel_logits,
            response_supports,
            response_cluster_index,
            response_mask,
            targets,
            cluster_mask,
            pseudo_mask,
        )
        if pseudo_mask is not None
        else criterion(
            q_rel_logits,
            response_supports,
            response_cluster_index,
            response_mask,
            targets,
            cluster_mask,
        )
    )


def density_batch_loss(
    criterion: torch.nn.Module,
    table: EPNTable,
    indices: np.ndarray,
    rho_logits: torch.Tensor,
    zeta: float,
    targets: Optional[torch.Tensor],
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return response_density_batch_loss(criterion, table, indices, rho_logits, zeta, targets, mask, device)


def table_loss(
    model: torch.nn.Module,
    table: EPNTable,
    criterion: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    zeta: float,
) -> float:
    density_target_array_for_loss(criterion, table)
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, table.question_count, int(batch_size)):
            idx = np.arange(start, min(start + int(batch_size), table.question_count), dtype=np.int64)
            logits = density_logits_for_indices(model, table, idx, device)
            mask = torch.from_numpy(table.cluster_mask[idx].astype(np.float32)).to(device)
            loss = density_batch_loss(criterion, table, idx, logits, float(zeta), None, mask, device)
            total_loss += float(loss.item()) * int(idx.shape[0])
            total_count += int(idx.shape[0])
    return total_loss / float(max(1, total_count))


def response_density_eval_tensors(
    table: EPNTable,
    indices: np.ndarray,
    rho_logits: torch.Tensor,
    zeta: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    response_supports, _, response_cluster_index, response_mask = response_density_tensors_for_indices(
        table,
        indices,
        device,
    )
    rho_logits_by_response = rho_logits_by_response_from_logits(rho_logits, response_supports)
    rho_by_response = density_rho_from_logits(rho_logits_by_response)
    evidence, q_rel, rho_by_response, retained = response_cluster_evidence(
        rho_by_response,
        response_supports,
        float(zeta),
        response_cluster_index,
        response_mask,
        torch.from_numpy(table.cluster_mask[np.asarray(indices, dtype=np.int64)].astype(np.float32)).to(device),
    )
    return rho_by_response, q_rel, retained, response_mask


def epn_d_selection_family(label_source: str) -> str:
    return "dissonance" if str(label_source).split("+", 1)[0] == "ambig_label.npy" else "vacuity"


def select_epn_d_zeta_index(
    grid_results: Sequence[Dict[str, float]],
    zeta_grid: Sequence[float],
    valid_labels: np.ndarray,
    valid_metric_eval_mask: np.ndarray,
) -> Tuple[int, str, str]:
    labels = np.asarray(valid_labels, dtype=np.int64).reshape(-1)
    mask = np.asarray(valid_metric_eval_mask, dtype=bool).reshape(-1)
    if labels.shape[0] != mask.shape[0]:
        raise ValueError(
            "EPN-D zeta selection label/mask length mismatch: "
            f"labels={labels.shape[0]}, mask={mask.shape[0]}."
        )
    if len(grid_results) != len(zeta_grid):
        raise ValueError(
            "EPN-D zeta selection grid/result length mismatch: "
            f"grid_results={len(grid_results)}, zeta_grid={len(zeta_grid)}."
        )

    eval_labels = labels[mask]
    if np.unique(eval_labels).size >= 2:
        finite_scores = [
            (idx, float(row["selection_score"]))
            for idx, row in enumerate(grid_results)
            if np.isfinite(float(row["selection_score"]))
        ]
        if finite_scores:
            return max(finite_scores, key=lambda item: item[1])[0], "valid_auroc", "binary_validation_labels"
        selection_metric = "valid_auroc"
        selection_reason = "binary_validation_labels_no_finite_valid_auroc"
    else:
        finite_losses = [
            (idx, float(row["best_valid_loss"]))
            for idx, row in enumerate(grid_results)
            if np.isfinite(float(row["best_valid_loss"]))
        ]
        if finite_losses:
            return min(finite_losses, key=lambda item: item[1])[0], "valid_loss", "single_class_validation_labels"
        selection_metric = "valid_loss"
        selection_reason = "single_class_validation_labels_no_finite_valid_loss"

    selected_index = 0
    for idx, zeta in enumerate(zeta_grid):
        if np.isclose(float(zeta), 1.0, rtol=0.0, atol=1e-8):
            selected_index = int(idx)
            break
    return int(selected_index), selection_metric, selection_reason


def _train_density_probe_for_zeta(
    args,
    train_table: EPNTable,
    valid_table: EPNTable,
    device: torch.device,
    criterion: torch.nn.Module,
    zeta: float,
    zeta_index: int,
    wandb_step_offset: int,
) -> Dict[str, object]:
    seed_everything(int(args.seed))
    density_target_array_for_loss(criterion, train_table)
    density_target_array_for_loss(criterion, valid_table)
    model = ScalarProbe(
        probe_type=str(args.density_probe),
        input_dim=density_input_dim(train_table),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    use_scheduler = bool(getattr(args, "scheduler", False))
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        if use_scheduler
        else None
    )
    rng = np.random.default_rng(int(args.seed))
    history: List[Dict[str, float]] = []
    prev_train_loss: Optional[float] = None
    converged = False
    best_valid_loss = float("inf")
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0

    desc = f"[density_probe:epn_d zeta={float(zeta):g}] training"
    with tqdm(total=int(args.n_epochs), desc=desc, unit="epoch", file=sys.stdout) as epoch_bar:
        for epoch in range(1, int(args.n_epochs) + 1):
            model.train()
            current_lr = float(optimizer.param_groups[0]["lr"])
            order = rng.permutation(train_table.question_count)
            total_loss = 0.0
            total_count = 0
            for start in range(0, train_table.question_count, int(args.batch_size)):
                idx = order[start : start + int(args.batch_size)]
                logits = density_logits_for_indices(model, train_table, idx, device)
                mask = torch.from_numpy(train_table.cluster_mask[idx].astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = density_batch_loss(criterion, train_table, idx, logits, float(zeta), None, mask, device)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(idx.shape[0])
                total_count += int(idx.shape[0])

            train_loss = total_loss / float(max(1, total_count))
            valid_loss = table_loss(model, valid_table, criterion, int(args.batch_size), device, float(zeta))
            history.append(
                {
                    "epoch": float(epoch),
                    "zeta": float(zeta),
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss),
                    "lr": current_lr,
                }
            )
            if np.isfinite(valid_loss) and valid_loss < best_valid_loss:
                best_valid_loss = float(valid_loss)
                best_epoch = int(epoch)
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epoch_bar.set_postfix(
                train_loss=f"{float(train_loss):.4f}",
                valid_loss=f"{float(valid_loss):.4f}",
                best_valid=f"{float(best_valid_loss):.4f}",
                lr=f"{current_lr:.2e}",
            )
            if wandb.run is not None:
                log_prefix = "density_probe/epn_d"
                wandb.log(
                    {
                        "density_probe/epn_d_epoch": int(epoch),
                        "density_probe/epn_d_train_loss": float(train_loss),
                        "density_probe/epn_d_valid_loss": float(valid_loss),
                        "density_probe/epn_d_best_valid_loss": float(best_valid_loss),
                        "density_probe/epn_d_lr": current_lr,
                        "density_probe/epn_d_zeta": float(zeta),
                        "density_probe/epn_d_zeta_index": int(zeta_index),
                        f"{log_prefix}/epoch": int(epoch),
                        f"{log_prefix}/zeta": float(zeta),
                        f"{log_prefix}/zeta_index": int(zeta_index),
                        f"{log_prefix}/train_loss": float(train_loss),
                        f"{log_prefix}/valid_loss": float(valid_loss),
                        f"{log_prefix}/best_valid_loss": float(best_valid_loss),
                        f"{log_prefix}/lr": current_lr,
                    },
                    step=int(wandb_step_offset) + int(zeta_index) * int(args.n_epochs) + int(epoch),
                )
            should_stop = False
            if prev_train_loss is not None:
                improvement = float(prev_train_loss) - float(train_loss)
                if 0.0 <= improvement < float(args.tol):
                    converged = True
                    should_stop = True
            prev_train_loss = float(train_loss)
            if scheduler is not None:
                scheduler.step()
            epoch_bar.update(1)
            if should_stop:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_valid_loss = table_loss(model, valid_table, criterion, int(args.batch_size), device, float(zeta))

    return {
        "model": model,
        "state_dict": best_state_dict,
        "zeta": float(zeta),
        "zeta_index": int(zeta_index),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "converged": bool(converged),
        "n_iter": int(history[-1]["epoch"]) if history else 0,
        "history": history,
    }


def train_density_probe(
    args,
    train_table: EPNTable,
    valid_table: EPNTable,
    device: torch.device,
    semantic_prob_source: str = EPN_SEMANTIC_PROB_SOURCE,
    wandb_step_offset: int = 0,
) -> Dict[str, object]:
    prob_source = normalize_epn_semantic_prob_source(semantic_prob_source)
    if train_table.semantic_prob_source != prob_source or valid_table.semantic_prob_source != prob_source:
        raise ValueError(
            "Density probe semantic probability source mismatch: "
            f"requested={prob_source}, train={train_table.semantic_prob_source}, "
            f"valid={valid_table.semantic_prob_source}."
        )
    if train_table.semantic_supervision_target != valid_table.semantic_supervision_target:
        raise ValueError(
            "Density probe semantic supervision target mismatch: "
            f"train={train_table.semantic_supervision_target}, "
            f"valid={valid_table.semantic_supervision_target}."
        )
    activation = normalize_density_activation(getattr(args, "activation", RHO_ACTIVATION))
    zeta_grid = normalize_zeta_grid(getattr(args, "zeta_grid", EPN_D_DEFAULT_ZETA_GRID))
    use_scheduler = bool(getattr(args, "scheduler", False))
    train_pseudo_ood_count = (
        0
        if train_table.pseudo_ood_mask is None
        else int(np.sum(np.asarray(train_table.pseudo_ood_mask, dtype=bool)))
    )
    valid_pseudo_ood_count = (
        0
        if valid_table.pseudo_ood_mask is None
        else int(np.sum(np.asarray(valid_table.pseudo_ood_mask, dtype=bool)))
    )
    use_pseudo_ood_loss = train_pseudo_ood_count > 0 or valid_pseudo_ood_count > 0
    loss_name = normalize_density_loss_name(str(args.loss))
    criterion = build_density_loss(
        loss_name,
        float(args.lambda_coef),
        semantic_prob_source=prob_source,
        use_pseudo_ood=use_pseudo_ood_loss,
    )
    density_output = (
        "hallucination_probability"
        if is_hallucination_bce_loss(criterion)
        else "rho"
    )
    effective_loss_name = criterion.__class__.__name__
    base_effective_loss_name = (
        criterion.base_loss.__class__.__name__
        if hasattr(criterion, "base_loss")
        else effective_loss_name
    )
    selection_family = epn_d_selection_family(valid_table.label_source)
    candidate_states: List[Dict[str, torch.Tensor]] = []
    candidate_payloads: List[Dict[str, object]] = []
    grid_results: List[Dict[str, float]] = []

    for zeta_index, zeta in enumerate(zeta_grid):
        candidate = _train_density_probe_for_zeta(
            args,
            train_table,
            valid_table,
            device,
            criterion,
            float(zeta),
            int(zeta_index),
            int(wandb_step_offset),
        )
        valid_pred = predict_density(
            candidate["model"],
            valid_table,
            int(args.batch_size),
            device,
            zeta=float(zeta),
            activation=activation,
        )
        valid_eval = evaluate_epn_table(valid_table, valid_pred)
        selection_auroc = float(valid_eval["AUROC"][selection_family])
        selection_aupr = float(valid_eval["AUPR"][selection_family])
        grid_row = {
            "zeta": float(zeta),
            "selection_score": selection_auroc,
            "selection_aupr": selection_aupr,
            "valid_auroc_vacuity": float(valid_eval["AUROC"]["vacuity"]),
            "valid_aupr_vacuity": float(valid_eval["AUPR"]["vacuity"]),
            "valid_auroc_dissonance": float(valid_eval["AUROC"]["dissonance"]),
            "valid_aupr_dissonance": float(valid_eval["AUPR"]["dissonance"]),
            "best_valid_loss": float(candidate["best_valid_loss"]),
            "best_epoch": float(candidate["best_epoch"]),
            "n_iter": float(candidate["n_iter"]),
        }
        grid_results.append(grid_row)
        if wandb.run is not None:
            log_prefix = "density_probe/epn_d_zeta_grid"
            wandb.log(
                {
                    f"{log_prefix}/zeta": float(zeta),
                    f"{log_prefix}/zeta_index": int(zeta_index),
                    f"{log_prefix}/selection_score": selection_auroc,
                    f"{log_prefix}/selection_aupr": selection_aupr,
                    f"{log_prefix}/valid_auroc_vacuity": float(valid_eval["AUROC"]["vacuity"]),
                    f"{log_prefix}/valid_aupr_vacuity": float(valid_eval["AUPR"]["vacuity"]),
                    f"{log_prefix}/valid_auroc_dissonance": float(valid_eval["AUROC"]["dissonance"]),
                    f"{log_prefix}/valid_aupr_dissonance": float(valid_eval["AUPR"]["dissonance"]),
                    f"{log_prefix}/best_valid_loss": float(candidate["best_valid_loss"]),
                },
                step=int(wandb_step_offset) + int(zeta_index + 1) * int(args.n_epochs),
            )
        candidate_states.append(candidate["state_dict"])
        candidate_payloads.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in {"model", "state_dict"}
            }
        )
        del candidate["model"]

    selected_index, selection_metric, selection_reason = select_epn_d_zeta_index(
        grid_results,
        zeta_grid,
        valid_table.labels,
        valid_table.metric_eval_mask,
    )

    selected_state_dict = candidate_states[int(selected_index)]
    model = ScalarProbe(
        probe_type=str(args.density_probe),
        input_dim=density_input_dim(train_table),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    model.load_state_dict(selected_state_dict)
    selected_payload = candidate_payloads[int(selected_index)]
    selected_zeta = float(zeta_grid[int(selected_index)])

    return {
        "model": model,
        "state_dict": selected_state_dict,
        "semantic_prob_source": prob_source,
        "semantic_supervision_target": train_table.semantic_supervision_target,
        "train_support_zero_count": int(train_table.semantic_support_zero_count),
        "valid_support_zero_count": int(valid_table.semantic_support_zero_count),
        "train_zero_denominator_count": int(train_table.semantic_prob_zero_denominator_count),
        "valid_zero_denominator_count": int(valid_table.semantic_prob_zero_denominator_count),
        "input_dim": density_input_dim(train_table),
        "probe_type": str(args.density_probe),
        "feature_kind": str(args.density_probe_feature),
        "hidden_dim": int(args.hidden_dim),
        "activation": activation,
        "rho_activation": activation,
        "zeta_grid": [float(zeta) for zeta in zeta_grid],
        "selected_zeta": float(selected_zeta),
        "selected_zeta_index": int(selected_index),
        "selection_family": str(selection_family),
        "selection_metric": str(selection_metric),
        "selection_reason": str(selection_reason),
        "zeta_grid_results": grid_results,
        "loss": str(loss_name),
        "effective_loss": effective_loss_name,
        "base_effective_loss": base_effective_loss_name,
        "density_output": density_output,
        "lambda_coef": float(args.lambda_coef),
        "pseudo_ood_enabled": bool(use_pseudo_ood_loss),
        "train_pseudo_ood_count": int(train_pseudo_ood_count),
        "valid_pseudo_ood_count": int(valid_pseudo_ood_count),
        "lr": float(args.lr),
        "scheduler": bool(use_scheduler),
        "n_epochs": int(args.n_epochs),
        "tol": float(args.tol),
        "best_epoch": int(selected_payload["best_epoch"]),
        "best_valid_loss": float(selected_payload["best_valid_loss"]),
        "converged": bool(selected_payload["converged"]),
        "n_iter": int(selected_payload["n_iter"]),
        "history": selected_payload["history"],
        "per_zeta_history": candidate_payloads,
    }


def predict_density(
    model: torch.nn.Module,
    table: EPNTable,
    batch_size: int,
    device: torch.device,
    zeta: float,
    activation: str = RHO_ACTIVATION,
) -> Dict[str, np.ndarray]:
    activation_norm = normalize_density_activation(activation)
    model.eval()
    if table.response_supports is None:
        raise ValueError("Missing response supports for EPN-D rho prediction.")
    max_resp = max(
        1,
        max((int(np.asarray(row).reshape(-1).shape[0]) for row in table.response_supports), default=0),
    )
    rho_logits_arr = np.zeros((table.question_count, max_resp), dtype=np.float32)
    rho_arr = np.zeros((table.question_count, max_resp), dtype=np.float32)
    q_rel_arr = np.zeros((table.question_count, max_resp), dtype=np.float32)
    response_mask = np.zeros((table.question_count, max_resp), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, table.question_count, int(batch_size)):
            idx = np.arange(start, min(start + int(batch_size), table.question_count), dtype=np.int64)
            logits = density_logits_for_indices(model, table, idx, device)
            response_supports, _, _, batch_response_mask = response_density_tensors_for_indices(table, idx, device)
            logits_by_response = rho_logits_by_response_from_logits(logits, response_supports)
            rho = density_rho_from_logits(logits_by_response, activation_norm)
            q_rel = torch.exp(-float(zeta) * rho) * batch_response_mask.to(dtype=rho.dtype)
            logits_np = logits_by_response.detach().cpu().numpy().astype(np.float32, copy=False)
            rho_np = (rho * batch_response_mask.to(dtype=rho.dtype)).detach().cpu().numpy().astype(np.float32, copy=False)
            q_rel_np = q_rel.detach().cpu().numpy().astype(np.float32, copy=False)
            mask_np = batch_response_mask.detach().cpu().numpy().astype(np.float32, copy=False)
            for local_row, qid in enumerate(idx.tolist()):
                n_resp = int(np.asarray(table.response_supports[int(qid)]).reshape(-1).shape[0])
                if n_resp == 0:
                    continue
                rho_logits_arr[int(qid), :n_resp] = logits_np[local_row, :n_resp]
                rho_arr[int(qid), :n_resp] = rho_np[local_row, :n_resp]
                q_rel_arr[int(qid), :n_resp] = q_rel_np[local_row, :n_resp]
                response_mask[int(qid), :n_resp] = mask_np[local_row, :n_resp]
    return {
        "rho_logits": rho_logits_arr,
        "rho_logits_by_response": rho_logits_arr,
        "density_logits_by_response": rho_logits_arr,
        "rho": rho_arr,
        "density_score": rho_arr,
        "density_score_by_response": rho_arr,
        "q_rel": q_rel_arr,
        "selected_zeta": np.asarray([float(zeta)], dtype=np.float32),
        "response_mask": response_mask,
    }


def evaluate_epn_table(
    table: EPNTable,
    density_outputs: Dict[str, np.ndarray],
) -> Dict[str, object]:
    rho_raw = np.asarray(density_outputs["rho"], dtype=np.float32)
    selected_zeta = float(np.asarray(density_outputs["selected_zeta"], dtype=np.float32).reshape(-1)[0])
    q_rel_raw = np.exp(-selected_zeta * np.clip(rho_raw, 0.0, 1.0)).astype(np.float32, copy=False)
    evidence_all: List[Dict[int, float]] = []
    vacuity_scores = np.zeros(table.question_count, dtype=np.float32)
    dissonance_scores = np.zeros(table.question_count, dtype=np.float32)
    alpha0 = np.zeros(table.question_count, dtype=np.float32)
    formula_vacuity = np.zeros(table.question_count, dtype=np.float32)
    if table.response_supports is None or table.response_aux_probs is None or table.response_cluster_index is None:
        raise ValueError("EPN-D evaluation requires response supports, aux probabilities, and cluster indices.")

    max_resp = max(
        1,
        max((int(np.asarray(row).reshape(-1).shape[0]) for row in table.response_supports), default=0),
    )
    rho_by_response_all = np.zeros((table.question_count, max_resp), dtype=np.float32)
    q_rel_all = np.zeros((table.question_count, max_resp), dtype=np.float32)
    response_aux_all = np.zeros((table.question_count, max_resp), dtype=np.float32)
    response_support_all = np.zeros((table.question_count, max_resp), dtype=np.float32)
    retained_by_response_all = np.zeros((table.question_count, max_resp), dtype=np.float32)
    retained_evidence_by_cluster = np.zeros_like(table.semantic_supports, dtype=np.float32)
    retained_totals = np.zeros(table.question_count, dtype=np.float32)
    projected_prob_by_cluster = np.zeros_like(table.semantic_supports, dtype=np.float32)
    neg_proj_prob_conf_scores = np.ones(table.question_count, dtype=np.float32)
    proj_prob_entropy_scores = np.zeros(table.question_count, dtype=np.float32)
    valid_rho_values: List[np.ndarray] = []
    valid_q_rel_values: List[np.ndarray] = []

    rho_arr = np.asarray(rho_raw, dtype=np.float32)
    q_rel_arr = np.asarray(q_rel_raw, dtype=np.float32)
    for qid in range(table.question_count):
        mask = table.cluster_mask[qid]
        ids = table.cluster_ids[qid, mask]
        K = int(mask.sum())
        resp_support = np.asarray(table.response_supports[qid], dtype=np.float32).reshape(-1)
        resp_aux = np.asarray(table.response_aux_probs[qid], dtype=np.float32).reshape(-1)
        resp_cluster = np.asarray(table.response_cluster_index[qid], dtype=np.int64).reshape(-1)
        if resp_support.shape[0] != resp_cluster.shape[0]:
            raise ValueError(
                f"Response support/cluster-index mismatch at split={table.split}, qid={qid}: "
                f"supports={resp_support.shape[0]}, clusters={resp_cluster.shape[0]}."
            )
        if resp_aux.shape[0] != resp_support.shape[0]:
            raise ValueError(
                f"Response aux/support mismatch at split={table.split}, qid={qid}: "
                f"aux={resp_aux.shape[0]}, supports={resp_support.shape[0]}."
            )
        n_resp = int(resp_support.shape[0])
        if rho_arr.ndim != 2 or q_rel_arr.ndim != 2:
            raise ValueError(
                f"EPN-D evaluation expects rho and q_rel with shape [N, R], "
                f"got rho={rho_arr.shape}, q_rel={q_rel_arr.shape}."
            )
        if qid >= rho_arr.shape[0] or qid >= q_rel_arr.shape[0] or n_resp > rho_arr.shape[1] or n_resp > q_rel_arr.shape[1]:
            raise ValueError(
                f"Response rho/q_rel width mismatch at split={table.split}, qid={qid}: "
                f"rho={rho_arr.shape}, q_rel={q_rel_arr.shape}, n_resp={n_resp}."
            )
        rho_q = np.clip(rho_arr[qid, :n_resp], 0.0, 1.0).astype(np.float32, copy=False)
        q_rel_q = np.clip(q_rel_arr[qid, :n_resp], 0.0, 1.0).astype(np.float32, copy=False)
        evidence_for_projection = np.zeros(K, dtype=np.float32)

        if K <= 0 or n_resp <= 0:
            evidence_dict = {}
            alpha0[qid] = 0.0
            formula_vacuity[qid] = 0.0
        else:
            aux_q = np.clip(resp_aux, 0.0, 1.0).astype(np.float32, copy=False)
            retained_by_response = np.clip(resp_support, 0.0, None) * q_rel_q
            evidence = np.zeros(K, dtype=np.float32)
            for resp_idx, cluster_idx in enumerate(resp_cluster.tolist()):
                if 0 <= int(cluster_idx) < K:
                    evidence[int(cluster_idx)] += float(retained_by_response[resp_idx])
            evidence_for_projection = evidence
            evidence_dict = {int(cid): float(val) for cid, val in zip(ids.tolist(), evidence.tolist())}
            alpha0[qid] = float(K) + float(np.sum(evidence, dtype=np.float64))
            formula_vacuity[qid] = float(K) / max(alpha0[qid], 1e-12)
            rho_by_response_all[qid, :n_resp] = rho_q
            q_rel_all[qid, :n_resp] = q_rel_q
            response_aux_all[qid, :n_resp] = aux_q
            response_support_all[qid, :n_resp] = resp_support
            retained_by_response_all[qid, :n_resp] = retained_by_response
            retained_evidence_by_cluster[qid, :K] = evidence
            retained_totals[qid] = float(np.sum(evidence, dtype=np.float64))
            valid_rho_values.append(rho_q)
            valid_q_rel_values.append(q_rel_q)
        if K > 0:
            projected_prob = subjective_logic_projected_prob_from_evidence(evidence_for_projection)
            projected_prob_by_cluster[qid, :K] = projected_prob
            neg_proj_prob_conf_scores[qid] = float(1.0 - np.max(projected_prob))
            proj_prob_entropy_scores[qid] = categorical_entropy(projected_prob)
        evidence_all.append(evidence_dict)
        vacuity_scores[qid] = float(vacuity(evidence_dict)) if evidence_dict else 0.0
        dissonance_scores[qid] = float(dissonance(evidence_dict)) if evidence_dict else 0.0

    rho_for_summary = (
        np.concatenate(valid_rho_values, axis=0)
        if valid_rho_values
        else np.zeros(0, dtype=np.float32)
    )
    q_rel_for_summary = (
        np.concatenate(valid_q_rel_values, axis=0)
        if valid_q_rel_values
        else np.zeros(0, dtype=np.float32)
    )

    metric_mask = np.asarray(table.metric_eval_mask, dtype=bool).reshape(-1)
    auroc_vac, aupr_vac = safe_binary_metrics(table.labels[metric_mask], vacuity_scores[metric_mask])
    auroc_dis, aupr_dis = safe_binary_metrics(table.labels[metric_mask], dissonance_scores[metric_mask])
    auroc_neg_proj, aupr_neg_proj = safe_binary_metrics(
        table.labels[metric_mask],
        neg_proj_prob_conf_scores[metric_mask],
    )
    auroc_proj_entropy, aupr_proj_entropy = safe_binary_metrics(
        table.labels[metric_mask],
        proj_prob_entropy_scores[metric_mask],
    )
    return {
        "split": table.split,
        "semantic_prob_source": table.semantic_prob_source,
        "semantic_supervision_target": table.semantic_supervision_target,
        "selected_zeta": float(selected_zeta),
        "rho": rho_raw,
        "rho_by_response": rho_by_response_all,
        "density_score": rho_raw,
        "density_score_by_response": rho_by_response_all,
        "q_rel": q_rel_all,
        "response_aux_probs": response_aux_all,
        "response_supports": response_support_all,
        "retained_evidence_by_response": retained_by_response_all,
        "retained_evidence_by_cluster": retained_evidence_by_cluster,
        "retained_evidence_total": retained_totals,
        "projected_prob_by_cluster": projected_prob_by_cluster,
        "alpha": 1.0 + retained_evidence_by_cluster,
        "alpha0": alpha0,
        "formula_vacuity": formula_vacuity,
        "evidence": evidence_all,
        "vacuity": vacuity_scores,
        "dissonance": dissonance_scores,
        "neg_proj_prob_conf": neg_proj_prob_conf_scores,
        "proj_prob_entropy": proj_prob_entropy_scores,
        "uncertainty_labels": table.labels,
        "y": table.y,
        "metric_eval_mask": metric_mask,
        "ood_refusal_eval": table.ood_refusal_eval,
        "AUROC": {
            "vacuity": auroc_vac,
            "dissonance": auroc_dis,
            "neg_proj_prob_conf": auroc_neg_proj,
            "proj_prob_entropy": auroc_proj_entropy,
        },
        "AUPR": {
            "vacuity": aupr_vac,
            "dissonance": aupr_dis,
            "neg_proj_prob_conf": aupr_neg_proj,
            "proj_prob_entropy": aupr_proj_entropy,
        },
        "mean_rho": float(np.mean(rho_for_summary)) if rho_for_summary.size > 0 else float("nan"),
        "mean_density_score": float(np.mean(rho_for_summary)) if rho_for_summary.size > 0 else float("nan"),
        "mean_q_rel": float(np.mean(q_rel_for_summary)) if q_rel_for_summary.size > 0 else float("nan"),
        "mean_retained_evidence": float(np.mean(retained_totals)) if retained_totals.size > 0 else float("nan"),
        "mean_vacuity": float(np.mean(vacuity_scores)) if vacuity_scores.size > 0 else float("nan"),
        "mean_dissonance": float(np.mean(dissonance_scores)) if dissonance_scores.size > 0 else float("nan"),
        "mean_neg_proj_prob_conf": (
            float(np.mean(neg_proj_prob_conf_scores)) if neg_proj_prob_conf_scores.size > 0 else float("nan")
        ),
        "mean_proj_prob_entropy": (
            float(np.mean(proj_prob_entropy_scores)) if proj_prob_entropy_scores.size > 0 else float("nan")
        ),
        "semantic_support_zero_count": int(table.semantic_support_zero_count),
        "semantic_prob_zero_denominator_count": int(table.semantic_prob_zero_denominator_count),
    }


def epn_metric_stem(semantic_prob_source: str) -> str:
    normalize_epn_semantic_prob_source(semantic_prob_source)
    return "epn_d"


def epn_summary_metrics(
    split: str,
    results: Dict[str, object],
    semantic_prob_source: str = EPN_SEMANTIC_PROB_SOURCE,
) -> Dict[str, float]:
    stem = epn_metric_stem(semantic_prob_source)
    return {
        f"{split}/auroc_{stem}_vacuity": float(results["AUROC"]["vacuity"]),
        f"{split}/aupr_{stem}_vacuity": float(results["AUPR"]["vacuity"]),
        f"{split}/auroc_{stem}_dissonance": float(results["AUROC"]["dissonance"]),
        f"{split}/aupr_{stem}_dissonance": float(results["AUPR"]["dissonance"]),
        f"{split}/auroc_{stem}_neg_proj_prob_conf": float(results["AUROC"]["neg_proj_prob_conf"]),
        f"{split}/aupr_{stem}_neg_proj_prob_conf": float(results["AUPR"]["neg_proj_prob_conf"]),
        f"{split}/auroc_{stem}_proj_prob_entropy": float(results["AUROC"]["proj_prob_entropy"]),
        f"{split}/aupr_{stem}_proj_prob_entropy": float(results["AUPR"]["proj_prob_entropy"]),
        f"{split}/selected_{stem}_zeta": float(results["selected_zeta"]),
        f"{split}/mean_{stem}_rho": float(results["mean_rho"]),
        f"{split}/mean_{stem}_density_score": float(results["mean_density_score"]),
        f"{split}/mean_{stem}_q_rel": float(results["mean_q_rel"]),
        f"{split}/mean_{stem}_retained_evidence": float(results["mean_retained_evidence"]),
        f"{split}/mean_{stem}_vacuity": float(results["mean_vacuity"]),
        f"{split}/mean_{stem}_dissonance": float(results["mean_dissonance"]),
        f"{split}/mean_{stem}_neg_proj_prob_conf": float(results["mean_neg_proj_prob_conf"]),
        f"{split}/mean_{stem}_proj_prob_entropy": float(results["mean_proj_prob_entropy"]),
    }


def build_predicted_rho_histogram_wandb_image(
    rho: np.ndarray,
    split: str,
    bins: int = 40,
) -> Optional[object]:
    arr = np.asarray(rho, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True, dpi=320)
    ax.hist(
        arr,
        bins=np.linspace(0.0, 1.0, int(bins) + 1),
        alpha=0.75,
        color="#4e79a7",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(f"{split} predicted rho histogram (epn_d)")
    ax.set_xlabel("predicted rho")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    image = wandb.Image(fig)
    plt.close(fig)
    return image


def save_predictions(path: str, table: EPNTable, density_outputs: Dict[str, np.ndarray], results: Dict[str, object]) -> None:
    np.savez_compressed(
        path,
        cluster_ids=table.cluster_ids,
        semantic_probs=table.semantic_probs,
        semantic_supports=table.semantic_supports,
        semantic_support_totals=table.semantic_support_totals,
        semantic_prob_source=np.asarray([table.semantic_prob_source], dtype=object),
        semantic_supervision_target=np.asarray(
            [table.semantic_supervision_target],
            dtype=object,
        ),
        semantic_support_zero_count=np.asarray(
            [table.semantic_support_zero_count],
            dtype=np.int64,
        ),
        semantic_prob_zero_denominator_count=np.asarray(
            [table.semantic_prob_zero_denominator_count],
            dtype=np.int64,
        ),
        cluster_targets=table.cluster_targets,
        augmented_targets=(
            np.zeros((table.question_count, table.cluster_targets.shape[1] + 1), dtype=np.float32)
            if table.augmented_targets is None
            else np.asarray(table.augmented_targets, dtype=np.float32)
        ),
        uce_targets=(
            np.zeros((table.question_count, table.cluster_targets.shape[1] + 1), dtype=np.float32)
            if table.uce_targets is None
            else np.asarray(table.uce_targets, dtype=np.float32)
        ),
        cluster_mask=table.cluster_mask,
        response_supports=np.asarray(
            [] if table.response_supports is None else table.response_supports,
            dtype=object,
        ),
        response_aux_probs=np.asarray(
            [] if table.response_aux_probs is None else table.response_aux_probs,
            dtype=object,
        ),
        response_cluster_index=np.asarray(
            [] if table.response_cluster_index is None else table.response_cluster_index,
            dtype=object,
        ),
        pseudo_ood_mask=(
            np.zeros(table.question_count, dtype=bool)
            if table.pseudo_ood_mask is None
            else np.asarray(table.pseudo_ood_mask, dtype=bool).reshape(-1)
        ),
        K=table.K,
        rho_logits_by_response=np.asarray(
            density_outputs.get("rho_logits_by_response", density_outputs["rho_logits"]),
            dtype=np.float32,
        ),
        rho_by_response=np.asarray(results["rho_by_response"], dtype=np.float32),
        density_logits_by_response=np.asarray(
            density_outputs.get(
                "density_logits_by_response",
                density_outputs.get("rho_logits_by_response", density_outputs["rho_logits"]),
            ),
            dtype=np.float32,
        ),
        density_score_by_response=np.asarray(
            results.get("density_score_by_response", results["rho_by_response"]),
            dtype=np.float32,
        ),
        selected_zeta=np.asarray([float(results["selected_zeta"])], dtype=np.float32),
        zeta_grid=np.asarray(density_outputs.get("zeta_grid", []), dtype=np.float32),
        density_response_mask=np.asarray(
            density_outputs.get("response_mask", np.ones(table.question_count, dtype=np.float32)),
            dtype=np.float32,
        ),
        q_rel=np.asarray(results["q_rel"], dtype=np.float32),
        response_supports_padded=np.asarray(results["response_supports"], dtype=np.float32),
        retained_evidence_by_response=np.asarray(results["retained_evidence_by_response"], dtype=np.float32),
        retained_evidence_by_cluster=np.asarray(results["retained_evidence_by_cluster"], dtype=np.float32),
        retained_evidence_total=np.asarray(results["retained_evidence_total"], dtype=np.float32),
        projected_prob_by_cluster=np.asarray(results["projected_prob_by_cluster"], dtype=np.float32),
        alpha=np.asarray(results["alpha"], dtype=np.float32),
        alpha0=np.asarray(results["alpha0"], dtype=np.float32),
        vacuity=np.asarray(results["vacuity"], dtype=np.float32),
        formula_vacuity=np.asarray(results["formula_vacuity"], dtype=np.float32),
        dissonance=np.asarray(results["dissonance"], dtype=np.float32),
        neg_proj_prob_conf=np.asarray(results["neg_proj_prob_conf"], dtype=np.float32),
        proj_prob_entropy=np.asarray(results["proj_prob_entropy"], dtype=np.float32),
        uncertainty_labels=table.labels,
        y=table.y,
        metric_eval_mask=table.metric_eval_mask,
        label_source=np.asarray([table.label_source], dtype=object),
    )


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


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def save_checkpoint(path: str, state_dict: Dict[str, torch.Tensor], metadata: Dict[str, object]) -> None:
    torch.save({"state_dict": state_dict, "metadata": metadata}, path)


def resolve_density_checkpoint_path(run_dir: str) -> str:
    return os.path.join(os.path.abspath(run_dir), "density_probe.pt")


def resolve_run_dir(args) -> str:
    model_id = str(args.model_name).rstrip("/").split("/")[-1]
    return os.path.join(
        os.path.abspath(args.output_dir),
        f"{model_id}_{args.dataset_name}_{int(args.n_samples)}_{args.judge_type}",
        (
            f"epn_d_{args.layer_range}_aux_{args.aux_probe}_{args.aux_probe_feature}_{args.aux_probe_type}"
            f"_density_{args.density_probe_feature}_{args.density_probe}"
        ),
    )


def eud_compatible_args(args, train_bundle: SplitBundle):
    ns = argparse.Namespace(**vars(args))
    ns.probe_target = str(args.aux_probe)
    ns.probe_feature = str(args.aux_probe_feature)
    ns.probe_type = str(args.aux_probe_type)
    feature_for_last_k = "slt" if str(args.aux_probe_feature) == "slt" else "tbg"
    if feature_for_last_k in train_bundle.X:
        ns.hallucination_probe_last_k = layer_range_to_last_k(train_bundle.X[feature_for_last_k], args.layer_range)
    else:
        ns.hallucination_probe_last_k = 5
    ns.eval_split = "test"
    return ns


def aux_result_to_eud_probe_dict(result: AuxiliaryProbeResult) -> Dict[str, object]:
    return {
        "clf": result.clf,
        "scores": result.scores,
        "raw_scores": result.scores,
        "logits": np.zeros_like(result.scores, dtype=float),
        "train_labels": np.zeros(result.train_count, dtype=np.int64),
        "eval_labels": result.eval_labels,
        "metric_eval_mask": np.ones(result.eval_labels.shape[0], dtype=bool),
        "AUROC": float(result.AUROC),
        "AUPR": float(result.AUPR),
        "probe_ece": float("nan"),
        "calibration": None,
        "calibration_enabled": False,
        "calibration_available": False,
        "temperature": float(result.temperature),
        "temperature_fit": None,
        "valid_probe_ece": float("nan"),
        "valid_probe_calibration": None,
        "valid_calibrated_probe_ece": float("nan"),
        "valid_calibration": None,
        "last_k": 0,
        "feature_dim": int(result.feature_dim),
        "train_count": int(result.train_count),
        "eval_count": int(result.eval_count),
        "probe_target": str(result.probe_target),
        "probe_label_source": str(result.probe_label_source),
        "probe_feature": str(result.probe_feature),
        "probe_type": str(result.probe_type),
        "probe_converged": bool(result.probe_converged),
        "probe_n_iter": int(result.probe_n_iter),
        "train_paths": {},
        "eval_paths": {},
    }


def print_aux_probe_summary(result: AuxiliaryProbeResult) -> None:
    constant_text = "none" if result.constant_prob is None else f"{float(result.constant_prob):.4f}"
    print(
        "[Aux Probe] "
        f"target={result.probe_target} | "
        f"feature={result.probe_feature} | "
        f"type={result.probe_type} | "
        f"labels={result.probe_label_source} | "
        f"train_n={result.train_count} | "
        f"test_n={result.eval_count} | "
        f"constant_prob={constant_text} | "
        f"converged={result.probe_converged} | "
        f"n_iter={result.probe_n_iter}",
        flush=True,
    )
    print(
        f"[Aux Probe] test AUROC={float(result.AUROC):.4f} | test AUPR={float(result.AUPR):.4f}",
        flush=True,
    )


def build_primary_question_discounted_bundle(
    args,
    artifacts: Any,
    X: Dict[str, np.ndarray],
    *,
    dataset_name: str,
    feature_kind: str,
) -> DiscountedSplitBundle:
    feature = normalize_feature_kind(feature_kind)
    if feature not in X:
        raise ValueError(f"aux_probe_feature={feature!r} requires X['{feature}'] for split={artifacts.split}.")
    question_features = flatten_question_features(X[feature], args.layer_range, split=artifacts.split, feature=feature)
    return DiscountedSplitBundle(
        dataset_name=str(dataset_name),
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
        response_slt_all=None,
        response_slt_lastk_path=None,
        question_probe_features=question_features,
        question_probe_feature_path=os.path.join(artifacts.split_dir, "X.npz"),
        disambig_question_semantic_ids=artifacts.disambig_question_semantic_ids,
        load_dir=artifacts.load_dir,
        split_dir=artifacts.split_dir,
        metadata={
            "base_split": str(artifacts.split),
            "artifact_dataset_name": str(dataset_name),
            "primary_probe_feature": feature,
        },
    )


def load_primary_question_discounted_bundle(
    args,
    dataset_name: str,
    split: str,
    feature_kind: str,
) -> DiscountedSplitBundle:
    artifacts = _load_split_artifacts_for(args, str(dataset_name), str(split))
    X_path = os.path.join(artifacts.split_dir, "X.npz")
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Missing X artifact: {X_path}")
    X = load_X_npz(X_path)
    return build_primary_question_discounted_bundle(
        args,
        artifacts,
        X,
        dataset_name=str(dataset_name),
        feature_kind=feature_kind,
    )


def resolve_primary_question_discounted_selection_bundle(
    args,
    eval_bundle: DiscountedSplitBundle,
    eval_metric_mask: np.ndarray,
    eval_ood_refusal_eval: Dict[str, object],
    feature_kind: str,
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
            },
        )

    selection_bundle = load_primary_question_discounted_bundle(args, str(args.dataset_name), "valid", feature_kind)

    selection_metadata = {
        "zeta_selection_enabled": True,
        "selection_split": str(selection_bundle.split),
        "selection_dataset_name": str(selection_bundle.dataset_name),
        "selection_label_source": str(selection_bundle.label_source),
    }
    return (
        selection_bundle,
        np.ones(len(selection_bundle.responses), dtype=bool),
        _default_refusal_stats(len(selection_bundle.responses)),
        selection_metadata,
    )


def evaluate_discounted_eud_with_primary_question_probe(
    args,
    artifacts: Any,
    metric_eval_mask: np.ndarray,
    ood_refusal_eval: Dict[str, object],
    probe_results: Dict[str, object],
    eval_bundle: SplitBundle,
) -> Dict[str, object]:
    feature = normalize_feature_kind(probe_results.get("probe_feature", "slt"))
    n_questions = len(artifacts.responses)
    selection_task, selection_family = _discount_selection_spec(artifacts.label_source)
    zeta_grid = normalize_zeta_grid(getattr(args, "zeta_grid", DEFAULT_DISCOUNT_ZETA_GRID))
    probe_target = normalize_probe_target(probe_results.get("probe_target", "hallucination"))
    probe_label_source = str(probe_results.get("probe_label_source", "1-y"))
    discounted_eval_bundle = build_primary_question_discounted_bundle(
        args,
        artifacts,
        eval_bundle.X,
        dataset_name=str(args.dataset_name),
        feature_kind=feature,
    )
    selection_bundle, selection_metric_mask, selection_refusal_eval, selection_metadata = (
        resolve_primary_question_discounted_selection_bundle(
            args,
            discounted_eval_bundle,
            metric_eval_mask,
            ood_refusal_eval,
            feature,
        )
    )

    clf = probe_results.get("clf")
    if clf is None:
        print(
            "[EUD Discounted/Weighted Branches] Skipping probe-derived metrics "
            f"because the {probe_target} probe was not fit."
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

    question_probe_results = dict(probe_results)
    question_probe_results["probe_feature"] = "tbg"
    question_probe_results["source_probe_feature"] = feature
    eval_inputs = _prepare_discounted_split_bundle(
        discounted_eval_bundle,
        question_probe_results,
        metric_eval_mask=metric_eval_mask,
        ood_refusal_eval=ood_refusal_eval,
    )
    if selection_bundle is discounted_eval_bundle and np.array_equal(
        selection_metric_mask, np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    ):
        selection_inputs = eval_inputs
    else:
        selection_inputs = _prepare_discounted_split_bundle(
            selection_bundle,
            question_probe_results,
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
        response_slt_lastk_path=None,
        zeta_grid=zeta_grid,
        selection_task=selection_task,
        selection_family=selection_family,
        selection_metadata=selection_metadata,
        branch_results=branch_results,
    )


def maybe_init_accelerator():
    if Accelerator is None:
        return None
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    if world_size > 1 or "LOCAL_RANK" in os.environ:
        return Accelerator()
    return None


def is_distributed_accelerate(accelerator) -> bool:
    return accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1


def is_main_process(accelerator) -> bool:
    return accelerator is None or bool(getattr(accelerator, "is_main_process", True))


def cleanup_distributed(accelerator) -> None:
    if not is_distributed_accelerate(accelerator):
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def make_baseline_args(args) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.eval_split = str(getattr(ns, "eval_split", "test"))
    ns.probe_target = str(getattr(ns, "probe_target", getattr(ns, "aux_probe", "hallucination")))
    ns.probe_feature = str(getattr(ns, "probe_feature", getattr(ns, "aux_probe_feature", "tbg")))
    ns.probe_type = str(getattr(ns, "probe_type", getattr(ns, "aux_probe_type", "lr")))
    return ns


def single_gpu_device_map(accelerator) -> Dict[str, object]:
    device = getattr(accelerator, "device", None)
    if isinstance(device, torch.device):
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            return {"": int(index)}
        if device.type == "cpu":
            return {"": "cpu"}
        return {"": str(device)}
    if torch.cuda.is_available():
        return {"": int(torch.cuda.current_device())}
    return {"": "cpu"}


def load_causal_lm_for_accelerated_baseline(model_name: str, accelerator):
    resolved_model_name = _resolve_hf_model_name(model_name)
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map = single_gpu_device_map(accelerator)
    rank = int(getattr(accelerator, "process_index", 0))
    if _needs_8bit_quantized_loading(model_name):
        _ensure_bitsandbytes_available()
        print(f"[Baseline][rank={rank}] [int8] Loading {resolved_model_name} with device_map={device_map}.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    elif _is_gemma_4_e4b_model_name(model_name):
        if Gemma4ForCausalLM is None:
            raise ImportError("Gemma4ForCausalLM is unavailable in the installed transformers package.")
        model = Gemma4ForCausalLM.from_pretrained(
            resolved_model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
    return model, tokenizer


def shard_indices(n_items: int, accelerator) -> np.ndarray:
    if not is_distributed_accelerate(accelerator):
        return np.arange(int(n_items), dtype=np.int64)
    shards = np.array_split(np.arange(int(n_items), dtype=np.int64), int(accelerator.num_processes))
    return np.asarray(shards[int(accelerator.process_index)], dtype=np.int64)


def slice_by_indices(value: Any, indices: np.ndarray) -> Any:
    if value is None:
        return None
    idx = np.asarray(indices, dtype=np.int64)
    if isinstance(value, np.ndarray):
        return value[idx]
    if isinstance(value, dict):
        return {key: slice_by_indices(item, idx) for key, item in value.items()}
    return [value[int(i)] for i in idx.tolist()]


def shard_baseline_data(data: Dict[str, object], indices: np.ndarray) -> Dict[str, object]:
    sharded = dict(data)
    for key in (
        "se_resp",
        "questions",
        "references",
        "preds",
        "contexts",
        "y",
        "semantic_ids",
        "hallucination_labels",
        "uncertainty_labels_by_kind",
    ):
        if key in sharded:
            sharded[key] = slice_by_indices(sharded[key], indices)
    sharded["original_indices"] = np.asarray(indices, dtype=np.int64)
    return sharded


def gather_records_to_main(local_records: List[Dict[str, object]], accelerator) -> Optional[List[Dict[str, object]]]:
    if not is_distributed_accelerate(accelerator):
        return local_records
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("Distributed baseline gather requires `accelerate launch --num_processes N ...`.")
    if accelerator.is_main_process:
        gathered: List[Optional[List[Dict[str, object]]]] = [None for _ in range(int(accelerator.num_processes))]
        torch.distributed.gather_object(local_records, gathered, dst=0)
        merged: List[Dict[str, object]] = []
        for shard in gathered:
            if shard:
                merged.extend(shard)
        return merged
    torch.distributed.gather_object(local_records, None, dst=0)
    return None


def ordered_values_from_records(records: Sequence[Dict[str, object]], n_items: int, key: str) -> List[object]:
    values: List[object] = [None for _ in range(int(n_items))]
    for record in records:
        values[int(record["idx"])] = record.get(key)
    return values


def ordered_float_array_from_records(records: Sequence[Dict[str, object]], n_items: int, key: str) -> np.ndarray:
    values = ordered_values_from_records(records, n_items, key)
    return np.asarray([0.0 if value is None else float(value) for value in values], dtype=float)


def ordered_matrix_from_records(
    records: Sequence[Dict[str, object]],
    n_items: int,
    key: str,
    width: int,
) -> np.ndarray:
    values = ordered_values_from_records(records, n_items, key)
    rows = []
    for value in values:
        if value is None:
            rows.append(np.zeros(int(width), dtype=float))
        else:
            rows.append(np.asarray(value, dtype=float).reshape(-1)[: int(width)])
    return np.vstack(rows).astype(float, copy=False) if rows else np.zeros((0, int(width)), dtype=float)


def finalize_binary_baseline_results(
    args,
    data: Dict[str, object],
    scores: np.ndarray,
    method_name: str,
    responses: Optional[List[object]] = None,
) -> Dict[str, object]:
    from baseline import (
        _compute_no_clear_err_auroc_aupr,
        _compute_no_clear_large_n_cluster_auroc_aupr,
        _safe_binary_auroc_aupr,
    )

    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    labels_arr = np.asarray(data["hallucination_labels"], dtype=np.int64).reshape(-1)
    split_name = str(data.get("split", getattr(args, "eval_split", "test")))
    metric_eval_mask, ood_refusal_eval = compute_ood_refusal_filter(
        labels=labels_arr,
        dataset_name=str(getattr(args, "dataset_name", "")),
        split=split_name,
        remove_refusal=bool(getattr(args, "remove_refusal", False)),
        load_dir=data.get("load_dir"),
        preds=data.get("preds"),
    )
    auroc, aupr = _safe_binary_auroc_aupr(
        labels_arr[metric_eval_mask],
        scores_arr[metric_eval_mask],
        dataset_name=str(getattr(args, "dataset_name", "")),
        split=split_name,
        method_name=method_name,
    )
    auroc_no_clear_err, aupr_no_clear_err = _compute_no_clear_err_auroc_aupr(
        ambig_labels=labels_arr,
        y=np.asarray(data.get("y", []), dtype=np.int64),
        scores=scores_arr,
        metric_eval_mask=metric_eval_mask,
        dataset_name=str(getattr(args, "dataset_name", "")),
        split=split_name,
        method_name=method_name,
    )
    auroc_no_clear_large_n_cluster, aupr_no_clear_large_n_cluster = _compute_no_clear_large_n_cluster_auroc_aupr(
        ambig_labels=labels_arr,
        semantic_ids=data.get("semantic_ids"),
        scores=scores_arr,
        metric_eval_mask=metric_eval_mask,
        dataset_name=str(getattr(args, "dataset_name", "")),
        split=split_name,
        method_name=method_name,
    )
    result: Dict[str, object] = {
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
    if responses is not None:
        result["responses"] = responses
    return result


def finalize_verbalized_results(
    args,
    data: Dict[str, object],
    records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    from baseline import Verbalized

    n_items = int(len(data["questions"]))
    responses = ordered_values_from_records(records, n_items, "response")
    parsed_confidences_arr = ordered_matrix_from_records(records, n_items, "parsed_confidence_pair", 2)
    knowledge_confidences_arr = parsed_confidences_arr[:, 0]
    ambiguity_confidences_arr = parsed_confidences_arr[:, 1]
    epistemic_scores_arr = 1.0 - knowledge_confidences_arr
    aleatoric_scores_arr = ambiguity_confidences_arr

    verbalized = Verbalized(args, model=None, tokenizer=None)
    verbalized.data = data
    task_kind = str(data.get("uncertainty_label_kind", "fallback"))
    labels_by_kind = data.get("uncertainty_labels_by_kind") or {}
    epistemic_labels = verbalized._slice_optional_labels(labels_by_kind.get("epistemic"), n_items)
    aleatoric_labels = verbalized._slice_optional_labels(labels_by_kind.get("aleatoric"), n_items)
    default_labels = np.asarray(data["hallucination_labels"], dtype=np.int64).reshape(-1)[:n_items]
    if epistemic_labels is None and task_kind in {"ood", "fallback"}:
        epistemic_labels = default_labels
    if aleatoric_labels is None and task_kind == "ambig":
        aleatoric_labels = default_labels
    if epistemic_labels is None and aleatoric_labels is not None:
        epistemic_labels = np.asarray(aleatoric_labels, dtype=np.int64).reshape(-1)[:n_items]
    if aleatoric_labels is None and epistemic_labels is not None:
        aleatoric_labels = np.asarray(epistemic_labels, dtype=np.int64).reshape(-1)[:n_items]

    split_name = str(data.get("split", getattr(args, "eval_split", "test")))
    preds_eval = list(data["preds"]) if data.get("preds") is not None else None
    epistemic_metrics = verbalized._evaluate_family_scores(
        family_name="epistemic",
        labels=epistemic_labels,
        scores=epistemic_scores_arr,
        preds=preds_eval,
        split_name=split_name,
    )
    aleatoric_metrics = verbalized._evaluate_family_scores(
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


def run_accelerated_optional_baselines(args, artifacts: Optional[Any], accelerator, step: int = 0) -> None:
    from baseline import LinearProbe, PTrue, SelfCheckGPT, Verbalized, _log_baseline_summary

    baseline_args = make_baseline_args(args)
    if artifacts is None:
        artifacts = _load_split_artifacts_for(baseline_args, str(baseline_args.dataset_name), "test")

    rank = int(getattr(accelerator, "process_index", 0))
    world = int(getattr(accelerator, "num_processes", 1))
    if accelerator.is_main_process:
        print(f"[Baseline][Accelerate] Running optional baselines across {world} processes.", flush=True)

    ptrue_for_data = PTrue(baseline_args, model=None, tokenizer=None)
    data = _load_baseline_data(baseline_args, artifacts, ptrue_for_data)
    n_items = int(len(data["questions"]))
    local_indices = shard_indices(n_items, accelerator)
    local_data = shard_baseline_data(data, local_indices)
    print(f"[Baseline][rank={rank}] shard size={len(local_indices)}/{n_items}", flush=True)

    model, tokenizer = load_causal_lm_for_accelerated_baseline(str(baseline_args.model_name), accelerator)
    verbalized = None
    ptrue = PTrue(baseline_args, model=model, tokenizer=tokenizer)
    ptrue.data = local_data
    ptrue_local = ptrue.evaluate(
        local_data["questions"],
        local_data["preds"],
        local_data["hallucination_labels"],
    )
    ptrue_records = [
        {
            "idx": int(idx),
            "score": float(score),
            "response": response,
        }
        for idx, score, response in zip(
            local_indices.tolist(),
            np.asarray(ptrue_local["scores"], dtype=float).reshape(-1).tolist(),
            list(ptrue_local.get("responses", [])),
        )
    ]
    gathered_ptrue = gather_records_to_main(ptrue_records, accelerator)
    if accelerator.is_main_process:
        ptrue_scores = ordered_float_array_from_records(gathered_ptrue or [], n_items, "score")
        ptrue_responses = ordered_values_from_records(gathered_ptrue or [], n_items, "response")
        ptrue_results = finalize_binary_baseline_results(baseline_args, data, ptrue_scores, "PTrue", ptrue_responses)
        print(f"[{artifacts.split} PTrue] AUROC: {ptrue_results['AUROC']:.4f}, AUPR: {ptrue_results['AUPR']:.4f}")
        _log_baseline_summary(artifacts.split, "ptrue", ptrue_results)
        _log_metrics(_metric_dict_for_history(artifacts.split, "ptrue", ptrue_results), step=step)

    if data.get("uncertainty_label_kind") in {"ambig", "ood", "fallback"}:
        verbalized = Verbalized(baseline_args, model=model, tokenizer=tokenizer)
        verbalized.data = local_data
        verbalized_local = verbalized.evaluate(
            local_data["questions"],
            local_data["preds"],
            local_data["hallucination_labels"],
        )
        parsed_pairs = np.asarray(verbalized_local.get("parsed_confidence_pairs", np.zeros((0, 2))), dtype=float)
        verbalized_records = []
        for row_pos, idx in enumerate(local_indices.tolist()):
            verbalized_records.append(
                {
                    "idx": int(idx),
                    "response": list(verbalized_local.get("responses", []))[row_pos],
                    "parsed_confidence_pair": parsed_pairs[row_pos].astype(float).tolist(),
                }
            )
        gathered_verbalized = gather_records_to_main(verbalized_records, accelerator)
        if accelerator.is_main_process:
            verbalized_results = finalize_verbalized_results(baseline_args, data, gathered_verbalized or [])
            print(
                f"[{artifacts.split} Verbalized] "
                f"Epistemic AUROC: {verbalized_results['AUROC_epistemic']:.4f}, "
                f"AUPR: {verbalized_results['AUPR_epistemic']:.4f}; "
                f"Aleatoric AUROC: {verbalized_results['AUROC_aleatoric']:.4f}, "
                f"AUPR: {verbalized_results['AUPR_aleatoric']:.4f}"
            )
            _log_baseline_summary(artifacts.split, "verbalized", verbalized_results)
            _log_metrics(_metric_dict_for_history(artifacts.split, "verbalized", verbalized_results), step=step)

    ptrue.model = None
    ptrue.tokenizer = None
    ptrue.data = None
    del ptrue
    if verbalized is not None:
        verbalized.model = None
        verbalized.tokenizer = None
        verbalized.data = None
        del verbalized
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_distributed_accelerate(accelerator):
        accelerator.wait_for_everyone()

    selfcheck_device = str(getattr(accelerator, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    selfcheck_model, selfcheck_tokenizer, _ = SelfCheckGPT.instantiate_nli_model(device=selfcheck_device)
    selfcheck = SelfCheckGPT(baseline_args, model=selfcheck_model, tokenizer=selfcheck_tokenizer)
    selfcheck.data = local_data
    selfcheck_local = selfcheck.evaluate(
        local_data["questions"],
        local_data["preds"],
        local_data["hallucination_labels"],
    )
    selfcheck_records = [
        {
            "idx": int(idx),
            "score": float(score),
        }
        for idx, score in zip(
            local_indices.tolist(),
            np.asarray(selfcheck_local["scores"], dtype=float).reshape(-1).tolist(),
        )
    ]
    gathered_selfcheck = gather_records_to_main(selfcheck_records, accelerator)
    if accelerator.is_main_process:
        selfcheck_scores = ordered_float_array_from_records(gathered_selfcheck or [], n_items, "score")
        selfcheck_results = finalize_binary_baseline_results(baseline_args, data, selfcheck_scores, "SelfCheckGPT")
        print(
            f"[{artifacts.split} SelfCheckGPT] AUROC: {selfcheck_results['AUROC']:.4f}, "
            f"AUPR: {selfcheck_results['AUPR']:.4f}"
        )
        _log_baseline_summary(artifacts.split, "selfcheckgpt", selfcheck_results)
        _log_metrics(_metric_dict_for_history(artifacts.split, "selfcheckgpt", selfcheck_results), step=step)

        class _EUDLinearProbe(LinearProbe):
            def _load_hidden_and_labels(self, split: str) -> Dict[str, object]:
                return _load_linear_probe_split(self.args, split)

        linear_probe = _EUDLinearProbe(baseline_args)
        linear_probe_results = linear_probe.evaluate(eval_split=artifacts.split)
        print(
            f"[{artifacts.split} LinearProbe] AUROC: {linear_probe_results['AUROC']:.4f}, "
            f"AUPR: {linear_probe_results['AUPR']:.4f}"
        )
        _log_baseline_summary(artifacts.split, "linear_probe", linear_probe_results)
        _log_metrics(_metric_dict_for_history(artifacts.split, "linear_probe", linear_probe_results), step=step)

    del selfcheck
    del selfcheck_model
    del selfcheck_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EPN-D trainer with norm-sequence support and q_rel=exp(-zeta * rho) evidence."
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trivia_qa_split")
    parser.add_argument("--judge_type", type=str, default="rouge")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--n_resp", type=int, default=10)
    parser.add_argument("--valid_frac", "--valid_fra", dest="valid_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--test_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--fix_val_test_splits", type=_bool, default=False)
    parser.add_argument("--layer_range", type=str, default="last_5", choices=["last_1", "last_5", "all"])
    parser.add_argument("--aux_probe", type=str, default="hallucination", choices=["hallucination", "ambiguity"])
    parser.add_argument("--aux_probe_feature", type=str, default="tbg", choices=["tbg", "slt"])
    parser.add_argument("--aux_probe_type", type=str, default="lr", choices=["lr", "mlp"])
    parser.add_argument("--density_probe", type=str, default="lr", choices=["lr", "mlp"])
    parser.add_argument("--density_probe_feature", type=str, default="tbg", choices=["tbg", "slt"])
    parser.add_argument(
        "--activation",
        type=str,
        default=RHO_ACTIVATION,
        help="Compatibility option only; EPN-D forces rho activation to sigmoid.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=EPND_AUGMENTED_LOSS_NAME,
        choices=list(DENSITY_LOSS_CHOICES),
        help=(
            "Density-probe objective. BCE/HallucinationBCEWithLogitsLoss trains "
            "sigmoid(logit) directly as P(hall)."
        ),
    )
    parser.add_argument(
        "--semantic_supervision_target",
        type=str,
        default="norm",
        choices=list(SEMANTIC_SUPERVISION_TARGET_CHOICES),
        help="Semantic cluster supervision target y_k construction (default: norm).",
    )
    parser.add_argument("--lambda", dest="lambda_coef", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler",
        type=_bool,
        default=False,
        help="Use StepLR for density probe training: multiply lr by 0.1 every 20 epochs.",
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_pseudo_OOD", type=_bool, default=True)
    parser.add_argument(
        "--pseudo_OOD_dataset_name",
        type=str,
        default=None,
        choices=["temporalood", "alcuna"],
        help="Fixed pseudo-OOD artifact dataset: alcuna_trivia_qa requires temporalood; temporalood_trivia_qa requires alcuna.",
    )
    parser.add_argument("--remove_refusal", type=_bool, default=False)
    parser.add_argument("--run_baseline", type=_bool, default=True)
    parser.add_argument("--zeta_grid", type=float, nargs="+", default=list(EPN_D_DEFAULT_ZETA_GRID))
    parser.add_argument("--zeta_selection", type=_bool, default=False)
    parser.add_argument("--output_dir", type=str, default=os.path.join(WORKING_DIR, "epn_runs_ver3"))
    parser.add_argument(
        "--save",
        type=_bool,
        default=True,
        help="Save the trained density_probe checkpoint as density_probe.pt.",
    )
    return parser


def main(args) -> None:
    accelerator = maybe_init_accelerator()
    distributed_baselines = is_distributed_accelerate(accelerator)
    if distributed_baselines and not is_main_process(accelerator):
        try:
            if bool(args.run_baseline):
                accelerator.wait_for_everyone()
                run_accelerated_optional_baselines(args, artifacts=None, accelerator=accelerator, step=0)
                accelerator.wait_for_everyone()
            else:
                accelerator.wait_for_everyone()
        finally:
            cleanup_distributed(accelerator)
        return

    args.layer_range = normalize_layer_range(args.layer_range)
    args.aux_probe = normalize_probe_target(args.aux_probe)
    if args.aux_probe != "hallucination":
        raise ValueError("EPN-D requires --aux_probe=hallucination for hallucination-derived rho supervision.")
    args.aux_probe_feature = normalize_feature_kind(args.aux_probe_feature)
    args.density_probe_feature = normalize_feature_kind(args.density_probe_feature)
    args.activation = normalize_density_activation(args.activation)
    args.loss = normalize_density_loss_name(args.loss)
    if is_hallucination_bce_loss_name(args.loss) and bool(getattr(args, "use_pseudo_OOD", False)):
        raise ValueError(
            "HallucinationBCEWithLogitsLoss requires --use_pseudo_OOD=False; "
            "pseudo-OOD rows are not hallucination-label supervision."
        )
    args.semantic_supervision_target = normalize_semantic_supervision_target(args.semantic_supervision_target)
    validate_pseudo_ood_args(args)
    seed_everything(int(args.seed))
    validate_ambiguity_artifacts(args, ("train", "valid", "test"))

    need_response_slt = args.aux_probe_feature == "slt" or args.density_probe_feature == "slt"
    splits = ("train", "valid", "test")
    bundles = {split: load_split_bundle(args, split, need_response_slt=need_response_slt) for split in splits}
    response_slt_by_split: Dict[str, List[np.ndarray]] = {}
    if need_response_slt:
        response_slt_by_split = {
            split: response_slt_matrices(bundle, args.layer_range)
            for split, bundle in bundles.items()
        }
    pseudo_ood_bundles = load_pseudo_ood_bundles(args)
    pseudo_ood_enabled = bool(pseudo_ood_bundles)
    pseudo_ood_metadata: Dict[str, object] = {
        "enabled": bool(pseudo_ood_enabled),
        "dataset_name": str(args.pseudo_OOD_dataset_name) if pseudo_ood_enabled else "",
        "fixed_n_samples": int(PSEUDO_OOD_FIXED_N_SAMPLES) if pseudo_ood_enabled else 0,
        "load_dir": pseudo_ood_bundles["train"].load_dir if pseudo_ood_enabled else "",
        "expected_split_counts": dict(PSEUDO_OOD_EXPECTED_SPLIT_COUNTS) if pseudo_ood_enabled else {},
        "train_count": (
            int(pseudo_ood_bundles["train"].labels.shape[0])
            if pseudo_ood_enabled
            else 0
        ),
        "valid_count": (
            int(pseudo_ood_bundles["valid"].labels.shape[0])
            if pseudo_ood_enabled
            else 0
        ),
        "test_count": (
            int(pseudo_ood_bundles["test"].labels.shape[0])
            if pseudo_ood_enabled
            else 0
        ),
    }
    if pseudo_ood_enabled:
        print(
            "[Pseudo-OOD] "
            f"dataset={args.pseudo_OOD_dataset_name} | "
            f"load_dir={pseudo_ood_bundles['train'].load_dir} | "
            f"train={pseudo_ood_metadata['train_count']} | "
            f"valid={pseudo_ood_metadata['valid_count']} | "
            f"test={pseudo_ood_metadata['test_count']}",
            flush=True,
        )

    run_dir = resolve_run_dir(args)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    density_checkpoint_path = resolve_density_checkpoint_path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="EPN", config={**vars(args), "output_dir": run_dir})

    test_metric_mask, test_refusal_eval = compute_ood_refusal_filter(
        labels=bundles["test"].artifacts.labels,
        dataset_name=str(args.dataset_name),
        split="test",
        remove_refusal=bool(args.remove_refusal),
        load_dir=bundles["test"].artifacts.load_dir,
    )

    aux_result = fit_auxiliary_probe(
        args,
        train_bundle=bundles["train"],
        eval_bundle=bundles["test"],
        response_slt_by_split=response_slt_by_split,
        metric_eval_mask=test_metric_mask,
    )
    print_aux_probe_summary(aux_result)
    aux_prob_by_split: Dict[str, List[np.ndarray]] = {}
    reliability_by_split: Dict[str, List[np.ndarray]] = {}
    for split in splits:
        aux_prob_by_split[split], reliability_by_split[split] = predict_aux_reliability_by_question(
            args,
            aux_result,
            bundles[split],
            response_slt_by_split,
        )

    if aux_result.clf is not None:
        save_checkpoint(
            os.path.join(run_dir, "aux_probe.pt"),
            {key: value.detach().cpu().clone() for key, value in aux_result.clf.model.state_dict().items()},
            {
                "input_dim": aux_result.feature_dim,
                "probe_type": aux_result.probe_type,
                "probe_target": aux_result.probe_target,
                "probe_feature": aux_result.probe_feature,
                "probe_label_source": aux_result.probe_label_source,
                "activation": "sigmoid",
            },
        )

    metrics: Dict[str, object] = {
        "artifact_load_dir": bundles["train"].artifacts.load_dir,
        "output_dir": run_dir,
        "device": str(device),
        "args": vars(args),
        "method": "epn_d",
        "semantic_prob_source": EPN_SEMANTIC_PROB_SOURCE,
        "aux_probe": {
            "probe_target": aux_result.probe_target,
            "probe_feature": aux_result.probe_feature,
            "probe_type": aux_result.probe_type,
            "probe_label_source": aux_result.probe_label_source,
            "train_count": aux_result.train_count,
            "eval_count": aux_result.eval_count,
            "AUROC": aux_result.AUROC,
            "AUPR": aux_result.AUPR,
            "converged": aux_result.probe_converged,
            "n_iter": aux_result.probe_n_iter,
            "constant_prob": aux_result.constant_prob,
        },
        "density_probe": {},
        "pseudo_ood": pseudo_ood_metadata,
        "splits": {},
    }

    log_payload: Dict[str, float] = {
        "test/aux_probe_auroc": float(aux_result.AUROC),
        "test/aux_probe_aupr": float(aux_result.AUPR),
    }
    image_log_payload: Dict[str, object] = {}
    for split in splits:
        metrics["splits"][split] = {"summary": {}, "epn_d": {}}

    semantic_prob_source = EPN_SEMANTIC_PROB_SOURCE
    epn_tables = {
        split: build_epn_table(
            args,
            bundle=bundles[split],
            aux_prob_by_question=aux_prob_by_split[split],
            reliability_by_question=reliability_by_split[split],
            response_slt_by_split=response_slt_by_split,
            semantic_prob_source=semantic_prob_source,
        )
        for split in splits
    }
    zero_support_total = int(sum(table.semantic_support_zero_count for table in epn_tables.values()))
    if zero_support_total > 0:
        print(
            f"[EPN-D] {zero_support_total} questions had zero normalized sequence-probability support; "
            "no uniform fallback was applied.",
            flush=True,
        )

    train_loss_table = epn_tables["train"]
    valid_loss_table = epn_tables["valid"]
    if pseudo_ood_enabled:
        train_loss_table = append_pseudo_ood_rows(args, train_loss_table, pseudo_ood_bundles["train"])
        valid_loss_table = append_pseudo_ood_rows(args, valid_loss_table, pseudo_ood_bundles["valid"])
        print(
            "[EPN-D] pseudo-OOD loss rows appended: "
            f"train={int(np.sum(train_loss_table.pseudo_ood_mask))}, "
            f"valid={int(np.sum(valid_loss_table.pseudo_ood_mask))}.",
            flush=True,
        )

    density_results = train_density_probe(
        args,
        train_loss_table,
        valid_loss_table,
        device,
        semantic_prob_source=semantic_prob_source,
    )
    del train_loss_table, valid_loss_table
    density_checkpoint_metadata = {
        "method": "epn_d",
        "model_name": str(args.model_name),
        "dataset_name": str(args.dataset_name),
        "judge_type": str(args.judge_type),
        "n_samples": int(args.n_samples),
        "n_resp": int(args.n_resp),
        "layer_range": str(args.layer_range),
        "aux_probe": str(args.aux_probe),
        "aux_probe_feature": str(args.aux_probe_feature),
        "aux_probe_type": str(args.aux_probe_type),
        "semantic_prob_source": semantic_prob_source,
        "semantic_supervision_target": density_results["semantic_supervision_target"],
        "input_dim": density_results["input_dim"],
        "probe_type": density_results["probe_type"],
        "feature_kind": density_results["feature_kind"],
        "hidden_dim": density_results["hidden_dim"],
        "best_epoch": density_results["best_epoch"],
        "best_valid_loss": density_results["best_valid_loss"],
        "train_support_zero_count": density_results["train_support_zero_count"],
        "valid_support_zero_count": density_results["valid_support_zero_count"],
        "train_zero_denominator_count": density_results["train_zero_denominator_count"],
        "valid_zero_denominator_count": density_results["valid_zero_denominator_count"],
        "activation": density_results["rho_activation"],
        "rho_activation": density_results["rho_activation"],
        "zeta_grid": density_results["zeta_grid"],
        "selected_zeta": density_results["selected_zeta"],
        "selected_zeta_index": density_results["selected_zeta_index"],
        "selection_family": density_results["selection_family"],
        "selection_metric": density_results["selection_metric"],
        "selection_reason": density_results["selection_reason"],
        "zeta_grid_results": density_results["zeta_grid_results"],
        "loss": density_results["loss"],
        "effective_loss": density_results["effective_loss"],
        "base_effective_loss": density_results["base_effective_loss"],
        "density_output": density_results["density_output"],
        "regularization_target": density_results["density_output"],
        "lambda_coef": density_results["lambda_coef"],
        "pseudo_ood_enabled": density_results["pseudo_ood_enabled"],
        "pseudo_ood_dataset_name": str(args.pseudo_OOD_dataset_name) if pseudo_ood_enabled else "",
        "pseudo_ood_load_dir": pseudo_ood_metadata["load_dir"],
        "train_pseudo_ood_count": density_results["train_pseudo_ood_count"],
        "valid_pseudo_ood_count": density_results["valid_pseudo_ood_count"],
        "scheduler": density_results["scheduler"],
        "n_epochs": density_results["n_epochs"],
        "tol": density_results["tol"],
        "n_iter": density_results["n_iter"],
        "converged": density_results["converged"],
    }
    if bool(args.save):
        Path(os.path.dirname(density_checkpoint_path)).mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            density_checkpoint_path,
            density_results["state_dict"],
            density_checkpoint_metadata,
        )
    density_summary = {
        key: value
        for key, value in density_results.items()
        if key not in {"model", "state_dict"}
    }
    density_summary["checkpoint_written"] = bool(args.save)
    density_summary["checkpoint_path"] = density_checkpoint_path if bool(args.save) else ""
    metrics["density_probe"] = density_summary

    density_model = density_results["model"]
    for split in splits:
        density_pred = predict_density(
            density_model,
            epn_tables[split],
            int(args.batch_size),
            device,
            zeta=float(density_results["selected_zeta"]),
            activation=density_results["rho_activation"],
        )
        density_pred["zeta_grid"] = np.asarray(density_results["zeta_grid"], dtype=np.float32)
        epn_eval = evaluate_epn_table(epn_tables[split], density_pred)
        save_predictions(
            os.path.join(run_dir, f"predictions_{split}.npz"),
            epn_tables[split],
            density_pred,
            epn_eval,
        )
        split_metrics = epn_summary_metrics(split, epn_eval, semantic_prob_source)
        log_payload.update(split_metrics)
        metrics["splits"][split]["summary"].update(split_metrics)
        metrics["splits"][split]["epn_d"] = epn_eval

        stem = epn_metric_stem(semantic_prob_source)
        if split == "test" and wandb.run is not None:
            rho_hist_image = build_predicted_rho_histogram_wandb_image(
                np.asarray(epn_eval["rho_by_response"], dtype=np.float32),
                split=split,
            )
            if rho_hist_image is not None:
                image_log_payload[f"{split}/{stem}_predicted_rho_hist"] = rho_hist_image
        print(
            f"[{split}][epn_d] "
            f"auroc_vacuity={split_metrics[f'{split}/auroc_{stem}_vacuity']:.4f} | "
            f"aupr_vacuity={split_metrics[f'{split}/aupr_{stem}_vacuity']:.4f} | "
            f"auroc_dissonance={split_metrics[f'{split}/auroc_{stem}_dissonance']:.4f} | "
            f"aupr_dissonance={split_metrics[f'{split}/aupr_{stem}_dissonance']:.4f} | "
            f"auroc_neg_proj_prob_conf={split_metrics[f'{split}/auroc_{stem}_neg_proj_prob_conf']:.4f} | "
            f"aupr_neg_proj_prob_conf={split_metrics[f'{split}/aupr_{stem}_neg_proj_prob_conf']:.4f} | "
            f"auroc_proj_prob_entropy={split_metrics[f'{split}/auroc_{stem}_proj_prob_entropy']:.4f} | "
            f"aupr_proj_prob_entropy={split_metrics[f'{split}/aupr_{stem}_proj_prob_entropy']:.4f}",
            flush=True,
        )

    final_step = int(density_results.get("n_iter", 0)) + 1
    del density_model
    del density_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gt_evidence_all = _build_gt_evidence_dicts(bundles["test"].artifacts.semantic_ids)
    eud_results = _evaluate_eud(bundles["test"].artifacts, gt_evidence_all, test_metric_mask, test_refusal_eval)
    disambig_question_results = _evaluate_disambig_question_eud(
        bundles["test"].artifacts,
        test_metric_mask,
        test_refusal_eval,
    )
    artifact_baselines = _evaluate_artifact_baselines(
        bundles["test"].artifacts,
        gt_evidence_all,
        test_metric_mask,
        test_refusal_eval,
    )
    eud_args = eud_compatible_args(args, bundles["train"])
    eud_probe_dict = aux_result_to_eud_probe_dict(aux_result)
    if args.aux_probe_feature == "slt":
        discounted_eud_results = evaluate_discounted_eud_with_primary_question_probe(
            eud_args,
            bundles["test"].artifacts,
            test_metric_mask,
            test_refusal_eval,
            eud_probe_dict,
            bundles["test"],
        )
    else:
        discounted_eud_results = _evaluate_discounted_eud(
            eud_args,
            bundles["test"].artifacts,
            test_metric_mask,
            test_refusal_eval,
            eud_probe_dict,
        )
    eud_metric_payload = {}
    eud_metric_payload.update(_main_summary_metrics("test", eud_results))
    eud_metric_payload.update(_disambig_question_summary_metrics("test", disambig_question_results))
    eud_metric_payload.update(_artifact_baseline_metrics("test", artifact_baselines))
    eud_metric_payload.update(_discounted_summary_metrics("test", discounted_eud_results))
    log_payload.update(eud_metric_payload)
    metrics["eud"] = eud_results
    metrics["disambig_question_eud"] = disambig_question_results
    metrics["artifact_baselines"] = artifact_baselines
    metrics["discounted_eud"] = discounted_eud_results

    save_json(os.path.join(run_dir, "config.json"), {"args": vars(args), "output_dir": run_dir})
    save_json(os.path.join(run_dir, "metrics.json"), metrics)

    if wandb.run is not None:
        wandb.run.summary.update(log_payload)
        for key, image in image_log_payload.items():
            wandb.run.summary[key] = image
        wandb.run.summary["artifact_load_dir"] = bundles["test"].artifacts.load_dir
        wandb.run.summary["label_source"] = bundles["test"].artifacts.label_source
        wandb.run.summary["method"] = "epn_d"
        wandb.run.summary["aux_probe"] = aux_result.probe_target
        wandb.run.summary["aux_probe_feature"] = aux_result.probe_feature
        wandb.run.summary["density_probe"] = str(args.density_probe)
        wandb.run.summary["density_probe_feature"] = str(args.density_probe_feature)
        wandb.run.summary["density_loss"] = str(args.loss)
        wandb.run.summary["density_loss_lambda"] = float(args.lambda_coef)
        wandb.run.summary["density_scheduler"] = bool(getattr(args, "scheduler", False))
        wandb.run.summary["rho_activation"] = RHO_ACTIVATION
        wandb.run.summary["zeta_grid"] = [float(zeta) for zeta in density_summary.get("zeta_grid", [])]
        wandb.run.summary["selected_zeta"] = float(density_summary.get("selected_zeta", float("nan")))
        wandb.run.summary["selected_zeta_index"] = int(density_summary.get("selected_zeta_index", -1))
        wandb.run.summary["selection_family"] = str(density_summary.get("selection_family", ""))
        wandb.run.summary["selection_metric"] = str(density_summary.get("selection_metric", ""))
        wandb.run.summary["selection_reason"] = str(density_summary.get("selection_reason", ""))
        wandb.run.summary["semantic_supervision_target"] = str(args.semantic_supervision_target)
        wandb.run.summary["pseudo_ood_enabled"] = bool(pseudo_ood_metadata["enabled"])
        wandb.run.summary["pseudo_ood_dataset_name"] = str(pseudo_ood_metadata["dataset_name"])
        wandb.run.summary["pseudo_ood_load_dir"] = str(pseudo_ood_metadata["load_dir"])
        wandb.run.summary["pseudo_ood_train_count"] = int(pseudo_ood_metadata["train_count"])
        wandb.run.summary["pseudo_ood_valid_count"] = int(pseudo_ood_metadata["valid_count"])
        wandb.log({**log_payload, **image_log_payload}, step=final_step, commit=False)
    else:
        final_step = 0

    if bool(args.run_baseline):
        if distributed_baselines:
            accelerator.wait_for_everyone()
            run_accelerated_optional_baselines(
                eud_args,
                artifacts=bundles["test"].artifacts,
                accelerator=accelerator,
                step=final_step,
            )
            accelerator.wait_for_everyone()
        else:
            _run_optional_baselines(eud_args, bundles["test"].artifacts, step=final_step)
    elif distributed_baselines:
        accelerator.wait_for_everyone()

    if distributed_baselines:
        cleanup_distributed(accelerator)

    if wandb.run is not None:
        wandb.log({}, step=final_step, commit=True)
        wandb.finish()
    print(f"[EPN-D] Saved outputs to {run_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
