from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import timedelta
from distutils.util import strtobool as _bool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from config import ITPSamplingConfig
from epn_helper import (
    _ZeroEvidenceProbe,
    _build_itp_sampling_cfg,
    _load_language_model_and_datasets,
    _prepare_exchange_dir,
    compute_ood_refusal_filter,
    count_or_frac,
    normalize_feature_type,
)
from feature_collector import _needs_8bit_quantized_loading, _resolve_hf_model_name
from semantic_cluster_collector import (
    EntailmentDeberta,
    _split_indices_for_process,
    collect_semantic_cluster_eval,
)
from train_epn import (
    AUX_RELIABILITY_EPS,
    DENSITY_LOSS_CHOICES,
    EPND_AUGMENTED_LOSS_NAME,
    EPN_D_DEFAULT_ZETA_GRID,
    EPN_SEMANTIC_PROB_SOURCE,
    RHO_ACTIVATION,
    SEMANTIC_SUPERVISION_TARGET_CHOICES,
    WORKING_DIR,
    EPNTable,
    ScalarProbe,
    _predict_aux_probs,
    append_pseudo_ood_rows,
    build_epn_table,
    build_predicted_rho_histogram_wandb_image,
    density_input_dim,
    epn_metric_stem,
    epn_summary_metrics,
    evaluate_epn_table,
    fit_auxiliary_probe,
    flatten_question_features,
    is_hallucination_bce_loss_name,
    load_pseudo_ood_bundles,
    load_split_bundle,
    normalize_density_activation,
    normalize_density_loss_name,
    normalize_feature_kind,
    normalize_layer_range,
    normalize_epn_semantic_prob_source,
    normalize_probe_target,
    normalize_pseudo_ood_dataset_name,
    normalize_semantic_supervision_target,
    predict_aux_reliability_by_question,
    predict_density,
    print_aux_probe_summary,
    response_slt_matrices,
    save_checkpoint,
    save_json,
    save_predictions,
    seed_everything,
    train_density_probe,
    validate_ambiguity_artifacts,
    validate_pseudo_ood_args,
)


def resolve_run_dir(args) -> str:
    model_id = str(args.model_name).rstrip("/").split("/")[-1]
    loss_token = str(args.loss).replace("/", "_")
    target_token = str(args.semantic_supervision_target).replace("/", "_")
    return os.path.join(
        os.path.abspath(args.output_dir),
        f"{model_id}_{args.dataset_name}_{int(args.n_samples)}_{args.judge_type}",
        (
            f"epn_d_itp_{args.layer_range}"
            f"_aux_{args.aux_probe}_{args.aux_probe_feature}_{args.aux_probe_type}"
            f"_density_{args.density_probe_feature}_{args.density_probe}"
            f"_loss_{loss_token}_target_{target_token}"
        ),
    )


def _feature_record(record: Dict[Any, Any], resp_id: int, qid: int) -> Dict[str, object]:
    if resp_id in record:
        payload = record[resp_id]
    elif str(resp_id) in record:
        payload = record[str(resp_id)]
    else:
        raise ValueError(f"Missing ITP feature record at qid={qid}, resp_id={resp_id}.")
    if not isinstance(payload, dict):
        raise ValueError(f"ITP feature record must be a mapping at qid={qid}, resp_id={resp_id}.")
    return payload


def _stack_itp_features(
    record: Dict[Any, Any],
    resp_ids: Sequence[int],
    field: str,
    qid: int,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for resp_id in resp_ids:
        payload = _feature_record(record, int(resp_id), qid)
        if field not in payload:
            raise ValueError(
                f"Missing ITP feature field={field!r} at qid={qid}, resp_id={int(resp_id)}."
            )
        arr = np.asarray(payload[field], dtype=np.float32).reshape(-1)
        rows.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
    if not rows:
        dim = int(expected_dim or 0)
        return np.zeros((0, dim), dtype=np.float32)
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    if expected_dim is not None and int(matrix.shape[1]) != int(expected_dim):
        raise ValueError(
            f"ITP feature dim mismatch at qid={qid}, field={field!r}: "
            f"expected={int(expected_dim)}, got={int(matrix.shape[1])}."
        )
    return matrix


def _response_count_for_qid(responses: Sequence[Sequence[str]], qid: int) -> int:
    return int(len(responses[int(qid)]))


def _valid_semantic_ids(ids: object, qid: int, n_resp: int) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    if int(arr.shape[0]) != int(n_resp):
        raise ValueError(
            f"ITP semantic id count mismatch at qid={qid}: ids={int(arr.shape[0])}, "
            f"responses={int(n_resp)}."
        )
    if np.any(arr < 0):
        raise ValueError(f"ITP semantic ids must be non-negative at qid={qid}.")
    return arr


def _response_prob_row(
    probs: np.ndarray,
    qid: int,
    n_resp: int,
    field_name: str,
) -> np.ndarray:
    if probs.ndim != 2:
        raise ValueError(f"Expected rank-2 {field_name}, got shape={probs.shape}.")
    if int(qid) >= int(probs.shape[0]):
        raise ValueError(f"{field_name} is missing qid={qid}.")
    if int(n_resp) > int(probs.shape[1]):
        raise ValueError(
            f"{field_name} width mismatch at qid={qid}: n_resp={n_resp}, width={probs.shape[1]}."
        )
    row = np.asarray(probs[int(qid), : int(n_resp)], dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(row)):
        raise ValueError(f"Non-finite {field_name} values at qid={qid}.")
    if np.any(row < 0.0):
        raise ValueError(f"Negative {field_name} values at qid={qid}.")
    return row


def _itp_aux_prob_rows(
    args,
    aux_result,
    test_bundle,
    collector_results: Dict[str, object],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    responses = collector_results["responses"]
    aux_feature = normalize_feature_kind(args.aux_probe_feature)
    prob_rows: List[np.ndarray] = []
    rel_rows: List[np.ndarray] = []

    if aux_feature == "slt":
        feature_records = collector_results.get("features_itp")
        if feature_records is None:
            raise ValueError("aux_probe_feature='slt' requires ITP response features.")
        for qid in range(len(responses)):
            n_resp = _response_count_for_qid(responses, qid)
            if n_resp == 0:
                prob_rows.append(np.zeros(0, dtype=np.float32))
                rel_rows.append(np.zeros(0, dtype=np.float32))
                continue
            resp_ids = list(range(n_resp))
            record = feature_records[qid] or {}
            X_resp = _stack_itp_features(
                record=record,
                resp_ids=resp_ids,
                field="density_slt",
                qid=qid,
                expected_dim=int(aux_result.feature_dim),
            )
            probs = _predict_aux_probs(aux_result.clf, aux_result.constant_prob, X_resp)
            probs = np.clip(probs, 0.0, 1.0).astype(np.float32, copy=False)
            rel = np.clip(1.0 - probs, AUX_RELIABILITY_EPS, 1.0).astype(np.float32, copy=False)
            prob_rows.append(probs)
            rel_rows.append(rel)
        return prob_rows, rel_rows

    if aux_feature not in test_bundle.X:
        raise ValueError(f"aux_probe_feature={aux_feature!r} requires X['{aux_feature}'] for test.")
    X_q = flatten_question_features(
        test_bundle.X[aux_feature],
        args.layer_range,
        split="test",
        feature=aux_feature,
    )
    probs_q = _predict_aux_probs(aux_result.clf, aux_result.constant_prob, X_q)
    if int(probs_q.shape[0]) != len(responses):
        raise ValueError(
            f"Aux question-prob count mismatch: probs={int(probs_q.shape[0])}, "
            f"responses={len(responses)}."
        )
    for qid in range(len(responses)):
        n_resp = _response_count_for_qid(responses, qid)
        probs = np.full(n_resp, float(probs_q[qid]), dtype=np.float32)
        probs = np.clip(probs, 0.0, 1.0).astype(np.float32, copy=False)
        rel = np.clip(1.0 - probs, AUX_RELIABILITY_EPS, 1.0).astype(np.float32, copy=False)
        prob_rows.append(probs)
        rel_rows.append(rel)
    return prob_rows, rel_rows


def _build_itp_epn_table(
    args,
    test_bundle,
    collector_results: Dict[str, object],
    aux_result,
    density_input_dim_expected: int,
) -> EPNTable:
    responses = collector_results["responses"]
    semantic_ids_all = collector_results.get("semantic_ids_itp", collector_results.get("semantic_ids"))
    if semantic_ids_all is None:
        raise ValueError("ITP collector did not return semantic_ids_itp or semantic_ids.")
    if len(semantic_ids_all) != len(responses):
        raise ValueError(
            f"ITP semantic_ids question-count mismatch: ids={len(semantic_ids_all)}, "
            f"responses={len(responses)}."
        )
    response_norm_sequence_probs = collector_results.get("response_norm_sequence_probs")
    if response_norm_sequence_probs is None:
        raise ValueError("EPN-D ITP evidence requires response_norm_sequence_probs.")
    response_norm_sequence_probs = np.asarray(response_norm_sequence_probs, dtype=np.float32)

    aux_prob_rows, rel_rows = _itp_aux_prob_rows(
        args=args,
        aux_result=aux_result,
        test_bundle=test_bundle,
        collector_results=collector_results,
    )
    target_mode = normalize_semantic_supervision_target(args.semantic_supervision_target)
    cluster_rows: List[np.ndarray] = []
    prob_rows: List[np.ndarray] = []
    support_rows: List[np.ndarray] = []
    support_total_rows: List[float] = []
    target_rows: List[np.ndarray] = []
    unsupported_target_rows: List[float] = []
    response_support_rows: List[np.ndarray] = []
    response_aux_rows: List[np.ndarray] = []
    response_cluster_rows: List[np.ndarray] = []
    K_vals: List[int] = []
    zero_support_count = 0

    for qid in range(len(responses)):
        n_resp = _response_count_for_qid(responses, qid)
        ids = _valid_semantic_ids(semantic_ids_all[qid], qid=qid, n_resp=n_resp)
        aux_prob = np.asarray(aux_prob_rows[qid], dtype=np.float32).reshape(-1)
        rel = np.asarray(rel_rows[qid], dtype=np.float32).reshape(-1)
        if aux_prob.shape[0] != n_resp or rel.shape[0] != n_resp:
            raise ValueError(
                f"ITP aux/response mismatch at qid={qid}: aux={aux_prob.shape[0]}, "
                f"rel={rel.shape[0]}, responses={n_resp}."
            )
        supports_by_response = _response_prob_row(
            response_norm_sequence_probs,
            qid=qid,
            n_resp=n_resp,
            field_name="response_norm_sequence_probs",
        )
        if n_resp == 0:
            cluster_rows.append(np.zeros(0, dtype=np.int64))
            prob_rows.append(np.zeros(0, dtype=np.float32))
            support_rows.append(np.zeros(0, dtype=np.float32))
            support_total_rows.append(0.0)
            target_rows.append(np.zeros(0, dtype=np.float32))
            unsupported_target_rows.append(0.0)
            response_support_rows.append(np.zeros(0, dtype=np.float32))
            response_aux_rows.append(np.zeros(0, dtype=np.float32))
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
        target_weights = (
            supports_by_response
            if target_mode in {"norm_seq_prob", "raw_norm_seq_prob"}
            else np.ones(n_resp, dtype=np.float32)
        )
        unsupported_target = float(np.sum(target_weights * aux_prob, dtype=np.float64))
        support_total = float(np.sum(supports_by_response, dtype=np.float64))
        if support_total <= 0.0:
            zero_support_count += 1
        for idx, cid in enumerate(unique_ids.tolist()):
            mask = ids == int(cid)
            probs[idx] = float(np.mean(mask.astype(np.float32)))
            supports[idx] = float(np.sum(supports_by_response[mask], dtype=np.float64))
            targets[idx] = float(np.sum(target_weights[mask] * rel[mask], dtype=np.float64))
        if target_mode in {"norm", "norm_seq_prob"}:
            target_total = float(np.sum(targets, dtype=np.float64) + unsupported_target)
            if target_total <= 0.0 or not np.isfinite(target_total):
                raise ValueError(
                    f"ITP augmented supervision target normalization denominator is invalid at qid={qid}."
                )
            targets = (targets / target_total).astype(np.float32, copy=False)
            unsupported_target = float(unsupported_target / target_total)

        cluster_rows.append(unique_ids)
        prob_rows.append(probs)
        support_rows.append(supports)
        support_total_rows.append(support_total)
        target_rows.append(targets)
        unsupported_target_rows.append(float(unsupported_target))
        response_support_rows.append(supports_by_response.astype(np.float32, copy=False))
        response_aux_rows.append(aux_prob.astype(np.float32, copy=False))
        response_cluster_rows.append(response_cluster_index)
        K_vals.append(int(unique_ids.shape[0]))

    max_clusters = max(1, max(K_vals) if K_vals else 1)
    n_questions = len(responses)
    cluster_ids = np.full((n_questions, max_clusters), -1, dtype=np.int64)
    semantic_probs = np.zeros((n_questions, max_clusters), dtype=np.float32)
    semantic_supports = np.zeros((n_questions, max_clusters), dtype=np.float32)
    cluster_targets = np.zeros((n_questions, max_clusters), dtype=np.float32)
    augmented_targets = np.zeros((n_questions, max_clusters + 1), dtype=np.float32)
    cluster_mask = np.zeros((n_questions, max_clusters), dtype=bool)
    for qid, (ids, probs, supports, targets, unsupported_target) in enumerate(
        zip(cluster_rows, prob_rows, support_rows, target_rows, unsupported_target_rows)
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
        augmented_targets[qid, max_clusters] = float(unsupported_target)
        cluster_mask[qid, :width] = True

    density_feature = normalize_feature_kind(args.density_probe_feature)
    question_features = None
    response_features = None
    if density_feature == "slt":
        feature_records = collector_results.get("features_itp")
        if feature_records is None:
            raise ValueError("density_probe_feature='slt' requires ITP response features.")
        response_features = []
        for qid in range(n_questions):
            n_resp = _response_count_for_qid(responses, qid)
            record = feature_records[qid] or {}
            response_features.append(
                _stack_itp_features(
                    record=record,
                    resp_ids=list(range(n_resp)),
                    field="density_slt",
                    qid=qid,
                    expected_dim=int(density_input_dim_expected),
                )
            )
    else:
        if density_feature not in test_bundle.X:
            raise ValueError(f"density_probe_feature={density_feature!r} requires X['{density_feature}'] for test.")
        question_features = flatten_question_features(
            test_bundle.X[density_feature],
            args.layer_range,
            split="test",
            feature=density_feature,
        )
        if int(question_features.shape[0]) != int(n_questions):
            raise ValueError(
                f"Question density feature count mismatch: features={int(question_features.shape[0])}, "
                f"ITP questions={int(n_questions)}."
            )
        if int(question_features.shape[1]) != int(density_input_dim_expected):
            raise ValueError(
                f"Question density feature dim mismatch: expected={int(density_input_dim_expected)}, "
                f"got={int(question_features.shape[1])}."
            )

    labels = np.asarray(collector_results.get("uncertainty_labels", test_bundle.artifacts.labels), dtype=np.int64)
    labels = labels.reshape(-1)
    y = np.asarray(collector_results.get("y", test_bundle.artifacts.y), dtype=np.int64).reshape(-1)
    metric_eval_mask = np.asarray(
        collector_results.get(
            "metric_eval_mask",
            compute_ood_refusal_filter(
                labels=labels,
                dataset_name=str(args.dataset_name),
                split="test",
                remove_refusal=bool(args.remove_refusal),
                load_dir=test_bundle.artifacts.load_dir,
            )[0],
        ),
        dtype=bool,
    ).reshape(-1)
    if labels.shape[0] != n_questions or y.shape[0] != n_questions or metric_eval_mask.shape[0] != n_questions:
        raise ValueError(
            "ITP label/mask count mismatch: "
            f"labels={labels.shape[0]}, y={y.shape[0]}, mask={metric_eval_mask.shape[0]}, "
            f"questions={n_questions}."
        )

    return EPNTable(
        split="test_itp",
        semantic_prob_source=EPN_SEMANTIC_PROB_SOURCE,
        semantic_supervision_target=target_mode,
        question_count=n_questions,
        cluster_ids=cluster_ids,
        semantic_probs=semantic_probs,
        semantic_supports=semantic_supports,
        semantic_support_totals=np.asarray(support_total_rows, dtype=np.float32).reshape(-1),
        cluster_targets=cluster_targets,
        augmented_targets=augmented_targets,
        cluster_mask=cluster_mask,
        K=np.asarray(K_vals, dtype=np.float32),
        question_features=question_features,
        response_features=response_features,
        response_supports=response_support_rows,
        response_aux_probs=response_aux_rows,
        response_cluster_index=response_cluster_rows,
        labels=labels,
        y=y,
        label_source=str(test_bundle.artifacts.label_source),
        metric_eval_mask=metric_eval_mask,
        ood_refusal_eval=dict(collector_results.get("ood_refusal_eval", {})),
        semantic_support_zero_count=int(zero_support_count),
        semantic_prob_zero_denominator_count=int(zero_support_count),
        pseudo_ood_mask=np.zeros(n_questions, dtype=bool),
    )


def _load_epn_d_density_checkpoint(path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object]]:
    checkpoint_path = resolve_epn_d_density_checkpoint_path(path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload or "metadata" not in payload:
        raise ValueError(f"Invalid EPN-D density checkpoint format: {checkpoint_path}")
    metadata = _normalize_epn_d_density_checkpoint_metadata(
        dict(payload["metadata"]),
        checkpoint_path=checkpoint_path,
    )
    model = ScalarProbe(
        probe_type=str(metadata["probe_type"]),
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def resolve_epn_d_density_checkpoint_path(load_path: str) -> str:
    if load_path is None or not str(load_path).strip():
        raise ValueError("--load_pretrained_EPN=True requires --load_path.")
    path = os.path.abspath(os.path.expanduser(str(load_path).strip()))
    if os.path.isdir(path):
        path = os.path.join(path, "density_probe.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing EPN-D density checkpoint: {path}")
    return path


def _normalize_epn_d_density_checkpoint_metadata(
    metadata: Dict[str, object],
    checkpoint_path: str,
) -> Dict[str, object]:
    if "input_dim" not in metadata:
        raise ValueError(f"EPN-D density checkpoint metadata is missing input_dim: {checkpoint_path}")
    metadata = dict(metadata)
    metadata["checkpoint_path"] = checkpoint_path
    metadata["semantic_prob_source"] = normalize_epn_semantic_prob_source(
        str(metadata.get("semantic_prob_source", EPN_SEMANTIC_PROB_SOURCE))
    )
    metadata["semantic_supervision_target"] = normalize_semantic_supervision_target(
        str(metadata.get("semantic_supervision_target", "norm"))
    )
    metadata["feature_kind"] = normalize_feature_kind(str(metadata.get("feature_kind", "tbg")))
    probe_type = str(metadata.get("probe_type", "lr")).strip().lower()
    if probe_type not in {"lr", "mlp"}:
        raise ValueError(f"Unsupported EPN-D density probe_type in checkpoint: {probe_type!r}")
    metadata["probe_type"] = probe_type
    if "layer_range" in metadata:
        metadata["layer_range"] = normalize_layer_range(str(metadata["layer_range"]))
    if "aux_probe" in metadata:
        aux_probe = normalize_probe_target(str(metadata["aux_probe"]))
        if aux_probe != "hallucination":
            raise ValueError("EPN-D ITP requires pretrained EPN-D aux_probe='hallucination'.")
        metadata["aux_probe"] = aux_probe
    if "aux_probe_feature" in metadata:
        metadata["aux_probe_feature"] = normalize_feature_kind(str(metadata["aux_probe_feature"]))
    if "aux_probe_type" in metadata:
        aux_probe_type = str(metadata["aux_probe_type"]).strip().lower()
        if aux_probe_type not in {"lr", "mlp"}:
            raise ValueError(f"Unsupported EPN-D aux_probe_type in checkpoint: {aux_probe_type!r}")
        metadata["aux_probe_type"] = aux_probe_type
    metadata["input_dim"] = int(metadata["input_dim"])
    metadata["hidden_dim"] = int(metadata.get("hidden_dim", 512))
    activation = metadata.get("rho_activation", metadata.get("activation", RHO_ACTIVATION))
    metadata["rho_activation"] = normalize_density_activation(str(activation))
    metadata["activation"] = str(metadata["rho_activation"])

    raw_zeta_grid = metadata.get("zeta_grid", [])
    if isinstance(raw_zeta_grid, np.ndarray):
        zeta_grid = [float(value) for value in raw_zeta_grid.reshape(-1).tolist()]
    elif isinstance(raw_zeta_grid, (list, tuple)):
        zeta_grid = [float(value) for value in raw_zeta_grid]
    elif raw_zeta_grid in (None, ""):
        zeta_grid = []
    else:
        zeta_grid = [float(raw_zeta_grid)]
    if "selected_zeta" in metadata:
        selected_zeta = float(metadata["selected_zeta"])
    elif zeta_grid:
        selected_zeta = float(zeta_grid[0])
    else:
        selected_zeta = 1.0
    metadata["selected_zeta"] = selected_zeta
    metadata["zeta_grid"] = zeta_grid if zeta_grid else [selected_zeta]
    if "selected_zeta_index" not in metadata:
        metadata["selected_zeta_index"] = int(
            np.argmin(np.abs(np.asarray(metadata["zeta_grid"], dtype=np.float32) - selected_zeta))
        )
    else:
        metadata["selected_zeta_index"] = int(metadata["selected_zeta_index"])
    if "loss" in metadata:
        metadata["loss"] = normalize_density_loss_name(str(metadata["loss"]))
    return metadata


def _load_epn_d_density_checkpoint_metadata(path: str) -> Dict[str, object]:
    checkpoint_path = resolve_epn_d_density_checkpoint_path(path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError(f"Invalid EPN-D density checkpoint format: {checkpoint_path}")
    return _normalize_epn_d_density_checkpoint_metadata(
        dict(payload["metadata"]),
        checkpoint_path=checkpoint_path,
    )


def _apply_pretrained_density_metadata_to_args(args, metadata: Dict[str, object]) -> None:
    if str(metadata["semantic_prob_source"]) != EPN_SEMANTIC_PROB_SOURCE:
        raise ValueError(
            "EPN-D ITP only supports pretrained checkpoints with "
            f"semantic_prob_source={EPN_SEMANTIC_PROB_SOURCE!r}; "
            f"got {metadata['semantic_prob_source']!r}."
        )
    args.density_probe = str(metadata["probe_type"])
    args.density_probe_feature = str(metadata["feature_kind"])
    args.hidden_dim = int(metadata["hidden_dim"])
    args.activation = str(metadata["rho_activation"])
    args.semantic_supervision_target = str(metadata["semantic_supervision_target"])
    args.zeta_grid = [float(zeta) for zeta in metadata.get("zeta_grid", [])]
    if "layer_range" in metadata:
        args.layer_range = str(metadata["layer_range"])
    if "aux_probe" in metadata:
        args.aux_probe = str(metadata["aux_probe"])
    if "aux_probe_feature" in metadata:
        args.aux_probe_feature = str(metadata["aux_probe_feature"])
    if "aux_probe_type" in metadata:
        args.aux_probe_type = str(metadata["aux_probe_type"])
    if "loss" in metadata:
        args.loss = str(metadata["loss"])


def _save_itp_response_jsonl(path: str, responses: Sequence[Sequence[str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in responses:
            handle.write(json.dumps({"responses": list(row)}, ensure_ascii=False) + "\n")


def _save_itp_npz(
    path: str,
    table: EPNTable,
    collector_results: Dict[str, object],
    density_outputs: Dict[str, np.ndarray],
    results: Dict[str, object],
) -> None:
    np.savez_compressed(
        path,
        responses=np.asarray(collector_results["responses"], dtype=object),
        semantic_ids_itp=np.asarray(
            collector_results.get("semantic_ids_itp", collector_results.get("semantic_ids")),
            dtype=object,
        ),
        response_sequence_probs=np.asarray(
            collector_results.get("response_sequence_probs", np.zeros((table.question_count, 0), dtype=np.float32)),
            dtype=np.float32,
        ),
        response_norm_sequence_probs=np.asarray(
            collector_results.get(
                "response_norm_sequence_probs",
                np.zeros((table.question_count, 0), dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        itp_valid_response_counts=np.asarray(
            collector_results.get("itp_valid_response_counts", np.zeros(table.question_count, dtype=np.int64)),
            dtype=np.int64,
        ),
        itp_refusal_counts=np.asarray(
            collector_results.get("itp_refusal_counts", np.zeros(table.question_count, dtype=np.int64)),
            dtype=np.int64,
        ),
        itp_attempt_counts=np.asarray(
            collector_results.get("itp_attempt_counts", np.zeros(table.question_count, dtype=np.int64)),
            dtype=np.int64,
        ),
        itp_refusal_rates=np.asarray(
            collector_results.get("itp_refusal_rates", np.zeros(table.question_count, dtype=np.float32)),
            dtype=np.float32,
        ),
        cluster_ids=table.cluster_ids,
        cluster_mask=table.cluster_mask,
        response_cluster_index=np.asarray(table.response_cluster_index, dtype=object),
        response_aux_probs=np.asarray(table.response_aux_probs, dtype=object),
        response_supports=np.asarray(table.response_supports, dtype=object),
        response_supports_padded=np.asarray(results["response_supports"], dtype=np.float32),
        rho_logits_by_response=np.asarray(
            density_outputs.get("rho_logits_by_response", density_outputs["rho_logits"]),
            dtype=np.float32,
        ),
        rho_by_response=np.asarray(results["rho_by_response"], dtype=np.float32),
        q_rel=np.asarray(results["q_rel"], dtype=np.float32),
        retained_evidence_by_response=np.asarray(results["retained_evidence_by_response"], dtype=np.float32),
        retained_evidence_by_cluster=np.asarray(results["retained_evidence_by_cluster"], dtype=np.float32),
        retained_evidence_total=np.asarray(results["retained_evidence_total"], dtype=np.float32),
        alpha=np.asarray(results["alpha"], dtype=np.float32),
        alpha0=np.asarray(results["alpha0"], dtype=np.float32),
        vacuity=np.asarray(results["vacuity"], dtype=np.float32),
        dissonance=np.asarray(results["dissonance"], dtype=np.float32),
        selected_zeta=np.asarray([float(results["selected_zeta"])], dtype=np.float32),
        zeta_grid=np.asarray(density_outputs.get("zeta_grid", []), dtype=np.float32),
        uncertainty_labels=table.labels,
        y=table.y,
        metric_eval_mask=table.metric_eval_mask,
        label_source=np.asarray([table.label_source], dtype=object),
        semantic_prob_source=np.asarray([table.semantic_prob_source], dtype=object),
        semantic_supervision_target=np.asarray([table.semantic_supervision_target], dtype=object),
    )


def _itp_log_payload(results: Dict[str, object], collector_results: Dict[str, object]) -> Dict[str, float]:
    payload = epn_summary_metrics("test", results, EPN_SEMANTIC_PROB_SOURCE)
    aliases = {
        key.replace("test/", "test/itp_"): value
        for key, value in payload.items()
    }
    payload.update(aliases)
    valid_counts = np.asarray(
        collector_results.get("itp_valid_response_counts", []),
        dtype=np.float32,
    ).reshape(-1)
    refusal_rates = np.asarray(
        collector_results.get("itp_refusal_rates", []),
        dtype=np.float32,
    ).reshape(-1)
    payload["test/mean_itp_valid_response_count"] = (
        float(np.mean(valid_counts)) if valid_counts.size > 0 else 0.0
    )
    payload["test/mean_itp_refusal_rate"] = (
        float(np.mean(refusal_rates)) if refusal_rates.size > 0 else 0.0
    )
    payload["test/metric_eval_count"] = float(np.sum(np.asarray(results["metric_eval_mask"], dtype=bool)))
    return payload


def do_itp_inference(
    args,
    run_dir: str,
    load_dir: str,
    test_bundle,
    aux_result,
    density_model: torch.nn.Module,
    density_metadata: Dict[str, object],
    device: torch.device,
    accelerator: Optional[Accelerator] = None,
    exchange_dir: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    split = "test"
    if _needs_8bit_quantized_loading(args.model_name):
        if accelerator is None or bool(accelerator.is_main_process):
            resolved_model_name = _resolve_hf_model_name(args.model_name)
            print(
                f"[EPN-D-ITP][int8] Loading {resolved_model_name} with 8-bit quantization.",
                flush=True,
            )
    model, tokenizer, split_datasets, fewshot_pool, data_collator = _load_language_model_and_datasets(
        args=args,
        load_dir=load_dir,
        device=device,
        accelerator=accelerator,
    )
    model.eval()
    density_model.eval()

    question_count = int(len(test_bundle.artifacts.responses))
    distributed = accelerator is not None and int(accelerator.num_processes) > 1
    full_dataset = split_datasets[split]
    if distributed:
        local_indices = _split_indices_for_process(question_count, accelerator)
        shard_dataset = full_dataset.select(local_indices)
    else:
        local_indices = list(range(question_count))
        shard_dataset = full_dataset

    need_generated_slt = args.aux_probe_feature == "slt" or args.density_probe_feature == "slt"
    collector_density_feature = "slt" if need_generated_slt else str(args.density_probe_feature)
    layer_indices = np.arange(test_bundle.X["tbg"].shape[0], dtype=np.int64)
    if args.layer_range == "last_1":
        layer_indices = layer_indices[-1:]
    elif args.layer_range == "last_5":
        layer_indices = layer_indices[-5:]
    elif args.layer_range != "all":
        raise ValueError(f"Unsupported layer_range={args.layer_range!r}.")

    sampling_cfg = _build_itp_sampling_cfg(args, device)
    entailment_model = EntailmentDeberta(device=device) if bool(args.itp_semantic_clustering) else None
    collector_results = collect_semantic_cluster_eval(
        eup=_ZeroEvidenceProbe().to(device),
        model=model,
        load_dir=load_dir,
        split=split,
        layer_range=layer_indices,
        tokenizer=tokenizer,
        dataset=shard_dataset,
        collator=data_collator,
        fewshot_pool=fewshot_pool,
        n_resp=args.n_resp,
        fix_n_eval_resp=True,
        n_max_resp=args.n_eval_max_resp,
        conf_thres=float("inf"),
        cfg=sampling_cfg,
        output_feature=bool(need_generated_slt),
        MSE_type="response-wise",
        sanity_check=False,
        uncertainty_labels=np.asarray(test_bundle.artifacts.labels, dtype=np.int64),
        dataset_name=args.dataset_name,
        remove_refusal=bool(args.remove_refusal),
        feature_type=args.feature_type,
        density_probe_feature=collector_density_feature,
        need_response_embeddings=False,
        itp_semantic_clustering=bool(args.itp_semantic_clustering),
        entailment_model=entailment_model,
        strict_entailment=False,
        accelerator=accelerator,
        global_indices=local_indices,
        expected_total=question_count,
        exchange_dir=exchange_dir,
        split_name=split,
        idx_subset=local_indices if distributed else None,
    )
    if distributed and accelerator is not None:
        accelerator.wait_for_everyone()
    if collector_results is None:
        return None

    if aux_result is None:
        raise RuntimeError("Main process must have an auxiliary probe result for ITP EPN-D inference.")
    if len(collector_results["responses"]) != question_count:
        raise ValueError(
            f"ITP response/question-count mismatch: responses={len(collector_results['responses'])}, "
            f"questions={question_count}."
        )

    density_input_dim_expected = int(density_metadata["input_dim"])
    itp_table = _build_itp_epn_table(
        args=args,
        test_bundle=test_bundle,
        collector_results=collector_results,
        aux_result=aux_result,
        density_input_dim_expected=density_input_dim_expected,
    )
    if int(density_input_dim(itp_table)) != density_input_dim_expected:
        raise ValueError(
            f"ITP table density input dim mismatch: expected={density_input_dim_expected}, "
            f"got={int(density_input_dim(itp_table))}."
        )

    density_outputs = predict_density(
        density_model,
        itp_table,
        int(args.batch_size),
        device,
        zeta=float(density_metadata["selected_zeta"]),
        activation=str(density_metadata.get("rho_activation", RHO_ACTIVATION)),
    )
    density_outputs["zeta_grid"] = np.asarray(density_metadata.get("zeta_grid", []), dtype=np.float32)
    epn_eval = evaluate_epn_table(itp_table, density_outputs)

    predictions_path = os.path.join(run_dir, "predictions_itp_test.npz")
    save_predictions(predictions_path, itp_table, density_outputs, epn_eval)
    inference_path = os.path.join(run_dir, "inference_test.npz")
    _save_itp_npz(inference_path, itp_table, collector_results, density_outputs, epn_eval)
    responses_path = os.path.join(run_dir, "infer_resp.jsonl")
    _save_itp_response_jsonl(responses_path, collector_results["responses"])

    log_payload = _itp_log_payload(epn_eval, collector_results)
    if wandb.run is not None:
        wandb.run.summary.update(log_payload)
        wandb.run.summary["method"] = "epn_d_itp"
        wandb.run.summary["itp_semantic_clustering"] = bool(args.itp_semantic_clustering)
        wandb.run.summary["n_eval_max_resp"] = int(args.n_eval_max_resp)
        wandb.log(log_payload, step=int(density_metadata.get("n_iter", 0)) + 1, commit=False)

    stem = epn_metric_stem(EPN_SEMANTIC_PROB_SOURCE)
    print(
        f"[test][epn_d_itp] "
        f"auroc_vacuity={log_payload[f'test/auroc_{stem}_vacuity']:.4f} | "
        f"aupr_vacuity={log_payload[f'test/aupr_{stem}_vacuity']:.4f} | "
        f"auroc_dissonance={log_payload[f'test/auroc_{stem}_dissonance']:.4f} | "
        f"aupr_dissonance={log_payload[f'test/aupr_{stem}_dissonance']:.4f}",
        flush=True,
    )

    return {
        "split": "test",
        "method": "epn_d_itp",
        "n_eval_max_resp": int(args.n_eval_max_resp),
        "itp_semantic_clustering": bool(args.itp_semantic_clustering),
        "question_count": int(itp_table.question_count),
        "metric_eval_count": int(np.sum(itp_table.metric_eval_mask)),
        "artifact_path": inference_path,
        "predictions_path": predictions_path,
        "responses_path": responses_path,
        "collector": {
            "mean_valid_response_count": float(log_payload["test/mean_itp_valid_response_count"]),
            "mean_refusal_rate": float(log_payload["test/mean_itp_refusal_rate"]),
            "itp_valid_response_counts": np.asarray(
                collector_results.get("itp_valid_response_counts", []),
                dtype=np.int64,
            ),
            "itp_refusal_rates": np.asarray(
                collector_results.get("itp_refusal_rates", []),
                dtype=np.float32,
            ),
        },
        "epn_d": epn_eval,
        "summary": log_payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EPN-D trainer with explicit iterative test-time prompting."
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
    parser.add_argument(
        "--feature_type",
        type=str,
        default="tbg_concat",
        choices=["tbg_concat", "tbg", "slt", "tbg_slt_hidden", "tbg_slt_concat"],
        help="ITP collector feature mode. No semantic probability probe is trained in this script.",
    )
    parser.add_argument("--aux_probe", type=str, default="hallucination", choices=["hallucination", "ambiguity"])
    parser.add_argument("--aux_probe_feature", type=str, default="tbg", choices=["tbg", "slt"])
    parser.add_argument("--aux_probe_type", type=str, default="lr", choices=["lr", "mlp"])
    parser.add_argument("--density_probe", type=str, default="lr", choices=["lr", "mlp"])
    parser.add_argument("--density_probe_feature", type=str, default="tbg", choices=["tbg", "slt"])
    parser.add_argument(
        "--activation",
        type=str,
        default=RHO_ACTIVATION,
        help="Compatibility option; EPN-D rho activation is sigmoid.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=EPND_AUGMENTED_LOSS_NAME,
        choices=list(DENSITY_LOSS_CHOICES),
    )
    parser.add_argument(
        "--semantic_supervision_target",
        type=str,
        default="norm",
        choices=list(SEMANTIC_SUPERVISION_TARGET_CHOICES),
    )
    parser.add_argument("--lambda", dest="lambda_coef", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=_bool, default=False)
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
    )
    parser.add_argument("--remove_refusal", type=_bool, default=False)
    parser.add_argument("--zeta_grid", type=float, nargs="+", default=list(EPN_D_DEFAULT_ZETA_GRID))
    parser.add_argument("--zeta_selection", type=_bool, default=False)
    parser.add_argument("--n_eval_max_resp", type=int, default=5)
    parser.add_argument("--itp_max_new_tokens", type=int, default=int(ITPSamplingConfig.max_new_tokens))
    parser.add_argument("--itp_temperature", type=float, default=float(ITPSamplingConfig.temperature))
    parser.add_argument("--itp_top_p", type=float, default=float(ITPSamplingConfig.top_p))
    parser.add_argument("--itp_do_sample", type=_bool, default=bool(ITPSamplingConfig.do_sample))
    parser.add_argument("--itp_batch_size", type=int, default=int(ITPSamplingConfig.batch_size))
    parser.add_argument("--itp_semantic_clustering", type=_bool, default=False)
    parser.add_argument("--test", type=_bool, default=False)
    parser.add_argument(
        "--load_pretrained_EPN",
        type=_bool,
        default=False,
        help="Load a pretrained EPN-D density_probe checkpoint instead of training a new one.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to an EPN-D run directory or density_probe.pt when --load_pretrained_EPN=True.",
    )
    parser.add_argument("--output_dir", type=str, default=os.path.join(WORKING_DIR, "epn_runs_ver3"))
    parser.add_argument("--save_config", type=_bool, default=True)
    return parser


def _normalize_args(args) -> None:
    args.load_pretrained_EPN = bool(getattr(args, "load_pretrained_EPN", False))
    args.load_path = None if getattr(args, "load_path", None) is None else str(args.load_path).strip()
    if args.load_pretrained_EPN and not args.load_path:
        raise ValueError("--load_pretrained_EPN=True requires --load_path.")
    args.layer_range = normalize_layer_range(args.layer_range)
    args.feature_type = normalize_feature_type(args.feature_type)
    args.aux_probe = normalize_probe_target(args.aux_probe)
    if args.aux_probe != "hallucination":
        raise ValueError("EPN-D ITP requires --aux_probe=hallucination for rho supervision.")
    args.aux_probe_feature = normalize_feature_kind(args.aux_probe_feature)
    args.density_probe_feature = normalize_feature_kind(args.density_probe_feature)
    args.activation = normalize_density_activation(args.activation)
    args.loss = normalize_density_loss_name(args.loss)
    if (
        not args.load_pretrained_EPN
        and is_hallucination_bce_loss_name(args.loss)
        and bool(getattr(args, "use_pseudo_OOD", False))
    ):
        raise ValueError(
            "HallucinationBCEWithLogitsLoss requires --use_pseudo_OOD=False; "
            "pseudo-OOD rows are not hallucination-label supervision."
        )
    args.semantic_supervision_target = normalize_semantic_supervision_target(args.semantic_supervision_target)
    if args.load_pretrained_EPN:
        args.use_pseudo_OOD = False
        args.pseudo_OOD_dataset_name = normalize_pseudo_ood_dataset_name(
            getattr(args, "pseudo_OOD_dataset_name", None)
        )
    else:
        validate_pseudo_ood_args(args)


def main(args) -> None:
    _normalize_args(args)
    pretrained_density_checkpoint_path: Optional[str] = None
    pretrained_density_metadata: Dict[str, object] = {}
    if bool(args.load_pretrained_EPN):
        pretrained_density_checkpoint_path = resolve_epn_d_density_checkpoint_path(str(args.load_path))
        pretrained_density_metadata = _load_epn_d_density_checkpoint_metadata(pretrained_density_checkpoint_path)
        _apply_pretrained_density_metadata_to_args(args, pretrained_density_metadata)
        args.load_path = pretrained_density_checkpoint_path
    seed_everything(int(args.seed))
    timeout_min = int(os.environ.get("SE_DDP_TIMEOUT_MINUTES", "180"))
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=timeout_min))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    distributed = int(accelerator.num_processes) > 1
    device = accelerator.device if distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_process = bool(accelerator.is_main_process)

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

    run_dir = resolve_run_dir(args)
    if main_process:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    exchange_dir = os.path.join(run_dir, ".dist_exchange")
    if distributed and main_process:
        _prepare_exchange_dir(exchange_dir)
    accelerator.wait_for_everyone()

    run = wandb.init(project="EPN-IT", config={**vars(args), "output_dir": run_dir}) if main_process else None
    if run is not None:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("valid/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

    metrics: Optional[Dict[str, object]] = None
    aux_result = None
    density_summary: Dict[str, object] = {}
    density_checkpoint_path = (
        str(pretrained_density_checkpoint_path)
        if bool(args.load_pretrained_EPN)
        else os.path.join(run_dir, "density_probe.pt")
    )

    if main_process:
        if bool(args.load_pretrained_EPN):
            pseudo_ood_bundles = {}
            pseudo_ood_training_enabled = False
            pseudo_ood_metadata: Dict[str, object] = {
                "enabled": bool(pretrained_density_metadata.get("pseudo_ood_enabled", False)),
                "dataset_name": str(pretrained_density_metadata.get("pseudo_ood_dataset_name", "")),
                "load_dir": str(pretrained_density_metadata.get("pseudo_ood_load_dir", "")),
                "train_count": int(pretrained_density_metadata.get("train_pseudo_ood_count", 0)),
                "valid_count": int(pretrained_density_metadata.get("valid_pseudo_ood_count", 0)),
                "test_count": int(pretrained_density_metadata.get("test_pseudo_ood_count", 0)),
                "source": "pretrained_density_checkpoint",
            }
        else:
            pseudo_ood_bundles = load_pseudo_ood_bundles(args)
            pseudo_ood_training_enabled = bool(pseudo_ood_bundles)
            pseudo_ood_metadata = {
                "enabled": bool(pseudo_ood_training_enabled),
                "dataset_name": str(args.pseudo_OOD_dataset_name) if pseudo_ood_training_enabled else "",
                "load_dir": pseudo_ood_bundles["train"].load_dir if pseudo_ood_training_enabled else "",
                "train_count": (
                    int(pseudo_ood_bundles["train"].labels.shape[0]) if pseudo_ood_training_enabled else 0
                ),
                "valid_count": (
                    int(pseudo_ood_bundles["valid"].labels.shape[0]) if pseudo_ood_training_enabled else 0
                ),
                "test_count": (
                    int(pseudo_ood_bundles["test"].labels.shape[0]) if pseudo_ood_training_enabled else 0
                ),
            }
        if bool(pseudo_ood_metadata["enabled"]):
            print(
                "[Pseudo-OOD] "
                f"dataset={pseudo_ood_metadata['dataset_name']} | "
                f"load_dir={pseudo_ood_metadata['load_dir']} | "
                f"train={pseudo_ood_metadata['train_count']} | "
                f"valid={pseudo_ood_metadata['valid_count']} | "
                f"test={pseudo_ood_metadata['test_count']}",
                flush=True,
            )

        test_metric_mask, _ = compute_ood_refusal_filter(
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

        train_loss_table = epn_tables["train"]
        valid_loss_table = epn_tables["valid"]
        if pseudo_ood_training_enabled:
            train_loss_table = append_pseudo_ood_rows(args, train_loss_table, pseudo_ood_bundles["train"])
            valid_loss_table = append_pseudo_ood_rows(args, valid_loss_table, pseudo_ood_bundles["valid"])
            print(
                "[EPN-D] pseudo-OOD loss rows appended: "
                f"train={int(np.sum(train_loss_table.pseudo_ood_mask))}, "
                f"valid={int(np.sum(valid_loss_table.pseudo_ood_mask))}.",
                flush=True,
            )

        if bool(args.load_pretrained_EPN):
            density_model, density_metadata = _load_epn_d_density_checkpoint(
                density_checkpoint_path,
                device=device,
            )
            density_results = {
                "model": density_model,
                "selected_zeta": float(density_metadata["selected_zeta"]),
                "rho_activation": str(density_metadata.get("rho_activation", RHO_ACTIVATION)),
                "zeta_grid": [float(zeta) for zeta in density_metadata.get("zeta_grid", [])],
                "n_iter": int(density_metadata.get("n_iter", 0)),
            }
            density_summary = dict(density_metadata)
            density_summary["loaded_pretrained"] = True
            density_summary["checkpoint_written"] = False
            print(
                f"[EPN-D-ITP] Loaded pretrained density_probe from {density_checkpoint_path}",
                flush=True,
            )
        else:
            density_results = train_density_probe(
                args,
                train_loss_table,
                valid_loss_table,
                device,
                semantic_prob_source=semantic_prob_source,
            )
            save_checkpoint(
                density_checkpoint_path,
                density_results["state_dict"],
                {
                    "method": "epn_d_itp",
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
                    "activation": density_results["rho_activation"],
                    "rho_activation": density_results["rho_activation"],
                    "zeta_grid": density_results["zeta_grid"],
                    "selected_zeta": density_results["selected_zeta"],
                    "selected_zeta_index": density_results["selected_zeta_index"],
                    "selection_family": density_results["selection_family"],
                    "zeta_grid_results": density_results["zeta_grid_results"],
                    "loss": density_results["loss"],
                    "effective_loss": density_results["effective_loss"],
                    "base_effective_loss": density_results["base_effective_loss"],
                    "density_output": density_results["density_output"],
                    "lambda_coef": density_results["lambda_coef"],
                    "pseudo_ood_enabled": density_results["pseudo_ood_enabled"],
                    "pseudo_ood_dataset_name": (
                        str(args.pseudo_OOD_dataset_name) if pseudo_ood_training_enabled else ""
                    ),
                    "pseudo_ood_load_dir": pseudo_ood_metadata["load_dir"],
                    "train_pseudo_ood_count": density_results["train_pseudo_ood_count"],
                    "valid_pseudo_ood_count": density_results["valid_pseudo_ood_count"],
                    "scheduler": density_results["scheduler"],
                    "n_epochs": density_results["n_epochs"],
                    "tol": density_results["tol"],
                    "n_iter": density_results["n_iter"],
                    "converged": density_results["converged"],
                },
            )
            density_summary = {
                key: value
                for key, value in density_results.items()
                if key not in {"model", "state_dict"}
            }
            density_summary["loaded_pretrained"] = False
            density_summary["checkpoint_written"] = True
            density_summary["checkpoint_path"] = density_checkpoint_path
        del train_loss_table, valid_loss_table

        metrics = {
            "artifact_load_dir": bundles["train"].artifacts.load_dir,
            "output_dir": run_dir,
            "device": str(device),
            "args": vars(args),
            "method": "epn_d_itp",
            "semantic_probability_probe": {"enabled": False},
            "semantic_prob_source": semantic_prob_source,
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
            "density_probe": density_summary,
            "pseudo_ood": pseudo_ood_metadata,
            "splits": {},
        }

        log_payload: Dict[str, float] = {
            "test/aux_probe_auroc": float(aux_result.AUROC),
            "test/aux_probe_aupr": float(aux_result.AUPR),
        }
        image_log_payload: Dict[str, object] = {}
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
            if split == "test":
                log_payload.update({
                    key.replace("test/", "artifact_test/"): value
                    for key, value in split_metrics.items()
                })
            metrics["splits"][split] = {
                "summary": split_metrics,
                "epn_d": epn_eval,
            }
            stem = epn_metric_stem(semantic_prob_source)
            if split == "test" and wandb.run is not None:
                rho_hist_image = build_predicted_rho_histogram_wandb_image(
                    np.asarray(epn_eval["rho_by_response"], dtype=np.float32),
                    split=split,
                )
                if rho_hist_image is not None:
                    image_log_payload[f"{split}/{stem}_artifact_predicted_rho_hist"] = rho_hist_image
            print(
                f"[{split}][epn_d artifact] "
                f"auroc_vacuity={split_metrics[f'{split}/auroc_{stem}_vacuity']:.4f} | "
                f"aupr_vacuity={split_metrics[f'{split}/aupr_{stem}_vacuity']:.4f} | "
                f"auroc_dissonance={split_metrics[f'{split}/auroc_{stem}_dissonance']:.4f} | "
                f"aupr_dissonance={split_metrics[f'{split}/aupr_{stem}_dissonance']:.4f}",
                flush=True,
            )

        if bool(args.save_config):
            save_json(os.path.join(run_dir, "config.json"), {"args": vars(args), "output_dir": run_dir})
        if wandb.run is not None:
            wandb.run.summary.update(log_payload)
            for key, image in image_log_payload.items():
                wandb.run.summary[key] = image
            wandb.run.summary["artifact_load_dir"] = bundles["test"].artifacts.load_dir
            wandb.run.summary["label_source"] = bundles["test"].artifacts.label_source
            wandb.run.summary["method"] = "epn_d_itp"
            wandb.run.summary["semantic_probability_probe_enabled"] = False
            wandb.run.summary["aux_probe"] = aux_result.probe_target
            wandb.run.summary["aux_probe_feature"] = aux_result.probe_feature
            wandb.run.summary["density_probe"] = str(args.density_probe)
            wandb.run.summary["density_probe_feature"] = str(args.density_probe_feature)
            wandb.run.summary["load_pretrained_EPN"] = bool(args.load_pretrained_EPN)
            wandb.run.summary["load_path"] = str(density_checkpoint_path if bool(args.load_pretrained_EPN) else "")
            wandb.run.summary["density_loss"] = str(args.loss)
            wandb.run.summary["density_loss_lambda"] = float(args.lambda_coef)
            wandb.run.summary["density_scheduler"] = bool(getattr(args, "scheduler", False))
            wandb.run.summary["rho_activation"] = RHO_ACTIVATION
            wandb.run.summary["zeta_grid"] = [float(zeta) for zeta in density_summary.get("zeta_grid", [])]
            wandb.run.summary["selected_zeta"] = float(density_summary.get("selected_zeta", float("nan")))
            wandb.run.summary["selected_zeta_index"] = int(density_summary.get("selected_zeta_index", -1))
            wandb.run.summary["selection_family"] = str(density_summary.get("selection_family", ""))
            wandb.run.summary["semantic_supervision_target"] = str(args.semantic_supervision_target)
            wandb.run.summary["pseudo_ood_enabled"] = bool(pseudo_ood_metadata["enabled"])
            wandb.run.summary["pseudo_ood_dataset_name"] = str(pseudo_ood_metadata["dataset_name"])
            wandb.run.summary["pseudo_ood_load_dir"] = str(pseudo_ood_metadata["load_dir"])
            wandb.run.summary["pseudo_ood_train_count"] = int(pseudo_ood_metadata["train_count"])
            wandb.run.summary["pseudo_ood_valid_count"] = int(pseudo_ood_metadata["valid_count"])
            wandb.run.summary["n_eval_max_resp"] = int(args.n_eval_max_resp)
            wandb.run.summary["itp_semantic_clustering"] = bool(args.itp_semantic_clustering)
            wandb.log({**log_payload, **image_log_payload}, step=int(density_summary.get("n_iter", 0)), commit=False)

        del density_model
        del density_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    if bool(args.test):
        density_model, density_metadata = _load_epn_d_density_checkpoint(density_checkpoint_path, device=device)
        inference_metrics = do_itp_inference(
            args=args,
            run_dir=run_dir,
            load_dir=bundles["test"].artifacts.load_dir,
            test_bundle=bundles["test"],
            aux_result=aux_result,
            density_model=density_model,
            density_metadata=density_metadata,
            device=device,
            accelerator=accelerator if distributed else None,
            exchange_dir=exchange_dir if distributed else None,
        )
        if main_process and metrics is not None and inference_metrics is not None:
            metrics["test_inference"] = inference_metrics
        del density_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not main_process:
        return

    if metrics is not None:
        save_json(os.path.join(run_dir, "metrics.json"), metrics)
    if wandb.run is not None:
        wandb.log({}, step=int(density_summary.get("n_iter", 0)) + 1, commit=True)
        wandb.finish()
    print(f"[EPN-D-ITP] Saved outputs to {run_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
