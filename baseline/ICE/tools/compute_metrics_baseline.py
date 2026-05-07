import argparse
import json
import os
import sys
import inspect
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
try:
    import wandb
except ImportError:
    wandb = None

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from src.evaluation import is_ambig

ALEATORIC_KEY = "data_uncertainty"
EPISTEMIC_KEY = "model_uncertainty"
THRESHOLDS = np.arange(1, 100) / 100.0


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def _parse_binary_scalar(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value > 0)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError("Label is NaN.")
        return int(value > 0)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "ambiguous", "ood", "multipleqas"}:
            return 1
        if normalized in {"0", "false", "no", "unambiguous", "id", "singleanswer"}:
            return 0
        try:
            return int(float(normalized) > 0)
        except ValueError as exc:
            raise ValueError(f"Cannot parse label value: {value}") from exc
    raise ValueError(f"Unsupported label type: {type(value)}")


def _to_binary_label(raw_label, dataset_name):
    ds = (dataset_name or "").lower()
    if ds in {"ambigqa", "ambig_qa"}:
        if isinstance(raw_label, list):
            return int(is_ambig(raw_label))
        return _parse_binary_scalar(raw_label)

    if isinstance(raw_label, list):
        if len(raw_label) == 0:
            raise ValueError("Received empty label list.")
        if len(raw_label) == 1:
            return _parse_binary_scalar(raw_label[0])
        # fallback: if list contains AmbigQA tags, still parse as ambiguity
        if "singleAnswer" in raw_label or "multipleQAs" in raw_label:
            return int(is_ambig(raw_label))
        raise ValueError(f"Cannot infer binary label from list for dataset `{dataset_name}`: {raw_label}")
    return _parse_binary_scalar(raw_label)


def _compute_score_metrics(labels, scores):
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    pos_mean = float(np.mean(pos_scores)) if len(pos_scores) > 0 else float("nan")
    neg_mean = float(np.mean(neg_scores)) if len(neg_scores) > 0 else float("nan")

    all_f1s = []
    all_precisions = []
    all_recalls = []
    for threshold in THRESHOLDS:
        preds = (scores > threshold).astype(np.int64)
        all_f1s.append(
            f1_score(labels, preds, zero_division=0)
        )
        all_precisions.append(
            precision_score(labels, preds, zero_division=0)
        )
        all_recalls.append(
            recall_score(labels, preds, zero_division=0)
        )

    best_idx = int(np.argmax(all_f1s))
    best_threshold = float(THRESHOLDS[best_idx])
    best_f1 = float(all_f1s[best_idx])
    best_precision = float(all_precisions[best_idx])
    best_recall = float(all_recalls[best_idx])

    pos_acc = float(np.mean(pos_scores > best_threshold)) if len(pos_scores) > 0 else float("nan")
    neg_acc = float(np.mean(neg_scores <= best_threshold)) if len(neg_scores) > 0 else float("nan")

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_threshold": best_threshold,
        "positive_acc_at_best_threshold": pos_acc,
        "negative_acc_at_best_threshold": neg_acc,
    }


def _filter_nan_scores(labels, scores):
    valid_mask = ~np.isnan(scores)
    filtered_labels = labels[valid_mask]
    filtered_scores = scores[valid_mask]
    skipped_count = int(np.sum(~valid_mask))
    return filtered_labels, filtered_scores, skipped_count, valid_mask


def _load_content_filter_skip_summary(path):
    summary = {
        "total": 0,
        "generate_clarification": 0,
        "answer_extraction": 0,
    }
    if not path:
        return summary
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing content_filter skip sidecar: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Skip sidecar is not a list: {path}")

    seen_total = set()
    seen_by_stage = {
        "generate_clarification": set(),
        "answer_extraction": set(),
    }
    for idx, row in enumerate(records):
        if "_source_idx" not in row:
            raise KeyError(f"Missing `_source_idx` in skip record index {idx}.")
        source_idx = int(row["_source_idx"])
        seen_total.add(source_idx)
        stages = row.get("stages")
        if isinstance(stages, list):
            normalized_stages = {str(stage) for stage in stages}
        else:
            stage = row.get("stage")
            normalized_stages = {str(stage)} if stage else set()
        for stage_name in seen_by_stage:
            if stage_name in normalized_stages:
                seen_by_stage[stage_name].add(source_idx)

    summary["total"] = len(seen_total)
    summary["generate_clarification"] = len(seen_by_stage["generate_clarification"])
    summary["answer_extraction"] = len(seen_by_stage["answer_extraction"])
    return summary


def _collect_labels_and_scores(records, label_key, dataset_name):
    labels = []
    aleatoric_scores = []
    epistemic_scores = []
    for idx, row in enumerate(records):
        if label_key not in row:
            raise KeyError(f"Missing `{label_key}` in record index {idx}.")
        if ALEATORIC_KEY not in row:
            raise KeyError(f"Missing `{ALEATORIC_KEY}` in record index {idx}.")
        if EPISTEMIC_KEY not in row:
            raise KeyError(f"Missing `{EPISTEMIC_KEY}` in record index {idx}.")

        labels.append(_to_binary_label(row[label_key], dataset_name))
        aleatoric_scores.append(float(row[ALEATORIC_KEY]))
        epistemic_scores.append(float(row[EPISTEMIC_KEY]))

    labels = np.asarray(labels, dtype=np.int64)
    aleatoric_scores = np.asarray(aleatoric_scores, dtype=float)
    epistemic_scores = np.asarray(epistemic_scores, dtype=float)
    return labels, aleatoric_scores, epistemic_scores


def _print_metrics(title, metrics):
    print(f"{title} AUROC: {metrics['auroc']}")
    print(f"{title} AUPR: {metrics['aupr']}")
    print(f"{title} positive_mean: {metrics['positive_mean']}")
    print(f"{title} negative_mean: {metrics['negative_mean']}")
    print(f"{title} best_f1: {metrics['best_f1']}")
    print(f"{title} best_precision: {metrics['best_precision']}")
    print(f"{title} best_recall: {metrics['best_recall']}")
    print(f"{title} best_threshold: {metrics['best_threshold']}")
    print(f"{title} positive_acc_at_best_threshold: {metrics['positive_acc_at_best_threshold']}")
    print(f"{title} negative_acc_at_best_threshold: {metrics['negative_acc_at_best_threshold']}")


def _build_wandb_summary(
    dataset_name,
    model_name,
    baseline_method,
    n_samples,
    num_examples,
    num_positive,
    num_negative,
    skipped_any_nan,
    aleatoric_skipped_nan,
    epistemic_skipped_nan,
    content_filter_skipped_total,
    content_filter_skipped_generate_clarification,
    content_filter_skipped_answer_extraction,
    num_examples_before_content_filter_skip,
    aleatoric_used,
    epistemic_used,
    aleatoric_metrics,
    epistemic_metrics,
):
    summary = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "baseline_method": baseline_method,
        "n_samples": int(n_samples),
        "num_examples": int(num_examples),
        "num_positive": int(num_positive),
        "num_negative": int(num_negative),
        "num_skipped_any_nan": int(skipped_any_nan),
        "num_skipped_aleatoric_nan": int(aleatoric_skipped_nan),
        "num_skipped_epistemic_nan": int(epistemic_skipped_nan),
        "num_content_filter_skipped_total": int(content_filter_skipped_total),
        "num_content_filter_skipped_generate_clarification": int(content_filter_skipped_generate_clarification),
        "num_content_filter_skipped_answer_extraction": int(content_filter_skipped_answer_extraction),
        "num_examples_before_content_filter_skip": int(num_examples_before_content_filter_skip),
        "num_used_aleatoric": int(aleatoric_used),
        "num_used_epistemic": int(epistemic_used),
        "aleatoric_auroc": aleatoric_metrics["auroc"],
        "aleatoric_aupr": aleatoric_metrics["aupr"],
        "epistemic_auroc": epistemic_metrics["auroc"],
        "epistemic_aupr": epistemic_metrics["aupr"],
    }
    for key, value in aleatoric_metrics.items():
        summary[f"aleatoric_{key}"] = value
    for key, value in epistemic_metrics.items():
        summary[f"epistemic_{key}"] = value
    return summary


def _resolve_wandb_run_url(run):
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


def _log_to_wandb(summary_metrics, args):
    if wandb is None:
        raise ImportError("wandb is not installed but --wandb_log=True was requested.")
    run_name = f"baseline_{args.dataset_name}_{args.n_samples}_{os.path.basename(args.log_path)}"
    cfg = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "baseline_method": args.baseline_method,
        "n_samples": int(args.n_samples),
        "log_path": args.log_path,
        "label_key": args.label_key,
        "content_filter_skip_path": args.content_filter_skip_path,
    }
    init_kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "config": cfg,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    run = wandb.init(**init_kwargs)
    run_url = _resolve_wandb_run_url(run)
    wandb.run.summary.update(summary_metrics)
    wandb.log(summary_metrics)
    run.finish()
    return run_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/uq_eval/ambigqa.json",
        help="Path to uncertainty evaluation JSON file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ambigqa",
        help="Dataset name used to parse the label semantics (e.g., ambigqa).",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="label",
        help="Field name in log file that stores the dataset label.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Original dataset sample size used in the experiment. Defaults to len(log).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unknown",
        help="Model name string to include in W&B summary metrics.",
    )
    parser.add_argument(
        "--baseline_method",
        type=str,
        default="ICE",
        help="Baseline method name to include in W&B summary metrics.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="EPN-Baselines",
        help="W&B project for logging summary metrics.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity/team.",
    )
    parser.add_argument(
        "--wandb_log",
        type=_str_to_bool,
        default=True,
        help="Whether to log this run to W&B. Use True or False.",
    )
    parser.add_argument(
        "--content_filter_skip_path",
        type=str,
        default=None,
        help="Optional JSON sidecar with content-filter skipped questions from the ICE pipeline.",
    )
    args = parser.parse_args()

    with open(args.log_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if len(content) == 0:
        raise ValueError(f"No records found in {args.log_path}.")

    if args.n_samples is None:
        args.n_samples = len(content)

    content_filter_skip_summary = _load_content_filter_skip_summary(args.content_filter_skip_path)
    content_filter_skipped_total = int(content_filter_skip_summary["total"])
    content_filter_skipped_generate_clarification = int(
        content_filter_skip_summary["generate_clarification"]
    )
    content_filter_skipped_answer_extraction = int(
        content_filter_skip_summary["answer_extraction"]
    )
    num_examples_before_content_filter_skip = int(len(content) + content_filter_skipped_total)

    labels, aleatoric_scores, epistemic_scores = _collect_labels_and_scores(
        records=content,
        label_key=args.label_key,
        dataset_name=args.dataset_name,
    )
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both positive and negative labels to compute AUROC/AUPR.")

    num_examples = len(content)
    num_positive = int(labels.sum())
    num_negative = int(num_examples - num_positive)

    (
        aleatoric_labels,
        aleatoric_scores_filtered,
        aleatoric_skipped_nan,
        aleatoric_valid_mask,
    ) = _filter_nan_scores(labels, aleatoric_scores)
    (
        epistemic_labels,
        epistemic_scores_filtered,
        epistemic_skipped_nan,
        epistemic_valid_mask,
    ) = _filter_nan_scores(labels, epistemic_scores)
    skipped_any_nan = int(np.sum(~(aleatoric_valid_mask & epistemic_valid_mask)))

    if aleatoric_skipped_nan > 0:
        print(f"[warn] skipped {aleatoric_skipped_nan} questions with NaN {ALEATORIC_KEY}.")
    if epistemic_skipped_nan > 0:
        print(f"[warn] skipped {epistemic_skipped_nan} questions with NaN {EPISTEMIC_KEY}.")
    if skipped_any_nan > 0:
        print(f"[warn] total unique questions skipped due to NaN in any uncertainty: {skipped_any_nan}")

    if len(aleatoric_labels) == 0 or len(np.unique(aleatoric_labels)) < 2:
        raise ValueError(
            "Need both positive and negative labels after NaN filtering for aleatoric metrics."
        )
    if len(epistemic_labels) == 0 or len(np.unique(epistemic_labels)) < 2:
        raise ValueError(
            "Need both positive and negative labels after NaN filtering for epistemic metrics."
        )

    aleatoric_metrics = _compute_score_metrics(aleatoric_labels, aleatoric_scores_filtered)
    epistemic_metrics = _compute_score_metrics(epistemic_labels, epistemic_scores_filtered)
    aleatoric_used = len(aleatoric_labels)
    epistemic_used = len(epistemic_labels)

    print("dataset_name:", args.dataset_name)
    print("model_name:", args.model_name)
    print("baseline_method:", args.baseline_method)
    print("n_samples:", int(args.n_samples))
    print("num_examples:", num_examples)
    print("num_examples_before_content_filter_skip:", num_examples_before_content_filter_skip)
    print("num_positive:", num_positive)
    print("num_negative:", num_negative)
    print("num_skipped_any_nan:", skipped_any_nan)
    print("num_content_filter_skipped_total:", content_filter_skipped_total)
    print(
        "num_content_filter_skipped_generate_clarification:",
        content_filter_skipped_generate_clarification,
    )
    print(
        "num_content_filter_skipped_answer_extraction:",
        content_filter_skipped_answer_extraction,
    )
    print("num_used_aleatoric:", aleatoric_used)
    print("num_used_epistemic:", epistemic_used)
    print()

    _print_metrics("aleatoric", aleatoric_metrics)
    print()
    _print_metrics("epistemic", epistemic_metrics)

    summary_metrics = _build_wandb_summary(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        baseline_method=args.baseline_method,
        n_samples=args.n_samples,
        num_examples=num_examples,
        num_positive=num_positive,
        num_negative=num_negative,
        skipped_any_nan=skipped_any_nan,
        aleatoric_skipped_nan=aleatoric_skipped_nan,
        epistemic_skipped_nan=epistemic_skipped_nan,
        content_filter_skipped_total=content_filter_skipped_total,
        content_filter_skipped_generate_clarification=content_filter_skipped_generate_clarification,
        content_filter_skipped_answer_extraction=content_filter_skipped_answer_extraction,
        num_examples_before_content_filter_skip=num_examples_before_content_filter_skip,
        aleatoric_used=aleatoric_used,
        epistemic_used=epistemic_used,
        aleatoric_metrics=aleatoric_metrics,
        epistemic_metrics=epistemic_metrics,
    )
    run_url = None
    if args.wandb_log:
        run_url = _log_to_wandb(summary_metrics, args)
    print()
    print("wandb_project:", args.wandb_project)
    print("wandb_logged:", args.wandb_log)
    if args.wandb_log and run_url:
        print("wandb_run_url:", run_url)
    elif args.wandb_log:
        print("wandb_run_url: unavailable")
    else:
        print("wandb_run_url: skipped (wandb_log=False)")


if __name__ == "__main__":
    main()
