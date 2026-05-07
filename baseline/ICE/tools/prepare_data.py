import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import datasets
import numpy as np
import json
import random
import argparse
from src.data_util import split_ambigqa_eup, parse_frac, _split_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(parentdir))
DEFAULT_JUDGE_TYPE = "gpt5-mini"
SITUATED_QA_DATASET_NAME = "situated_qa"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def _save_json(path, payload):
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _format_frac_tag(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    tag = f"{val:.6f}".rstrip("0").rstrip(".")
    return tag if tag else "0"


def _build_ambigqa_name(split_name, n_samples, val_frac, test_frac):
    val_tag = _format_frac_tag(val_frac)
    test_tag = _format_frac_tag(test_frac)
    return f"ambigqa_{split_name}_{n_samples}_{val_tag}_{test_tag}.json"


def _build_split_name(dataset_name, split_name, n_samples, val_frac, test_frac):
    val_tag = _format_frac_tag(val_frac)
    test_tag = _format_frac_tag(test_frac)
    return f"{dataset_name}_{split_name}_{n_samples}_{val_tag}_{test_tag}.json"


def _require_int_count(value, name):
    if not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be provided as an integer count when --fix_val_test_splits=True."
        )
    return int(value)


def _remove_dup_questions(dataset):
    seen_questions = set()

    def _is_unique(example):
        question = example.get("question")
        if question in seen_questions:
            return False
        seen_questions.add(question)
        return True

    return dataset.filter(_is_unique)


def _resolve_alcuna_path():
    path_candidates = [
        "/datasets/ALCUNA/id2question.json",
        os.path.join(PROJECT_ROOT, "datasets", "ALCUNA", "id2question.json"),
        os.path.join("datasets", "ALCUNA", "id2question.json"),
    ]
    for candidate in path_candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "ALCUNA id2question.json not found. Tried: " + ", ".join(path_candidates)
    )


def _load_alcuna_rows():
    alcuna_path = _resolve_alcuna_path()
    with open(alcuna_path, "r", encoding="utf-8") as f:
        raw_alcuna = json.load(f)

    rows = []
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
                rows.append(
                    {
                        "question": question,
                        "answer": refs,
                        "label": 1,
                        "ood_label": 1,
                    }
                )
    return rows


def _normalize_ref_list(values):
    if isinstance(values, list):
        refs = [str(item) for item in values if item is not None and str(item).strip() != ""]
    elif values is None:
        refs = []
    else:
        refs = [str(values)] if str(values).strip() != "" else []
    deduped = []
    for ref in refs:
        if ref not in deduped:
            deduped.append(ref)
    return deduped


def _resolve_temporalood_path():
    path_candidates = [
        os.path.join(PROJECT_ROOT, "datasets", "TemporalOOD", "post_nov_2025_open_domain_qa_2000.json"),
        os.path.join("datasets", "TemporalOOD", "post_nov_2025_open_domain_qa_2000.json"),
    ]
    for candidate in path_candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "TemporalOOD QA JSON not found. Tried: " + ", ".join(path_candidates)
    )


def _resolve_situated_qa_path():
    path_candidates = [
        os.path.join(PROJECT_ROOT, "datasets", "SituatedQA", "data", "situated_qa.jsonl"),
        os.path.join("datasets", "SituatedQA", "data", "situated_qa.jsonl"),
    ]
    for candidate in path_candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "SituatedQA JSONL not found. Tried: " + ", ".join(path_candidates)
    )


def _load_jsonl_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stringify_if_nonempty(value):
    if value is None:
        return ""
    return str(value).strip()


def _flatten_nested_text_values(value):
    if isinstance(value, dict):
        flattened = []
        for nested in value.values():
            flattened.extend(_flatten_nested_text_values(nested))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for nested in value:
            flattened.extend(_flatten_nested_text_values(nested))
        return flattened
    text = _stringify_if_nonempty(value)
    return [text] if text else []


def _dedupe_text_list(values):
    deduped = []
    for value in values:
        text = _stringify_if_nonempty(value)
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _normalize_situated_qa_is_dependent(value):
    if isinstance(value, bool):
        return int(bool(value))

    normalized = _stringify_if_nonempty(value).lower()
    if normalized in {"true", "1", "yes"}:
        return 1
    if normalized in {"false", "0", "no"}:
        return 0
    raise ValueError(f"SituatedQA row has invalid is_dependent value: {value!r}")


def _load_situated_qa_rows():
    situated_qa_path = _resolve_situated_qa_path()
    raw_rows = _load_jsonl_rows(situated_qa_path)
    if len(raw_rows) == 0:
        raise ValueError(f"SituatedQA combined dataset is empty: {situated_qa_path}")

    rows = []
    for item in raw_rows:
        question = _stringify_if_nonempty(item.get("question"))
        if not question:
            raise ValueError(f"SituatedQA row is missing question text: {item}")

        refs = _dedupe_text_list(_flatten_nested_text_values(item.get("any_answer", [])))
        if len(refs) == 0:
            refs = _dedupe_text_list(_flatten_nested_text_values(item.get("answer", [])))
        if len(refs) == 0:
            raise ValueError(
                "SituatedQA row is missing both any_answer and answer values: "
                f"{item.get('sample_id', '<missing sample_id>')}"
            )

        row = {
            "question": question,
            "answer": refs,
            "label": _normalize_situated_qa_is_dependent(item.get("is_dependent")),
        }
        sample_id = _stringify_if_nonempty(item.get("sample_id"))
        if sample_id:
            row["sample_id"] = sample_id
        rows.append(row)
    return rows


def _load_temporalood_rows():
    temporalood_path = _resolve_temporalood_path()
    with open(temporalood_path, "r", encoding="utf-8") as f:
        raw_temporalood = json.load(f)

    rows = []
    for item in raw_temporalood:
        refs = _normalize_ref_list(item.get("reference_answer", []))
        if refs:
            rows.append(
                {
                    "question": item.get("question", ""),
                    "answer": refs,
                    "label": 1,
                    "ood_label": 1,
                }
            )
    return rows


def _load_trivia_dataset():
    raw_trivia = datasets.load_dataset("mandarjoshi/trivia_qa", "rc", cache_dir="dataset_cache")
    trivia_full = raw_trivia["train"]
    trivia_full = _remove_dup_questions(trivia_full)
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
    return trivia_full


def _trivia_dataset_to_rows(trivia_dataset):
    rows = []
    for example in trivia_dataset:
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
        if not refs:
            continue
        rows.append(
            {
                "question": example["question"],
                "answer": refs,
                "label": 0,
                "ood_label": 0,
            }
        )
    return rows


def _load_trivia_rows(target_size):
    trivia_full = _load_trivia_dataset()

    if len(trivia_full) < target_size:
        raise ValueError(
            f"Requested {target_size} trivia_qa samples but only {len(trivia_full)} are available."
        )
    trivia_full = trivia_full.shuffle(seed=42).select(range(target_size))

    rows = _trivia_dataset_to_rows(trivia_full)
    if len(rows) != target_size:
        raise ValueError(
            f"Constructed {len(rows)} trivia_qa samples, expected {target_size}."
        )
    return rows


def _split_alcuna_trivia_remainder(remainder, test_frac, val_frac, seed):
    N = len(remainder)
    strat_remainder = remainder.class_encode_column("ood_label")
    try:
        if isinstance(test_frac, (int, np.integer)):
            test_size = int(test_frac)
            assert 0 <= test_size <= N, "test_size out of range"
            if test_size > 0:
                tv_split = strat_remainder.train_test_split(
                    test_size=test_size,
                    seed=seed,
                    stratify_by_column="ood_label",
                )
            else:
                tv_split = datasets.DatasetDict(
                    {
                        "train": strat_remainder,
                        "test": strat_remainder.select(range(0)),
                    }
                )

            if isinstance(val_frac, (int, np.integer)):
                val_size = int(val_frac)
                assert 0 <= val_size <= (N - test_size), "val_size too large for remaining train split"
                if val_size > 0:
                    train_val_split = tv_split["train"].train_test_split(
                        test_size=val_size,
                        seed=seed,
                        stratify_by_column="ood_label",
                    )
                else:
                    train_val_split = datasets.DatasetDict(
                        {
                            "train": tv_split["train"],
                            "test": tv_split["train"].select(range(0)),
                        }
                    )
            else:
                if not (0.0 <= float(val_frac) < 1.0):
                    raise ValueError("val_frac as float must be in [0.0, 1.0).")
                if float(val_frac) > 0.0:
                    train_val_split = tv_split["train"].train_test_split(
                        test_size=val_frac,
                        seed=seed,
                        stratify_by_column="ood_label",
                    )
                else:
                    train_val_split = datasets.DatasetDict(
                        {
                            "train": tv_split["train"],
                            "test": tv_split["train"].select(range(0)),
                        }
                    )
        else:
            test_frac = float(test_frac)
            if not (0.0 <= test_frac < 1.0):
                raise ValueError("test_frac as float must be in [0.0, 1.0).")
            val_frac_in_train = float(val_frac) / (1.0 - test_frac)
            if not (0.0 <= val_frac_in_train < 1.0):
                raise ValueError("val_frac leads to invalid validation split ratio.")

            if test_frac > 0.0:
                tv_split = strat_remainder.train_test_split(
                    test_size=test_frac,
                    seed=seed,
                    stratify_by_column="ood_label",
                )
            else:
                tv_split = datasets.DatasetDict(
                    {
                        "train": strat_remainder,
                        "test": strat_remainder.select(range(0)),
                    }
                )

            if val_frac_in_train > 0.0:
                train_val_split = tv_split["train"].train_test_split(
                    test_size=val_frac_in_train,
                    seed=seed,
                    stratify_by_column="ood_label",
                )
            else:
                train_val_split = datasets.DatasetDict(
                    {
                        "train": tv_split["train"],
                        "test": tv_split["train"].select(range(0)),
                    }
                )
    except ValueError as exc:
        raise ValueError(
            f"Stratified split failed for alcuna_trivia_qa with column `ood_label`: {exc}"
        ) from exc

    return train_val_split["train"], train_val_split["test"], tv_split["test"]


def _split_binary_label_remainder(
    remainder,
    dataset_name,
    label_column,
    test_frac,
    val_frac,
    seed,
    fix_val_test_splits=False,
    train_size_target=None,
):
    N = len(remainder)
    strat_remainder = remainder.class_encode_column(label_column)
    split_seed = lambda stage: _split_seed(seed, fix_val_test_splits, stage)

    try:
        if isinstance(test_frac, (int, np.integer)):
            test_size = int(test_frac)
            assert 0 <= test_size <= N, "test_size out of range"
            if test_size > 0:
                tv_split = strat_remainder.train_test_split(
                    test_size=test_size,
                    seed=split_seed("test_membership"),
                    stratify_by_column=label_column,
                )
            else:
                tv_split = datasets.DatasetDict(
                    {
                        "train": strat_remainder,
                        "test": strat_remainder.select(range(0)),
                    }
                )

            if isinstance(val_frac, (int, np.integer)):
                val_size = int(val_frac)
                assert 0 <= val_size <= (N - test_size), "val_size too large for remaining train split"
                if val_size > 0:
                    train_val_split = tv_split["train"].train_test_split(
                        test_size=val_size,
                        seed=split_seed("valid_membership"),
                        stratify_by_column=label_column,
                    )
                else:
                    train_val_split = datasets.DatasetDict(
                        {
                            "train": tv_split["train"],
                            "test": tv_split["train"].select(range(0)),
                        }
                    )
            else:
                val_frac = float(val_frac)
                if not (0.0 <= val_frac < 1.0):
                    raise ValueError("val_frac as float must be in [0.0, 1.0).")
                if val_frac > 0.0:
                    train_val_split = tv_split["train"].train_test_split(
                        test_size=val_frac,
                        seed=split_seed("valid_membership"),
                        stratify_by_column=label_column,
                    )
                else:
                    train_val_split = datasets.DatasetDict(
                        {
                            "train": tv_split["train"],
                            "test": tv_split["train"].select(range(0)),
                        }
                    )
        else:
            test_frac = float(test_frac)
            if not (0.0 <= test_frac < 1.0):
                raise ValueError("test_frac as float must be in [0.0, 1.0).")
            val_frac_in_train = float(val_frac) / (1.0 - test_frac)
            if not (0.0 <= val_frac_in_train < 1.0):
                raise ValueError("val_frac leads to invalid validation split ratio.")

            if test_frac > 0.0:
                tv_split = strat_remainder.train_test_split(
                    test_size=test_frac,
                    seed=split_seed("test_membership"),
                    stratify_by_column=label_column,
                )
            else:
                tv_split = datasets.DatasetDict(
                    {
                        "train": strat_remainder,
                        "test": strat_remainder.select(range(0)),
                    }
                )

            if val_frac_in_train > 0.0:
                train_val_split = tv_split["train"].train_test_split(
                    test_size=val_frac_in_train,
                    seed=split_seed("valid_membership"),
                    stratify_by_column=label_column,
                )
            else:
                train_val_split = datasets.DatasetDict(
                    {
                        "train": tv_split["train"],
                        "test": tv_split["train"].select(range(0)),
                    }
                )
    except ValueError as exc:
        raise ValueError(
            f"Stratified split failed for {dataset_name} with column `{label_column}`: {exc}"
        ) from exc

    train_ds = train_val_split["train"]
    valid_ds = train_val_split["test"]
    test_ds = tv_split["test"]

    if train_size_target is not None:
        if train_size_target > len(train_ds):
            raise ValueError(
                f"Requested train size {train_size_target} exceeds available pool {len(train_ds)} "
                "after reserving validation and test samples."
            )
        if train_size_target > 0:
            train_ds = train_ds.shuffle(seed=split_seed("train_membership")).select(range(train_size_target))
        else:
            train_ds = train_ds.select(range(0))

    if fix_val_test_splits:
        train_ds = train_ds.shuffle(seed=split_seed("train_order"))
        valid_ds = valid_ds.shuffle(seed=split_seed("valid_order"))
        test_ds = test_ds.shuffle(seed=split_seed("test_order"))

    return train_ds, valid_ds, test_ds


def _prepare_ood_trivia_qa_fixed(
    dataset_name,
    ood_rows,
    n_samples,
    test_frac,
    val_frac,
    seed,
    n_fewshot,
):
    if not isinstance(test_frac, (int, np.integer)):
        raise ValueError(
            "test_frac must be provided as an integer count when --fix_val_test_splits=True."
        )
    if not isinstance(val_frac, (int, np.integer)):
        raise ValueError(
            "valid_frac must be provided as an integer count when --fix_val_test_splits=True."
        )

    test_size = int(test_frac)
    val_size = int(val_frac)
    train_size = int(n_samples) - test_size - val_size
    if test_size < 0 or test_size > int(n_samples):
        raise ValueError("test_size out of range")
    if val_size < 0 or val_size > (int(n_samples) - test_size):
        raise ValueError("val_size too large for remaining train split")
    if train_size < 0:
        raise ValueError("n_samples is too small for requested val/test sizes.")

    test_ood_target = (test_size // 2) + (test_size % 2)
    test_trivia_target = test_size - test_ood_target
    train_trivia_target = train_size
    valid_trivia_target = val_size

    if len(ood_rows) < test_ood_target:
        raise ValueError(
            f"Need at least {test_ood_target} {dataset_name} OOD samples for test split, "
            f"but only {len(ood_rows)} are available."
        )

    trivia_full = _load_trivia_dataset()
    required_trivia = train_trivia_target + valid_trivia_target + test_trivia_target
    max_k_trivia = len(trivia_full) - required_trivia
    if max_k_trivia < 0:
        raise ValueError(
            f"Need at least {required_trivia} trivia_qa samples for train/validation/test splits, "
            f"but only {len(trivia_full)} are available."
        )

    k_trivia = min(int(n_fewshot), max_k_trivia)
    trivia_target = required_trivia + k_trivia

    ood_test_pool = datasets.Dataset.from_list(ood_rows).shuffle(seed=42).select(range(test_ood_target))
    trivia_selected = trivia_full.shuffle(seed=42).select(range(trivia_target))
    trivia_rows = _trivia_dataset_to_rows(trivia_selected)
    if len(trivia_rows) != trivia_target:
        raise ValueError(
            f"Constructed {len(trivia_rows)} trivia_qa rows, expected {trivia_target}."
        )

    trivia_selected_rows = datasets.Dataset.from_list(trivia_rows)
    trivia_nonfew_pool = trivia_selected_rows.select(
        range(
            k_trivia,
            k_trivia + required_trivia,
        )
    )

    valid_ds = trivia_nonfew_pool.select(range(0, valid_trivia_target))
    trivia_test_pool = trivia_nonfew_pool.select(
        range(valid_trivia_target, valid_trivia_target + test_trivia_target)
    )
    train_ds = trivia_nonfew_pool.select(
        range(
            valid_trivia_target + test_trivia_target,
            valid_trivia_target + test_trivia_target + train_trivia_target,
        )
    )

    train_ds = train_ds.shuffle(seed=_split_seed(seed, True, "train_order"))
    valid_ds = valid_ds.shuffle(seed=_split_seed(seed, True, "valid_order"))

    if test_size > 0:
        test_ds = datasets.concatenate_datasets([ood_test_pool, trivia_test_pool]).shuffle(
            seed=_split_seed(seed, True, "test_order")
        )
    else:
        test_ds = train_ds.select(range(0))

    return train_ds, valid_ds, test_ds


def _dataset_to_ood_trivia_rows(dataset):
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        answers = item.get("answer", [])
        if isinstance(answers, list):
            answer_values = [str(x) for x in answers if x is not None]
        elif answers is None:
            answer_values = []
        else:
            answer_values = [str(answers)]
        label = int(item.get("ood_label", item.get("label", 0)))
        rows.append(
            {
                "question": item.get("question", ""),
                "answer": answer_values,
                "label": label,
                "ood_label": label,
            }
        )
    return rows


def _dataset_to_binary_label_rows(dataset):
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        row = {
            "question": item.get("question", ""),
            "answer": _normalize_ref_list(item.get("answer", [])),
            "label": int(item.get("label", 0)),
        }
        sample_id = item.get("sample_id")
        if sample_id is not None and str(sample_id).strip() != "":
            row["sample_id"] = str(sample_id)
        rows.append(row)
    return rows


def _prepare_ood_trivia_qa_eup(
    dataset_name,
    ood_rows,
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    seed=42,
    n_fewshot=5,
    fix_val_test_splits=False,
):
    if fix_val_test_splits:
        train_ds, valid_ds, test_ds = _prepare_ood_trivia_qa_fixed(
            dataset_name=dataset_name,
            ood_rows=ood_rows,
            n_samples=n_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=seed,
            n_fewshot=n_fewshot,
        )
    else:
        if n_samples % 2 != 0:
            raise ValueError(f"{dataset_name} requires an even n_samples to split 50/50.")

        total_target = n_samples + n_fewshot
        ood_target = total_target // 2 + (total_target % 2)
        trivia_target = total_target - ood_target

        if len(ood_rows) < ood_target:
            raise ValueError(
                f"Requested {ood_target} {dataset_name} OOD samples but only {len(ood_rows)} are available."
            )
        ood_full = datasets.Dataset.from_list(ood_rows).shuffle(seed=42).select(range(ood_target))

        trivia_rows = _load_trivia_rows(trivia_target)

        mixed_rows = [row for row in ood_full] + trivia_rows
        if len(mixed_rows) != total_target:
            raise ValueError(
                f"Constructed {len(mixed_rows)} mixed samples, expected {total_target}."
            )

        full = datasets.Dataset.from_list(mixed_rows).shuffle(seed=seed)

        rng = np.random.default_rng(seed)
        k = min(n_fewshot, len(full))
        ood = np.asarray(full["ood_label"], dtype=np.int64)
        idx_ood = np.where(ood == 1)[0]
        idx_trivia = np.where(ood == 0)[0]
        rng.shuffle(idx_ood)
        rng.shuffle(idx_trivia)

        k_ood = min(len(idx_ood), (k // 2) + (k % 2))
        k_trivia = k - k_ood
        if k_trivia > len(idx_trivia):
            k_trivia = len(idx_trivia)
            k_ood = k - k_trivia
        if k_ood > len(idx_ood):
            raise ValueError(f"Not enough {dataset_name} OOD samples to build stratified few-shot pool.")

        few_idx = set(np.concatenate([idx_ood[:k_ood], idx_trivia[:k_trivia]]).tolist())
        keep_idx = [i for i in range(len(full)) if i not in few_idx]
        remainder = full.select(keep_idx)

        train_ds, valid_ds, test_ds = _split_alcuna_trivia_remainder(
            remainder=remainder,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=seed,
        )

    train_rows = _dataset_to_ood_trivia_rows(train_ds)
    valid_rows = _dataset_to_ood_trivia_rows(valid_ds)
    test_rows = _dataset_to_ood_trivia_rows(test_ds)

    train_name = _build_split_name(dataset_name, "train", n_samples, val_frac, test_frac)
    valid_name = _build_split_name(dataset_name, "validation", n_samples, val_frac, test_frac)
    test_name = _build_split_name(dataset_name, "test", n_samples, val_frac, test_frac)

    _save_json(f"logs/dataset/{dataset_name}/{train_name}", train_rows)
    _save_json(f"logs/dataset/{dataset_name}/{valid_name}", valid_rows)
    _save_json(f"logs/dataset/{dataset_name}/{test_name}", test_rows)


def _load_jsonl_field(path, field_name):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for default_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            item_idx = int(payload.get("i", default_idx))
            if field_name not in payload:
                raise KeyError(f"Missing `{field_name}` in {path}: {payload}")
            items.append((item_idx, payload[field_name]))
    items.sort(key=lambda x: x[0])
    return [value for _, value in items]


def _resolve_feature_dataset_dir(model_name, dataset_name, n_samples, judge_type):
    if not model_name:
        raise ValueError(
            f"--model_name is required to prepare {dataset_name} because it is sourced from "
            "model-specific artifacts in the main project."
        )
    model_id = str(model_name).strip().split("/")[-1]
    dataset_dir = os.path.join(
        PROJECT_ROOT,
        "input_hidden_states",
        f"{model_id}_{dataset_name}_{int(n_samples)}_{judge_type}",
    )
    if os.path.isdir(dataset_dir):
        return dataset_dir
    raise FileNotFoundError(
        f"Missing prepared source artifacts for {dataset_name}: {dataset_dir}. "
        "Run feature_collector.py in the main project first."
    )


def _load_ambig_filtered_split_rows(split_dir):
    questions = _load_jsonl_field(os.path.join(split_dir, "questions.jsonl"), "question")
    references = _load_jsonl_field(os.path.join(split_dir, "references.jsonl"), "ref")
    labels = np.asarray(np.load(os.path.join(split_dir, "ambig_label.npy")), dtype=np.int64).reshape(-1)

    if not (len(questions) == len(references) == len(labels)):
        raise ValueError(
            f"Filtered split size mismatch in {split_dir}: "
            f"questions={len(questions)}, references={len(references)}, labels={len(labels)}"
        )

    rows = []
    for question, ref_values, label in zip(questions, references, labels.tolist()):
        rows.append(
            {
                "question": "" if question is None else str(question),
                "answer": _normalize_ref_list(ref_values),
                "label": int(label),
            }
        )
    return rows


def prepare_ambig_filtered_eup(
    dataset_name,
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    model_name=None,
    judge_type=DEFAULT_JUDGE_TYPE,
):
    dataset_dir = _resolve_feature_dataset_dir(
        model_name=model_name,
        dataset_name=dataset_name,
        n_samples=n_samples,
        judge_type=judge_type,
    )

    split_dirnames = {
        "train": "train",
        "validation": "valid",
        "test": "test",
    }
    for split_name, disk_name in split_dirnames.items():
        split_rows = _load_ambig_filtered_split_rows(os.path.join(dataset_dir, disk_name))
        split_filename = _build_split_name(dataset_name, split_name, n_samples, val_frac, test_frac)
        _save_json(f"logs/dataset/{dataset_name}/{split_filename}", split_rows)

def prepare_ambigqa(per_class_size = 100):
    test_data = datasets.load_dataset('ambig_qa', 'light', split='validation', cache_dir = 'dataset_cache')

    ambig_examples = []
    unambig_examples = []

    for idx in range(len(test_data)):
        data = test_data[idx]
        question = data['question']
        annotate_dict = data['annotations']
        label = annotate_dict['type']

        if 'singleAnswer' in label and 'multipleQAs' in label:
            continue

        num_annotate = len(annotate_dict['qaPairs'])
        gt_rewrite = []
        gt_answers = []
        for anno_idx in range(num_annotate):
            curr_label = label[anno_idx]
            curr_single_answers = annotate_dict['answer'][anno_idx]
            if curr_label == 'singleAnswer':
                gt_rewrite += [question]
                gt_answers += [curr_single_answers]
            elif curr_label == 'multipleQAs':
                gt_rewrite += annotate_dict['qaPairs'][anno_idx]['question']
                gt_answers += annotate_dict['qaPairs'][anno_idx]['answer']

        log_dict = {
            'question': question,
            'label': label,
            'orig_annotation': annotate_dict,
            'gt_rewrite': gt_rewrite,
            'answer': gt_answers,
        }
        if 'singleAnswer' in label:
            unambig_examples.append(log_dict)
        else:
            ambig_examples.append(log_dict)

    np.random.seed(666)
    random.seed(666)

    unambig_rand_idxs = np.random.choice(len(unambig_examples), len(unambig_examples), replace = False)
    ambig_rand_idxs = np.random.choice(len(ambig_examples), len(ambig_examples), replace = False)

    selected_unambig_list = []
    selected_ambig_list = []
    for idx in unambig_rand_idxs[:per_class_size]:
        selected_unambig_list.append(unambig_examples[idx])

    for idx in ambig_rand_idxs[:per_class_size]:
        selected_ambig_list.append(ambig_examples[idx])

    final_list = selected_unambig_list + selected_ambig_list

    output_path = 'logs/dataset/ambigqa/ambigqa_dev_balance.json'
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(final_list, f, indent = 4)

def extract_candidate_data():
    train_data = datasets.load_dataset('ambig_qa', 'light', split='train', cache_dir = 'dataset_cache')

    ambig_examples = []
    unambig_examples = []

    for idx in range(len(train_data)):
        data = train_data[idx]
        question = data['question']
        annotate_dict = data['annotations']
        label = annotate_dict['type']

        if 'singleAnswer' in label and 'multipleQAs' in label:
            continue

        num_annotate = len(annotate_dict['qaPairs'])
        gt_rewrite = []
        gt_answers = []
        for anno_idx in range(num_annotate):
            curr_label = label[anno_idx]
            curr_single_answers = annotate_dict['answer'][anno_idx]
            if curr_label == 'singleAnswer':
                gt_rewrite += [question]
                gt_answers += [curr_single_answers]
            elif curr_label == 'multipleQAs':
                gt_rewrite += annotate_dict['qaPairs'][anno_idx]['question']
                gt_answers += annotate_dict['qaPairs'][anno_idx]['answer']

        log_dict = {
            'question': question,
            'label': label,
            'orig_annotation': annotate_dict,
            'gt_rewrite': gt_rewrite,
            'answer': gt_answers,
        }
        if 'singleAnswer' in label:
            unambig_examples.append(log_dict)
        else:
            ambig_examples.append(log_dict)

    np.random.seed(666)
    random.seed(666)

    unambig_rand_idxs = np.random.choice(len(unambig_examples), len(unambig_examples), replace = False)
    ambig_rand_idxs = np.random.choice(len(ambig_examples), len(ambig_examples), replace = False)

    selected_unambig_list = []
    selected_ambig_list = []
    for idx in unambig_rand_idxs:
        selected_unambig_list.append(unambig_examples[idx])

    for idx in ambig_rand_idxs:
        selected_ambig_list.append(ambig_examples[idx])

    output_path = 'logs/dataset/ambigqa/ambigqa_train_ambig.json'
    _save_json(output_path, selected_ambig_list)

    output_path = 'logs/dataset/ambigqa/ambigqa_train_unambig.json'
    _save_json(output_path, selected_unambig_list)


def prepare_ambigqa_eup(
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    seed=42,
    n_fewshot=5,
    fix_val_test_splits=False,
):
    splits = split_ambigqa_eup(
        n_samples=n_samples,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
        n_fewshot=n_fewshot,
        fix_val_test_splits=fix_val_test_splits,
    )

    train_name = _build_ambigqa_name("train", n_samples, val_frac, test_frac)
    valid_name = _build_ambigqa_name("validation", n_samples, val_frac, test_frac)
    test_name = _build_ambigqa_name("test", n_samples, val_frac, test_frac)

    _save_json(f"logs/dataset/ambigqa/{train_name}", splits["train"])
    _save_json(f"logs/dataset/ambigqa/{valid_name}", splits["validation"])
    _save_json(f"logs/dataset/ambigqa/{test_name}", splits["test"])

    train_ambig = [x for x in splits["train"] if "singleAnswer" not in x.get("label", [])]
    train_unambig = [x for x in splits["train"] if "singleAnswer" in x.get("label", [])]
    ambig_name = _build_ambigqa_name("train_ambig", n_samples, val_frac, test_frac)
    unambig_name = _build_ambigqa_name("train_unambig", n_samples, val_frac, test_frac)
    _save_json(f"logs/dataset/ambigqa/{ambig_name}", train_ambig)
    _save_json(f"logs/dataset/ambigqa/{unambig_name}", train_unambig)


def prepare_alcuna_trivia_qa_eup(
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    seed=42,
    n_fewshot=5,
    fix_val_test_splits=False,
):
    _prepare_ood_trivia_qa_eup(
        dataset_name="alcuna_trivia_qa",
        ood_rows=_load_alcuna_rows(),
        n_samples=n_samples,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
        n_fewshot=n_fewshot,
        fix_val_test_splits=fix_val_test_splits,
    )


def prepare_temporalood_trivia_qa_eup(
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    seed=42,
    n_fewshot=5,
    fix_val_test_splits=False,
):
    _prepare_ood_trivia_qa_eup(
        dataset_name="temporalood_trivia_qa",
        ood_rows=_load_temporalood_rows(),
        n_samples=n_samples,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
        n_fewshot=n_fewshot,
        fix_val_test_splits=fix_val_test_splits,
    )


def prepare_situated_qa_eup(
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    seed=42,
    n_fewshot=5,
    fix_val_test_splits=False,
):
    full = datasets.Dataset.from_list(_load_situated_qa_rows()).shuffle(seed=42)

    if fix_val_test_splits:
        test_size = _require_int_count(test_frac, "test_frac")
        val_size = _require_int_count(val_frac, "val_frac")
        train_size_target = int(n_samples) - test_size - val_size
        if test_size < 0 or test_size > int(n_samples):
            raise ValueError("test_size out of range")
        if val_size < 0:
            raise ValueError("val_size out of range")
        if train_size_target < 0:
            raise ValueError("n_samples is too small for requested val/test sizes.")
    else:
        requested_total = int(n_samples) + int(n_fewshot)
        if requested_total > len(full):
            raise ValueError(
                f"Requested {requested_total} SituatedQA rows (n_samples + n_fewshot), "
                f"but only {len(full)} are available in {_resolve_situated_qa_path()}."
            )
        full = full.select(range(requested_total))
        train_size_target = None

    rng = np.random.default_rng(_split_seed(seed, fix_val_test_splits, "fewshot_membership"))
    k = min(int(n_fewshot), len(full))
    few_idx = set(rng.choice(len(full), size=k, replace=False).tolist())
    keep_idx = [i for i in range(len(full)) if i not in few_idx]
    remainder = full.select(keep_idx)

    train_ds, valid_ds, test_ds = _split_binary_label_remainder(
        remainder=remainder,
        dataset_name=SITUATED_QA_DATASET_NAME,
        label_column="label",
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
        fix_val_test_splits=fix_val_test_splits,
        train_size_target=train_size_target,
    )

    train_rows = _dataset_to_binary_label_rows(train_ds)
    valid_rows = _dataset_to_binary_label_rows(valid_ds)
    test_rows = _dataset_to_binary_label_rows(test_ds)

    train_name = _build_split_name(SITUATED_QA_DATASET_NAME, "train", n_samples, val_frac, test_frac)
    valid_name = _build_split_name(SITUATED_QA_DATASET_NAME, "validation", n_samples, val_frac, test_frac)
    test_name = _build_split_name(SITUATED_QA_DATASET_NAME, "test", n_samples, val_frac, test_frac)

    _save_json(f"logs/dataset/{SITUATED_QA_DATASET_NAME}/{train_name}", train_rows)
    _save_json(f"logs/dataset/{SITUATED_QA_DATASET_NAME}/{valid_name}", valid_rows)
    _save_json(f"logs/dataset/{SITUATED_QA_DATASET_NAME}/{test_name}", test_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ambigqa",
        choices=[
            "ambigqa",
            "ambig_qa",
            "ambig_qa_filtered",
            "ambig_qa_filtered_ver3",
            "alcuna_trivia_qa",
            "situated_qa",
            "temporalood_trivia_qa",
        ],
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--test_frac", type=str, default=None)
    parser.add_argument("--valid_frac", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_fewshot", type=int, default=5)
    parser.add_argument("--fix_val_test_splits", type=str2bool, default=False)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--judge_type", type=str, default=DEFAULT_JUDGE_TYPE)
    args = parser.parse_args()
    dataset_name = "ambigqa" if args.dataset_name == "ambig_qa" else args.dataset_name

    if dataset_name == "ambigqa":
        if args.n_samples is None and args.test_frac is None and args.valid_frac is None:
            prepare_ambigqa()
            extract_candidate_data()
        else:
            if args.n_samples is None:
                raise ValueError("n_samples is required when using EUP-style splits.")
            test_frac = parse_frac(args.test_frac, default=0.10)
            val_frac = parse_frac(args.valid_frac, default=0.10)
            prepare_ambigqa_eup(
                n_samples=args.n_samples,
                test_frac=test_frac,
                val_frac=val_frac,
                seed=args.seed,
                n_fewshot=args.n_fewshot,
                fix_val_test_splits=args.fix_val_test_splits,
            )
    elif dataset_name == "alcuna_trivia_qa":
        if args.n_samples is None:
            raise ValueError("n_samples is required for alcuna_trivia_qa splits.")
        test_frac = parse_frac(args.test_frac, default=0.10)
        val_frac = parse_frac(args.valid_frac, default=0.10)
        prepare_alcuna_trivia_qa_eup(
            n_samples=args.n_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=args.seed,
            n_fewshot=args.n_fewshot,
            fix_val_test_splits=args.fix_val_test_splits,
        )
    elif dataset_name == "temporalood_trivia_qa":
        if args.n_samples is None:
            raise ValueError("n_samples is required for temporalood_trivia_qa splits.")
        test_frac = parse_frac(args.test_frac, default=0.10)
        val_frac = parse_frac(args.valid_frac, default=0.10)
        prepare_temporalood_trivia_qa_eup(
            n_samples=args.n_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=args.seed,
            n_fewshot=args.n_fewshot,
            fix_val_test_splits=args.fix_val_test_splits,
        )
    elif dataset_name == SITUATED_QA_DATASET_NAME:
        if args.n_samples is None:
            raise ValueError("n_samples is required for situated_qa splits.")
        test_frac = parse_frac(args.test_frac, default=0.10)
        val_frac = parse_frac(args.valid_frac, default=0.10)
        prepare_situated_qa_eup(
            n_samples=args.n_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=args.seed,
            n_fewshot=args.n_fewshot,
            fix_val_test_splits=args.fix_val_test_splits,
        )
    elif dataset_name in {"ambig_qa_filtered", "ambig_qa_filtered_ver3"}:
        if args.n_samples is None:
            raise ValueError(f"n_samples is required for {dataset_name} splits.")
        test_frac = parse_frac(args.test_frac, default=0.10)
        val_frac = parse_frac(args.valid_frac, default=0.10)
        prepare_ambig_filtered_eup(
            dataset_name=dataset_name,
            n_samples=args.n_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            model_name=args.model_name,
            judge_type=args.judge_type,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
