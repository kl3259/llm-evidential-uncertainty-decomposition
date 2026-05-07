import os
import json
import datasets
import numpy as np

_SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
}

FIXED_SPLIT_SEEDS = {
    "fewshot_membership": 131071,
    "test_membership": 131101,
    "valid_membership": 131111,
    "train_membership": 131131,
    "train_order": 131149,
    "valid_order": 131173,
    "test_order": 131183,
}


def _normalize_dataset_name(dataset_name):
    if dataset_name == "ambig_qa":
        return "ambigqa"
    return dataset_name


def _split_seed(base_seed: int, fix_val_test_splits: bool, stage: str) -> int:
    if fix_val_test_splits:
        return int(FIXED_SPLIT_SEEDS[stage])
    return int(base_seed)


def parse_frac(value, default=None):
    if value is None:
        return default
    if isinstance(value, (int, np.integer, float, np.floating)):
        return value
    value = str(value).strip()
    if value == "":
        return default
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid fraction/count value: {value}") from exc


def _normalize_split(split_name):
    if split_name is None:
        return None
    split_name = str(split_name).strip().lower()
    split_name = _SPLIT_ALIASES.get(split_name, split_name)
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported eval_split: {split_name}")
    return split_name


def _format_frac_tag(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    tag = f"{val:.6f}".rstrip("0").rstrip(".")
    return tag if tag else "0"


def _build_split_filename(dataset_name, split_name, n_samples, val_frac, test_frac):
    val_tag = _format_frac_tag(val_frac)
    test_tag = _format_frac_tag(test_frac)
    return f"{dataset_name}_{split_name}_{n_samples}_{val_tag}_{test_tag}.json"


def _load_prepared_split_dataset(dataset_name, eval_split, n_samples, test_frac, val_frac):
    eval_split = _normalize_split(eval_split or "test")
    if n_samples is None and test_frac is None and val_frac is None:
        data_path = f"logs/dataset/{dataset_name}/{dataset_name}_{eval_split}.json"
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(
            f"Missing split file {data_path}. "
            f"Run tools/prepare_data.py with --dataset_name={dataset_name} and split args, "
            "or pass n_samples/test_frac/valid_frac."
        )

    n_samples = int(n_samples) if n_samples is not None else 2000
    test_frac = parse_frac(test_frac, default=0.10)
    val_frac = parse_frac(val_frac, default=0.10)

    split_file = _build_split_filename(
        dataset_name=dataset_name,
        split_name=eval_split,
        n_samples=n_samples,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    data_path = os.path.join("logs", "dataset", dataset_name, split_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing split file {data_path}. "
            f"Run tools/prepare_data.py with --dataset_name={dataset_name} "
            f"--n_samples={n_samples} --test_frac={test_frac} --valid_frac={val_frac}."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_ambigqa_log_dict(data, skip_mixed=False):
    question = data.get("question", "")
    annotate_dict = data.get("annotations", {})
    label = annotate_dict.get("type", [])
    if skip_mixed and "singleAnswer" in label and "multipleQAs" in label:
        return None

    qa_pairs = annotate_dict.get("qaPairs", [])
    answers = annotate_dict.get("answer", [])
    if not isinstance(label, list):
        label = [label] if label is not None else []

    num_annotate = min(len(qa_pairs), len(answers), len(label))
    gt_rewrite = []
    gt_answers = []
    for anno_idx in range(num_annotate):
        curr_label = label[anno_idx]
        curr_single_answers = answers[anno_idx]
        if curr_label == "singleAnswer":
            gt_rewrite += [question]
            gt_answers += [curr_single_answers]
        elif curr_label == "multipleQAs":
            gt_rewrite += qa_pairs[anno_idx].get("question", [])
            gt_answers += qa_pairs[anno_idx].get("answer", [])

    return {
        "question": question,
        "label": label,
        "orig_annotation": annotate_dict,
        "gt_rewrite": gt_rewrite,
        "answer": gt_answers,
    }


def _ambigqa_dataset_to_list(dataset, skip_mixed=False):
    results = []
    for idx in range(len(dataset)):
        log_dict = _build_ambigqa_log_dict(dataset[idx], skip_mixed=skip_mixed)
        if log_dict is not None:
            results.append(log_dict)
    return results


def _require_int_count(value, name):
    if not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be provided as an integer count when --fix_val_test_splits=True."
        )
    return int(value)


def split_ambigqa_eup(
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    seed=42,
    n_fewshot=5,
    cache_dir="dataset_cache",
    skip_mixed=False,
    fix_val_test_splits=False,
):
    raw = datasets.load_dataset("sewon/ambig_qa", "full", cache_dir=cache_dir)

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

        full = raw["train"].shuffle(seed=42)
        fixed_test = raw["validation"]

        rng = np.random.default_rng(_split_seed(seed, True, "fewshot_membership"))
        k = min(n_fewshot, len(full))
        few_idx = set(rng.choice(len(full), size=k, replace=False).tolist())
        keep_idx = [i for i in range(len(full)) if i not in few_idx]
        remainder = full.select(keep_idx)

        if val_size > len(remainder):
            raise ValueError(
                f"val_size {val_size} exceeds available train pool size {len(remainder)}."
            )
        if val_size > 0:
            train_val_split = remainder.train_test_split(
                test_size=val_size,
                seed=_split_seed(seed, True, "valid_membership"),
            )
        else:
            train_val_split = datasets.DatasetDict(
                {
                    "train": remainder,
                    "test": remainder.select(range(0)),
                }
            )

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
            raise ValueError(
                f"test_size {test_size} exceeds fixed test split size {len(fixed_test)}"
            )
        if test_size > 0:
            test_ds = fixed_test.shuffle(
                seed=_split_seed(seed, True, "test_membership")
            ).select(range(test_size))
        else:
            test_ds = fixed_test.select(range(0))

        train_ds = train_ds.shuffle(seed=_split_seed(seed, True, "train_order"))
        valid_ds = valid_ds.shuffle(seed=_split_seed(seed, True, "valid_order"))
        test_ds = test_ds.shuffle(seed=_split_seed(seed, True, "test_order"))

        return {
            "train": _ambigqa_dataset_to_list(train_ds, skip_mixed=skip_mixed),
            "validation": _ambigqa_dataset_to_list(valid_ds, skip_mixed=skip_mixed),
            "test": _ambigqa_dataset_to_list(test_ds, skip_mixed=skip_mixed),
        }

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

    full = raw["train"].shuffle(seed=seed).select(range(train_target))
    fixed_test = raw["validation"]

    rng = np.random.default_rng(seed)
    k = min(n_fewshot, len(full))
    few_idx = set(rng.choice(len(full), size=k, replace=False).tolist())
    keep_idx = [i for i in range(len(full)) if i not in few_idx]
    remainder = full.select(keep_idx)

    N = len(remainder)
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

    return {
        "train": _ambigqa_dataset_to_list(train_ds, skip_mixed=skip_mixed),
        "validation": _ambigqa_dataset_to_list(valid_ds, skip_mixed=skip_mixed),
        "test": _ambigqa_dataset_to_list(test_ds, skip_mixed=skip_mixed),
    }


def ambigqa_train_pool_eup(
    n_samples,
    test_frac=0.10,
    val_frac=0.10,
    seed=42,
    n_fewshot=5,
    cache_dir="dataset_cache",
    skip_mixed=False,
    fix_val_test_splits=False,
):
    if fix_val_test_splits:
        test_frac = parse_frac(test_frac, default=0.10)
        val_frac = parse_frac(val_frac, default=0.10)
        splits = split_ambigqa_eup(
            n_samples=n_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=seed,
            n_fewshot=n_fewshot,
            cache_dir=cache_dir,
            skip_mixed=skip_mixed,
            fix_val_test_splits=True,
        )
        return splits["train"]

    raw = datasets.load_dataset("sewon/ambig_qa", "full", cache_dir=cache_dir)
    test_frac = parse_frac(test_frac, default=0.10)

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

    full = raw["train"].shuffle(seed=seed).select(range(train_target))
    return _ambigqa_dataset_to_list(full, skip_mixed=skip_mixed)


def load_data(
    dataset_name,
    eval_split=None,
    n_samples=None,
    test_frac=None,
    val_frac=None,
    seed=42,
    n_fewshot=5,
    cache_dir="dataset_cache",
    fix_val_test_splits=False,
):
    dataset_name = _normalize_dataset_name(dataset_name)
    if dataset_name == "ambigqa":
        if eval_split is None and n_samples is None and test_frac is None and val_frac is None:
            data_path = "logs/dataset/ambigqa/ambigqa_dev_balance.json"
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        if n_samples is None and test_frac is None and val_frac is None:
            eval_split = _normalize_split(eval_split or "test")
            data_path = f"logs/dataset/ambigqa/ambigqa_{eval_split}.json"
            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            raise FileNotFoundError(
                f"Missing split file {data_path}. "
                "Run tools/prepare_data.py with EUP split args or pass n_samples/test_frac/valid_frac."
            )

        eval_split = _normalize_split(eval_split or "test")
        n_samples = int(n_samples) if n_samples is not None else 2000
        test_frac = parse_frac(test_frac, default=0.10)
        val_frac = parse_frac(val_frac, default=0.10)

        splits = split_ambigqa_eup(
            n_samples=n_samples,
            test_frac=test_frac,
            val_frac=val_frac,
            seed=seed,
            n_fewshot=n_fewshot,
            cache_dir=cache_dir,
            fix_val_test_splits=fix_val_test_splits,
        )
        return splits[eval_split]
    elif dataset_name == "ambig_inst":
        data_path = "logs/dataset/ambiginst.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    elif dataset_name == "nq_open":
        all_logs = datasets.load_dataset("nq_open", cache_dir="dataset_cache")
        test_data = all_logs["validation"].select(np.arange(200))
        test_data = [test_data[x] for x in range(200)]
        return test_data
    elif dataset_name == "gsm8k":
        gsm8k = datasets.load_dataset("gsm8k", "main")
        test_data = gsm8k["test"].select(np.arange(200))
        test_data = [test_data[x] for x in range(200)]
        return test_data
    elif dataset_name == "alcuna_trivia_qa":
        return _load_prepared_split_dataset(dataset_name, eval_split, n_samples, test_frac, val_frac)
    elif dataset_name in {
        "temporalood_trivia_qa",
        "situated_qa",
        "ambig_qa_filtered",
        "ambig_qa_filtered_ver3",
    }:
        return _load_prepared_split_dataset(dataset_name, eval_split, n_samples, test_frac, val_frac)
