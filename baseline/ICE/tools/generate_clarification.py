import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import re
import json
import glob
import numpy as np
import copy
import tqdm
import argparse
from src.prompt_util import load_clarification_system_prompt, load_clarification_user_prompt
from src.data_util import load_data, ambigqa_train_pool_eup
from src.common import ContentFilterSkipError, completion_with_backoff
from src.shard_checkpoint import ensure_parent_dir, load_shard_checkpoint, write_json_list_atomic, write_shard_checkpoint

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type = str, default = None)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--sample", action = 'store_true')
parser.add_argument("--sample_n", type = int, default = 1)
parser.add_argument("--eval_split", type=str, default=None)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--test_frac", type=str, default=None)
parser.add_argument("--valid_frac", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_fewshot", type=int, default=5)
parser.add_argument("--fix_val_test_splits", type=str2bool, default=False)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--merge_shards", type=str2bool, default=False)
parser.add_argument("--shard_input_glob", type=str, default=None)
parser.add_argument("--skip_output_path", type=str, default=None)
parser.add_argument("--skip_shard_input_glob", type=str, default=None)
parser.add_argument("--source_index_key", type=str, default="_source_idx")
parser.add_argument("--merge_index_key", type=str, default="_merge_idx")

args = parser.parse_args()

AMBIG_QA_PROMPT_DATASETS = {
    "ambigqa",
    "ambig_qa",
    "ambig_qa_filtered",
    "ambig_qa_filtered_ver3",
}
OPEN_QA_PROMPT_DATASETS = {
    "nq_open",
    "alcuna_trivia_qa",
    "situated_qa",
    "temporalood_trivia_qa",
}


def extract_clarification_ambigqa(model_ans: str):
    lines = model_ans.split('\n')
    extract_list = []
    others = []
    pattern = r'^\d'
    for line in lines:
        if line.startswith('Clarifications'):
            continue
        match = re.match(pattern, line)
        if match:
            ext = line.strip()[len("1. "):]
            extract_list.append(ext)
        else:
            others.append(line.strip())
    return extract_list, others

def extract_clarification_ambiginst(model_output: str):
    if "No clarification needed" in model_output:
        return [],[]
    if not "Disambiguations:" in model_output:
        return [], []
    lines = model_output.split('Disambiguations:')[-1].strip().split('\n')
    all_exts = []
    others = []
    for line in lines:
        if len(line.strip()) == 0:
            break
        ext = line.strip()[len('1. '):]
        all_exts.append(ext)
    return all_exts, others

def extract_clarification_nq(model_ans: str):
    lines = model_ans.split('\n')
    extract_list = []
    others = []
    for line in lines:
        if line.startswith('Rephrase'):
            ext = line[len("Rephrase 1: "):]
            if 'specific' in ext.lower():
                continue
            extract_list.append(ext)
        elif line.startswith('Rephrased question'):
            ext = line[len("Rephrased question 1: "):]
            extract_list.append(ext)
        elif line.startswith('Rephrased'):
            ext = line[len("Rephrased 1: "):]
            extract_list.append(ext)
        elif line.startswith("Question: "):
            break
        else:
            others.append(line)
    return extract_list, others

def extract_clarification_gsm8k(model_output: str):
    extract_list = []
    lines = model_output.split('\n')
    for line in lines:
        if line.startswith("Rephrase "):
            line = line[len("Rephrase 1: "): ]
            extract_list.append(line)
    return extract_list, []

def extract_clarification(model_output, dataset_name):
    if dataset_name in AMBIG_QA_PROMPT_DATASETS:
        return extract_clarification_ambigqa(model_output)
    elif dataset_name == 'ambig_inst':
        return extract_clarification_ambiginst(model_output)
    elif dataset_name in OPEN_QA_PROMPT_DATASETS:
        return extract_clarification_nq(model_output)
    elif dataset_name == 'gsm8k':
        return extract_clarification_gsm8k(model_output)


def load_db(
    n_samples=None,
    test_frac=None,
    val_frac=None,
    seed=42,
    n_fewshot=5,
    fix_val_test_splits=False,
):
    if n_samples is None and test_frac is None and val_frac is None:
        pos_db_path = 'logs/dataset/ambigqa/ambigqa_train_ambig.json'
        neg_db_path = 'logs/dataset/ambigqa/ambigqa_train_unambig.json'
        with open(pos_db_path, 'r', encoding='utf-8') as f:
            pos_db = json.load(f)
        with open(neg_db_path, 'r', encoding='utf-8') as f:
            neg_db = json.load(f)
        return pos_db, neg_db

    train_data = ambigqa_train_pool_eup(
        n_samples=n_samples,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
        n_fewshot=n_fewshot,
        fix_val_test_splits=fix_val_test_splits,
    )
    pos_db = [x for x in train_data if "singleAnswer" not in x.get("label", [])]
    neg_db = [x for x in train_data if "singleAnswer" in x.get("label", [])]
    return pos_db, neg_db


def _build_icl_selector_for_ambigqa(pos_db, neg_db):
    # sentence_transformers is only needed for ambiguity datasets. Keep the
    # import local so open-QA datasets do not fail on unrelated env issues.
    try:
        from src.model_util import ICLModel
    except ImportError as exc:
        raise ImportError(
            "AmbigQA clarification generation requires src.model_util / "
            "sentence_transformers, but that import failed. The current "
            "environment likely has a transformers/peft mismatch."
        ) from exc
    return ICLModel(positive_db=pos_db, negative_db=neg_db)

def format_query(log_dict, dataset_name, user_prompt, icl_selector):
    if dataset_name in AMBIG_QA_PROMPT_DATASETS:
        q = log_dict['question']
        icl_examples = icl_selector.format_icl_prompt(question = q, ambig_num = 8, unambig_num = 8)
        icl_example_str = '\n\n'.join(icl_examples)
        prompt_q = 'Original Question: ' + q
        prompt_full = icl_example_str + '\n\n' + prompt_q
    elif dataset_name in OPEN_QA_PROMPT_DATASETS:
        q = log_dict['question']
        prompt_q = 'Question: ' + q
        prompt_full = user_prompt + '\n\n' + prompt_q
    elif dataset_name == 'gsm8k':
        q = log_dict['question']
        if not q.startswith('Question: '):
            q = 'Question: ' + q
        prompt_full = user_prompt + q
    elif dataset_name == 'alcuna_trivia_qa':
        q = log_dict['question']
        prompt_q = 'Question: ' + q
        prompt_full = user_prompt + '\n\n' + prompt_q
    elif dataset_name == 'ambig_inst':
        orig_inst = log_dict['orig_instruction']
        input_q = log_dict['input']
        prompt_full = user_prompt + '\n\n'
        prompt_full += 'Original Task Instruction: ' + orig_inst + '\n'
        prompt_full += f'Input: ' +  input_q.strip() + '\n\n'
    else:
        raise NotImplementedError(f"Unsupported dataset_name in format_query: {dataset_name}")

    return prompt_full


def _resolve_output_path(path: str):
    if path.endswith(".txt"):
        return path.replace(".txt", ".json")
    return path


def _resolve_skip_output_path(output_path: str, skip_output_path: str = None):
    if skip_output_path:
        return _resolve_output_path(skip_output_path)
    resolved_output = _resolve_output_path(output_path)
    if resolved_output.endswith(".json"):
        return resolved_output[:-5] + "_content_filter_skipped.json"
    return resolved_output + "_content_filter_skipped.json"


def _ensure_parent_dir(path: str):
    ensure_parent_dir(path)


def _write_json_list(path: str, records):
    write_json_list_atomic(path, records)


def _build_skip_record(args, stage_name: str, question: str, source_idx: int, merge_idx: int, exc: ContentFilterSkipError):
    return {
        args.source_index_key: int(source_idx),
        args.merge_index_key: int(merge_idx),
        "stage": stage_name,
        "reason": exc.metadata.get("reason", "content_filter"),
        "question": question,
        "content_filter": exc.metadata,
    }


def _validate_args(args):
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards.")
    if args.source_index_key == args.merge_index_key:
        raise ValueError("--source_index_key and --merge_index_key must be different.")
    if args.merge_shards:
        if not args.shard_input_glob:
            raise ValueError("--shard_input_glob is required when --merge_shards=True.")
        return
    if not args.dataset_name:
        raise ValueError("--dataset_name is required when --merge_shards=False.")
    if args.sample and args.sample_n < 1:
        raise ValueError("--sample_n must be >= 1 when --sample is set.")


def _get_shard_indices(total_size: int, num_shards: int, shard_id: int):
    return [idx for idx in range(total_size) if idx % num_shards == shard_id]


def _merge_shards(args):
    shard_files = sorted(glob.glob(args.shard_input_glob))
    if len(shard_files) == 0:
        raise ValueError(f"No shard files matched: {args.shard_input_glob}")

    merged = []
    merged_skips = []
    seen_indices = set()
    merge_key = args.merge_index_key
    skip_files = sorted(glob.glob(args.skip_shard_input_glob)) if args.skip_shard_input_glob else []

    for shard_file in shard_files:
        with open(shard_file, "r", encoding="utf-8") as f:
            shard_records = json.load(f)
        if not isinstance(shard_records, list):
            raise ValueError(f"Shard file is not a list: {shard_file}")
        for record in shard_records:
            if merge_key not in record:
                raise ValueError(f"Missing `{merge_key}` in shard file: {shard_file}")
            merge_idx = int(record[merge_key])
            if merge_idx in seen_indices:
                raise ValueError(f"Duplicate `{merge_key}`={merge_idx} found while merging.")
            seen_indices.add(merge_idx)
            merged.append(record)

    for skip_file in skip_files:
        with open(skip_file, "r", encoding="utf-8") as f:
            skip_records = json.load(f)
        if not isinstance(skip_records, list):
            raise ValueError(f"Skip shard file is not a list: {skip_file}")
        for record in skip_records:
            if merge_key not in record:
                raise ValueError(f"Missing `{merge_key}` in skip shard file: {skip_file}")
            merge_idx = int(record[merge_key])
            if merge_idx in seen_indices:
                raise ValueError(f"Duplicate `{merge_key}`={merge_idx} found while merging skips.")
            seen_indices.add(merge_idx)
            merged_skips.append(record)

    all_indices = sorted(seen_indices)
    if len(all_indices) > 0:
        min_idx = min(all_indices)
        max_idx = max(all_indices)
        expected = set(range(min_idx, max_idx + 1))
        missing = sorted(expected - set(all_indices))
        if missing:
            preview = missing[:10]
            raise ValueError(f"Missing source indices detected during merge (first 10): {preview}")

    merged.sort(key=lambda x: int(x[merge_key]))
    for record in merged:
        record.pop(merge_key, None)

    output_path = _resolve_output_path(args.output_path)
    skip_output_path = _resolve_skip_output_path(args.output_path, args.skip_output_path)
    merged_skips.sort(key=lambda x: int(x.get(merge_key, x.get(args.source_index_key, 0))))
    _write_json_list(output_path, merged)
    _write_json_list(skip_output_path, merged_skips)
    print(
        f"Merged {len(shard_files)} shard files -> {output_path} ({len(merged)} records); "
        f"merged {len(merged_skips)} skipped records -> {skip_output_path}"
    )


def main(args):
    _validate_args(args)
    if args.merge_shards:
        _merge_shards(args)
        return

    dataset_name = "ambigqa" if args.dataset_name == "ambig_qa" else args.dataset_name
    system_prompt = load_clarification_system_prompt(dataset_name)
    user_prompt = load_clarification_user_prompt(dataset_name)

    if dataset_name in AMBIG_QA_PROMPT_DATASETS:
        pos_db, neg_db = load_db(
            n_samples=args.n_samples,
            test_frac=args.test_frac,
            val_frac=args.valid_frac,
            seed=args.seed,
            n_fewshot=args.n_fewshot,
            fix_val_test_splits=args.fix_val_test_splits,
        )
        icl_selector = _build_icl_selector_for_ambigqa(pos_db, neg_db)
    else:
        icl_selector = None

    output_path = args.output_path
    output_path = _resolve_output_path(output_path)
    skip_output_path = _resolve_skip_output_path(output_path, args.skip_output_path)
    _ensure_parent_dir(output_path)
    _ensure_parent_dir(skip_output_path)
    print("save logs to ", output_path)
    # model_index = 'gpt-5-mini-2' if dataset_name == 'ambigqa' else 'gpt-4'
    model_index = 'gpt-5-mini-2'

    test_data = load_data(
        dataset_name,
        eval_split=args.eval_split,
        n_samples=args.n_samples,
        test_frac=args.test_frac,
        val_frac=args.valid_frac,
        seed=args.seed,
        n_fewshot=args.n_fewshot,
        fix_val_test_splits=args.fix_val_test_splits,
    )

    shard_indices = _get_shard_indices(len(test_data), args.num_shards, args.shard_id)
    all_results, skipped_records, processed_indices = load_shard_checkpoint(
        output_path,
        skip_output_path,
        args.merge_index_key,
        args.source_index_key,
    )
    processed_shard_indices = {idx for idx in shard_indices if idx in processed_indices}
    remaining_indices = [idx for idx in shard_indices if idx not in processed_indices]
    if processed_indices:
        print(
            f"[resume] loaded {len(all_results)} completed records and "
            f"{len(skipped_records)} skipped records from checkpoint."
        )
    if processed_shard_indices:
        print(
            f"[resume] skipping {len(processed_shard_indices)} previously completed "
            f"indices for shard {args.shard_id}."
        )
    progress_desc = f"generate_clarification[{dataset_name}]"
    progress_desc += f" shard {args.shard_id + 1}/{args.num_shards}"

    progress_bar = tqdm.tqdm(
        remaining_indices,
        total=len(shard_indices),
        desc=progress_desc,
        unit="sample",
        dynamic_ncols=True,
        initial=len(processed_shard_indices),
    )
    for idx in progress_bar:
        case = test_data[idx]
        source_idx = int(case.get(args.source_index_key, idx))
        prompt_full = format_query(case, dataset_name, user_prompt, icl_selector)

        max_tokens=512
        if args.sample:
            temperature=1.0
            sample_n = args.sample_n
        else:
            temperature=0
            sample_n = 1

        if len(system_prompt) > 1:
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_full},
            ]
        else:
            messages=[
                {"role": "user", "content": prompt_full},
            ]
        try:
            response = completion_with_backoff(
                model=model_index,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=sample_n,
                content_filter_mode="skip_question",
                )
        except ContentFilterSkipError as exc:
            skipped_records.append(
                _build_skip_record(
                    args=args,
                    stage_name="generate_clarification",
                    question=case.get("question", ""),
                    source_idx=source_idx,
                    merge_idx=idx,
                    exc=exc,
                )
            )
            print(
                f"[warn] skipping source_idx={source_idx} merge_idx={idx} in generate_clarification "
                "due to content_filter."
            )
            write_shard_checkpoint(
                output_path,
                all_results,
                skip_output_path,
                skipped_records,
                args.merge_index_key,
                args.source_index_key,
            )
            continue
        ans_model_list = []
        other_outputs = []
        for sample_id in range(sample_n):
            ans_model = response['choices'][sample_id]['message']['content']
            extraction, others = extract_clarification(ans_model, dataset_name)
            ans_model_list += extraction
            other_outputs.append(others)

        result = copy.deepcopy(case)
        if dataset_name == 'ambig_inst':
            result['orig_inst'] = case['orig_instruction']            
        result['self_clarification'] = ans_model_list
        result['others'] = other_outputs
        result[args.source_index_key] = source_idx
        result[args.merge_index_key] = idx
        all_results.append(result)
        write_shard_checkpoint(
            output_path,
            all_results,
            skip_output_path,
            skipped_records,
            args.merge_index_key,
            args.source_index_key,
        )

    progress_bar.close()

    write_shard_checkpoint(
        output_path,
        all_results,
        skip_output_path,
        skipped_records,
        args.merge_index_key,
        args.source_index_key,
    )

if __name__ == '__main__':
    main(args)
