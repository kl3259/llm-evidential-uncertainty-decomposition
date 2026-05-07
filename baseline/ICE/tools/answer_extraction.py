import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import json
import glob
import numpy as np
import copy
from typing import List
import tqdm
import argparse
from transformers import GPT2Tokenizer
from src.common import ContentFilterSkipError, completion_with_backoff
from src.shard_checkpoint import ensure_parent_dir, load_shard_checkpoint, write_json_list_atomic, write_shard_checkpoint

INVLAID_RESUTL, EXCEED_LENGTH_LIMIT, SUCCESS = 1111, 2222, 3333


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
parser.add_argument("--log_path", type = str, default = None)
parser.add_argument("--prompt_path", type = str, default = None)
parser.add_argument("--answer_key", type = str, default = None)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--azure_config", type=str, default="config/azure_openai.json")
parser.add_argument(
    "--parse_retry_max",
    type=int,
    default=3,
    help="Maximum retries for format/parsing mismatch in extraction outputs.",
)
parser.add_argument(
    "--rewrite_retry_max",
    type=int,
    default=1,
    help="Maximum retries for each rewrite in fallback per-rewrite extraction path.",
)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--merge_shards", type=str2bool, default=False)
parser.add_argument("--shard_input_glob", type=str, default=None)
parser.add_argument("--skip_output_path", type=str, default=None)
parser.add_argument("--skip_shard_input_glob", type=str, default=None)
parser.add_argument("--source_index_key", type=str, default="_source_idx")
parser.add_argument("--merge_index_key", type=str, default="_merge_idx")

args = parser.parse_args()


def extract_rewrite(model_ans: str):
    exts = []
    answer_sets = []
    lines = model_ans.split('\n')
    for raw_line in lines:
        line = raw_line.strip()
        if line.lower().startswith("extraction"):
            try:
                ans = line.split(":", 1)[1].strip()
                # ans = line.strip()[len("Extraction 1/N: "):].strip()
            except:
                print(f"{line} cannot be splitted!")
                ans = line[len("Extraction"):].strip()
            exts.append(ans)
        else:
            ans_set = line
            answer_sets.append(ans_set)
    return exts, answer_sets

def format_message(system_prompt, user_prompt):
    if len(system_prompt) > 1:
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages=[
            {"role": "user", "content": user_prompt},
        ]
    return messages

def megabatch_forward(q: str, all_rewrite_answers: List[str]):
    ans_num_per_question = len(all_rewrite_answers[0])

    concate_all_rewrite_answers = []
    for ans_list in all_rewrite_answers:
        concate_all_rewrite_answers += ans_list

    ans2id = {}
    cleaned_answers = []
    for cdt_ans in concate_all_rewrite_answers:
        if cdt_ans.startswith('A: '):
            cdt_ans = cdt_ans[len('A: '):]
        cleaned_answers.append(cdt_ans)
        if cdt_ans not in ans2id: ans2id[cdt_ans] = len(ans2id)

    ext_rewrite_all_ans = []
    prompt_q = 'Q: ' + q

    for ans_id, cdt_ans in enumerate(list(ans2id.keys())):
        prompt_q = prompt_q + f'\nA{ans_id+1}: ' + cdt_ans
    prompt_q = prompt_q + '\nAnswer set at the begining: []'
    user_prompt = prompt_q

    tokens = tokenizer.tokenize(system_prompt + user_prompt)
    print(q)
    print(len(tokens))
    if len(tokens) > 2500:
        return EXCEED_LENGTH_LIMIT, None, None

    messages = format_message(system_prompt, user_prompt)

    parse_retry_max = max(1, int(args.parse_retry_max))
    ext_ans, ans_sets = [], []
    ans_model = ""
    for attempt_idx in range(parse_retry_max):
        response = completion_with_backoff(
            model="gpt-5-mini-2",
            messages=messages,
            max_completion_tokens=max_tokens,
            config_path=args.azure_config,
            content_filter_mode="skip_question",
            n=1,
            )
        ans_model = response['choices'][0]['message']['content']
        ext_ans, ans_sets = extract_rewrite(ans_model)
        if len(ext_ans) == len(ans2id):
            print(ans_model)
            break
        print(
            f"[Warning] parse mismatch ({attempt_idx+1}/{parse_retry_max}) "
            f"for question: {q[:80]}... expected {len(ans2id)} extractions, got {len(ext_ans)}"
        )

    if len(ext_ans) != len(ans2id):
        print("failed!")
        print(ans_model)
        ext_rewrite_all_ans.append(cleaned_answers)
        return INVLAID_RESUTL, None, None

    map_back_list = []
    for ans_index, ans in enumerate(cleaned_answers):
        map_back_list.append(ext_ans[ans2id[ans]])
        if ans_index % ans_num_per_question == ans_num_per_question - 1:
            ext_rewrite_all_ans.append(map_back_list)
            map_back_list = []

    return SUCCESS, ext_rewrite_all_ans, ans_sets

def batch_forward(q: str, rewrite_answers, prev_ans_set):
    prompt_q = 'Q: ' + q

    ans2id = {}
    cleaned_answers = []
    for cdt_ans in rewrite_answers:
        if cdt_ans.startswith('A: '):
            cdt_ans = cdt_ans[len('A: '):]
        cleaned_answers.append(cdt_ans)
        if cdt_ans not in ans2id: ans2id[cdt_ans] = len(ans2id)

    for ans_id, cdt_ans in enumerate(list(ans2id.keys())):
        prompt_q = prompt_q + f'\nA{ans_id+1}: ' + cdt_ans
    prompt_q = prompt_q + f'\nAnswer set at the begining: {prev_ans_set}'
    user_prompt = prompt_q

    tokens = tokenizer.tokenize(system_prompt + user_prompt)
    print(q)
    print(len(tokens))

    _max_tokens = 2000
    curr_model = "gpt-5-mini-2"
    if len(tokens) > 2000:
        curr_model = "gpt-5-mini-2"


    print(f"Answer set at the begining: {prev_ans_set}")
    messages = format_message(system_prompt, user_prompt)

    parse_retry_max = max(1, int(args.parse_retry_max))
    ext_ans, ans_sets = [], []
    ans_model = ""
    for attempt_idx in range(parse_retry_max):
        response = completion_with_backoff(
            model=curr_model,
            messages=messages,
            max_completion_tokens=_max_tokens,
            config_path=args.azure_config,
            content_filter_mode="skip_question",
            n=1,
            )
        ans_model = response['choices'][0]['message']['content']
        ext_ans, ans_sets = extract_rewrite(ans_model)
        if len(ext_ans) == len(ans2id):
            break
        print(
            f"[warn] parse mismatch ({attempt_idx+1}/{parse_retry_max}) "
            f"for question: {q[:80]}... expected {len(ans2id)} extractions, got {len(ext_ans)}"
        )

    if len(ext_ans) != len(ans2id):
        print("failed!")
        print(ans_model)
        return INVLAID_RESUTL, [], prev_ans_set

    map_back_list = []
    for ans in cleaned_answers:
        map_back_list.append(ext_ans[ans2id[ans]])
    return SUCCESS, map_back_list, ans_sets[-1]


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
    missing = []
    if not args.log_path:
        missing.append("--log_path")
    if not args.prompt_path:
        missing.append("--prompt_path")
    if not args.answer_key:
        missing.append("--answer_key")
    if missing:
        raise ValueError(f"Missing required args when --merge_shards=False: {', '.join(missing)}")


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


if __name__ == '__main__':
    _validate_args(args)
    if args.merge_shards:
        _merge_shards(args)
        sys.exit(0)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # estimate prompt length

    system_prompt = open(args.prompt_path,'r',encoding='utf-8').read().strip()

    output_path = _resolve_output_path(args.output_path)
    skip_output_path = _resolve_skip_output_path(output_path, args.skip_output_path)
    _ensure_parent_dir(output_path)
    _ensure_parent_dir(skip_output_path)
    print("save logs to ", output_path)

    with open(args.log_path,'r',encoding='utf-8') as f:
        test_data = json.load(f)

    multiple_outputs = type(test_data[0][args.answer_key][0]) == list
    if multiple_outputs:
        print("There are more than one output distribution; Extract them together.")

    max_tokens=1500
    sample_n = 1
    extract_ans_key = 'ext_' + args.answer_key


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
    progress_bar = tqdm.tqdm(
        remaining_indices,
        total=len(shard_indices),
        desc=f"answer_extraction shard {args.shard_id + 1}/{args.num_shards}",
        unit="sample",
        dynamic_ncols=True,
        initial=len(processed_shard_indices),
    )
    for idx in progress_bar:
        case = test_data[idx]
        q = case['question']
        all_rewrite_answers = case[args.answer_key]
        source_idx = int(case.get(args.source_index_key, idx))

        result = copy.deepcopy(case)
        result[args.source_index_key] = source_idx
        result[args.merge_index_key] = idx

        try:
            if multiple_outputs:
                FLAG, ext_rewrite_all_ans, ans_sets = megabatch_forward(q, all_rewrite_answers)
                if FLAG == SUCCESS:
                    result[extract_ans_key] = ext_rewrite_all_ans
                    result['ext_ans_sets'] = ans_sets
                    all_results.append(result)
                    write_shard_checkpoint(
                        output_path,
                        all_results,
                        skip_output_path,
                        skipped_records,
                        args.merge_index_key,
                        args.source_index_key,
                    )
                elif FLAG == EXCEED_LENGTH_LIMIT or FLAG == INVLAID_RESUTL:
                    ext_rewrite_all_ans = []
                    ans_sets = []
                    prev_ans_set = "[]"
                    rewrite_retry_max = max(1, int(args.rewrite_retry_max))
                    for rewrite_answers in all_rewrite_answers:
                        for rewrite_attempt_idx in range(rewrite_retry_max):
                            FLAG, curr_ext_res, curr_ans_set = batch_forward(q, rewrite_answers, prev_ans_set)
                            if FLAG != INVLAID_RESUTL:
                                break
                            print(
                                f"[warn] rewrite retry ({rewrite_attempt_idx+1}/{rewrite_retry_max}) failed "
                                f"for question: {q[:80]}..."
                            )
                        print(curr_ans_set)
                        if curr_ans_set.startswith('Final answer set: '):
                            processed_curr_ans_set = curr_ans_set.strip()[len('Final answer set: '):]
                        elif curr_ans_set.startswith("Updated answer set"):
                            processed_curr_ans_set = curr_ans_set.strip()[len('Updated answer set: '):]
                        else:
                            processed_curr_ans_set = curr_ans_set.strip()
                        print("answer set after process: ", processed_curr_ans_set)

                        if FLAG == INVLAID_RESUTL:
                            ext_rewrite_all_ans.append([])
                            ans_sets.append(None)
                        else:
                            ext_rewrite_all_ans.append(curr_ext_res)
                            prev_ans_set = processed_curr_ans_set
                            ans_sets.append(curr_ans_set)
                    result[extract_ans_key] = ext_rewrite_all_ans
                    result['ext_ans_sets'] = ans_sets
                    all_results.append(result)
                    write_shard_checkpoint(
                        output_path,
                        all_results,
                        skip_output_path,
                        skipped_records,
                        args.merge_index_key,
                        args.source_index_key,
                    )
                else:
                    raise NotImplementedError
            else:
                prev_ans_set = "[]"
                rewrite_answers = all_rewrite_answers
                FLAG, curr_ext_res, curr_ans_set = batch_forward(q, rewrite_answers, prev_ans_set)
                print(curr_ans_set)
                if curr_ans_set.startswith('Final answer set: '):
                    processed_curr_ans_set = curr_ans_set.strip()[len('Final answer set: '):]
                elif curr_ans_set.startswith("Updated answer set"):
                    processed_curr_ans_set = curr_ans_set.strip()[len('Updated answer set: '):]
                else:
                    processed_curr_ans_set = curr_ans_set.strip()
                print("answer set after process: ", processed_curr_ans_set)

                if FLAG == INVLAID_RESUTL:
                    ext_rewrite_all_ans = all_rewrite_answers
                    ans_sets = None
                else:
                    ext_rewrite_all_ans = curr_ext_res
                    prev_ans_set = processed_curr_ans_set
                    ans_sets = curr_ans_set
                result[extract_ans_key] = ext_rewrite_all_ans
                result['ext_ans_sets'] = ans_sets
                all_results.append(result)
                write_shard_checkpoint(
                    output_path,
                    all_results,
                    skip_output_path,
                    skipped_records,
                    args.merge_index_key,
                    args.source_index_key,
                )
        except ContentFilterSkipError as exc:
            skipped_records.append(
                _build_skip_record(
                    args=args,
                    stage_name="answer_extraction",
                    question=q,
                    source_idx=source_idx,
                    merge_idx=idx,
                    exc=exc,
                )
            )
            print(
                f"[warn] skipping source_idx={source_idx} merge_idx={idx} in answer_extraction "
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

    progress_bar.close()
    write_shard_checkpoint(
        output_path,
        all_results,
        skip_output_path,
        skipped_records,
        args.merge_index_key,
        args.source_index_key,
    )
