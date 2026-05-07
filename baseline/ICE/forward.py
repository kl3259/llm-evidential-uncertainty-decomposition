import re
import os
import random
from tqdm.auto import tqdm as auto_tqdm
import copy
import json
import numpy as np
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None
try:
    from accelerate import Accelerator
except Exception:
    Accelerator = None
try:
    from transformers import Gemma3ForCausalLM
except Exception:
    Gemma3ForCausalLM = None
from src.data_util import load_data
from src.prompt_util import load_fewshot_prompt, load_system_prompt, inst_transform
from src.config import MAX_CLARIFICATION, SAMPLE_N

ALLOWED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "google/gemma-3-12b-it",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type = str, required = True)
parser.add_argument("--clarification_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    choices=ALLOWED_MODELS,
)
parser.add_argument(
    "--generate_config",
    type=str,
    default="original",
    choices=["original", "eup"],
)
args = parser.parse_args()

SOURCE_INDEX_KEY = "_source_idx"
ORDER_INDEX_KEY = "_forward_order_idx"
AMBIG_QA_PROMPT_DATASETS = {
    "ambigqa",
    "ambig_qa",
    "ambig_qa_filtered",
    "ambig_qa_filtered_ver3",
}
OPEN_QA_PROMPT_DATASETS = {
    "nq_open",
    "situated_qa",
    "alcuna_trivia_qa",
    "temporalood_trivia_qa",
}


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

def build_prompt(messages, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    contents = []
    for msg in messages:
        contents.append(msg["content"])
    return "\n\n".join(contents)


def _dedup_and_limit(items):
    unique_items = []
    for item in items:
        if item not in unique_items:
            unique_items.append(item)
    return unique_items[:MAX_CLARIFICATION]


def _format_eup_query(question_text, dataset_name):
    question_prompt = question_text
    if dataset_name in {"nq_open", "situated_qa"}:
        question_prompt = question_text + "? "
    return f"{question_prompt} Response with the answer only, do not explain."


def format_query(log_dict, dataset_name, prompt, generate_config):
    if dataset_name in AMBIG_QA_PROMPT_DATASETS or dataset_name in {"nq_open", "situated_qa"}:
        orig_q = log_dict['question']
        rewrite_qs = log_dict['self_clarification']
        if len(rewrite_qs) == 0:
            rewrite_qs = [orig_q]
        rewrite_qs = _dedup_and_limit(rewrite_qs)

        if generate_config == "eup":
            clarified_inputs = [_format_eup_query(x, dataset_name) for x in rewrite_qs]
        else:
            clarified_inputs = [prompt.strip() + '\n\nQ: ' + x for x in rewrite_qs]
    elif dataset_name == 'ambig_inst':
        orig_inst = log_dict['orig_instruction']
        all_clarifications = log_dict['self_clarification']
        q = log_dict['input']
        if len(all_clarifications) == 0:
            all_clarifications = [log_dict['orig_instruction']]
        all_clarifications = [x.strip() for x in all_clarifications if len(x.strip()) > 0]
        all_clarifications = [inst_transform(orig_inst, x) for x in all_clarifications][:MAX_CLARIFICATION]
        clarified_inputs = []
        for clarified_inst in all_clarifications:
            prompt_q = "Task: " + clarified_inst + '\nInput: ' + q + "\nOutput: "
            clarified_inputs.append(prompt_q)
    elif dataset_name == 'gsm8k':
        orig_q = log_dict['question']
        rewrite_qs = log_dict['self_clarification']
        if len(rewrite_qs) == 0:
            rewrite_qs = [orig_q]
        rewrite_qs = _dedup_and_limit(rewrite_qs)
        if generate_config == "eup":
            clarified_inputs = [_format_eup_query(x, dataset_name) for x in rewrite_qs]
        else:
            clarified_inputs = [prompt.strip() + '\n\nQuestion: ' + orig_q + '\n' for x in rewrite_qs]
    elif dataset_name in {"alcuna_trivia_qa", "temporalood_trivia_qa"}:
        orig_q = log_dict['question']
        rewrite_qs = log_dict.get('self_clarification', [])
        if len(rewrite_qs) == 0:
            rewrite_qs = [orig_q]
        rewrite_qs = _dedup_and_limit(rewrite_qs)
        # Match EUP training prompt style used in feature collection.
        clarified_inputs = [f"{x} Response with the answer only, do not explain." for x in rewrite_qs]

    else:
        raise NotImplementedError
    return clarified_inputs

def post_process(model_response, dataset_name):
    if dataset_name in AMBIG_QA_PROMPT_DATASETS:
        if len(model_response.split('\n')) >= 2:
            return model_response.split('\n')[0]
        else:
            return model_response
    elif dataset_name == 'ambig_inst':
        return model_response
    elif dataset_name == 'gsm8k':
        return model_response
    elif dataset_name in {"nq_open", "situated_qa"}:
        return model_response
    elif dataset_name in {"alcuna_trivia_qa", "temporalood_trivia_qa"}:
        return model_response


def _build_accelerator():
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    distributed_env = world_size > 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ
    if not distributed_env:
        return None
    if Accelerator is None:
        raise ImportError(
            "accelerate is required for multi-process forward inference. "
            "Install `accelerate` and launch with `accelerate launch --num_processes N ...`."
        )
    return Accelerator()


def _split_indices_for_process(n_total, accelerator):
    if accelerator is None or accelerator.num_processes <= 1:
        return list(range(n_total))
    shards = np.array_split(np.arange(n_total, dtype=np.int64), accelerator.num_processes)
    return shards[int(accelerator.process_index)].tolist()


def _results_shard_dir(output_path):
    return f"{output_path}.parts"


def _results_shard_path(output_path, rank):
    shard_dir = _results_shard_dir(output_path)
    return os.path.join(shard_dir, f"rank{int(rank):05d}.json")


def _write_local_results_shard(local_results, output_path, accelerator):
    shard_dir = _results_shard_dir(output_path)
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = _results_shard_path(output_path, accelerator.process_index)
    tmp_path = f"{shard_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f)
    os.replace(tmp_path, shard_path)


def _gather_results_to_main(local_results, accelerator, output_path):
    if accelerator is None or accelerator.num_processes <= 1:
        return local_results

    # Avoid NCCL object collectives for large Python payloads; shard to disk and merge on rank 0.
    _write_local_results_shard(local_results, output_path, accelerator)
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return None

    merged = []
    for rank in range(int(accelerator.num_processes)):
        shard_path = _results_shard_path(output_path, rank)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing distributed result shard: {shard_path}")
        with open(shard_path, "r", encoding="utf-8") as f:
            shard_records = json.load(f)
        if shard_records:
            merged.extend(shard_records)
    return merged


def _finalize_order(records):
    if not records:
        return records
    if all(ORDER_INDEX_KEY not in row for row in records):
        return records
    ordered = sorted(records, key=lambda row: int(row[ORDER_INDEX_KEY]))
    for row in ordered:
        row.pop(ORDER_INDEX_KEY, None)
    return ordered


def _load_generation_model(model_name, dtype, accelerator):
    use_multi_process = accelerator is not None and accelerator.num_processes > 1

    if "gemma-3" in model_name.lower():
        if Gemma3ForCausalLM is None:
            raise RuntimeError("Gemma3ForCausalLM is unavailable; upgrade transformers for gemma-3 support.")
        model_cls = Gemma3ForCausalLM
    else:
        model_cls = AutoModelForCausalLM

    if use_multi_process:
        try:
            model = model_cls.from_pretrained(model_name, torch_dtype=dtype)
            model.to(accelerator.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    "Model must fit on a single GPU per process for Accelerator multi-GPU forward inference. "
                    "If you need cross-GPU model sharding instead, use the single-process path with "
                    "`device_map=\"auto\"`."
                ) from e
            raise
        return model

    return model_cls.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)


def _resolve_pretrained_path(model_name):
    if os.path.isdir(model_name) or snapshot_download is None:
        return model_name
    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        return model_name


def _tokenizer_load_kwargs(model_name):
    kwargs = {
        "padding_side": "left",
        "verbose": False,
    }
    if "mistral" in model_name.lower():
        kwargs["fix_mistral_regex"] = True
    return kwargs


def main(args):
    accelerator = _build_accelerator()
    use_multi_process = accelerator is not None and accelerator.num_processes > 1
    dataset_name = "ambigqa" if args.dataset_name == "ambig_qa" else args.dataset_name
    with open(args.clarification_path,'r',encoding='utf-8') as f:
        all_logs = json.load(f)
    if args.generate_config == "eup":
        # Match EUP generation prompt style: user-only prompt without baseline few-shot block.
        system_prompt = ""
        fewshot_prompt = ""
    else:
        system_prompt = load_system_prompt(dataset_name)
        fewshot_prompt = load_fewshot_prompt(dataset_name)
    best_n = SAMPLE_N

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_load_path = _resolve_pretrained_path(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, **_tokenizer_load_kwargs(args.model_name))
    model = _load_generation_model(model_load_path, dtype, accelerator)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    device = accelerator.device if accelerator is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    output_path = args.output_path
    save_dir = os.path.dirname(output_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if accelerator is None or accelerator.is_main_process:
        print("save logs to ", output_path)
        if model_load_path != args.model_name:
            print(f"[forward] using cached model snapshot: {model_load_path}")
    if use_multi_process and accelerator.is_main_process:
        print(
            f"[forward] Accelerator multi-process enabled: world_size={int(accelerator.num_processes)}. "
            "Each process loads one model replica and evaluates a disjoint shard of the input."
        )

    local_indices = _split_indices_for_process(len(all_logs), accelerator)
    prog_bar = auto_tqdm(
        local_indices,
        disable=accelerator is not None and not accelerator.is_local_main_process,
    )

    all_res = []
    for q_idx in prog_bar:
        log_dict = all_logs[q_idx]
        copy_log_dict = copy.deepcopy(log_dict)
        copy_log_dict[SOURCE_INDEX_KEY] = int(log_dict.get(SOURCE_INDEX_KEY, q_idx))
        if use_multi_process:
            copy_log_dict[ORDER_INDEX_KEY] = q_idx

        clarified_inputs = format_query(
            log_dict,
            dataset_name,
            prompt=fewshot_prompt,
            generate_config=args.generate_config,
        )
        copy_log_dict['clarified_all_ans'] = []
        for clarify_idx in range(len(clarified_inputs)):
            curr_input = clarified_inputs[clarify_idx]
            messages = format_message(system_prompt, curr_input)
            prompt_text = build_prompt(messages, tokenizer)
            enc = tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            prompt_len = int(attn.sum(dim=1)[0])

            do_sample = True
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=512,
                    do_sample=do_sample,
                    temperature=0.5,
                    top_p=1.0,
                    num_return_sequences=best_n,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen = gen[:, prompt_len:]
            model_outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)
            processed = [post_process(x, dataset_name) for x in model_outputs]

            copy_log_dict['clarified_all_ans'].append(processed)

        all_res.append(copy_log_dict)

    merged_results = _gather_results_to_main(all_res, accelerator, output_path)

    if accelerator is not None and not accelerator.is_main_process:
        return

    final_results = _finalize_order(merged_results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent = 4)

if __name__ == '__main__':
    main(args)
