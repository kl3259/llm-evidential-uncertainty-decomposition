#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_ICE.sh [options]

Core options:
  --dataset_name, --datset_name  Dataset name for the ICE pipeline.
  --eval_split                   Dataset split to run: train, validation, or test.
  --n_samples                    Total prepared dataset size used in split file naming.
  --test_frac                    Test split size or fraction used by the prepared dataset.
  --valid_frac                   Validation split size or fraction used by the prepared dataset.
  --model_name                   HF model name for forward generation.

Optional execution options:
  --num_shards                   Number of shards for step 1 and step 3. Default: 8.
                                 Set to 1 for serial single-worker execution.
  --num_processes                Number of accelerate processes for step 2. Default: 4
  --sample_n                     Number of sampled clarification completions. Default: 2
  --n_fewshot                    Few-shot examples for clarification generation. Default: 5
  --judge_type                   Main-project artifact judge suffix used by filtered datasets. Default: gpt5-mini
  --generate_config              Forward generation config. Default: eup
  --conda_env                    Conda env to activate. Default: semantic_uncertainty
  --cuda_visible_devices         CUDA_VISIBLE_DEVICES to export. Default: 4,5,6,7
  --fix_val_test_splits          Passed to baseline split generation/loading. Default: True
  --ignore_failed_cases          Passed to evaluate_uq_qa.py. Default: False
  --wandb_log                    Passed to compute_metrics_baseline.py. Default: True
  -h, --help                     Show this message

Example:
  bash run_ICE.sh \
    --n_samples=5000 \
    --test_frac=500 \
    --valid_frac=500 \
    --dataset_name=alcuna_trivia_qa \
    --eval_split=test \
    --model_name=google/gemma-3-12b-it \
    --cuda_visible_devices=4,5,6,7
EOF
}

ICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ICE_ROOT}/../.." && pwd)"
BASELINE_ROOT="${ICE_ROOT}"

dataset_name="alcuna_trivia_qa"
eval_split="test"
n_samples="10500"
test_frac="1000"
valid_frac="500"
model_name="google/gemma-3-12b-it"

num_shards="8"
num_processes="4"
sample_n="2"
n_fewshot="5"
judge_type="gpt5-mini"
generate_config="eup"
conda_env="semantic_uncertainty"
cuda_visible_devices="4,5,6,7"
fix_val_test_splits="True"
ignore_failed_cases="False"
wandb_log="True"

sanitize_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//\//-}"
  value="${value// /-}"
  printf '%s\n' "${value}"
}

normalize_split() {
  local split_name
  split_name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${split_name}" in
    val|valid|dev)
      printf '%s\n' "validation"
      ;;
    train|validation|test)
      printf '%s\n' "${split_name}"
      ;;
    *)
      echo "Unsupported --eval_split: ${split_name}" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_name|--datset_name)
      dataset_name="${2:-}"
      shift 2
      ;;
    --dataset_name=*|--datset_name=*)
      dataset_name="${1#*=}"
      shift 1
      ;;
    --eval_split)
      eval_split="${2:-}"
      shift 2
      ;;
    --eval_split=*)
      eval_split="${1#*=}"
      shift 1
      ;;
    --n_samples)
      n_samples="${2:-}"
      shift 2
      ;;
    --n_samples=*)
      n_samples="${1#*=}"
      shift 1
      ;;
    --test_frac)
      test_frac="${2:-}"
      shift 2
      ;;
    --test_frac=*)
      test_frac="${1#*=}"
      shift 1
      ;;
    --valid_frac)
      valid_frac="${2:-}"
      shift 2
      ;;
    --valid_frac=*)
      valid_frac="${1#*=}"
      shift 1
      ;;
    --model_name)
      model_name="${2:-}"
      shift 2
      ;;
    --model_name=*)
      model_name="${1#*=}"
      shift 1
      ;;
    --num_shards)
      num_shards="${2:-}"
      shift 2
      ;;
    --num_shards=*)
      num_shards="${1#*=}"
      shift 1
      ;;
    --num_processes)
      num_processes="${2:-}"
      shift 2
      ;;
    --num_processes=*)
      num_processes="${1#*=}"
      shift 1
      ;;
    --sample_n)
      sample_n="${2:-}"
      shift 2
      ;;
    --sample_n=*)
      sample_n="${1#*=}"
      shift 1
      ;;
    --n_fewshot)
      n_fewshot="${2:-}"
      shift 2
      ;;
    --n_fewshot=*)
      n_fewshot="${1#*=}"
      shift 1
      ;;
    --judge_type)
      judge_type="${2:-}"
      shift 2
      ;;
    --judge_type=*)
      judge_type="${1#*=}"
      shift 1
      ;;
    --generate_config)
      generate_config="${2:-}"
      shift 2
      ;;
    --generate_config=*)
      generate_config="${1#*=}"
      shift 1
      ;;
    --conda_env)
      conda_env="${2:-}"
      shift 2
      ;;
    --conda_env=*)
      conda_env="${1#*=}"
      shift 1
      ;;
    --cuda_visible_devices)
      cuda_visible_devices="${2:-}"
      shift 2
      ;;
    --cuda_visible_devices=*)
      cuda_visible_devices="${1#*=}"
      shift 1
      ;;
    --fix_val_test_splits)
      fix_val_test_splits="${2:-}"
      shift 2
      ;;
    --fix_val_test_splits=*)
      fix_val_test_splits="${1#*=}"
      shift 1
      ;;
    --ignore_failed_cases)
      ignore_failed_cases="${2:-}"
      shift 2
      ;;
    --ignore_failed_cases=*)
      ignore_failed_cases="${1#*=}"
      shift 1
      ;;
    --wandb_log)
      wandb_log="${2:-}"
      shift 2
      ;;
    --wandb_log=*)
      wandb_log="${1#*=}"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${BASELINE_ROOT}" ]]; then
  echo "Baseline repo not found: ${BASELINE_ROOT}" >&2
  exit 2
fi

eval_split="$(normalize_split "${eval_split}")"

if ! [[ "${num_shards}" =~ ^[0-9]+$ ]] || (( num_shards < 1 )); then
  echo "--num_shards must be an integer >= 1." >&2
  exit 2
fi
if ! [[ "${num_processes}" =~ ^[0-9]+$ ]] || (( num_processes < 1 )); then
  echo "--num_processes must be an integer >= 1." >&2
  exit 2
fi

run_tag="${dataset_name}_${eval_split}_${n_samples}_v$(sanitize_tag "${valid_frac}")_t$(sanitize_tag "${test_frac}")"
model_slug="$(sanitize_tag "${model_name//:/-}")"

clarification_path="logs/clarification/${run_tag}.json"
# Forward outputs depend on the HF backbone, so keep them model-scoped.
forward_path="logs/forward/${model_slug}_${run_tag}_forward.json"
extract_path="logs/forward/${model_slug}_${run_tag}_forward_ext.json"
uq_path="logs/uq_eval/${model_slug}_${run_tag}.json"
clarification_skip_path="logs/clarification/${run_tag}_content_filter_skipped.json"
extract_skip_path="logs/forward/${model_slug}_${run_tag}_forward_ext_content_filter_skipped.json"
content_filter_skip_path="logs/uq_eval/${model_slug}_${run_tag}_content_filter_skipped.json"
clarification_run_dir="logs/shard_runs/generate_clarification/${run_tag}"
extract_run_dir="logs/shard_runs/answer_extraction/${model_slug}_${run_tag}"

echo "ICE pipeline config:"
echo "  dataset_name: ${dataset_name}"
echo "  eval_split: ${eval_split}"
echo "  n_samples: ${n_samples}"
echo "  test_frac: ${test_frac}"
echo "  valid_frac: ${valid_frac}"
echo "  model_name: ${model_name}"
echo "  num_shards: ${num_shards}"
echo "  num_processes: ${num_processes}"
echo "  cuda_visible_devices: ${cuda_visible_devices}"
echo "  judge_type: ${judge_type}"
echo "  fix_val_test_splits: ${fix_val_test_splits}"
echo "  output tag: ${run_tag}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH. Activate ${conda_env} manually before running." >&2
  exit 2
fi

conda_base="$(conda info --base)"
# Use the activated env's executables directly so the pipeline can't fall back
# to a different Python/user-site installation.
. "${conda_base}/etc/profile.d/conda.sh"
conda deactivate >/dev/null 2>&1 || true
conda activate "${conda_env}"
hash -r

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Failed to activate conda env: ${conda_env}" >&2
  exit 2
fi

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
ACCELERATE_BIN="${CONDA_PREFIX}/bin/accelerate"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found in activated env: ${PYTHON_BIN}" >&2
  exit 2
fi
if (( num_processes > 1 )) && [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate not found in activated env: ${ACCELERATE_BIN}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1

export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"

cd "${BASELINE_ROOT}"
mkdir -p logs/clarification logs/forward logs/uq_eval

if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="${BASELINE_ROOT}/logs/tmp"
fi
mkdir -p "${TMPDIR}"
echo "[ICE] TMPDIR=${TMPDIR}"
echo "[ICE] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
echo "[ICE] python=${PYTHON_BIN}"
if (( num_processes > 1 )); then
  echo "[ICE] accelerate=${ACCELERATE_BIN}"
fi
"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import jinja2
except Exception as exc:
    print(f"[ICE] jinja2 import failed: {exc}")
else:
    print(f"[ICE] jinja2={jinja2.__version__} ({jinja2.__file__})")
print(f"[ICE] sys.executable={sys.executable}")
PY

combine_content_filter_skips() {
  local output_path="$1"
  shift
  "${PYTHON_BIN}" - "${output_path}" "$@" <<'PY'
import json
import os
import sys

output_path = sys.argv[1]
input_paths = [p for p in sys.argv[2:] if p]
merged = {}
for path in input_paths:
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Skip sidecar is not a list: {path}")
    for record in records:
        source_idx = record.get("_source_idx")
        if source_idx is None:
            raise ValueError(f"Missing `_source_idx` in skip sidecar: {path}")
        source_idx = int(source_idx)
        current = dict(record)
        stage = current.get("stage")
        if stage:
            current["stages"] = [stage]
        existing = merged.get(source_idx)
        if existing is None:
            merged[source_idx] = current
            continue
        stages = existing.get("stages", [])
        for stage_name in current.get("stages", []):
            if stage_name not in stages:
                stages.append(stage_name)
        if stages:
            existing["stages"] = stages

ordered = [merged[idx] for idx in sorted(merged)]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ordered, f, indent=4)
print(f"[content_filter_skip] wrote {len(ordered)} records -> {output_path}")
PY
}

run_generate_clarification_stage() {
  bash scripts/run_ice_sharded_stage.sh \
    --stage=generate_clarification \
    --num_shards="${num_shards}" \
    --run_dir="${clarification_run_dir}" \
    --output_path="${clarification_path}" \
    -- \
    --dataset_name="${dataset_name}" \
    --eval_split="${eval_split}" \
    --sample \
    --sample_n="${sample_n}" \
    --n_fewshot="${n_fewshot}" \
    --n_samples="${n_samples}" \
    --test_frac="${test_frac}" \
    --valid_frac="${valid_frac}" \
    --fix_val_test_splits="${fix_val_test_splits}"
}

run_answer_extraction_stage() {
  bash scripts/run_ice_sharded_stage.sh \
    --stage=answer_extraction \
    --num_shards="${num_shards}" \
    --run_dir="${extract_run_dir}" \
    --output_path="${extract_path}" \
    -- \
    --log_path="${forward_path}" \
    --prompt_path=lib_prompt/common/answer_extraction.txt \
    --answer_key=clarified_all_ans \
    --azure_config=config/azure_openai.json \
    --parse_retry_max=1 \
    --rewrite_retry_max=1
}

# 0) Prepare dataset splits used by downstream stages.
"${PYTHON_BIN}" tools/prepare_data.py \
  --dataset_name="${dataset_name}" \
  --n_samples="${n_samples}" \
  --test_frac="${test_frac}" \
  --valid_frac="${valid_frac}" \
  --n_fewshot="${n_fewshot}" \
  --model_name="${model_name}" \
  --judge_type="${judge_type}" \
  --fix_val_test_splits="${fix_val_test_splits}"

# 1) Generate clarifications.
run_generate_clarification_stage
combine_content_filter_skips "${content_filter_skip_path}" "${clarification_skip_path}"

# 2) Collect answers conditioned on clarifications.
if (( num_processes > 1 )); then
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes="${num_processes}" \
    forward.py \
    --dataset_name="${dataset_name}" \
    --clarification_path="${clarification_path}" \
    --output_path="${forward_path}" \
    --model_name="${model_name}" \
    --generate_config="${generate_config}"
else
  "${PYTHON_BIN}" forward.py \
    --dataset_name="${dataset_name}" \
    --clarification_path="${clarification_path}" \
    --output_path="${forward_path}" \
    --model_name="${model_name}" \
    --generate_config="${generate_config}"
fi

# 3) Answer canonicalization / semantic grouping.
run_answer_extraction_stage
combine_content_filter_skips "${content_filter_skip_path}" "${clarification_skip_path}" "${extract_skip_path}"

# 4) Uncertainty quantification.
"${PYTHON_BIN}" evaluate_uq_qa.py \
  --log_path="${extract_path}" \
  --output_path="${uq_path}" \
  --answer_key=ext_clarified_all_ans \
  --ignore_failed_cases="${ignore_failed_cases}"

# 5) Ambiguity detection evaluation.
"${PYTHON_BIN}" tools/compute_metrics_baseline.py \
  --log_path="${uq_path}" \
  --dataset_name="${dataset_name}" \
  --baseline_method=ICE \
  --model_name="${model_name}" \
  --content_filter_skip_path="${content_filter_skip_path}" \
  --wandb_log="${wandb_log}"
