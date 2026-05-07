#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_ice_sharded_stage.sh --stage <generate_clarification|answer_extraction> \
    --output_path <final_merged_output.json> [options] -- <stage_passthrough_args>

Options:
  --stage            Stage to run. Choices: generate_clarification, answer_extraction
  --num_shards       Number of shards to run in parallel (default: 4)
  --output_path      Final merged output path (required)
  --run_dir          Run directory for shard outputs/logs/manifest
  --python_bin       Python executable to use (default: python)
  --workdir          Working directory where stage script runs (default: repo root)
  --max_shard_restarts
                     Number of launcher-level retries per shard after the initial
                     attempt fails (default: 3)
  --shard_restart_base_sleep_sec
                     Base sleep in seconds before relaunching a failed shard.
                     The launcher sleeps base * attempt_number (default: 15)
  --dry_run          Print commands only; do not execute shard jobs
  -h, --help         Show this message

Pass-through args:
  Everything after `--` is passed to the stage script.
  Do not pass: --output_path, --num_shards, --shard_id, --merge_shards, --shard_input_glob, --skip_output_path, --skip_shard_input_glob, --source_index_key, --merge_index_key
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

stage=""
num_shards=4
output_path=""
run_dir=""
python_bin="python"
workdir="${REPO_ROOT}"
max_shard_restarts=3
shard_restart_base_sleep_sec=15
dry_run="false"
passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      stage="${2:-}"
      shift 2
      ;;
    --stage=*)
      stage="${1#*=}"
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
    --output_path)
      output_path="${2:-}"
      shift 2
      ;;
    --output_path=*)
      output_path="${1#*=}"
      shift 1
      ;;
    --run_dir)
      run_dir="${2:-}"
      shift 2
      ;;
    --run_dir=*)
      run_dir="${1#*=}"
      shift 1
      ;;
    --python_bin)
      python_bin="${2:-}"
      shift 2
      ;;
    --python_bin=*)
      python_bin="${1#*=}"
      shift 1
      ;;
    --workdir)
      workdir="${2:-}"
      shift 2
      ;;
    --workdir=*)
      workdir="${1#*=}"
      shift 1
      ;;
    --max_shard_restarts)
      max_shard_restarts="${2:-}"
      shift 2
      ;;
    --max_shard_restarts=*)
      max_shard_restarts="${1#*=}"
      shift 1
      ;;
    --shard_restart_base_sleep_sec)
      shard_restart_base_sleep_sec="${2:-}"
      shift 2
      ;;
    --shard_restart_base_sleep_sec=*)
      shard_restart_base_sleep_sec="${1#*=}"
      shift 1
      ;;
    --dry_run)
      dry_run="true"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      passthrough=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${stage}" ]]; then
  echo "--stage is required." >&2
  exit 2
fi
if [[ "${stage}" != "generate_clarification" && "${stage}" != "answer_extraction" ]]; then
  echo "Invalid --stage: ${stage}. Must be generate_clarification or answer_extraction." >&2
  exit 2
fi
if [[ -z "${output_path}" ]]; then
  echo "--output_path is required." >&2
  exit 2
fi
if ! [[ "${num_shards}" =~ ^[0-9]+$ ]]; then
  echo "--num_shards must be an integer." >&2
  exit 2
fi
if (( num_shards < 1 )); then
  echo "--num_shards must be >= 1." >&2
  exit 2
fi
if ! [[ "${max_shard_restarts}" =~ ^[0-9]+$ ]]; then
  echo "--max_shard_restarts must be an integer." >&2
  exit 2
fi
if (( max_shard_restarts < 0 )); then
  echo "--max_shard_restarts must be >= 0." >&2
  exit 2
fi
if ! [[ "${shard_restart_base_sleep_sec}" =~ ^[0-9]+$ ]]; then
  echo "--shard_restart_base_sleep_sec must be an integer." >&2
  exit 2
fi
if (( shard_restart_base_sleep_sec < 1 )); then
  echo "--shard_restart_base_sleep_sec must be >= 1." >&2
  exit 2
fi

if [[ ! -d "${workdir}" ]]; then
  echo "Invalid --workdir (not a directory): ${workdir}" >&2
  exit 2
fi

case "${stage}" in
  generate_clarification)
    stage_script="${workdir}/tools/generate_clarification.py"
    ;;
  answer_extraction)
    stage_script="${workdir}/tools/answer_extraction.py"
    ;;
esac
if [[ ! -f "${stage_script}" ]]; then
  echo "Stage script not found: ${stage_script}" >&2
  exit 2
fi

if [[ -z "${run_dir}" ]]; then
  run_id="$(date +%Y%m%d_%H%M%S)"
  run_dir="${workdir}/logs/shard_runs/${stage}/${run_id}"
fi

for arg in "${passthrough[@]}"; do
  case "${arg}" in
    --output_path|--output_path=*|--num_shards|--num_shards=*|--shard_id|--shard_id=*|--merge_shards|--merge_shards=*|--shard_input_glob|--shard_input_glob=*|--skip_output_path|--skip_output_path=*|--skip_shard_input_glob|--skip_shard_input_glob=*|--source_index_key|--source_index_key=*|--merge_index_key|--merge_index_key=*)
      echo "Pass-through args must not include shard/merge managed flags: ${arg}" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "${output_path}")"
mkdir -p "${run_dir}/shards" "${run_dir}/skip_shards" "${run_dir}/logs" "${run_dir}/status"
if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="${run_dir}/tmp"
fi
mkdir -p "${TMPDIR}"

fingerprint_path="${run_dir}/invocation_fingerprint.json"
fingerprint_tmp="${run_dir}/.fingerprint.current.json"
"${python_bin}" - "${fingerprint_tmp}" "${stage}" "${num_shards}" "${output_path}" "${workdir}" "${python_bin}" "${stage_script}" "${passthrough[@]}" <<'PY'
import json
import os
import sys

fingerprint_path = sys.argv[1]
stage = sys.argv[2]
num_shards = int(sys.argv[3])
output_path = sys.argv[4]
workdir = sys.argv[5]
python_bin = sys.argv[6]
stage_script = sys.argv[7]
passthrough = sys.argv[8:]

payload = {
    "stage": stage,
    "num_shards": num_shards,
    "output_path": os.path.abspath(output_path),
    "workdir": os.path.abspath(workdir),
    "python_bin": python_bin,
    "stage_script": os.path.abspath(stage_script),
    "passthrough_args": passthrough,
}
with open(fingerprint_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY
if [[ -f "${fingerprint_path}" ]]; then
  if ! cmp -s "${fingerprint_path}" "${fingerprint_tmp}"; then
    echo "Existing run_dir fingerprint does not match current invocation: ${run_dir}" >&2
    echo "Fingerprint file: ${fingerprint_path}" >&2
    rm -f "${fingerprint_tmp}"
    exit 2
  fi
  rm -f "${fingerprint_tmp}"
else
  mv "${fingerprint_tmp}" "${fingerprint_path}"
fi

if [[ "${output_path}" == *.json ]]; then
  skip_output_path="${output_path%.json}_content_filter_skipped.json"
else
  skip_output_path="${output_path}_content_filter_skipped.json"
fi

commands_tsv="${run_dir}/shard_commands.tsv"
status_tsv="${run_dir}/shard_status.tsv"
manifest_path="${run_dir}/manifest.json"
: > "${commands_tsv}"
: > "${status_tsv}"

start_epoch="$(date +%s)"
start_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

declare -a pids=()
declare -a shard_ids=()

echo "Stage: ${stage}"
echo "Workdir: ${workdir}"
echo "Run dir: ${run_dir}"
echo "Num shards: ${num_shards}"
echo "Max shard restarts: ${max_shard_restarts}"
echo "Shard restart base sleep (sec): ${shard_restart_base_sleep_sec}"
echo "Dry run: ${dry_run}"

for (( sid=0; sid<num_shards; sid++ )); do
  shard_output="${run_dir}/shards/${stage}.shard${sid}.json"
  shard_skip_output="${run_dir}/skip_shards/${stage}.shard${sid}.skipped.json"
  shard_log="${run_dir}/logs/shard${sid}.log"
  shard_status="${run_dir}/status/shard${sid}.status"
  cmd=(
    "${python_bin}" "${stage_script}" "${passthrough[@]}"
    --num_shards "${num_shards}"
    --shard_id "${sid}"
    --output_path "${shard_output}"
    --skip_output_path "${shard_skip_output}"
    --source_index_key "_source_idx"
    --merge_index_key "_merge_idx"
  )
  cmd_str="$(printf '%q ' "${cmd[@]}")"
  cmd_str="${cmd_str% }"

  printf '%s\t%s\t%s\t%s\n' "${sid}" "${shard_output}" "${shard_log}" "${cmd_str}" >> "${commands_tsv}"
  echo "[launch] shard ${sid}: ${cmd_str}"

  if [[ "${dry_run}" == "true" ]]; then
    continue
  fi

  shard_cmd=("${cmd[@]}")
  shard_cmd_str="${cmd_str}"
  (
    attempt=1
    max_attempts=$((max_shard_restarts + 1))
    while true; do
      {
        printf '===== shard %s attempt %s/%s [%s] =====\n' \
          "${sid}" "${attempt}" "${max_attempts}" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        printf 'command: %s\n' "${shard_cmd_str}"
      } >> "${shard_log}"

      if (
        cd "${workdir}"
        "${shard_cmd[@]}"
      ) >> "${shard_log}" 2>&1; then
        rc=0
      else
        rc=$?
      fi

      if (( rc == 0 )); then
        printf '%s\t%s\n' "${rc}" "${attempt}" > "${shard_status}"
        exit 0
      fi

      if (( attempt >= max_attempts )); then
        printf '%s\t%s\n' "${rc}" "${attempt}" > "${shard_status}"
        exit "${rc}"
      fi

      sleep_sec=$((shard_restart_base_sleep_sec * attempt))
      printf '[restart] shard %s failed with exit code %s on attempt %s/%s; retrying in %ss\n' \
        "${sid}" "${rc}" "${attempt}" "${max_attempts}" "${sleep_sec}" >> "${shard_log}"
      attempt=$((attempt + 1))
      sleep "${sleep_sec}"
    done
  ) &
  pids+=("$!")
  shard_ids+=("${sid}")
done

merge_glob="${run_dir}/shards/${stage}.shard*.json"
skip_merge_glob="${run_dir}/skip_shards/${stage}.shard*.skipped.json"
merge_cmd=(
  "${python_bin}" "${stage_script}"
  --merge_shards True
  --shard_input_glob "${merge_glob}"
  --skip_shard_input_glob "${skip_merge_glob}"
  --output_path "${output_path}"
  --skip_output_path "${skip_output_path}"
  --source_index_key "_source_idx"
  --merge_index_key "_merge_idx"
)
merge_cmd_str="$(printf '%q ' "${merge_cmd[@]}")"
merge_cmd_str="${merge_cmd_str% }"
echo "[merge] ${merge_cmd_str}"

write_manifest() {
  local status="$1"
  local merge_status="$2"
  local final_output="$3"
  local end_epoch end_iso
  end_epoch="$(date +%s)"
  end_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  "${python_bin}" - "${manifest_path}" "${stage}" "${num_shards}" "${start_iso}" "${end_iso}" "${start_epoch}" "${end_epoch}" "${workdir}" "${output_path}" "${status}" "${merge_status}" "${final_output}" "${commands_tsv}" "${status_tsv}" "${merge_cmd_str}" <<'PY'
import json
import sys

(
    manifest_path,
    stage,
    num_shards,
    start_iso,
    end_iso,
    start_epoch,
    end_epoch,
    workdir,
    output_path,
    status,
    merge_status,
    final_output,
    commands_tsv,
    status_tsv,
    merge_cmd,
) = sys.argv[1:]

num_shards = int(num_shards)
start_epoch = int(start_epoch)
end_epoch = int(end_epoch)

commands = {}
with open(commands_tsv, "r") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        shard_id, shard_output, shard_log, cmd = line.split("\t", 3)
        commands[int(shard_id)] = {
            "shard_id": int(shard_id),
            "shard_output": shard_output,
            "shard_log": shard_log,
            "command": cmd,
        }

statuses = {}
with open(status_tsv, "r") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            shard_id, exit_code = parts
            attempts = None
        elif len(parts) == 3:
            shard_id, exit_code, attempts = parts
        else:
            raise ValueError(f"Unexpected shard status format: {line}")
        statuses[int(shard_id)] = {
            "exit_code": int(exit_code),
            "attempts": int(attempts) if attempts is not None else None,
        }

shards = []
for shard_id in sorted(commands.keys()):
    item = dict(commands[shard_id])
    shard_status = statuses.get(shard_id, {})
    item["exit_code"] = shard_status.get("exit_code")
    item["attempts"] = shard_status.get("attempts")
    shards.append(item)

manifest = {
    "stage": stage,
    "num_shards": num_shards,
    "start_time_utc": start_iso,
    "end_time_utc": end_iso,
    "duration_sec": max(0, end_epoch - start_epoch),
    "workdir": workdir,
    "status": status,
    "merge_status": merge_status,
    "output_path": output_path,
    "final_merged_output": final_output if final_output else None,
    "merge_command": merge_cmd,
    "shards": shards,
}

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
PY
}

if [[ "${dry_run}" == "true" ]]; then
  write_manifest "dry_run" "dry_run" ""
  echo "Dry run complete. Manifest: ${manifest_path}"
  echo "Tail hint: tail -f ${run_dir}/logs/shard0.log"
  exit 0
fi

declare -a failed_shards=()
for i in "${!pids[@]}"; do
  sid="${shard_ids[$i]}"
  if wait "${pids[$i]}"; then
    rc=0
  else
    rc=$?
  fi
  shard_status="${run_dir}/status/shard${sid}.status"
  attempts=0
  if [[ -f "${shard_status}" ]]; then
    IFS=$'\t' read -r rc attempts < "${shard_status}"
  fi
  printf '%s\t%s\t%s\n' "${sid}" "${rc}" "${attempts}" >> "${status_tsv}"
  if (( rc != 0 )); then
    failed_shards+=("${sid}")
  fi
done

if (( ${#failed_shards[@]} > 0 )); then
  echo "One or more shards failed. Merge is skipped." >&2
  for sid in "${failed_shards[@]}"; do
    echo "  shard ${sid} log: ${run_dir}/logs/shard${sid}.log" >&2
  done
  write_manifest "failed" "skipped" ""
  echo "Manifest: ${manifest_path}" >&2
  exit 1
fi

(
  cd "${workdir}"
  "${merge_cmd[@]}"
)
write_manifest "success" "success" "${output_path}"

end_epoch="$(date +%s)"
duration="$(( end_epoch - start_epoch ))"
echo "Completed successfully."
echo "Merged output: ${output_path}"
echo "Merged skip output: ${skip_output_path}"
echo "Runtime (sec): ${duration}"
echo "Shard logs: ${run_dir}/logs"
echo "Manifest: ${manifest_path}"
echo "Tail hint: tail -f ${run_dir}/logs/shard0.log"
