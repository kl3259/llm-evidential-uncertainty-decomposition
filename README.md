# LLM Evidential Uncertainty Decomposition

This repository contains the EPN pipeline for collecting question-answering
features, collecting sampled semantic-cluster artifacts, training EPN, training
EPN-IT, and running the EoE and ICE baselines.

## File Map

| File | Role |
| --- | --- |
| `config.py` | Shared generation, sampling, and ITP defaults. |
| `openai_api.json` | Placeholder Azure/OpenAI config for GPT judging or embedding-backed utilities. |
| `feature_collector.py` | Builds dataset splits, runs base generations, saves hidden states, sequence scores, predictions, and labels. |
| `semantic_cluster_collector.py` | Samples multiple responses per prompt, assigns semantic IDs, computes semantic entropy, and saves response-level sequence and hidden-state artifacts. |
| `train_epn.py` | Trains and evaluates the EPN density/evidence model from collector artifacts. |
| `train_epn_itp.py` | Trains or reloads EPN and evaluates with explicit iterative test-time prompting. |
| `epn_helper.py` | Artifact loading contracts, split helpers, EUD evaluation helpers, baseline metric helpers, and ITP-side utilities. |
| `helper_functions.py` | JSONL, NPZ, sequence-score, and semantic-artifact save/load helpers. |
| `uncertainty.py` | Uncertainty and calibration metric utilities. |
| `epn_loss.py` | EPN loss functions. |
| `eud_helper.py` | Auxiliary probe helpers. |
| `baseline/eoe/run_eoe.py` | Standalone EoE baseline runner. |
| `baseline/ICE/run_ICE.sh` | ICE baseline entrypoint. |

## Configuration

The main API config is `openai_api.json`; fill in your own API key and endpoint
to use API-backed judging or embedding paths.

The ICE baseline has its own config at
`baseline/ICE/config/azure_openai.json`:

```json
{
  "api_key": "Fill your key here.",
  "azure_endpoint": "Fill your endpoint here.",
  "api_version": "Fill your API version here.",
  "deployment_name": "Fill your deployment name here."
}
```

Do not commit filled API configs, W&B run directories, model caches, generated
hidden states, or run outputs.

## Data And Artifacts

The collectors and training scripts use the same artifact root:

```text
input_hidden_states/{model_id}_{dataset_name}_{n_samples}_{judge_type}/
  train/
  valid/
  test/
```

`model_id` is the last path component of `--model_name`. For example,
`meta-llama/Llama-3.2-3B-Instruct` writes under a directory beginning with
`Llama-3.2-3B-Instruct_...`.

`feature_collector.py` writes base artifacts such as:

```text
X.npz
y.npy
acc.npy
f1.npy
preds.jsonl
preds_sequence_scores.npz
questions.jsonl
references.jsonl
```

`semantic_cluster_collector.py` adds semantic-sampling artifacts such as:

```text
se.npy
semantic_ids_{n_resp}.npy
se_resp_res.jsonl
se_resp_res_sequence_scores_{n_resp}.npz
response_slt_lastk_{n_resp}.npz
```

Use the same `--model_name`, `--dataset_name`, `--n_samples`,
`--judge_type`, `--valid_frac`, `--test_frac`, and `--fix_val_test_splits`
across collection and training so the training scripts load the intended
artifacts.

## Example Setup

Run the main pipeline commands from the repository root.

The examples below use fixed validation and test counts so all scripts share
the same split membership.

```bash
MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET="alcuna_trivia_qa"
JUDGE="rouge"
N_SAMPLES=5000
VALID_SIZE=500
TEST_SIZE=500
N_RESP=10
```

## 1. Collect Input Feature Artifacts

Run `feature_collector.py` first. It prepares splits, runs the base model, and
saves hidden states, predictions, answer scores, labels, and sequence scores.

```bash
python feature_collector.py \
  --model_name "${MODEL}" \
  --dataset_name "${DATASET}" \
  --judge_type "${JUDGE}" \
  --n_samples "${N_SAMPLES}" \
  --valid_frac "${VALID_SIZE}" \
  --test_frac "${TEST_SIZE}" \
  --fix_val_test_splits True
```

## 2. Collect Sampled Semantic Artifacts

Run `semantic_cluster_collector.py` after feature collection. It reuses the
same split contract and adds sampled responses, semantic IDs, semantic entropy,
sampled-response sequence scores, and response hidden-state summaries.

```bash
python semantic_cluster_collector.py \
  --model_name "${MODEL}" \
  --dataset_name "${DATASET}" \
  --judge_type "${JUDGE}" \
  --n_samples "${N_SAMPLES}" \
  --valid_frac "${VALID_SIZE}" \
  --test_frac "${TEST_SIZE}" \
  --fix_val_test_splits True \
  --n_resp "${N_RESP}"
```

## 3. Train EPN

Run `train_epn.py` after both collector steps complete.

```bash
WANDB_MODE=disabled python train_epn.py \
  --model_name "${MODEL}" \
  --dataset_name "${DATASET}" \
  --judge_type "${JUDGE}" \
  --n_samples "${N_SAMPLES}" \
  --valid_frac "${VALID_SIZE}" \
  --test_frac "${TEST_SIZE}" \
  --fix_val_test_splits True \
  --n_resp "${N_RESP}" \
  --use_pseudo_OOD False \
  --layer_range last_5 \
  --aux_probe hallucination \
  --aux_probe_feature tbg \
  --density_probe_feature tbg \
  --n_epochs 100 \
  --lr 1e-5
```

For pseudo-OOD training, collect artifacts for the pseudo-OOD dataset first and
then set `--use_pseudo_OOD True`. Use `--pseudo_OOD_dataset_name temporalood`
with `alcuna_trivia_qa`, and `--pseudo_OOD_dataset_name alcuna` with
`temporalood_trivia_qa`.

EPN outputs are written under `epn_runs_ver3/` by default.

## 4. Train Or Evaluate EPN-IT

`train_epn_itp.py` uses the same collector artifacts and can either train a new
EPN density probe or reload an existing EPN checkpoint before iterative
test-time prompting.

```bash
WANDB_MODE=disabled python train_epn_itp.py \
  --model_name "${MODEL}" \
  --dataset_name "${DATASET}" \
  --judge_type "${JUDGE}" \
  --n_samples "${N_SAMPLES}" \
  --valid_frac "${VALID_SIZE}" \
  --test_frac "${TEST_SIZE}" \
  --fix_val_test_splits True \
  --n_resp "${N_RESP}" \
  --use_pseudo_OOD False \
  --layer_range last_5 \
  --n_eval_max_resp 5 \
  --itp_batch_size 4 \
  --itp_semantic_clustering False
```

To reuse a trained EPN checkpoint:

```bash
WANDB_MODE=disabled python train_epn_itp.py \
  --model_name "${MODEL}" \
  --dataset_name "${DATASET}" \
  --judge_type "${JUDGE}" \
  --n_samples "${N_SAMPLES}" \
  --valid_frac "${VALID_SIZE}" \
  --test_frac "${TEST_SIZE}" \
  --fix_val_test_splits True \
  --n_resp "${N_RESP}" \
  --use_pseudo_OOD False \
  --load_pretrained_EPN True \
  --load_path epn_runs_ver3/<run_dir_or_density_probe.pt> \
  --test True
```

## 5. Run The EoE Baseline

The EoE baseline is in `baseline/eoe/run_eoe.py`. It uses the main project
dataset splitter and writes `eoe_uncertainty.npz` into the matching
`input_hidden_states/.../{eval_split}/` directory.

```bash
python baseline/eoe/run_eoe.py \
  --model_name "${MODEL}" \
  --dataset_name "${DATASET}" \
  --judge_type "${JUDGE}" \
  --n_samples "${N_SAMPLES}" \
  --valid_frac "${VALID_SIZE}" \
  --test_frac "${TEST_SIZE}" \
  --eval_split test \
  --fix_val_test_splits True \
  --wandb_log False
```

## 6. Run The ICE Baseline

The ICE baseline is isolated under `baseline/ICE/`. Its entrypoint prepares the
dataset, generates clarifications, runs model forwarding, extracts answers,
evaluates uncertainty, and computes baseline metrics.

Fill `baseline/ICE/config/azure_openai.json` before running the answer
extraction stage.

```bash
cd baseline/ICE

bash run_ICE.sh \
  --model_name "${MODEL}" \
  --dataset_name "${DATASET}" \
  --eval_split test \
  --n_samples "${N_SAMPLES}" \
  --valid_frac "${VALID_SIZE}" \
  --test_frac "${TEST_SIZE}" \
  --judge_type "${JUDGE}" \
  --num_shards 1 \
  --num_processes 1 \
  --wandb_log False
```

ICE writes runtime outputs under `baseline/ICE/logs/`, which is ignored by the
ICE `.gitignore`.
