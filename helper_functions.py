import numpy as np
from typing import List, Dict, Optional, Sequence, Tuple, Union
import json, argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import StoppingCriteria
import os
from collections import Counter
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score
from uncertainty import vacuity, dissonance

class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def save_cluster_embeddings_npz(path: str, clusters_per_example: List[Dict[int, np.ndarray]]) -> None:
    """
    Save list[dict{cluster_id -> (D,) float32}] into a single NPZ file.

    NPZ contents:
      - offsets:     int64, shape [N+1]
      - cluster_ids: int64, shape [M]
      - emb:         float32, shape [M, D]
      - dim:         int64 scalar (D)
      - N:           int64 scalar (#examples)
    """
    N = len(clusters_per_example)
    # Infer dimension from the first available vector
    D = None
    for d in clusters_per_example:
        if d:
            first = next(iter(d.values()))
            D = int(np.asarray(first).shape[-1])
            break
    if D is None:
        # All empty – store a trivial file
        np.savez_compressed(path,
                            offsets=np.zeros(1, dtype=np.int64),
                            cluster_ids=np.zeros(0, dtype=np.int64),
                            emb=np.zeros((0, 0), dtype=np.float32),
                            dim=np.array(0, dtype=np.int64),
                            N=np.array(N, dtype=np.int64))
        return
    offsets = np.zeros(N + 1, dtype=np.int64)
    flat_ids = []
    flat_vecs = []

    running = 0
    for i, d in enumerate(clusters_per_example):
        cids = sorted(int(k) for k in d.keys())  # stable, deterministic order
        vecs = [np.asarray(d[cid], dtype=np.float32).reshape(-1) for cid in cids]
        if any(v.shape[0] != D for v in vecs):
            raise ValueError(f"Example {i}: inconsistent dim; expected {D}")
        flat_ids.extend(cids)
        if vecs:
            flat_vecs.append(np.vstack(vecs))
            running += len(vecs)
        offsets[i + 1] = running

    M = offsets[-1]
    if M == 0:
        emb = np.zeros((0, D), dtype=np.float32)
        cluster_ids = np.zeros((0,), dtype=np.int64)
    else:
        emb = np.vstack(flat_vecs).astype(np.float32) if flat_vecs else np.zeros((0, D), dtype=np.float32)
        cluster_ids = np.asarray(flat_ids, dtype=np.int64)

    np.savez_compressed(
        path,
        offsets=offsets,
        cluster_ids=cluster_ids,
        emb=emb,
        dim=np.array(D, dtype=np.int64),
        N=np.array(N, dtype=np.int64),
    )


def load_cluster_embeddings_npz(path: str) -> List[Dict[int, np.ndarray]]:
    """
    Load the NPZ produced by save_cluster_embeddings_npz() back into list[dict].
    """
    z = np.load(path, allow_pickle=False)
    offsets = z["offsets"].astype(np.int64)
    cluster_ids = z["cluster_ids"].astype(np.int64)
    emb = z["emb"].astype(np.float32)
    N = int(z["N"])
    D = int(z["dim"])

    out: List[Dict[int, np.ndarray]] = []
    for i in range(N):
        s, e = int(offsets[i]), int(offsets[i + 1])
        d: Dict[int, np.ndarray] = {}
        if e > s:
            cids = cluster_ids[s:e]
            vecs = emb[s:e, :D]
            for cid, v in zip(cids, vecs):
                d[int(cid)] = v
        out.append(d)
    return out


def get_sampled_sequence_scores_filename(resp_jsonl_name: str, n_resp_suffix: Union[str, int]) -> str:
    if resp_jsonl_name == "se_resp_res.jsonl":
        stem = "se_resp_res_sequence_scores"
    elif resp_jsonl_name == "se_resp.jsonl":
        stem = "se_resp_sequence_scores"
    else:
        raise ValueError(f"Unsupported sampled response JSONL name for sequence scores: {resp_jsonl_name}")
    return f"{stem}_{n_resp_suffix}.npz"


def save_sequence_scores_npz(
    path: str,
    sequence_logprob,
    sequence_prob,
    length_normalized_sequence_logprob,
) -> None:
    seq_logprob = np.asarray(sequence_logprob, dtype=np.float32)
    seq_prob = np.asarray(sequence_prob, dtype=np.float32)
    seq_logprob_norm = np.asarray(length_normalized_sequence_logprob, dtype=np.float32)

    if not (seq_logprob.shape == seq_prob.shape == seq_logprob_norm.shape):
        raise ValueError(
            "Sequence score shape mismatch: "
            f"logprob={seq_logprob.shape}, prob={seq_prob.shape}, "
            f"length_normalized={seq_logprob_norm.shape}"
        )
    if seq_logprob.ndim not in (1, 2):
        raise ValueError(
            "Sequence score arrays must be rank-1 or rank-2. "
            f"Got shape {seq_logprob.shape}."
        )

    payload = {
        "sequence_logprob": seq_logprob,
        "sequence_prob": seq_prob,
        "length_normalized_sequence_logprob": seq_logprob_norm,
        "N": np.array(seq_logprob.shape[0], dtype=np.int64),
    }
    if seq_logprob.ndim == 2:
        payload["n_responses"] = np.array(seq_logprob.shape[1], dtype=np.int64)

    np.savez_compressed(path, **payload)


def load_sequence_scores_npz(path: str) -> Dict[str, Union[np.ndarray, int]]:
    with np.load(path, allow_pickle=False) as z:
        seq_logprob = z["sequence_logprob"].astype(np.float32, copy=False)
        seq_prob = z["sequence_prob"].astype(np.float32, copy=False)
        seq_logprob_norm = z["length_normalized_sequence_logprob"].astype(np.float32, copy=False)
        stored_n = int(z["N"]) if "N" in z.files else int(seq_logprob.shape[0])
        stored_n_responses = (
            int(z["n_responses"])
            if "n_responses" in z.files
            else (int(seq_logprob.shape[1]) if seq_logprob.ndim == 2 else 0)
        )

    if not (seq_logprob.shape == seq_prob.shape == seq_logprob_norm.shape):
        raise ValueError(
            f"Sequence score NPZ shape mismatch in {path}: "
            f"logprob={seq_logprob.shape}, prob={seq_prob.shape}, "
            f"length_normalized={seq_logprob_norm.shape}"
        )
    if seq_logprob.ndim not in (1, 2):
        raise ValueError(
            f"Sequence score NPZ in {path} must be rank-1 or rank-2, got {seq_logprob.shape}."
        )
    if stored_n != int(seq_logprob.shape[0]):
        raise ValueError(
            f"Sequence score NPZ N mismatch in {path}: stored={stored_n}, actual={seq_logprob.shape[0]}"
        )
    if seq_logprob.ndim == 2 and stored_n_responses != int(seq_logprob.shape[1]):
        raise ValueError(
            "Sequence score NPZ n_responses mismatch in "
            f"{path}: stored={stored_n_responses}, actual={seq_logprob.shape[1]}"
        )

    return {
        "sequence_logprob": seq_logprob,
        "sequence_prob": seq_prob,
        "length_normalized_sequence_logprob": seq_logprob_norm,
        "N": stored_n,
        "n_responses": stored_n_responses,
    }


def _save_preds_jsonl(path: str, preds):
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(preds):
            f.write(json.dumps({"i": i, "pred": p}, ensure_ascii=False) + "\n")

def _save_questions_jsonl(path: str, preds):
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(preds):
            f.write(json.dumps({"i": i, "question": p}, ensure_ascii=False) + "\n")

def _save_refs_jsonl(path: str, preds):
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(preds):
            f.write(json.dumps({"i": i, "ref": p}, ensure_ascii=False) + "\n")

def _save_X_npz(path: str, X):
    np.savez_compressed(path,
                        tbg=X["tbg"].astype(np.float32, copy=False),
                        slt=X["slt"].astype(np.float32, copy=False))

def load_X_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float32, copy=False) for k in z.files}

def load_pred_json(path: str):
    preds = pd.read_json(path, lines=True).sort_values("i")["pred"].tolist()
    return preds

def load_question_json(path: str):
    preds = pd.read_json(path, lines=True).sort_values("i")["question"].tolist()
    return preds

def load_ref_json(path: str):
    preds = pd.read_json(path, lines=True).sort_values("i")["ref"].tolist()
    return preds

def load_resp_json(path: str):
    df = pd.read_json(path, lines=True)
    resp_per_q = df["responses"].tolist() # list[list[str]]
    return resp_per_q

def count_or_frac(s: str):
    """Parse either an absolute count (int) or a fraction (float)."""
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be int (count) or float (fraction)")



REFUSAL_ANS = [
    "I cannot help with",
    "I cannot assist with",
    "I cannot answer",
    "I cannot provide", 
    "I couldn't find",
    "I cannot verify",
    "I couldn't verify",
    "I can't recall",
    "I don't have information",
    "I can't find any information",
    "Not found in",
    "I don't know.",
    "I cannot verify",
    "I cannot recall",
    "I'm sorry", 
    "I am sorry",
    "I apologize",
]


def detect_refusal_responses(
    preds: Sequence[str],
    refusal_answers: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = list(refusal_answers) if refusal_answers is not None else list(REFUSAL_ANS)
    candidates = [("" if c is None else str(c)).lower() for c in candidates]
    preds_list = [("" if p is None else str(p)).lower() for p in preds]
    match_scores = np.zeros(len(preds_list), dtype=np.float32)
    refusal_mask = np.zeros(len(preds_list), dtype=bool)
    if not candidates:
        return refusal_mask, match_scores
    for i, pred in enumerate(preds_list):
        matched = any(candidate and candidate in pred for candidate in candidates)
        refusal_mask[i] = bool(matched)
        match_scores[i] = 1.0 if matched else 0.0
    return refusal_mask, match_scores


def compute_ood_refusal_filter(
    labels: Sequence[int],
    dataset_name: str,
    split: str,
    remove_refusal: bool,
    load_dir: Optional[str] = None,
    preds: Optional[Sequence[str]] = None,
    idx_subset: Optional[Sequence[int]] = None,
    refusal_answers: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    total = int(labels_arr.shape[0])
    mask_all = np.ones(total, dtype=bool)
    stats = {
        "applied": False,
        "accuracy": float("nan"),
        "excluded_ood_count": 0,
        "excluded_id_count": 0,
        "excluded_total_count": 0,
        "remaining_count": total,
        "total_count": total,
    }

    ds_name = str(dataset_name or "").strip().lower()
    split_name = str(split or "").strip().lower()
    if not (
        bool(remove_refusal)
        and ds_name in {"alcuna_trivia_qa", "repli_qa_trivia_qa", "temporalood_trivia_qa"}
        and split_name == "test"
    ):
        return mask_all, stats

    preds_list = None
    if preds is not None:
        preds_list = ["" if p is None else str(p) for p in preds]
    elif load_dir:
        preds_path = os.path.join(load_dir, split_name, "preds.jsonl")
        if os.path.exists(preds_path):
            try:
                preds_list = load_pred_json(preds_path)
            except Exception as exc:
                print(f"[Refusal Filter] Failed to read {preds_path}: {exc}. Skipping refusal filtering.")
                return mask_all, stats
        else:
            print(f"[Refusal Filter] Missing {preds_path}. Skipping refusal filtering.")
            return mask_all, stats
    else:
        print("[Refusal Filter] Missing both preds and load_dir. Skipping refusal filtering.")
        return mask_all, stats

    if preds_list is None:
        return mask_all, stats

    if idx_subset is not None:
        idx_arr = np.asarray(idx_subset, dtype=np.int64).reshape(-1)
        if idx_arr.size > 0 and len(preds_list) > int(idx_arr.max()) and len(preds_list) != total:
            preds_list = [preds_list[int(i)] for i in idx_arr.tolist()]

    if len(preds_list) != total:
        print(
            "[Refusal Filter] preds/labels length mismatch "
            f"(preds={len(preds_list)}, labels={total}). Skipping refusal filtering."
        )
        return mask_all, stats

    try:
        refusal_mask, _ = detect_refusal_responses(
            preds=preds_list,
            refusal_answers=refusal_answers,
        )
    except Exception as exc:
        print(f"[Refusal Filter] Failed refusal substring matching: {exc}. Skipping refusal filtering.")
        return mask_all, stats
    refusal_mask = np.asarray(refusal_mask, dtype=bool)
    eval_mask = ~refusal_mask
    excluded_total = int(refusal_mask.sum())
    excluded_ood = int(np.logical_and(refusal_mask, labels_arr == 1).sum())
    excluded_id = int(np.logical_and(refusal_mask, labels_arr == 0).sum())
    # Report how accurately refused samples correspond to OOD items.
    # This is precision on the refusal-positive set: TP / (TP + FP).
    if excluded_total > 0:
        acc = float(excluded_ood / excluded_total)
    else:
        acc = float("nan")

    stats.update(
        {
            "applied": True,
            "accuracy": acc,
            "excluded_ood_count": excluded_ood,
            "excluded_id_count": excluded_id,
            "excluded_total_count": excluded_total,
            "remaining_count": int(eval_mask.sum()),
            "total_count": total,
        }
    )
    return eval_mask, stats


def combine_optional_eval_mask(
    metric_eval_mask: Sequence[bool],
    extra_eval_mask: Optional[Sequence[bool]] = None,
    idx_subset: Optional[Sequence[int]] = None,
    label: str = "eval",
) -> np.ndarray:
    base_mask = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)
    if extra_eval_mask is None:
        return base_mask

    extra_mask = np.asarray(extra_eval_mask, dtype=bool).reshape(-1)
    if idx_subset is not None:
        idx_arr = np.asarray(idx_subset, dtype=np.int64).reshape(-1)
        if idx_arr.size > 0:
            max_idx = int(idx_arr.max())
            if extra_mask.shape[0] <= max_idx:
                print(
                    f"[EvalMask] Skipping extra eval mask for {label}: "
                    f"mask_len={extra_mask.shape[0]}, max_idx={max_idx}."
                )
                return base_mask
            extra_mask = extra_mask[idx_arr]

    if extra_mask.shape[0] != base_mask.shape[0]:
        print(
            f"[EvalMask] Skipping extra eval mask for {label}: "
            f"base_len={base_mask.shape[0]}, extra_len={extra_mask.shape[0]}."
        )
        return base_mask

    return np.logical_and(base_mask, extra_mask)


def load_response_embeddings_split(
    load_dir: str,
    split: str,
    n_resp: int,
) -> List[Dict[int, np.ndarray]]:
    path = os.path.join(load_dir, split, f"response_embeddings_{n_resp}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing response-level embedding artifact: {path}. "
            "Regenerate semantic clusters with response-wise outputs to create response_embeddings artifacts."
        )
    return load_cluster_embeddings_npz(path)


def _nan_skyline_metric_dict() -> Dict[str, float]:
    return {"vacuity": float("nan"), "dissonance": float("nan")}


def _make_empty_train_response_skyline_payload(n_questions: int = 0) -> Dict[str, object]:
    count = max(int(n_questions), 0)
    return {
        "available": False,
        "evidence": [{} for _ in range(count)],
        "vacuity": [float("nan")] * count,
        "dissonance": [float("nan")] * count,
        "AUROC": _nan_skyline_metric_dict(),
        "AUPR": _nan_skyline_metric_dict(),
    }


def _resolve_metric_labels_and_mask(
    results: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    labels_raw = results.get("uncertainty_labels")
    if labels_raw is not None:
        labels = np.asarray(labels_raw, dtype=np.int64).reshape(-1)
    else:
        y_raw = results.get("y")
        if y_raw is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
        labels = np.asarray(np.asarray(y_raw).reshape(-1) == 0, dtype=np.int64)
    metric_eval_mask = np.asarray(
        results.get("metric_eval_mask", np.ones(labels.size, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if metric_eval_mask.shape[0] != labels.shape[0]:
        metric_eval_mask = np.ones(labels.size, dtype=bool)
    return labels, metric_eval_mask


def build_train_response_skyline_bank(
    semantic_ids: np.ndarray,
    response_embeddings: List[Dict[int, np.ndarray]],
) -> Dict[str, np.ndarray]:
    embeddings: List[np.ndarray] = []
    cluster_counts: List[float] = []
    qids: List[int] = []
    resp_ids: List[int] = []
    cluster_ids: List[int] = []
    dim: Optional[int] = None

    n_questions = min(len(semantic_ids), len(response_embeddings))
    for qid in range(n_questions):
        ids_row = np.asarray(semantic_ids[qid], dtype=np.int64).reshape(-1)
        if ids_row.size == 0:
            continue
        count_by_cluster = Counter(ids_row.tolist())
        emb_dict = response_embeddings[qid] if response_embeddings[qid] is not None else {}
        if not isinstance(emb_dict, dict) or not emb_dict:
            continue
        for raw_resp_id in sorted(emb_dict.keys(), key=lambda item: int(item)):
            resp_id = int(raw_resp_id)
            if resp_id < 0 or resp_id >= ids_row.size:
                continue
            vec = np.asarray(emb_dict[raw_resp_id], dtype=np.float32).reshape(-1)
            if vec.size == 0:
                continue
            if dim is None:
                dim = int(vec.size)
            elif vec.size != dim:
                raise ValueError(
                    "Inconsistent response embedding dimension in training skyline bank: "
                    f"expected={dim}, found={vec.size} at qid={qid}, resp_id={resp_id}."
                )
            norm = float(np.linalg.norm(vec))
            if norm <= 0.0:
                continue
            cid = int(ids_row[resp_id])
            embeddings.append((vec / norm).astype(np.float32, copy=False))
            cluster_counts.append(float(count_by_cluster[cid]))
            qids.append(int(qid))
            resp_ids.append(int(resp_id))
            cluster_ids.append(cid)

    emb_dim = int(dim or 0)
    emb_matrix = (
        np.stack(embeddings, axis=0).astype(np.float32, copy=False)
        if embeddings
        else np.zeros((0, emb_dim), dtype=np.float32)
    )
    return {
        "embeddings": emb_matrix,
        "cluster_counts": np.asarray(cluster_counts, dtype=np.float32),
        "qids": np.asarray(qids, dtype=np.int64),
        "resp_ids": np.asarray(resp_ids, dtype=np.int64),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64),
    }


def load_train_response_skyline_bank(
    load_dir: str,
    n_resp: int,
    semantic_ids: Optional[np.ndarray] = None,
    split: str = "train",
) -> Dict[str, np.ndarray]:
    semantic_ids_arr = semantic_ids
    if semantic_ids_arr is None:
        semantic_ids_arr = np.load(
            os.path.join(load_dir, split, f"semantic_ids_{n_resp}.npy")
        ).astype(np.int64)
    response_embeddings = load_response_embeddings_split(load_dir, split, n_resp)
    return build_train_response_skyline_bank(
        semantic_ids=np.asarray(semantic_ids_arr, dtype=np.int64),
        response_embeddings=response_embeddings,
    )


def _flatten_itp_response_query_embeddings(
    results: Dict[str, object],
    expected_size: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    semantic_cluster_embeddings = results.get("semantic_cluster_embeddings")
    if semantic_cluster_embeddings is None:
        return None
    try:
        n_questions = len(semantic_cluster_embeddings)
    except TypeError:
        return None
    if expected_size is not None and n_questions != int(expected_size):
        return None

    embeddings: List[np.ndarray] = []
    qids: List[int] = []
    cluster_ids: List[int] = []
    dim: Optional[int] = None

    for qid, emb_dict in enumerate(semantic_cluster_embeddings):
        if not isinstance(emb_dict, dict) or not emb_dict:
            continue
        for raw_cid in sorted(emb_dict.keys(), key=lambda item: int(item)):
            try:
                cid = int(raw_cid)
            except (TypeError, ValueError):
                continue
            vec = np.asarray(emb_dict[raw_cid], dtype=np.float32).reshape(-1)
            if vec.size == 0:
                continue
            if dim is None:
                dim = int(vec.size)
            elif vec.size != dim:
                return None
            norm = float(np.linalg.norm(vec))
            if norm <= 0.0:
                continue
            embeddings.append((vec / norm).astype(np.float32, copy=False))
            qids.append(int(qid))
            cluster_ids.append(int(cid))

    emb_dim = int(dim or 0)
    emb_matrix = (
        np.stack(embeddings, axis=0).astype(np.float32, copy=False)
        if embeddings
        else np.zeros((0, emb_dim), dtype=np.float32)
    )
    return {
        "embeddings": emb_matrix,
        "qids": np.asarray(qids, dtype=np.int64),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int64),
    }


def _resolve_train_response_skyline_block_sizes(
    n_queries: int,
    n_pool: int,
    dim: int,
    device: torch.device,
) -> Tuple[int, int]:
    if n_queries <= 0 or n_pool <= 0:
        return 1, 1
    if device.type != "cuda":
        return min(n_queries, 256), min(n_pool, 2048)

    max_query_block = min(n_queries, 1024)
    max_pool_block = min(n_pool, 8192)
    try:
        free_mem, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return min(max_query_block, 512), min(max_pool_block, 4096)

    bytes_per_float = 4
    budget = int(max(free_mem * 0.04, 32 * 1024 * 1024))
    query_block = max_query_block
    while query_block > 1:
        numer = (budget // bytes_per_float) - query_block * (dim + 2)
        denom = max(dim + query_block + 1, 1)
        pool_block = int(numer // denom)
        if pool_block > 0:
            return query_block, min(max_pool_block, max(1, pool_block))
        query_block //= 2
    return 1, min(max_pool_block, 1024)


def _find_closest_train_response_matches(
    query_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_queries = int(query_embeddings.shape[0])
    n_pool = int(train_embeddings.shape[0])
    if n_queries == 0 or n_pool == 0:
        return (
            np.full(n_queries, -1, dtype=np.int64),
            np.full(n_queries, -np.inf, dtype=np.float32),
        )

    scan_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if scan_device.type == "cuda" and not torch.cuda.is_available():
        scan_device = torch.device("cpu")

    q_block, pool_block = _resolve_train_response_skyline_block_sizes(
        n_queries=n_queries,
        n_pool=n_pool,
        dim=int(query_embeddings.shape[1]),
        device=scan_device,
    )
    best_indices = np.full(n_queries, -1, dtype=np.int64)
    best_sims = np.full(n_queries, -np.inf, dtype=np.float32)

    for q_start in range(0, n_queries, q_block):
        q_end = min(q_start + q_block, n_queries)
        q_cpu = torch.from_numpy(query_embeddings[q_start:q_end])
        q_block_tensor = (
            q_cpu.to(scan_device, non_blocking=True)
            if scan_device.type == "cuda"
            else q_cpu
        )
        local_best_sim = torch.full(
            (q_end - q_start,),
            -float("inf"),
            dtype=torch.float32,
            device=scan_device,
        )
        local_best_idx = torch.full(
            (q_end - q_start,),
            -1,
            dtype=torch.int64,
            device=scan_device,
        )

        for pool_start in range(0, n_pool, pool_block):
            pool_end = min(pool_start + pool_block, n_pool)
            pool_cpu = torch.from_numpy(train_embeddings[pool_start:pool_end])
            pool_block_tensor = (
                pool_cpu.to(scan_device, non_blocking=True)
                if scan_device.type == "cuda"
                else pool_cpu
            )
            sim_block = q_block_tensor @ pool_block_tensor.t()
            block_best_sim, block_best_idx = torch.max(sim_block, dim=1)
            better_mask = block_best_sim > local_best_sim
            local_best_sim = torch.where(better_mask, block_best_sim, local_best_sim)
            local_best_idx = torch.where(
                better_mask,
                block_best_idx.to(torch.int64) + int(pool_start),
                local_best_idx,
            )
            if scan_device.type == "cuda":
                del pool_block_tensor, sim_block, block_best_sim, block_best_idx, better_mask

        best_indices[q_start:q_end] = local_best_idx.detach().cpu().numpy()
        best_sims[q_start:q_end] = local_best_sim.detach().cpu().numpy()
        if scan_device.type == "cuda":
            del q_block_tensor, local_best_sim, local_best_idx
            torch.cuda.empty_cache()

    return best_indices, best_sims


def attach_train_response_skyline_metrics(
    results: Dict[str, object],
    split: str,
    scope: str,
    train_response_bank: Optional[Dict[str, np.ndarray]],
) -> Dict[str, object]:
    labels, metric_eval_mask = _resolve_metric_labels_and_mask(results)
    n_questions = int(labels.size)
    skyline_payload = _make_empty_train_response_skyline_payload(n_questions=n_questions)
    skyline_payload["metric_eval_mask"] = metric_eval_mask
    results["skyline"] = skyline_payload

    if n_questions == 0:
        return skyline_payload
    if train_response_bank is None:
        print(f"[Skyline] Skipping {scope} for {split} split (missing train response bank).")
        return skyline_payload

    train_embeddings = np.asarray(train_response_bank.get("embeddings"), dtype=np.float32)
    train_cluster_counts = np.asarray(train_response_bank.get("cluster_counts"), dtype=np.float32).reshape(-1)
    if train_embeddings.ndim != 2 or train_cluster_counts.size != train_embeddings.shape[0]:
        print(
            f"[Skyline] Skipping {scope} for {split} split "
            "(malformed train response bank)."
        )
        return skyline_payload
    if train_embeddings.shape[0] == 0:
        print(f"[Skyline] Skipping {scope} for {split} split (empty train response bank).")
        return skyline_payload

    query_bank = _flatten_itp_response_query_embeddings(results, expected_size=n_questions)
    if query_bank is None:
        print(
            f"[Skyline] Skipping {scope} for {split} split "
            "(missing or misaligned ITP response embeddings)."
        )
        return skyline_payload

    query_embeddings = np.asarray(query_bank.get("embeddings"), dtype=np.float32)
    query_qids = np.asarray(query_bank.get("qids"), dtype=np.int64).reshape(-1)
    query_cluster_ids = np.asarray(query_bank.get("cluster_ids"), dtype=np.int64).reshape(-1)
    if query_embeddings.ndim != 2:
        print(
            f"[Skyline] Skipping {scope} for {split} split "
            "(malformed flattened ITP response embeddings)."
        )
        return skyline_payload
    if query_embeddings.shape[0] != query_qids.size or query_embeddings.shape[0] != query_cluster_ids.size:
        print(
            f"[Skyline] Skipping {scope} for {split} split "
            "(flattened ITP response metadata mismatch)."
        )
        return skyline_payload
    if query_embeddings.shape[1] != train_embeddings.shape[1]:
        print(
            f"[Skyline] Skipping {scope} for {split} split "
            f"(embedding dim mismatch: test={query_embeddings.shape[1]}, train={train_embeddings.shape[1]})."
        )
        return skyline_payload

    evidence_all: List[Dict[int, float]] = [{} for _ in range(n_questions)]
    if query_embeddings.shape[0] > 0:
        best_indices, _best_sims = _find_closest_train_response_matches(
            query_embeddings=query_embeddings,
            train_embeddings=train_embeddings,
        )
        for idx, best_idx in enumerate(best_indices.tolist()):
            if best_idx < 0:
                continue
            qid = int(query_qids[idx])
            cid = int(query_cluster_ids[idx])
            evidence_all[qid][cid] = float(train_cluster_counts[best_idx])

    vacuity_scores = np.asarray([vacuity(e_dict) for e_dict in evidence_all], dtype=np.float32)
    dissonance_scores = np.asarray([dissonance(e_dict) for e_dict in evidence_all], dtype=np.float32)
    auroc = _nan_skyline_metric_dict()
    aupr = _nan_skyline_metric_dict()

    if metric_eval_mask.shape[0] != labels.shape[0]:
        metric_eval_mask = np.ones(labels.shape[0], dtype=bool)

    for metric_name, scores in {
        "vacuity": vacuity_scores,
        "dissonance": dissonance_scores,
    }.items():
        labels_eval = labels[metric_eval_mask]
        scores_eval = scores[metric_eval_mask]
        if labels_eval.size == 0:
            continue
        unique_labels = np.unique(labels_eval)
        if unique_labels.size < 2:
            print(
                f"[Skyline] Skipping {scope}:{metric_name} AUROC/AUPR for {split} split "
                f"(only one class present: {unique_labels.tolist()})."
            )
            continue
        auroc[metric_name] = float(roc_auc_score(labels_eval, scores_eval))
        aupr[metric_name] = float(average_precision_score(labels_eval, scores_eval))

    skyline_payload = {
        "available": True,
        "evidence": evidence_all,
        "vacuity": vacuity_scores,
        "dissonance": dissonance_scores,
        "metric_eval_mask": np.asarray(metric_eval_mask, dtype=bool),
        "AUROC": auroc,
        "AUPR": aupr,
    }
    results["skyline"] = skyline_payload
    return skyline_payload


def log_train_response_skyline_metrics(
    split: str,
    results: Dict[str, object],
    scope: str,
    train_response_bank: Optional[Dict[str, np.ndarray]],
    step: Optional[int] = None,
    summary: bool = False,
) -> Dict[str, float]:
    skyline_payload = results.get("skyline")
    if (
        not isinstance(skyline_payload, dict)
        or (train_response_bank is not None and not bool(skyline_payload.get("available", False)))
    ):
        skyline_payload = attach_train_response_skyline_metrics(
            results=results,
            split=split,
            scope=scope,
            train_response_bank=train_response_bank,
        )
    metrics = {
        f"{split}/skyline_auroc_vacuity": float(skyline_payload.get("AUROC", {}).get("vacuity", float("nan"))),
        f"{split}/skyline_auroc_dissonance": float(skyline_payload.get("AUROC", {}).get("dissonance", float("nan"))),
        f"{split}/skyline_aupr_vacuity": float(skyline_payload.get("AUPR", {}).get("vacuity", float("nan"))),
        f"{split}/skyline_aupr_dissonance": float(skyline_payload.get("AUPR", {}).get("dissonance", float("nan"))),
    }
    print(
        f"[Skyline] {split} AUROC vacuity: {metrics[f'{split}/skyline_auroc_vacuity']:.4f}, "
        f"dissonance: {metrics[f'{split}/skyline_auroc_dissonance']:.4f}"
    )
    print(
        f"[Skyline] {split} AUPR vacuity: {metrics[f'{split}/skyline_aupr_vacuity']:.4f}, "
        f"dissonance: {metrics[f'{split}/skyline_aupr_dissonance']:.4f}"
    )
    if summary and wandb.run is not None:
        wandb.run.summary.update(metrics)
    if step is not None and wandb.run is not None:
        history_metrics = dict(metrics)
        history_metrics["epoch"] = int(step)
        wandb.log(history_metrics, step=int(step), commit=False)
    return metrics
