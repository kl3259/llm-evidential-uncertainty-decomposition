"""
Define datasets for cluster-wise training
"""
import os, json, argparse, time, math, gc
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
from bisect import bisect_right
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from distutils.util import strtobool as _bool
from helper_functions import (
    count_or_frac,
    load_cluster_embeddings_npz,
    load_X_npz,
    load_question_json,
    load_resp_json,
)
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

try:
    from transformers import Gemma4ForCausalLM
except ImportError:
    Gemma4ForCausalLM = None

from feature_collector import _resolve_hf_model_name, _is_gemma_4_e4b_model_name

WORKING_DIR = "./"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _layer_index_list(layer_range) -> List[int]:
    if torch.is_tensor(layer_range):
        arr = layer_range.detach().cpu().numpy()
    else:
        arr = np.asarray(layer_range)
    return np.atleast_1d(arr).astype(np.int64).tolist()


@dataclass(slots=True)
class ScanMetadata:
    dataset_idx: Optional[int]
    sample_idx: Optional[int]
    qid: Optional[int]
    resp_id: Optional[int]
    cid: Optional[int]
    reg_idx: Optional[int]
    y_val: float
    is_reg: bool
    display_qid: Optional[int]
    display_resp_id: Optional[int]
    display_cid: Optional[int]
    source_qid: Optional[int]
    source_cid: Optional[int]
    source_resp_id: Optional[int]

def _is_response_level_mode(mode: str) -> bool:
    return (mode or "").lower() in {"response-wise", "ranking"}


def normalize_target_type(target_type: str) -> str:
    norm = str(target_type or "sc_count").strip().lower()
    if norm not in {"sc_count", "sc_prob_mass"}:
        raise ValueError(f"Unsupported target_type: {target_type}")
    return norm


def build_target_evidence_by_question(
    semantic_ids: np.ndarray,
    target_type: str = "sc_count",
    response_sequence_probs: Optional[np.ndarray] = None,
) -> List[Dict[int, float]]:
    semantic_ids_np = np.asarray(semantic_ids, dtype=np.int64)
    if semantic_ids_np.ndim != 2:
        raise ValueError(f"semantic_ids must be 2D, got shape={semantic_ids_np.shape}.")

    target_type_norm = normalize_target_type(target_type)
    if target_type_norm == "sc_prob_mass":
        if response_sequence_probs is None:
            raise ValueError("target_type=sc_prob_mass requires response_sequence_probs.")
        response_sequence_probs_np = np.asarray(response_sequence_probs, dtype=np.float32)
        if response_sequence_probs_np.ndim != 2:
            raise ValueError(
                "response_sequence_probs must be 2D for sc_prob_mass, "
                f"got shape={response_sequence_probs_np.shape}."
            )
        if response_sequence_probs_np.shape != semantic_ids_np.shape:
            raise ValueError(
                "response_sequence_probs / semantic_ids shape mismatch: "
                f"response_sequence_probs={response_sequence_probs_np.shape}, "
                f"semantic_ids={semantic_ids_np.shape}"
            )
        if not np.all(np.isfinite(response_sequence_probs_np)):
            raise ValueError("response_sequence_probs contains non-finite values.")
    else:
        response_sequence_probs_np = None

    target_by_question: List[Dict[int, float]] = []
    for qid in range(semantic_ids_np.shape[0]):
        ids_row = semantic_ids_np[qid]
        target_row: Dict[int, float] = {}
        if ids_row.size == 0:
            target_by_question.append(target_row)
            continue
        if target_type_norm == "sc_count":
            vals, cnts = np.unique(ids_row, return_counts=True)
            target_row = {int(cid): float(cnt) for cid, cnt in zip(vals, cnts)}
        else:
            probs_row = response_sequence_probs_np[qid]
            for resp_id, cid in enumerate(ids_row.tolist()):
                cid_int = int(cid)
                target_row[cid_int] = target_row.get(cid_int, 0.0) + float(probs_row[resp_id])
        target_by_question.append(target_row)
    return target_by_question


def _build_tbg_slt_hidden_features(
    H: Union[np.ndarray, torch.Tensor],
    response_hidden_embs: List[Dict[int, np.ndarray]],
    semantic_ids: np.ndarray,
    hidden_layer_count: int,
) -> List[Dict[int, np.ndarray]]:
    """
    Build per-(question,response) hidden feature vectors:
        [u, v, |u-v|, u*v, cos_layerwise(u, v)]
    where u is question TBG hidden and v is response SLT hidden.
    """
    if hidden_layer_count is None or int(hidden_layer_count) <= 0:
        raise ValueError("feature_type=tbg_slt_hidden requires hidden_layer_count > 0.")
    k_layers = int(hidden_layer_count)

    if torch.is_tensor(H):
        H_np = H.detach().cpu().numpy().astype(np.float32)
    else:
        H_np = np.asarray(H, dtype=np.float32)
    if H_np.ndim != 2:
        raise ValueError(f"H must be a 2D matrix, got shape={H_np.shape}.")

    semantic_ids_np = np.asarray(semantic_ids, dtype=np.int64)
    if semantic_ids_np.ndim != 2:
        raise ValueError(f"semantic_ids must be 2D, got shape={semantic_ids_np.shape}.")

    n_questions, h_dim = H_np.shape
    if semantic_ids_np.shape[0] != n_questions:
        raise ValueError(
            "H / semantic_ids size mismatch: "
            f"H={n_questions}, semantic_ids={semantic_ids_np.shape[0]}"
        )
    if len(response_hidden_embs) != n_questions:
        raise ValueError(
            "response_hidden_embs / semantic_ids size mismatch: "
            f"response_hidden_embs={len(response_hidden_embs)}, semantic_ids={n_questions}"
        )
    if h_dim % k_layers != 0:
        raise ValueError(
            "H hidden dimension is not divisible by hidden_layer_count: "
            f"h_dim={h_dim}, hidden_layer_count={k_layers}"
        )

    hidden_size = h_dim // k_layers
    n_resp = int(semantic_ids_np.shape[1])
    eps = 1e-12
    composed_hidden: List[Dict[int, np.ndarray]] = []

    for qid in range(n_questions):
        u_flat = np.asarray(H_np[qid], dtype=np.float32).reshape(-1)
        u_layerwise = u_flat.reshape(k_layers, hidden_size)

        resp_map_raw = response_hidden_embs[qid]
        if resp_map_raw is None:
            composed_hidden.append({})
            continue
        if not isinstance(resp_map_raw, dict):
            raise TypeError(
                "response_hidden_embs entries must be dict[int, np.ndarray] or None; "
                f"got type={type(resp_map_raw)} for qid={qid}"
            )

        qid_composed: Dict[int, np.ndarray] = {}
        for resp_id in range(n_resp):
            v_raw = resp_map_raw.get(int(resp_id))
            if v_raw is None:
                continue

            v_flat = np.asarray(v_raw, dtype=np.float32).reshape(-1)
            if int(v_flat.size) != h_dim:
                raise ValueError(
                    "Response hidden size mismatch for feature_type=tbg_slt_hidden: "
                    f"qid={qid}, resp_id={resp_id}, expected={h_dim}, got={int(v_flat.size)}"
                )
            v_layerwise = v_flat.reshape(k_layers, hidden_size)

            abs_diff_flat = np.abs(u_layerwise - v_layerwise).reshape(-1)
            prod_flat = (u_layerwise * v_layerwise).reshape(-1)
            dot = np.sum(u_layerwise * v_layerwise, axis=1)
            u_norm = np.linalg.norm(u_layerwise, axis=1)
            v_norm = np.linalg.norm(v_layerwise, axis=1)
            cos_layerwise = dot / (u_norm * v_norm + eps)
            cos_layerwise = np.clip(cos_layerwise, -1.0, 1.0).astype(np.float32)

            feat = np.concatenate(
                (u_flat, v_flat, abs_diff_flat, prod_flat, cos_layerwise),
                axis=0,
            )
            feat = np.nan_to_num(
                feat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
            qid_composed[int(resp_id)] = feat

        composed_hidden.append(qid_composed)

    return composed_hidden


def _build_tbg_slt_concat_features(
    H: Union[np.ndarray, torch.Tensor],
    question_slt_hidden: Union[np.ndarray, torch.Tensor],
    cluster_embs: List[Dict[int, np.ndarray]],
    semantic_ids: np.ndarray,
) -> List[Dict[int, np.ndarray]]:
    """
    Build per-(question,response) hidden feature vectors:
      [u, v, e]
    where
      u: selected-layer TBG hidden (question-level)
      v: selected-layer SLT hidden (question-level, from X["slt"])
      e: response embedding
    """
    if torch.is_tensor(H):
        H_np = H.detach().cpu().numpy().astype(np.float32)
    else:
        H_np = np.asarray(H, dtype=np.float32)
    if H_np.ndim != 2:
        raise ValueError(f"H must be a 2D matrix, got shape={H_np.shape}.")

    if torch.is_tensor(question_slt_hidden):
        S_np = question_slt_hidden.detach().cpu().numpy().astype(np.float32)
    else:
        S_np = np.asarray(question_slt_hidden, dtype=np.float32)
    if S_np.ndim != 2:
        raise ValueError(f"question_slt_hidden must be a 2D matrix, got shape={S_np.shape}.")

    semantic_ids_np = np.asarray(semantic_ids, dtype=np.int64)
    if semantic_ids_np.ndim != 2:
        raise ValueError(f"semantic_ids must be 2D, got shape={semantic_ids_np.shape}.")

    n_questions, h_dim = H_np.shape
    n_questions_slt, slt_dim = S_np.shape
    if semantic_ids_np.shape[0] != n_questions:
        raise ValueError(
            "H / semantic_ids size mismatch: "
            f"H={n_questions}, semantic_ids={semantic_ids_np.shape[0]}"
        )
    if n_questions_slt != n_questions:
        raise ValueError(
            "question_slt_hidden / semantic_ids size mismatch: "
            f"question_slt_hidden={n_questions_slt}, semantic_ids={n_questions}"
        )
    if len(cluster_embs) != n_questions:
        raise ValueError(
            "cluster_embs / semantic_ids size mismatch: "
            f"cluster_embs={len(cluster_embs)}, semantic_ids={n_questions}"
        )
    if int(slt_dim) != int(h_dim):
        raise ValueError(
            "Selected-layer TBG and SLT dims must match for feature_type=tbg_slt_concat: "
            f"tbg={int(h_dim)}, slt={int(slt_dim)}"
        )

    n_resp = int(semantic_ids_np.shape[1])
    composed_hidden: List[Dict[int, np.ndarray]] = []

    for qid in range(n_questions):
        u_flat = np.asarray(H_np[qid], dtype=np.float32).reshape(-1)
        v_flat = np.asarray(S_np[qid], dtype=np.float32).reshape(-1)
        emb_map_raw = cluster_embs[qid]
        if not isinstance(emb_map_raw, dict):
            raise TypeError(
                "cluster_embs entries must be dict[int, np.ndarray]; "
                f"got type={type(emb_map_raw)} for qid={qid}"
            )

        qid_composed: Dict[int, np.ndarray] = {}
        for resp_id in range(n_resp):
            e_raw = emb_map_raw.get(int(resp_id))
            if e_raw is None:
                raise ValueError(
                    "Missing response embedding for feature_type=tbg_slt_concat: "
                    f"qid={qid}, resp_id={resp_id}"
                )
            e_flat = np.asarray(e_raw, dtype=np.float32).reshape(-1)
            feat = np.concatenate((u_flat, v_flat, e_flat), axis=0)
            feat = np.nan_to_num(
                feat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
            qid_composed[int(resp_id)] = feat

        composed_hidden.append(qid_composed)

    return composed_hidden


def load_split(load_dir, split, n_resp, MSE_type: str = "cluster-wise"):
    split_dir = os.path.join(load_dir, split)

    semantic_ids = np.load(os.path.join(split_dir, f"semantic_ids_{n_resp}.npy")).astype(np.int64) # semantic_ids

    emb_name = (
        f"response_embeddings_{n_resp}.npz"
        if _is_response_level_mode(MSE_type)
        else f"cluster_embeddings_{n_resp}.npz"
    )
    emb_path = os.path.join(split_dir, emb_name)
    semantic_cluster_embeddings = load_cluster_embeddings_npz(emb_path) # semantic cluster embeddings

    X = load_X_npz(os.path.join(split_dir, "X.npz")) # hidden_states
    y = np.load(os.path.join(split_dir, f"y.npy")) # labels
    se_scores = np.load(os.path.join(split_dir, f"se.npy")) # se_scores

    return semantic_ids, semantic_cluster_embeddings, se_scores, X, y


class ClusterCountDataset(Dataset):
    def __init__(self, H, semantic_ids, cluster_embs, c_dim: int):
        """
        H            : np.ndarray [N, h_dim]  hidden states for chosen layer_range
        semantic_ids : np.ndarray [N, n_responses]  (cluster id per response)
        cluster_embs : list[dict[int -> np.ndarray(3072,)]]  (per-example cluster embeddings)
        c_dim        : int, e.g. 3072 for text-embedding-3-large
        """
        self.H = torch.as_tensor(H, dtype=torch.float32)
        self.semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
        self.cluster_embs = cluster_embs
        self.index = [(i, cid) for i, d in enumerate(cluster_embs) for cid in d.keys()]
        self.c_dim = c_dim
        assert len(self.H) == len(self.semantic_ids) == len(self.cluster_embs)

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        i, cid = self.index[idx]
        h = self.H[i]
        c = torch.tensor(self.cluster_embs[i][cid], dtype=torch.float32)
        y = torch.tensor(float(np.count_nonzero(self.semantic_ids[i] == cid)), dtype=torch.float32)

        return h, c, y, (i, cid)


def cluster_count_collate(batch): 
    Hb   = torch.stack([b[0] for b in batch], 0)  # [B, h_dim]
    Cb   = torch.stack([b[1] for b in batch], 0)  # [B, c_dim]
    yb   = torch.stack([b[2] for b in batch], 0)  # [B]
    idxb = [b[3] for b in batch]                  # list[(qid, cid)]
    return Hb, Cb, yb, idxb

def _dominant_cid(ids_row: np.ndarray) -> int:
    vals, cnts = np.unique(ids_row, return_counts=True)
    return int(vals[np.argmax(cnts)])

class WeightedResponseClusterCountDataset(Dataset):
    def __init__(self, H, semantic_ids, response_embs, is_reg,
                 c_dim: int, n_reg: int, index_offset: int = 0,
                 metadata_map: Optional[Dict[Tuple[int, ...], Dict[str, int]]] = None,
                 norm: bool = False, allowed_qids: Optional[Set[int]] = None,
                 cnt_weighted: bool = False, cluster_counts: Optional[List[int]] = None,
                 target_by_question: Optional[List[Dict[int, float]]] = None,
                 target_type: str = "sc_count",
                 domain_labels: Optional[np.ndarray] = None):
        """
        H              : np.ndarray [N, h_dim] hidden states for chosen layer_range
        semantic_ids   : np.ndarray [N, n_responses] (cluster id per response)
        response_embs  : list of per-response embeddings aligned with semantic_ids
        is_reg         : np.ndarray[N,] question-wise indicator of regularization samples
        c_dim          : int, e.g. 3072 for text-embedding-3-large
        n_reg          : int, number of regularization samples per response
        allowed_qids   : optional set of question indices to keep; if provided, only items whose
                         (i + index_offset) belong to this set are retained.
        """
        self.H = torch.as_tensor(H, dtype=torch.float32)
        self.semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
        self.response_embs = response_embs
        self.is_reg = is_reg
        self.c_dim = c_dim
        self.n_reg = n_reg
        self.index_offset = index_offset
        self.metadata_map = metadata_map if metadata_map is not None else {}
        self.norm = norm
        self.cnt_weighted = cnt_weighted
        self.target_by_question = target_by_question
        self.target_type = normalize_target_type(target_type)
        if cluster_counts is None:
            cluster_counts = [len(np.unique(row)) for row in self.semantic_ids]
        self.cluster_counts = cluster_counts
        self.n_resp = self.semantic_ids.shape[1]
        if domain_labels is not None:
            if torch.is_tensor(domain_labels):
                domain_labels = domain_labels.detach().cpu().numpy()
            self.domain_labels = np.asarray(domain_labels, dtype=np.int64)
            if len(self.domain_labels) != len(self.H):
                raise ValueError(
                    f"domain_labels length mismatch: {len(self.domain_labels)} vs {len(self.H)}"
                )
        else:
            self.domain_labels = None

        assert len(self.H) == len(self.semantic_ids) == len(self.response_embs)

        self.index: List[Union[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        for i, resp_list in enumerate(self.response_embs):
            if allowed_qids is not None and (i + self.index_offset) not in allowed_qids:
                continue
            if resp_list is None:
                continue
            limit = min(len(resp_list), self.semantic_ids.shape[1])
            for resp_id in range(limit):
                emb_entry = resp_list[resp_id]
                if emb_entry is None:
                    continue
                cid = int(self.semantic_ids[i][resp_id])
                if isinstance(emb_entry, (list, tuple)):
                    for reg_idx, emb in enumerate(emb_entry):
                        if emb is None:
                            continue
                        self.index.append((i, resp_id, cid, reg_idx))
                else:
                    self.index.append((i, resp_id, cid))

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        reg_idx = None
        if isinstance(entry, (list, tuple)) and len(entry) >= 4:
            i, resp_id, cid, reg_idx = entry[0], entry[1], entry[2], entry[3]
        else:
            i, resp_id, cid = entry
        h = self.H[i]
        emb_entry = self.response_embs[i][resp_id]
        if isinstance(emb_entry, (list, tuple)):
            idx_val = 0 if reg_idx is None else int(reg_idx)
            emb_val = emb_entry[idx_val]
        else:
            emb_val = emb_entry
        c = torch.tensor(emb_val, dtype=torch.float32)
        if self.is_reg[i] == 0:
            cnt = float(np.count_nonzero(self.semantic_ids[i] == cid))
            denom = float(self.semantic_ids.shape[1])
            if self.target_by_question is not None:
                y_base = float(self.target_by_question[i].get(int(cid), 0.0))
                if self.target_type == "sc_count" and self.norm and denom > 0:
                    y_val = y_base / denom
                else:
                    y_val = y_base
            else:
                if self.norm and denom > 0:
                    y_val = cnt / denom
                else:
                    y_val = cnt

            w_val = 1.0 / denom if denom > 0 else 0.0
        else:
            y_val = 0.0
            if len(self.semantic_ids[i]) > 0:
                w_val = 1.0 / float(self.n_reg * self.n_resp)
            else:
                w_val = 0.0

        y = torch.tensor(y_val, dtype=torch.float32)
        w = torch.tensor(w_val, dtype=torch.float32)

        idx_entry = (
            (i + self.index_offset, resp_id, cid, reg_idx)
            if reg_idx is not None
            else (i + self.index_offset, resp_id, cid)
        )
        if self.domain_labels is None:
            return h, c, y, w, self.is_reg[i], idx_entry
        domain_label = int(self.domain_labels[i])
        return h, c, y, w, self.is_reg[i], domain_label, idx_entry


class HiddenStateClusterCountDataset(Dataset):
    """
    Hidden-only response-wise dataset.
    Each sample is (qid, resp_id) -> SLT hidden vector with target cluster count for its cid.
    """

    def __init__(
        self,
        response_hidden_embs: List[Dict[int, np.ndarray]],
        semantic_ids: np.ndarray,
        is_reg: np.ndarray,
        index_offset: int = 0,
        metadata_map: Optional[Dict[Tuple[int, ...], Dict[str, int]]] = None,
        norm: bool = False,
        allowed_qids: Optional[Set[int]] = None,
        target_by_question: Optional[List[Dict[int, float]]] = None,
        target_type: str = "sc_count",
        domain_labels: Optional[np.ndarray] = None,
    ) -> None:
        self.response_hidden_embs = response_hidden_embs
        self.semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
        self.is_reg = np.asarray(is_reg, dtype=np.int64)
        self.index_offset = int(index_offset)
        self.metadata_map = metadata_map if metadata_map is not None else {}
        self.norm = bool(norm)
        self.target_by_question = target_by_question
        self.target_type = normalize_target_type(target_type)
        if domain_labels is not None:
            if torch.is_tensor(domain_labels):
                domain_labels = domain_labels.detach().cpu().numpy()
            self.domain_labels = np.asarray(domain_labels, dtype=np.int64)
            if len(self.domain_labels) != len(self.semantic_ids):
                raise ValueError(
                    f"domain_labels length mismatch: {len(self.domain_labels)} vs {len(self.semantic_ids)}"
                )
        else:
            self.domain_labels = None

        if len(self.response_hidden_embs) != len(self.semantic_ids):
            raise ValueError(
                "HiddenStateClusterCountDataset size mismatch: "
                f"response_hidden_embs={len(self.response_hidden_embs)}, semantic_ids={len(self.semantic_ids)}"
            )

        self.index: List[Tuple[int, int, int]] = []
        for qid, ids_row in enumerate(self.semantic_ids):
            if allowed_qids is not None and (qid + self.index_offset) not in allowed_qids:
                continue
            emb_map = self.response_hidden_embs[qid]
            if emb_map is None:
                continue
            for resp_id in range(len(ids_row)):
                h_vec = emb_map.get(int(resp_id)) if isinstance(emb_map, dict) else None
                if h_vec is None:
                    continue
                cid = int(ids_row[resp_id])
                self.index.append((qid, resp_id, cid))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        qid, resp_id, cid = self.index[idx]
        emb_map = self.response_hidden_embs[qid]
        h_vec = emb_map.get(int(resp_id)) if isinstance(emb_map, dict) else None
        if h_vec is None:
            raise KeyError(f"Missing hidden embedding for qid={qid}, resp_id={resp_id}")

        h = torch.as_tensor(np.asarray(h_vec, dtype=np.float32), dtype=torch.float32).view(-1)
        c = torch.empty(0, dtype=torch.float32)
        if self.is_reg[qid] == 0:
            cnt = float(np.count_nonzero(self.semantic_ids[qid] == cid))
            denom = float(self.semantic_ids.shape[1])
            if self.target_by_question is not None:
                y_base = float(self.target_by_question[qid].get(int(cid), 0.0))
                if self.target_type == "sc_count" and self.norm and denom > 0:
                    y_val = y_base / denom
                else:
                    y_val = y_base
            else:
                if self.norm and denom > 0:
                    y_val = cnt / denom
                else:
                    y_val = cnt
            w_val = 1.0 / denom if denom > 0 else 0.0
        else:
            y_val = 0.0
            w_val = 0.0

        y = torch.tensor(y_val, dtype=torch.float32)
        w = torch.tensor(w_val, dtype=torch.float32)
        idx_entry = (qid + self.index_offset, resp_id, cid)

        if self.domain_labels is None:
            return h, c, y, w, self.is_reg[qid], idx_entry
        domain_label = int(self.domain_labels[qid])
        return h, c, y, w, self.is_reg[qid], domain_label, idx_entry


class ResponseRankingDataset(Dataset):
    """Pairwise ranking dataset over response embeddings within each question."""

    def __init__(
        self,
        H,
        semantic_ids: np.ndarray,
        response_embs: List[Dict[int, np.ndarray]],
        c_dim: int,
        allowed_qids: Optional[Set[int]] = None,
        max_pairs_per_question: Optional[int] = None,
    ) -> None:
        self.H = torch.as_tensor(H, dtype=torch.float32)
        self.semantic_ids = np.asarray(semantic_ids, dtype=np.int64)
        self.response_embs = response_embs
        self.c_dim = int(c_dim)
        self.max_pairs_per_question = (
            None if max_pairs_per_question is None else int(max_pairs_per_question)
        )
        if self.max_pairs_per_question is not None and self.max_pairs_per_question <= 0:
            raise ValueError(
                "max_pairs_per_question must be a positive integer when provided."
            )

        if len(self.H) != len(self.semantic_ids) or len(self.H) != len(self.response_embs):
            raise ValueError(
                "ResponseRankingDataset input size mismatch: "
                f"H={len(self.H)}, semantic_ids={len(self.semantic_ids)}, "
                f"response_embs={len(self.response_embs)}"
            )

        # index entries:
        # (qid, resp_a, resp_b, y_ab, cid_a, cid_b, cnt_a, cnt_b)
        self.index: List[Tuple[int, int, int, float, int, int, int, int]] = []
        self.total_pairs_before_cap = 0
        self.total_pairs_after_cap = 0
        self.total_pairs_dropped = 0
        self.pairs_before_cap_by_qid: Dict[int, int] = {}
        self.pairs_after_cap_by_qid: Dict[int, int] = {}
        for qid in range(len(self.H)):
            if allowed_qids is not None and qid not in allowed_qids:
                continue

            resp_emb_map = self.response_embs[qid]
            if resp_emb_map is None:
                continue

            ids_row = self.semantic_ids[qid]
            if ids_row.size == 0:
                continue
            unique_cids, counts = np.unique(ids_row, return_counts=True)
            count_by_cid = {int(cid): int(cnt) for cid, cnt in zip(unique_cids, counts)}

            valid_resp_ids: List[int] = []
            for resp_id, emb in resp_emb_map.items():
                rid = int(resp_id)
                if rid < 0 or rid >= len(ids_row):
                    continue
                if emb is None:
                    continue
                valid_resp_ids.append(rid)
            valid_resp_ids = sorted(set(valid_resp_ids))

            n_valid = len(valid_resp_ids)
            question_candidates: List[Tuple[int, int, int, float, int, int, int, int]] = []
            for a_idx in range(n_valid):
                resp_a = valid_resp_ids[a_idx]
                cid_a = int(ids_row[resp_a])
                cnt_a = int(count_by_cid.get(cid_a, 0))
                for b_idx in range(a_idx + 1, n_valid):
                    resp_b = valid_resp_ids[b_idx]
                    cid_b = int(ids_row[resp_b])
                    cnt_b = int(count_by_cid.get(cid_b, 0))

                    # Keep only cross-cluster pairs with non-equal frequencies.
                    if cid_a == cid_b or cnt_a == cnt_b:
                        continue

                    y_ab = 1.0 if cnt_a > cnt_b else 0.0
                    question_candidates.append(
                        (qid, resp_a, resp_b, y_ab, cid_a, cid_b, cnt_a, cnt_b)
                    )

            pairs_before_cap = len(question_candidates)
            self.total_pairs_before_cap += pairs_before_cap
            self.pairs_before_cap_by_qid[qid] = pairs_before_cap

            if (
                self.max_pairs_per_question is not None
                and pairs_before_cap > self.max_pairs_per_question
            ):
                question_candidates = sorted(
                    question_candidates,
                    key=lambda p: (
                        -abs(p[6] - p[7]),
                        p[1],
                        p[2],
                        p[4],
                        p[5],
                    ),
                )[: self.max_pairs_per_question]

            pairs_after_cap = len(question_candidates)
            self.total_pairs_after_cap += pairs_after_cap
            self.pairs_after_cap_by_qid[qid] = pairs_after_cap
            self.index.extend(question_candidates)

        self.total_pairs_dropped = self.total_pairs_before_cap - self.total_pairs_after_cap

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        qid, resp_a, resp_b, y_ab, cid_a, cid_b, cnt_a, cnt_b = self.index[idx]
        h = self.H[qid]
        c_a = torch.as_tensor(self.response_embs[qid][resp_a], dtype=torch.float32).view(-1)
        c_b = torch.as_tensor(self.response_embs[qid][resp_b], dtype=torch.float32).view(-1)
        y = torch.tensor(float(y_ab), dtype=torch.float32)
        meta = (qid, resp_a, resp_b, cid_a, cid_b, cnt_a, cnt_b)
        return h, c_a, c_b, y, meta


class RegularizedResponseRankingDataset(Dataset):
    """Prebuilt response-pair dataset for ranking with base+regularization samples."""

    def __init__(
        self,
        H,
        pairs: List[Tuple[int, np.ndarray, np.ndarray, float, Tuple[Any, ...]]],
        total_pairs_before_cap: int,
        total_pairs_after_cap: int,
        total_pairs_dropped: int,
        base_base_pairs_before_cap: int,
        base_reg_pairs_before_cap: int,
        base_base_pairs_after_cap: int,
        base_reg_pairs_after_cap: int,
    ) -> None:
        self.H = torch.as_tensor(H, dtype=torch.float32)
        self.pairs = pairs
        self.total_pairs_before_cap = int(total_pairs_before_cap)
        self.total_pairs_after_cap = int(total_pairs_after_cap)
        self.total_pairs_dropped = int(total_pairs_dropped)
        self.base_base_pairs_before_cap = int(base_base_pairs_before_cap)
        self.base_reg_pairs_before_cap = int(base_reg_pairs_before_cap)
        self.base_base_pairs_after_cap = int(base_base_pairs_after_cap)
        self.base_reg_pairs_after_cap = int(base_reg_pairs_after_cap)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        qid, emb_a, emb_b, y_ab, meta = self.pairs[idx]
        h = self.H[int(qid)]
        c_a = torch.as_tensor(np.asarray(emb_a, dtype=np.float32), dtype=torch.float32).view(-1)
        c_b = torch.as_tensor(np.asarray(emb_b, dtype=np.float32), dtype=torch.float32).view(-1)
        y = torch.tensor(float(y_ab), dtype=torch.float32)
        return h, c_a, c_b, y, meta


def build_regularized_response_ranking_dataset(
    H,
    semantic_ids: np.ndarray,
    response_concat_dataset: Dataset,
    max_pairs_per_question: Optional[int] = None,
) -> RegularizedResponseRankingDataset:
    if not isinstance(response_concat_dataset, ConcatDataset):
        raise TypeError(
            "Expected response_concat_dataset to be ConcatDataset built by build_dataset(..., response-wise)."
        )
    if max_pairs_per_question is not None and int(max_pairs_per_question) <= 0:
        raise ValueError("max_pairs_per_question must be a positive integer when provided.")

    semantic_ids_np = np.asarray(semantic_ids, dtype=np.int64)
    if semantic_ids_np.ndim != 2:
        raise ValueError(f"semantic_ids must be 2D, got shape={semantic_ids_np.shape}")
    n_questions = int(semantic_ids_np.shape[0])

    counts_by_qid: List[Dict[int, int]] = []
    for qid in range(n_questions):
        cids, cnts = np.unique(semantic_ids_np[qid], return_counts=True)
        counts_by_qid.append({int(cid): int(cnt) for cid, cnt in zip(cids, cnts)})

    samples_by_qid: Dict[int, List[Dict[str, Any]]] = {}

    def _parse_index_entry(
        entry: object,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        if not isinstance(entry, (list, tuple)):
            return None, None, None, None
        if len(entry) >= 4:
            return int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3])
        if len(entry) == 3:
            return int(entry[0]), int(entry[1]), int(entry[2]), None
        return None, None, None, None

    def _sample_sort_key(sample: Dict[str, Any]) -> Tuple[int, int, int, int]:
        kind_order = 0 if str(sample.get("kind", "base")) == "base" else 1
        resp_id = int(sample.get("resp_id", -1))
        cid = int(sample.get("cid", -1))
        reg_idx_raw = sample.get("reg_idx", None)
        reg_idx = -1 if reg_idx_raw is None else int(reg_idx_raw)
        return kind_order, resp_id, cid, reg_idx

    for ds in response_concat_dataset.datasets:
        index_entries = getattr(ds, "index", None)
        response_embs = getattr(ds, "response_embs", None)
        if index_entries is None or response_embs is None:
            continue
        index_offset = int(getattr(ds, "index_offset", 0))
        is_reg_attr = getattr(ds, "is_reg", None)

        for entry in index_entries:
            local_qid, resp_id, cid, reg_idx = _parse_index_entry(entry)
            if local_qid is None or resp_id is None or cid is None:
                continue
            if local_qid < 0 or local_qid >= len(response_embs):
                continue
            resp_row = response_embs[local_qid]
            if resp_row is None or resp_id < 0 or resp_id >= len(resp_row):
                continue
            emb_entry = resp_row[resp_id]
            emb_val = emb_entry
            reg_idx_int = None if reg_idx is None else int(reg_idx)
            if isinstance(emb_entry, (list, tuple)):
                if reg_idx_int is None:
                    reg_idx_int = 0
                if reg_idx_int < 0 or reg_idx_int >= len(emb_entry):
                    continue
                emb_val = emb_entry[reg_idx_int]
            if emb_val is None:
                continue

            global_qid = int(local_qid + index_offset)
            base_qid = int(global_qid % n_questions)
            is_reg_sample = global_qid >= n_questions
            if (not is_reg_sample) and is_reg_attr is not None:
                try:
                    is_reg_sample = bool(np.asarray(is_reg_attr)[int(local_qid)])
                except Exception:
                    try:
                        is_reg_sample = bool(is_reg_attr[int(local_qid)])
                    except Exception:
                        is_reg_sample = bool(is_reg_attr)

            kind = "reg" if is_reg_sample else "base"
            count_val = 0
            if kind == "base":
                count_val = int(counts_by_qid[base_qid].get(int(cid), 0))

            sample_info = {
                "kind": kind,
                "resp_id": int(resp_id),
                "cid": int(cid),
                "reg_idx": reg_idx_int if kind == "reg" else None,
                "count": int(count_val),
                "emb": np.asarray(emb_val, dtype=np.float32),
            }
            samples_by_qid.setdefault(base_qid, []).append(sample_info)

    prebuilt_pairs: List[Tuple[int, np.ndarray, np.ndarray, float, Tuple[Any, ...]]] = []
    total_pairs_before_cap = 0
    total_pairs_after_cap = 0
    base_base_pairs_before_cap = 0
    base_reg_pairs_before_cap = 0
    base_base_pairs_after_cap = 0
    base_reg_pairs_after_cap = 0

    for qid in range(n_questions):
        q_samples = samples_by_qid.get(qid, [])
        if not q_samples:
            continue
        q_samples = sorted(q_samples, key=_sample_sort_key)
        question_candidates: List[Tuple[int, np.ndarray, np.ndarray, float, Tuple[Any, ...], str]] = []

        n_q_samples = len(q_samples)
        for a_idx in range(n_q_samples):
            sample_a = q_samples[a_idx]
            for b_idx in range(a_idx + 1, n_q_samples):
                sample_b = q_samples[b_idx]
                kind_a = str(sample_a["kind"])
                kind_b = str(sample_b["kind"])
                if kind_a == "reg" and kind_b == "reg":
                    continue

                cnt_a = int(sample_a["count"])
                cnt_b = int(sample_b["count"])
                if kind_a == "base" and kind_b == "base":
                    if int(sample_a["cid"]) == int(sample_b["cid"]) or cnt_a == cnt_b:
                        continue
                    pair_kind = "base_base"
                else:
                    if cnt_a == cnt_b:
                        continue
                    pair_kind = "base_reg"

                y_ab = 1.0 if cnt_a > cnt_b else 0.0
                meta = (
                    int(qid),
                    int(sample_a["resp_id"]),
                    int(sample_b["resp_id"]),
                    int(sample_a["cid"]),
                    int(sample_b["cid"]),
                    cnt_a,
                    cnt_b,
                    kind_a,
                    kind_b,
                    sample_a["reg_idx"],
                    sample_b["reg_idx"],
                    pair_kind,
                )
                question_candidates.append(
                    (
                        int(qid),
                        sample_a["emb"],
                        sample_b["emb"],
                        float(y_ab),
                        meta,
                        pair_kind,
                    )
                )

        q_pairs_before_cap = len(question_candidates)
        total_pairs_before_cap += q_pairs_before_cap
        for pair_entry in question_candidates:
            if pair_entry[5] == "base_base":
                base_base_pairs_before_cap += 1
            else:
                base_reg_pairs_before_cap += 1

        if (
            max_pairs_per_question is not None
            and q_pairs_before_cap > int(max_pairs_per_question)
        ):
            question_candidates = sorted(
                question_candidates,
                key=lambda p: (
                    -abs(int(p[4][5]) - int(p[4][6])),
                    int(p[4][1]),
                    int(p[4][3]),
                    -1 if p[4][9] is None else int(p[4][9]),
                    str(p[4][7]),
                    int(p[4][2]),
                    int(p[4][4]),
                    -1 if p[4][10] is None else int(p[4][10]),
                    str(p[4][8]),
                ),
            )[: int(max_pairs_per_question)]

        q_pairs_after_cap = len(question_candidates)
        total_pairs_after_cap += q_pairs_after_cap
        for pair_entry in question_candidates:
            if pair_entry[5] == "base_base":
                base_base_pairs_after_cap += 1
            else:
                base_reg_pairs_after_cap += 1
            prebuilt_pairs.append((pair_entry[0], pair_entry[1], pair_entry[2], pair_entry[3], pair_entry[4]))

    total_pairs_dropped = int(total_pairs_before_cap - total_pairs_after_cap)
    return RegularizedResponseRankingDataset(
        H=H,
        pairs=prebuilt_pairs,
        total_pairs_before_cap=total_pairs_before_cap,
        total_pairs_after_cap=total_pairs_after_cap,
        total_pairs_dropped=total_pairs_dropped,
        base_base_pairs_before_cap=base_base_pairs_before_cap,
        base_reg_pairs_before_cap=base_reg_pairs_before_cap,
        base_base_pairs_after_cap=base_base_pairs_after_cap,
        base_reg_pairs_after_cap=base_reg_pairs_after_cap,
    )


class WeightedClusterCountDataset(Dataset):
    def __init__(self, H, semantic_ids, cluster_embs, is_reg, 
                 c_dim: int, n_reg: int, index_offset: int = 0, 
                 metadata_map: Optional[Dict[Tuple[int, ...], Dict[str, int]]] = None,
                 norm: bool = False, allowed_qids: Optional[Set[int]] = None,
                 cnt_weighted: bool = False,
                 target_by_question: Optional[List[Dict[int, float]]] = None,
                 target_type: str = "sc_count",
                 domain_labels: Optional[np.ndarray] = None):
        """
        H            : np.ndarray [N, h_dim]  hidden states for chosen layer_range
        semantic_ids : np.ndarray [N, n_responses]  (cluster id per response)
        cluster_embs : list[dict[int -> np.ndarray(3072,)]]  (per-example cluster embeddings)
        is_reg       : np.ndarray[N,] quesiton-wise indicator vector of whether a sample is for regularization pairs
        c_dim        : int, e.g. 3072 for text-embedding-3-large
        n_reg        : int, number of regularization sampled per training question
        allowed_qids : optional set of question indices to keep; if provided, only items whose
                       (i + index_offset) belong to this set are retained.
        """
        self.H = torch.as_tensor(H, dtype=torch.float32)
        self.semantic_ids = semantic_ids
        self.cluster_embs = cluster_embs
        self.is_reg = is_reg
        self.c_dim = c_dim
        self.n_reg = n_reg
        self.index_offset = index_offset
        self.metadata_map = metadata_map if metadata_map is not None else {}
        self.norm = norm
        self.cnt_weighted = cnt_weighted
        self.target_by_question = target_by_question
        self.target_type = normalize_target_type(target_type)
        if domain_labels is not None:
            if torch.is_tensor(domain_labels):
                domain_labels = domain_labels.detach().cpu().numpy()
            self.domain_labels = np.asarray(domain_labels, dtype=np.int64)
            if len(self.domain_labels) != len(self.H):
                raise ValueError(
                    f"domain_labels length mismatch: {len(self.domain_labels)} vs {len(self.H)}"
                )
        else:
            self.domain_labels = None

        assert len(self.H) == len(self.semantic_ids) == len(self.cluster_embs)

        if allowed_qids is not None:
            self.index = [
                (i, cid)
                for i, d in enumerate(cluster_embs)
                if (i + self.index_offset) in allowed_qids
                for cid in d.keys()
            ]
        else:
            self.index = [(i, cid) for i, d in enumerate(cluster_embs) for cid in d.keys()]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        i, cid = self.index[idx] # i --> qid
        h = self.H[i]
        c = torch.tensor(self.cluster_embs[i][cid], dtype=torch.float32)
        if self.is_reg[i] == 0:
            cnt = float(np.count_nonzero(self.semantic_ids[i] == cid))
            total_cnt = float(self.semantic_ids.shape[1])
            if self.target_by_question is not None:
                y_base = float(self.target_by_question[i].get(int(cid), 0.0))
                if self.target_type == "sc_count" and self.norm and total_cnt > 0:
                    y_val = y_base / total_cnt
                else:
                    y_val = y_base
            else:
                if self.norm and total_cnt > 0:
                    y_val = cnt / total_cnt
                else:
                    y_val = cnt
            if self.cnt_weighted:
                w_val = float(cnt / total_cnt) if total_cnt > 0 else 0.0 # n_k / n
            else:
                denom = float(len(self.cluster_embs[i]))
                w_val = 1.0 / denom if denom > 0 else 0.0 # 1 / K or 1 / n_resp
        else:
            y_val = 0.0
            if len(self.semantic_ids[i]) > 0:
                w_val = 1.0 / float(self.n_reg)
            else:
                w_val = 0.0

        y = torch.tensor(y_val, dtype=torch.float32)
        w = torch.tensor(w_val, dtype=torch.float32)

        if self.domain_labels is None:
            return h, c, y, w, self.is_reg[i], (i + self.index_offset, cid)
        domain_label = int(self.domain_labels[i])
        return h, c, y, w, self.is_reg[i], domain_label, (i + self.index_offset, cid)

def weighted_cluster_count_collate(batch):
    """
    Batch items from WeightedClusterCountDataset.
    """
    has_domain = len(batch[0]) == 7
    H_list, C_list, y_list, w_list, is_reg_list, idx_list = [], [], [], [], [], []
    domain_list = [] if has_domain else None

    if has_domain:
        for h, c, y, w, is_reg, domain_label, idx in batch:
            H_list.append(h if isinstance(h, torch.Tensor) else torch.tensor(h, dtype=torch.float32))
            C_list.append(c if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32))
            y_list.append(y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32))
            w_list.append(w if isinstance(w, torch.Tensor) else torch.tensor(w, dtype=torch.float32))
            is_reg_list.append(is_reg if isinstance(is_reg, torch.Tensor) else torch.tensor(is_reg, dtype=torch.int))
            domain_list.append(
                domain_label if isinstance(domain_label, torch.Tensor)
                else torch.tensor(domain_label, dtype=torch.long)
            )
            idx_list.append(idx)
    else:
        for h, c, y, w, is_reg, idx in batch:
            H_list.append(h if isinstance(h, torch.Tensor) else torch.tensor(h, dtype=torch.float32))
            C_list.append(c if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=torch.float32))
            y_list.append(y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32))
            w_list.append(w if isinstance(w, torch.Tensor) else torch.tensor(w, dtype=torch.float32))
            is_reg_list.append(is_reg if isinstance(is_reg, torch.Tensor) else torch.tensor(is_reg, dtype=torch.int))
            idx_list.append(idx)

    Hb = torch.stack(H_list, dim=0)              # [B, h_dim]
    Cb = torch.stack(C_list, dim=0)              # [B, c_dim]
    yb = torch.stack(y_list, dim=0).view(-1)     # [B]
    wb = torch.stack(w_list, dim=0).view(-1)     # [B]
    is_regb = torch.stack(is_reg_list, dim=0).view(-1)

    if has_domain:
        domainb = torch.stack(domain_list, dim=0).view(-1)
        return Hb, Cb, yb, wb, is_regb, domainb, idx_list

    return Hb, Cb, yb, wb, is_regb, idx_list


def response_ranking_collate(batch):
    Hb = torch.stack([item[0] for item in batch], dim=0)
    C_ab = torch.stack([item[1] for item in batch], dim=0)
    C_bb = torch.stack([item[2] for item in batch], dim=0)
    yb = torch.stack([item[3] for item in batch], dim=0).view(-1)
    idxb = [item[4] for item in batch]
    return Hb, C_ab, C_bb, yb, idxb


def build_response_ranking_dataset(
    H,
    semantic_ids: np.ndarray,
    response_embs: List[Dict[int, np.ndarray]],
    c_dim: int,
    subset: bool = False,
    idx_subset: Optional[np.ndarray] = None,
    max_pairs_per_question: Optional[int] = None,
) -> ResponseRankingDataset:
    allowed_qids: Optional[Set[int]] = None
    if subset:
        if idx_subset is None:
            raise ValueError("subset=True but idx_subset is None for ranking dataset.")
        allowed_qids = set(int(i) for i in np.asarray(idx_subset, dtype=int).tolist())
    return ResponseRankingDataset(
        H=H,
        semantic_ids=semantic_ids,
        response_embs=response_embs,
        c_dim=c_dim,
        allowed_qids=allowed_qids,
        max_pairs_per_question=max_pairs_per_question,
    )

def build_dataset(
    H,                        # flattened hidden_states
    semantic_ids: np.ndarray,            # [N, n_responses]
    cluster_embs: List[Dict[int, np.ndarray]],  # len N, per-q dict[cid]->(c_dim,)
    c_dim, 
    MSE_type: str = "response-wise",
    n_reg: int = 1,                            # regularization samples per question/response
    max_cos: float = 0.6,
    seed: int = 42, 
    regularized: bool = True, 
    questions_jsonl_path: Optional[str] = None,
    se_resp_jsonl_path: Optional[str] = None,
    scan_output_path: Optional[str] = None,
    norm: bool = False,
    cnt_weighted: bool = False,
    subset: bool = False,
    idx_subset: np.ndarray = None,
    output_dist: bool = False,
    reduce_scan: bool = False,
    skip_final_scan: bool = False,
    skip_sample_stats: bool = False,
    domain_labels: Optional[np.ndarray] = None,
    feature_type: str = "tbg_concat",
    response_hidden_embs: Optional[List[Dict[int, np.ndarray]]] = None,
    hidden_layer_count: Optional[int] = None,
    question_slt_embs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    target_type: str = "sc_count",
    response_sequence_probs: Optional[np.ndarray] = None,
) -> Tuple[
    ConcatDataset,
    List[np.ndarray],
    List[np.ndarray],
    List[Dict[str, object]],
    Dict[str, Any],
]:
    feature_type_norm = (feature_type or "tbg_concat").strip().lower()
    if feature_type_norm == "tbg":
        feature_type_norm = "tbg_concat"
    if feature_type_norm not in {"tbg_concat", "slt", "tbg_slt_hidden", "tbg_slt_concat"}:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    if (MSE_type or "").lower() == "ranking":
        # Ranking training uses ResponseRankingDataset; keep build_dataset on response-wise
        # behavior for evaluation and analysis utilities.
        MSE_type = "response-wise"

    target_type_norm = normalize_target_type(target_type)
    target_by_question = build_target_evidence_by_question(
        semantic_ids=semantic_ids,
        target_type=target_type_norm,
        response_sequence_probs=response_sequence_probs,
    )

    N = semantic_ids.shape[0]

    if subset:
        if idx_subset is None:
            raise ValueError("subset=True but idx_subset is None")
        # idx_subset refers to base-question ids in [0..N-1]
        selected_qids: Set[int] = set(int(i) for i in idx_subset.tolist())
        print(f"[Build Dataset] Using subset of questions: {sorted(selected_qids)}")
        selected_reg_qids: Set[int] = set(int(i + N) for i in selected_qids)
        allowed_qids_global: Set[int] = selected_qids.union(selected_reg_qids)
    else:
        selected_qids = set(range(N))
        selected_reg_qids = set(range(N, 2 * N))
        allowed_qids_global: Set[int] = selected_qids.union(selected_reg_qids)

    reg_metadata_map: Dict[Tuple[int, ...], Dict[str, int]] = {}
    response_reg_selection_by_qid: Dict[int, List[Dict[str, int]]] = {}
    response_embs: Optional[List[List[Optional[np.ndarray]]]] = None
    dataset_reg: Optional[Dataset] = None
    cos_sims_all, euc_dists_all = [], []
    max_cos_sim_selected, min_cos_sim_selected = 0.0, 1.0
    regularization_enabled = bool(regularized)
    effective_reduce_scan = bool(reduce_scan or (not regularization_enabled))
    effective_skip_final_scan = bool(skip_final_scan or (not regularization_enabled))

    if regularization_enabled:
        print(
            f"[Build Dataset] feature_type={feature_type_norm} | "
            f"Selecting {MSE_type} regularization samples..."
        )
    else:
        print(
            f"[Build Dataset] feature_type={feature_type_norm} | "
            "regularized=False, skipping regularization sample selection and "
            "duplicate/conflict scans."
        )

    if feature_type_norm == "slt":
        if MSE_type != "response-wise":
            raise ValueError("feature_type=slt is only supported with MSE_type='response-wise'.")
        if regularized:
            raise ValueError("feature_type=slt only supports non-regularized training (set regularized=False).")
        if response_hidden_embs is None:
            raise ValueError("feature_type=slt requires response_hidden_embs.")

        dataset_train = HiddenStateClusterCountDataset(
            response_hidden_embs=response_hidden_embs,
            semantic_ids=semantic_ids,
            is_reg=np.zeros(N, dtype=np.int64),
            index_offset=0,
            norm=norm,
            allowed_qids=selected_qids,
            target_by_question=target_by_question,
            target_type=target_type_norm,
            domain_labels=domain_labels,
        )
    elif feature_type_norm == "tbg_slt_hidden":
        if MSE_type != "response-wise":
            raise ValueError("feature_type=tbg_slt_hidden is only supported with MSE_type='response-wise'.")
        if regularized:
            raise ValueError(
                "feature_type=tbg_slt_hidden only supports non-regularized training "
                "(set regularized=False)."
            )
        if response_hidden_embs is None:
            raise ValueError("feature_type=tbg_slt_hidden requires response_hidden_embs.")
        if hidden_layer_count is None:
            raise ValueError("feature_type=tbg_slt_hidden requires hidden_layer_count.")

        composed_hidden_embs = _build_tbg_slt_hidden_features(
            H=H,
            response_hidden_embs=response_hidden_embs,
            semantic_ids=semantic_ids,
            hidden_layer_count=int(hidden_layer_count),
        )
        dataset_train = HiddenStateClusterCountDataset(
            response_hidden_embs=composed_hidden_embs,
            semantic_ids=semantic_ids,
            is_reg=np.zeros(N, dtype=np.int64),
            index_offset=0,
            norm=norm,
            allowed_qids=selected_qids,
            target_by_question=target_by_question,
            target_type=target_type_norm,
            domain_labels=domain_labels,
        )
    elif feature_type_norm == "tbg_slt_concat":
        if MSE_type != "response-wise":
            raise ValueError("feature_type=tbg_slt_concat is only supported with MSE_type='response-wise'.")
        if regularized:
            raise ValueError(
                "feature_type=tbg_slt_concat only supports non-regularized training "
                "(set regularized=False)."
            )
        if question_slt_embs is None:
            raise ValueError("feature_type=tbg_slt_concat requires question_slt_embs derived from X['slt'].")

        composed_hidden_embs = _build_tbg_slt_concat_features(
            H=H,
            question_slt_hidden=question_slt_embs,
            cluster_embs=cluster_embs,
            semantic_ids=semantic_ids,
        )
        dataset_train = HiddenStateClusterCountDataset(
            response_hidden_embs=composed_hidden_embs,
            semantic_ids=semantic_ids,
            is_reg=np.zeros(N, dtype=np.int64),
            index_offset=0,
            norm=norm,
            allowed_qids=selected_qids,
            target_by_question=target_by_question,
            target_type=target_type_norm,
            domain_labels=domain_labels,
        )
    elif MSE_type == "cluster-wise":
        dataset_train = WeightedClusterCountDataset(
            H=H,
            semantic_ids=semantic_ids,
            cluster_embs=cluster_embs,
            is_reg=np.zeros(N, dtype=np.int64),
            c_dim=c_dim,
            n_reg=n_reg,
            index_offset=0,
            norm=norm,
            cnt_weighted=cnt_weighted,
            allowed_qids=allowed_qids_global,
            target_by_question=target_by_question,
            target_type=target_type_norm,
            domain_labels=domain_labels,
        )

        if regularization_enabled:
            candidate_pool = {"embs": [], "qids": [], "cids": []}
            for j, d in enumerate(cluster_embs): # j-th question, cid-th cluster
                for cid, vec in d.items():
                    candidate_pool["embs"].append(vec.astype(np.float32))
                    candidate_pool["qids"].append(j)
                    candidate_pool["cids"].append(cid)

            if len(candidate_pool["embs"]) == 0:
                raise ValueError("No candidate cluster embeddings found.")

            # embedding matrix for all clusters
            E = np.stack(candidate_pool["embs"], axis=0).astype(np.float32)  # [n_total_cluster, c_dim]
            candidate_pool["qids"] = np.asarray(candidate_pool["qids"], dtype=np.int64)      # [n_total_cluster,]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            E_tensor = torch.from_numpy(E).to(device)
            E_norms = torch.norm(E_tensor, dim=1)
            E_norm_tensor = E_tensor / (E_norms[:, None] + 1e-12)
            candidate_qids_tensor = torch.from_numpy(candidate_pool["qids"]).to(device)
            rng = torch.Generator(device=torch.device(device))
            rng.manual_seed(seed)

            cluster_embs_reg: List[Dict[int, np.ndarray]] = [dict() for _ in range(N)] # per-question selected cluster ids
            semantic_ids_reg: List = []

            for i in tqdm(range(N)): # for all questions
                gt_cid = _dominant_cid(semantic_ids[i])
                gt_emb = cluster_embs[i][gt_cid]
                gt_emb_tensor = torch.from_numpy(gt_emb.astype(np.float32)).to(device)
                gt_norm = torch.norm(gt_emb_tensor)
                gt_emb_norm = gt_emb_tensor / (gt_norm + 1e-12)

                cos_sims = E_norm_tensor @ gt_emb_norm
                sims_masked = cos_sims.clone()
                sims_masked[candidate_qids_tensor == i] = float("-inf") # exclude same-question clusters
                sims_masked[cos_sims > max_cos] = float("-inf") # exclude too large cosine similarities

                valid_idx = torch.nonzero(torch.isfinite(sims_masked), as_tuple=False).squeeze(1)

                selected_indices: List[int] = []
                target = min(n_reg, valid_idx.numel())

                if target > 0:
                    first_idx_local = int(valid_idx[torch.argmax(cos_sims[valid_idx])].item())
                    selected_indices.append(first_idx_local)

                    while len(selected_indices) < target:
                        valid_idx_list = valid_idx.tolist()
                        candidate_pool_indices = [
                            idx
                            for idx in valid_idx_list
                            if idx not in selected_indices
                        ] # non-overlapping idx

                        if not candidate_pool_indices:
                            break

                        candidate_pool_indices = torch.tensor(
                            candidate_pool_indices, device=device, dtype=torch.long
                        )
                        candidate_added = False

                        for idx in candidate_pool_indices[torch.randperm(
                            candidate_pool_indices.numel(), generator=rng, device=device
                        )]:
                            idx_int = int(idx.item())
                            selected_norms = E_norm_tensor[
                                torch.tensor(selected_indices, device=device, dtype=torch.long)
                            ]
                            cos_to_selected = selected_norms @ E_norm_tensor[idx_int]

                            if torch.all(cos_to_selected <= max_cos).item():
                                selected_indices.append(idx_int)
                                candidate_added = True
                                break

                        if not candidate_added:
                            break

                chosen_idx = torch.tensor(selected_indices, device=device, dtype=torch.long) # chosen indices in candidate pool per question
                chosen_idx_np = np.asarray(selected_indices, dtype=int)

                semantic_ids_reg.append(list(range(len(selected_indices)))) # creating semantic ids for reg samples

                if chosen_idx.numel() > 0:
                    cos_vals = cos_sims[chosen_idx]
                    max_cos_sim_selected = max(max_cos_sim_selected, float(torch.max(cos_vals).item()))
                    min_cos_sim_selected = min(min_cos_sim_selected, float(torch.min(cos_vals).item()))

                cos_sims_all.append(cos_sims[chosen_idx].detach().cpu().numpy())
                if chosen_idx.numel() > 0:
                    base_norms = E_norms[chosen_idx]
                    cos_vals = cos_sims[chosen_idx]
                    dist_sq = (
                        gt_norm * gt_norm
                        + base_norms * base_norms
                        - 2.0 * cos_vals * gt_norm * base_norms
                    )
                    dist_sq = torch.clamp(dist_sq, min=0.0)
                    euc_selected = torch.sqrt(dist_sq).detach().cpu().numpy().astype(np.float32)
                else:
                    euc_selected = np.asarray([], dtype=np.float32)
                euc_dists_all.append(euc_selected)

                for t, idx_global in enumerate(chosen_idx_np):
                    cluster_embs_reg[i][t] = E[idx_global] # get all semantic clusters for the regularization sample
                    src_qid = int(candidate_pool["qids"][idx_global])
                    src_cid = int(candidate_pool["cids"][idx_global])
                    reg_metadata_map[(i + N, t)] = {"source_qid": src_qid, "source_cid": src_cid}

            dataset_reg = WeightedClusterCountDataset(
                H=H,
                semantic_ids=semantic_ids_reg,
                cluster_embs=cluster_embs_reg,
                is_reg=np.ones(N, dtype=np.int64),
                c_dim=c_dim,
                n_reg=n_reg,
                index_offset=N,
                metadata_map=reg_metadata_map,
                norm=norm,
                cnt_weighted=cnt_weighted,
                allowed_qids=allowed_qids_global,
                target_by_question=None,
                target_type=target_type_norm,
                domain_labels=domain_labels,
            )

    elif MSE_type == "response-wise":
        response_embs = []
        missing_embeddings = 0
        for qid in range(N):
            ids_row = semantic_ids[qid]
            resp_row: List[Optional[np.ndarray]] = []
            for resp_id in range(len(ids_row)):
                emb = cluster_embs[qid].get(int(resp_id))
                if emb is None:
                    missing_embeddings += 1
                    resp_row.append(None)
                    continue
                resp_row.append(np.asarray(emb, dtype=np.float32))
            response_embs.append(resp_row)

        if missing_embeddings:
            print(
                f"[Build Dataset] Warning: {missing_embeddings} responses missing response embeddings; "
                "skipping those responses."
            )

        cluster_counts = [len(np.unique(row)) for row in semantic_ids]

        dataset_train = WeightedResponseClusterCountDataset(
            H=H,
            semantic_ids=semantic_ids,
            response_embs=response_embs,
            is_reg=np.zeros(N, dtype=np.int64),
            c_dim=c_dim,
            n_reg=n_reg,
            index_offset=0,
            norm=norm,
            cnt_weighted=cnt_weighted,
            allowed_qids=allowed_qids_global,
            cluster_counts=cluster_counts,
            target_by_question=target_by_question,
            target_type=target_type_norm,
            domain_labels=domain_labels,
        )

        if regularization_enabled:
            candidate_pool = {"embs": [], "qids": [], "resp_ids": [], "cids": []}
            candidate_indices_by_qid: List[List[int]] = [[] for _ in range(N)]
            for qid, resp_row in enumerate(response_embs):
                for resp_id, emb in enumerate(resp_row):
                    if emb is None:
                        continue
                    pool_idx = len(candidate_pool["embs"])
                    candidate_pool["embs"].append(emb.astype(np.float32))
                    candidate_pool["qids"].append(qid)
                    candidate_pool["resp_ids"].append(resp_id)
                    candidate_pool["cids"].append(int(semantic_ids[qid][resp_id]))
                    candidate_indices_by_qid[qid].append(pool_idx)

            if len(candidate_pool["embs"]) == 0:
                raise ValueError("No candidate response embeddings found.")

            E = np.stack(candidate_pool["embs"], axis=0).astype(np.float32)
            candidate_pool["qids"] = np.asarray(candidate_pool["qids"], dtype=np.int64)
            candidate_pool["resp_ids"] = np.asarray(candidate_pool["resp_ids"], dtype=np.int64)
            candidate_pool["cids"] = np.asarray(candidate_pool["cids"], dtype=np.int64)
            response_embs_reg: List[List[List[np.ndarray]]] = [
                [[] for _ in range(len(response_embs[qid]))] for qid in range(N)
            ]
            used_reg_indices_by_qid: Dict[int, Set[int]] = {}
            candidate_question_ids = np.asarray(sorted(selected_qids), dtype=np.int64)
            sampled_question_count = min(200, int(candidate_question_ids.size))
            question_rng = np.random.default_rng(seed)
            print(
                "[Build Dataset] Response-wise candidate pool: "
                f"{sampled_question_count} sampled questions per base question"
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Build Dataset] Using device: {device}")
            E_tensor = torch.from_numpy(E).to(device)
            E_norms = torch.norm(E_tensor, dim=1)
            E_norm_tensor = E_tensor / (E_norms[:, None] + 1e-12)
            candidate_qids_tensor = torch.from_numpy(candidate_pool["qids"]).to(device)
            rng = torch.Generator(device=torch.device(device))
            rng.manual_seed(seed)

            if sampled_question_count > 0:
                for qid in tqdm(sorted(selected_qids)):
                    base_indices_list = candidate_indices_by_qid[qid]
                    if not base_indices_list:
                        continue

                    if sampled_question_count < candidate_question_ids.size:
                        sampled_qids = question_rng.choice(
                            candidate_question_ids,
                            size=sampled_question_count,
                            replace=False,
                        )
                    else:
                        sampled_qids = candidate_question_ids

                    candidate_indices_list: List[int] = []
                    for sampled_qid in sampled_qids.tolist():
                        candidate_indices_list.extend(candidate_indices_by_qid[int(sampled_qid)])
                    if not candidate_indices_list:
                        continue

                    candidate_idx_tensor = torch.tensor(
                        candidate_indices_list, device=device, dtype=torch.long
                    )
                    candidate_qids_local = candidate_qids_tensor[candidate_idx_tensor]
                    same_qid_mask = candidate_qids_local == int(qid)

                    base_idx_tensor = torch.tensor(
                        base_indices_list, device=device, dtype=torch.long
                    )
                    base_norm = E_norm_tensor[base_idx_tensor]
                    candidate_norm = E_norm_tensor[candidate_idx_tensor]
                    cos_batch = base_norm @ candidate_norm.T
                    cos_batch = cos_batch.masked_fill(same_qid_mask.unsqueeze(0), float("-inf"))
                    cos_batch = cos_batch.masked_fill(cos_batch > max_cos, float("-inf"))

                    max_vals, _ = torch.max(cos_batch, dim=1)
                    valid_mask = torch.isfinite(max_vals)
                    if not torch.any(valid_mask):
                        continue

                    valid_local = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                    used_indices = used_reg_indices_by_qid.setdefault(qid, set())
                    local_pos_by_global = {
                        int(idx_global): pos for pos, idx_global in enumerate(candidate_indices_list)
                    }

                    for idx_pos in range(valid_local.numel()):
                        base_local = int(valid_local[idx_pos].item())
                        base_idx = int(base_indices_list[base_local])
                        resp_id = int(candidate_pool["resp_ids"][base_idx])

                        row = cos_batch[base_local].clone()
                        if used_indices:
                            for used_idx in used_indices:
                                local_pos = local_pos_by_global.get(int(used_idx))
                                if local_pos is not None:
                                    row[local_pos] = float("-inf")

                        valid_idx = torch.nonzero(torch.isfinite(row), as_tuple=False).squeeze(1)
                        if valid_idx.numel() == 0:
                            continue

                        target = min(n_reg, int(valid_idx.numel()))
                        if target <= 0:
                            continue

                        selected_local_indices: List[int] = []
                        if n_reg == 1:
                            first_local_idx = int(valid_idx[torch.argmax(row[valid_idx])].item())
                            selected_local_indices.append(first_local_idx)
                        else:
                            perm = torch.randperm(
                                valid_idx.numel(), generator=rng, device=device
                            )
                            chosen = valid_idx[perm[:target]]
                            selected_local_indices = [int(idx.item()) for idx in chosen]

                        if not selected_local_indices:
                            continue

                        cos_vals_response: List[float] = []
                        dist_vals_response: List[float] = []
                        base_norm_val = E_norms[base_idx]

                        for reg_idx, local_idx in enumerate(selected_local_indices):
                            idx_global = int(candidate_indices_list[local_idx])
                            used_indices.add(idx_global)
                            response_embs_reg[qid][resp_id].append(E[idx_global])

                            src_qid = int(candidate_pool["qids"][idx_global])
                            src_resp_id = int(candidate_pool["resp_ids"][idx_global])
                            src_cid = int(candidate_pool["cids"][idx_global])
                            reg_metadata_map[(qid + N, resp_id, reg_idx)] = {
                                "source_qid": src_qid,
                                "source_resp_id": src_resp_id,
                                "source_cid": src_cid,
                            }
                            response_reg_selection_by_qid.setdefault(qid, []).append(
                                {
                                    "base_resp_id": int(resp_id),
                                    "reg_idx": int(reg_idx),
                                    "source_qid": src_qid,
                                    "source_resp_id": src_resp_id,
                                    "source_cid": src_cid,
                                    "candidate_idx": idx_global,
                                }
                            )

                            cos_val = row[local_idx]
                            cos_val_f = float(cos_val.item())
                            max_cos_sim_selected = max(max_cos_sim_selected, cos_val_f)
                            min_cos_sim_selected = min(min_cos_sim_selected, cos_val_f)
                            cos_vals_response.append(cos_val_f)

                            cand_norm_val = E_norms[idx_global]
                            dist_sq = (
                                base_norm_val * base_norm_val
                                + cand_norm_val * cand_norm_val
                                - 2.0 * cos_val * base_norm_val * cand_norm_val
                            )
                            dist_sq = torch.clamp(dist_sq, min=0.0)
                            dist_vals_response.append(float(torch.sqrt(dist_sq).item()))

                        cos_sims_all.append(np.asarray(cos_vals_response, dtype=np.float32))
                        euc_dists_all.append(np.asarray(dist_vals_response, dtype=np.float32))

            if device == "cuda":
                E_tensor = E_tensor.cpu()
                E_norms = E_norms.cpu()
                E_norm_tensor = E_norm_tensor.cpu()
                candidate_qids_tensor = candidate_qids_tensor.cpu()
                torch.cuda.empty_cache()

            dataset_reg = WeightedResponseClusterCountDataset(
                H=H,
                semantic_ids=semantic_ids,
                response_embs=response_embs_reg,
                is_reg=np.ones(N, dtype=np.int64),
                c_dim=c_dim,
                n_reg=n_reg,
                index_offset=N,
                metadata_map=reg_metadata_map,
                norm=norm,
                cnt_weighted=cnt_weighted,
                allowed_qids=allowed_qids_global,
                cluster_counts=cluster_counts,
                target_by_question=None,
                target_type=target_type_norm,
                domain_labels=domain_labels,
            )

    else:
        raise ValueError(f"Unsupported MSE_type: {MSE_type}")

    if regularization_enabled:
        print(
            f"[Build Dataset] Max Cluster Cosine Similarity Sampled: {max_cos_sim_selected:.4f}, "
            f"Min Cluster Cosine Similarity Sampled: {min_cos_sim_selected:.4f}"
        )
    else:
        print("[Build Dataset] regularized=False: no regularization samples were constructed.")

    if regularized:
        if dataset_reg is None:
            raise ValueError("regularized=True requires a regularization dataset.")
        dataset_concat = ConcatDataset([dataset_train, dataset_reg])
    else:
        dataset_concat = ConcatDataset([dataset_train])

    datasets_list = list(dataset_concat.datasets)

    def _refresh_concat_dataset():
        nonlocal dataset_concat, datasets_list
        dataset_concat = ConcatDataset(datasets_list)
        datasets_list = list(dataset_concat.datasets)
    
    print(f"[Build Dataset] Dataset length: {len(dataset_concat)}")

    questions_data: Optional[List[str]] = None
    if questions_jsonl_path is not None:
        if os.path.exists(questions_jsonl_path):
            try:
                questions_data = load_question_json(questions_jsonl_path)
            except Exception as exc:
                print(
                    f"[Build Dataset] Warning: Failed to load questions from {questions_jsonl_path}: {exc}"
                )
        else:
            print(f"[Build Dataset] Warning: questions file not found at {questions_jsonl_path}")

    se_resp_data: Optional[List[List[str]]] = None
    if se_resp_jsonl_path is not None:
        if os.path.exists(se_resp_jsonl_path):
            try:
                se_resp_data = load_resp_json(se_resp_jsonl_path)
            except Exception as exc:
                print(
                    f"[Build Dataset] Warning: Failed to load semantic responses from {se_resp_jsonl_path}: {exc}"
                )
        else:
            print(f"[Build Dataset] Warning: semantic response file not found at {se_resp_jsonl_path}")

    def _get_question_text(qid: Optional[int]) -> Optional[str]:
        if qid is None:
            return None
        if questions_data is None:
            return None
        if not (0 <= int(qid) < len(questions_data)):
            return None
        return questions_data[int(qid)]


    def _get_semantic_pairs(qid: Optional[int], cid: Optional[int]) -> List[Tuple[str, int]]:
        if qid is None or cid is None:
            return []
        if se_resp_data is None:
            return []
        if qid >= len(se_resp_data) or qid >= len(semantic_ids):
            return []
        responses = se_resp_data[qid]
        ids_row = semantic_ids[qid]
        limit = min(len(responses), len(ids_row))
        pairs: List[Tuple[str, int]] = []
        for resp_idx in range(limit):
            sid = int(ids_row[resp_idx])
            if sid == int(cid):
                resp_text = responses[resp_idx]
                pairs.append((resp_text, sid))
        return pairs

    def _parse_index_entry(
        entry: object,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        qid = resp_id = cid = reg_idx = None
        if entry is None:
            return qid, resp_id, cid, reg_idx
        if isinstance(entry, (list, tuple)):
            if len(entry) >= 4:
                return entry[0], entry[1], entry[2], entry[3]
            if len(entry) == 3:
                return entry[0], entry[1], entry[2], reg_idx
            if len(entry) == 2:
                return entry[0], None, entry[1], reg_idx
            if len(entry) == 1:
                return entry[0], None, None, reg_idx
        return entry, None, None, reg_idx

    def _scan_dataset():
        scan_start = time.perf_counter()
        duplicate_distance_threshold = 1e-2
        duplicate_cosine_threshold = 1 - 1e-2
        forced_scan_device = os.getenv("EUP_SCAN_DEVICE", "").strip().lower()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if forced_scan_device == "cpu":
            device = torch.device("cpu")
        elif forced_scan_device == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
                print("[Build Dataset] EUP_SCAN_DEVICE=cuda but CUDA is unavailable; using CPU.")
        batch_size = min(1024, max(1, len(dataset_concat)))
        loader = DataLoader(
            dataset_concat,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=weighted_cluster_count_collate,
        )

        cumulative_sizes = dataset_concat.cumulative_sizes

        def _locate_dataset_entry(global_idx: int) -> Tuple[int, int]:
            dataset_idx = bisect_right(cumulative_sizes, global_idx)
            if dataset_idx >= len(datasets_list):
                dataset_idx = len(datasets_list) - 1
            prev_cum = 0 if dataset_idx == 0 else cumulative_sizes[dataset_idx - 1]
            sample_idx = global_idx - prev_cum
            return dataset_idx, sample_idx

        n_total = len(dataset_concat)
        features: List[torch.Tensor] = []
        y_vals = torch.empty(n_total, dtype=torch.float32)
        none_val = -1
        dataset_idx_arr = np.full(n_total, none_val, dtype=np.int32)
        sample_idx_arr = np.full(n_total, none_val, dtype=np.int32)
        qid_arr = np.full(n_total, none_val, dtype=np.int32)
        resp_id_arr = np.full(n_total, none_val, dtype=np.int32)
        cid_arr = np.full(n_total, none_val, dtype=np.int32)
        reg_idx_arr = np.full(n_total, none_val, dtype=np.int32)
        is_reg_arr = np.zeros(n_total, dtype=bool)
        source_qid_arr = np.full(n_total, none_val, dtype=np.int32)
        source_cid_arr = np.full(n_total, none_val, dtype=np.int32)
        source_resp_id_arr = np.full(n_total, none_val, dtype=np.int32)

        running_idx = 0
        for batch in loader:
            h_batch, c_batch, y_batch = batch[:3]

            h_batch = h_batch.to(device, non_blocking=True)
            c_batch = c_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            combined = torch.cat((h_batch, c_batch), dim=1)
            combined_cpu = combined.detach().to("cpu")
            y_cpu = y_batch.detach().to("cpu")

            for local_idx in range(combined_cpu.shape[0]):
                global_idx = running_idx + local_idx
                x_vector = combined_cpu[local_idx]
                y_val = float(y_cpu[local_idx].item())
                y_vals[global_idx] = y_cpu[local_idx]

                dataset_idx, sample_idx = _locate_dataset_entry(global_idx)
                dataset = datasets_list[dataset_idx]
                if hasattr(dataset, "index") and sample_idx < len(dataset.index):
                    qid, resp_id, cid, reg_idx = _parse_index_entry(dataset.index[sample_idx])
                else:
                    qid = resp_id = cid = reg_idx = None
                dataset_idx_arr[global_idx] = int(dataset_idx)
                sample_idx_arr[global_idx] = int(sample_idx)
                if qid is not None:
                    qid_arr[global_idx] = int(qid)
                if resp_id is not None:
                    resp_id_arr[global_idx] = int(resp_id)
                if cid is not None:
                    cid_arr[global_idx] = int(cid)
                if reg_idx is not None:
                    reg_idx_arr[global_idx] = int(reg_idx)

                is_reg_sample = False
                is_reg_attr = getattr(dataset, "is_reg", None)
                if qid is not None and is_reg_attr is not None:
                    if isinstance(is_reg_attr, np.ndarray):
                        if 0 <= int(qid) < len(is_reg_attr):
                            is_reg_sample = bool(is_reg_attr[int(qid)])
                        else:
                            is_reg_sample = bool(is_reg_attr.any())
                    else:
                        try:
                            is_reg_sample = bool(is_reg_attr[int(qid)])
                        except Exception:
                            is_reg_sample = bool(is_reg_attr)
                is_reg_arr[global_idx] = is_reg_sample

                source_qid = source_cid = source_resp_id = None
                metadata_map = getattr(dataset, "metadata_map", None)

                if (
                    qid is not None
                    and isinstance(metadata_map, dict)
                    and (cid is not None or resp_id is not None)
                ):
                    qid_int = int(qid)
                    cid_int = int(cid) if cid is not None else None
                    resp_id_int = int(resp_id) if resp_id is not None else None
                    reg_idx_int = int(reg_idx) if reg_idx is not None else None

                    # Try with index_offset first (for reg dataset), then without (for safety)
                    qid_with_offset = qid_int
                    try:
                        qid_with_offset += int(getattr(dataset, "index_offset", 0))
                    except Exception:
                        pass

                    source_meta = None
                    if resp_id_int is not None:
                        if reg_idx_int is not None:
                            source_meta = metadata_map.get((qid_with_offset, resp_id_int, reg_idx_int))
                            if source_meta is None:
                                source_meta = metadata_map.get((qid_int, resp_id_int, reg_idx_int))
                        if source_meta is None:
                            source_meta = metadata_map.get((qid_with_offset, resp_id_int))
                            if source_meta is None:
                                source_meta = metadata_map.get((qid_int, resp_id_int))
                    if source_meta is None:
                        source_meta = metadata_map.get((qid_with_offset, cid_int))
                        if source_meta is None:
                            source_meta = metadata_map.get((qid_int, cid_int))

                    if isinstance(source_meta, dict):
                        source_qid = source_meta.get("source_qid")
                        source_cid = source_meta.get("source_cid")
                        source_resp_id = source_meta.get("source_resp_id")

                if source_qid is not None:
                    source_qid_arr[global_idx] = int(source_qid)
                if source_cid is not None:
                    source_cid_arr[global_idx] = int(source_cid)
                if source_resp_id is not None:
                    source_resp_id_arr[global_idx] = int(source_resp_id)

                features.append(x_vector)


            running_idx += combined_cpu.shape[0]

        X_concat_cpu = torch.stack(features, dim=0)

        def _rowwise_sq_norms(x: torch.Tensor, block_size: int) -> torch.Tensor:
            n_rows = x.shape[0]
            out = torch.empty((n_rows, 1), device=x.device, dtype=x.dtype)
            for start in range(0, n_rows, block_size):
                end = min(start + block_size, n_rows)
                block = x[start:end]
                out[start:end] = torch.sum(block * block, dim=1, keepdim=True)
            return out

        def _rowwise_norms(x: torch.Tensor, block_size: int) -> torch.Tensor:
            n_rows = x.shape[0]
            out = torch.empty((n_rows, 1), device=x.device, dtype=x.dtype)
            for start in range(0, n_rows, block_size):
                end = min(start + block_size, n_rows)
                block = x[start:end]
                out[start:end] = torch.norm(block, dim=1, keepdim=True)
            return out

        scan_device = device

        print(f"[Build Dataset] Duplicate & conflict scan using device: {scan_device}.")

        # using tiled block-wise calculation to avoid OOM
        n_samples = X_concat_cpu.shape[0]
        norm_block_size = min(1024, max(1, n_samples))
        norms_cpu = _rowwise_sq_norms(X_concat_cpu, norm_block_size)
        x_norms_cpu = _rowwise_norms(X_concat_cpu, norm_block_size)

        def _tile_block_size(total: int, dim: int) -> int:
            max_block = 4096
            if total <= 0:
                return 1
            if scan_device.type != "cuda":
                return min(total, max_block)
            try:
                free_mem, _total_mem = torch.cuda.mem_get_info(scan_device)
            except Exception:
                free_mem = None
            if free_mem is None or free_mem <= 0:
                return min(total, max_block)

            bytes_per_float = 4
            max_bytes = int(free_mem * 0.02)
            if max_bytes <= bytes_per_float:
                return 1
            max_elems = max_bytes / float(bytes_per_float)
            m_val = max_elems / 2.0
            disc = float(dim * dim) + 4.0 * m_val
            if disc <= 0.0:
                return 1
            block = int((math.sqrt(disc) - float(dim)) / 2.0)
            if block < 1:
                block = 1
            return min(total, max_block, block)

        feat_dim = int(X_concat_cpu.shape[1])
        block_size = _tile_block_size(n_samples, feat_dim)
        print(f"[Build Dataset] Duplicate & conflict scan block size: {block_size}")

        parent = list(range(n_samples))

        def _find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def _union(a: int, b: int):
            root_a = _find(a)
            root_b = _find(b)
            if root_a == root_b:
                return
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

        distance_thresh_sq = duplicate_distance_threshold * duplicate_distance_threshold
        for i_start in tqdm(
            range(0, n_samples, block_size),
            desc="[Build Dataset] Duplicate & conflict scan",
            total=math.ceil(n_samples / block_size),
        ):
            i_end = min(i_start + block_size, n_samples)
            Xi_cpu = X_concat_cpu[i_start:i_end]
            ni_sq_cpu = norms_cpu[i_start:i_end]
            ni_cpu = x_norms_cpu[i_start:i_end]

            if scan_device.type == "cuda":
                Xi = Xi_cpu.to(scan_device, non_blocking=True)
                ni_sq = ni_sq_cpu.to(scan_device, non_blocking=True)
                ni = ni_cpu.to(scan_device, non_blocking=True)
            else:
                Xi = Xi_cpu
                ni_sq = ni_sq_cpu
                ni = ni_cpu

            for j_start in range(i_start, n_samples, block_size):
                j_end = min(j_start + block_size, n_samples)
                Xj_cpu = X_concat_cpu[j_start:j_end]
                nj_sq_cpu = norms_cpu[j_start:j_end]
                nj_cpu = x_norms_cpu[j_start:j_end]

                if scan_device.type == "cuda":
                    Xj = Xj_cpu.to(scan_device, non_blocking=True)
                    nj_sq = nj_sq_cpu.to(scan_device, non_blocking=True)
                    nj = nj_cpu.to(scan_device, non_blocking=True)
                else:
                    Xj = Xj_cpu
                    nj_sq = nj_sq_cpu
                    nj = nj_cpu

                dot = Xi @ Xj.t()

                dist_sq = ni_sq + nj_sq.t() - 2.0 * dot
                dist_sq = torch.clamp(dist_sq, min=0.0)

                denom = ni * nj.t()
                cos_block = dot / (denom + 1e-12)
                cos_block = torch.clamp(cos_block, -1.0, 1.0)

                mask = (dist_sq < distance_thresh_sq) | (cos_block > duplicate_cosine_threshold)
                if i_start == j_start:
                    row_idx = torch.arange(i_start, i_end, device=scan_device).unsqueeze(1)
                    col_idx = torch.arange(j_start, j_end, device=scan_device).unsqueeze(0)
                    mask &= col_idx > row_idx

                if scan_device.type == "cuda":
                    mask_cpu = mask.detach().to("cpu")
                else:
                    mask_cpu = mask.detach()
                pairs_cpu = torch.nonzero(mask_cpu, as_tuple=False)
                if pairs_cpu.numel() > 0:
                    for k in range(pairs_cpu.shape[0]):
                        i_idx = int(pairs_cpu[k, 0].item())
                        j_idx = int(pairs_cpu[k, 1].item())
                        _union(i_start + i_idx, j_start + j_idx)

                if scan_device.type == "cuda":
                    del Xj, nj_sq, nj, dot, dist_sq, denom, cos_block, mask

            if scan_device.type == "cuda":
                del Xi, ni_sq, ni
                torch.cuda.empty_cache()

        def _meta_from_index(idx: int) -> ScanMetadata:
            qid_val = int(qid_arr[idx]) if qid_arr[idx] >= 0 else None
            resp_id_val = int(resp_id_arr[idx]) if resp_id_arr[idx] >= 0 else None
            cid_val = int(cid_arr[idx]) if cid_arr[idx] >= 0 else None
            reg_idx_val = int(reg_idx_arr[idx]) if reg_idx_arr[idx] >= 0 else None
            dataset_idx_val = int(dataset_idx_arr[idx]) if dataset_idx_arr[idx] >= 0 else None
            sample_idx_val = int(sample_idx_arr[idx]) if sample_idx_arr[idx] >= 0 else None
            is_reg_val = bool(is_reg_arr[idx])
            source_qid_val = int(source_qid_arr[idx]) if source_qid_arr[idx] >= 0 else None
            source_cid_val = int(source_cid_arr[idx]) if source_cid_arr[idx] >= 0 else None
            source_resp_id_val = (
                int(source_resp_id_arr[idx]) if source_resp_id_arr[idx] >= 0 else None
            )
            display_qid_val = (
                source_qid_val if is_reg_val and source_qid_val is not None else qid_val
            )
            display_cid_val = (
                source_cid_val if is_reg_val and source_cid_val is not None else cid_val
            )
            display_resp_id_val = (
                source_resp_id_val
                if is_reg_val and source_resp_id_val is not None
                else resp_id_val
            )
            return ScanMetadata(
                dataset_idx=dataset_idx_val,
                sample_idx=sample_idx_val,
                qid=qid_val,
                resp_id=resp_id_val,
                cid=cid_val,
                reg_idx=reg_idx_val,
                y_val=float(y_vals[idx].item()),
                is_reg=is_reg_val,
                display_qid=display_qid_val,
                display_resp_id=display_resp_id_val,
                display_cid=display_cid_val,
                source_qid=source_qid_val,
                source_cid=source_cid_val,
                source_resp_id=source_resp_id_val,
            )

        clusters: Dict[int, List[int]] = {}
        def _isclose(a: float, b: float, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
            return abs(a - b) <= (atol + rtol * abs(b))

        for idx in range(len(X_concat_cpu)):
            root = _find(idx)
            clusters.setdefault(root, []).append(idx)

        duplicate_count = 0
        duplicate_base_cnt = 0
        duplicate_reg_cnt = 0
        conflict_count = 0
        conflict_base_base_cnt = 0
        conflict_base_reg_cnt = 0
        conflicting_x: List[Tuple[float, ...]] = []
        x_metadata: Dict[Tuple[float, ...], List[ScanMetadata]] = {}
        xy_to_metadata: Dict[Tuple[Tuple[float, ...], float], List[ScanMetadata]] = {}

        for root_idx, indices in tqdm(
            clusters.items(),
            desc="[Build Dataset] Aggregate duplicate/conflict stats",
            total=len(clusters),
        ):
            if len(indices) <= 1:
                continue
            # rep_key = tuple(float(v) for v in np.round(Xy_concat[root_idx], decimals=6))
            label_groups: List[Tuple[float, List[int]]] = []
            for idx in indices:
                label_val = float(y_vals[idx].item())
                placed = False
                for group_idx, (existing_label, member_indices) in enumerate(label_groups):
                    if _isclose(existing_label, label_val):
                        label_groups[group_idx][1].append(idx)
                        placed = True
                        break
                if not placed:
                    label_groups.append((label_val, [idx]))

            has_conflict = len(label_groups) > 1
            has_duplicates = any(len(member_indices) > 1 for _, member_indices in label_groups)

            if not has_conflict and not has_duplicates:
                continue

            rep_vec = X_concat_cpu[root_idx]
            rep_key = tuple(
                float(v)
                for v in (torch.round(rep_vec * 1e6) / 1e6).tolist()
            )

            entries_by_idx = None
            if has_conflict:
                conflict_incr = len(label_groups) - 1
                conflict_count += conflict_incr
                has_base = any(not is_reg_arr[i] for i in indices)
                has_reg = any(is_reg_arr[i] for i in indices)
                if has_base and has_reg:
                    conflict_base_reg_cnt += conflict_incr
                else:
                    conflict_base_base_cnt += conflict_incr
                conflicting_x.append(rep_key)
                entries_all = [_meta_from_index(i) for i in indices]
                x_metadata[rep_key] = entries_all
                if has_duplicates:
                    entries_by_idx = dict(zip(indices, entries_all))

            for label_val, member_indices in label_groups:
                if len(member_indices) > 1:
                    if entries_by_idx is not None:
                        entries = [entries_by_idx[i] for i in member_indices]
                    else:
                        entries = [_meta_from_index(i) for i in member_indices]
                    duplicate_count += len(member_indices) - 1
                    base_count = sum(1 for entry in entries if not entry.is_reg)
                    reg_count = len(entries) - base_count
                    if base_count > 1:
                        duplicate_base_cnt += base_count - 1
                    if reg_count > 1:
                        duplicate_reg_cnt += reg_count - 1
                    xy_to_metadata[(rep_key, float(label_val))] = entries

        duplicate_metadata = {k: v for k, v in xy_to_metadata.items() if len(v) > 1}

        elapsed = time.perf_counter() - scan_start
        print(
            f"[Build Dataset] _scan_dataset time cost: {elapsed:.2f}s "
            f"({elapsed / 60.0:.2f} min)"
        )
        return {
            "duplicate_count": duplicate_count,
            "duplicate_base_cnt": duplicate_base_cnt,
            "duplicate_reg_cnt": duplicate_reg_cnt,
            "conflict_count": conflict_count,
            "conflict_base_base_cnt": conflict_base_base_cnt,
            "conflict_base_reg_cnt": conflict_base_reg_cnt,
            "conflicting_x": conflicting_x,
            "x_metadata": x_metadata,
            "duplicate_metadata": duplicate_metadata,
        }

    def _apply_removals(removals_per_dataset: Dict[int, Set[int]]) -> int:
        total_removed = 0
        for dataset_idx, removals in removals_per_dataset.items():
            if dataset_idx < 0 or dataset_idx >= len(datasets_list):
                continue
            dataset = datasets_list[dataset_idx]
            if not hasattr(dataset, "index"):
                continue
            keep_index = [entry for j, entry in enumerate(dataset.index) if j not in removals]
            removed_here = len(dataset.index) - len(keep_index)
            if removed_here > 0:
                total_removed += removed_here
                dataset.index = keep_index
        if total_removed > 0:
            _refresh_concat_dataset()
        return total_removed

    def _log_duplicate_details(
        duplicate_metadata: Dict[Tuple[Tuple[float, ...], float], List[ScanMetadata]]
    ):
        base_groups = []
        for meta_list in duplicate_metadata.values():
            base_entries = [m for m in meta_list if not m.is_reg]
            if len(base_entries) > 1:
                base_groups.append(base_entries)

        if not base_groups:
            return

        print("[Build Dataset] Duplicate base samples detected (before removal):")
        for group_idx, base_entries in enumerate(base_groups, start=1):
            y_val = base_entries[0].y_val
            print(f"  Group {group_idx}: {len(base_entries)} duplicates with y={y_val:.4f}")
            per_cluster: Dict[Tuple[int, int], int] = {}
            for entry in base_entries:
                qid = entry.display_qid
                cid = entry.display_cid
                if qid is None or cid is None:
                    continue
                per_cluster.setdefault((qid, cid), 0)
                per_cluster[(qid, cid)] += 1
            for (qid, cid), count in per_cluster.items():
                print(f"    qid={qid}, cid={cid}, duplicate_count={count}")
                question_text = None
                if questions_data is not None and qid is not None and 0 <= qid < len(questions_data):
                    question_text = questions_data[qid]
                if question_text is not None:
                    print(f"      Question: {question_text}")
                else:
                    print("      Question: <unavailable>")
                semantic_pairs = _get_semantic_pairs(qid, cid)
                if semantic_pairs:
                    print("      Semantic pairs:")
                    for resp_text, sid in semantic_pairs:
                        print(f"        - cluster_id={sid}: {resp_text}")
                elif se_resp_data is None:
                    print("      Semantic pairs: <unavailable>")
                else:
                    print("      Semantic pairs: []")

    scan_initial = None
    scan_after_duplicates = None
    scan_final = None

    def _release_scan(scan: Optional[Dict[str, Any]]) -> None:
        if not scan:
            return
        x_meta = scan.get("x_metadata")
        if isinstance(x_meta, dict):
            x_meta.clear()
        dup_meta = scan.get("duplicate_metadata")
        if isinstance(dup_meta, dict):
            dup_meta.clear()
        if "conflicting_x" in scan:
            scan["conflicting_x"] = []

    if effective_reduce_scan:
        print("[Build Dataset] reduce_scan=True: skipping initial duplicate/conflict scans.")
    else:
        scan_initial = _scan_dataset()
        scan_after_duplicates = scan_initial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if MSE_type != "response-wise" and scan_initial["duplicate_metadata"]:
            _log_duplicate_details(scan_initial["duplicate_metadata"])
            base_removals: Dict[int, Set[int]] = {}
            reg_removals: Dict[int, Set[int]] = {}
            base_dup_candidates = 0
            reg_dup_candidates = 0
            for meta_list in scan_initial["duplicate_metadata"].values():
                base_entries = [m for m in meta_list if not m.is_reg]
                reg_entries = [m for m in meta_list if m.is_reg]
                if len(base_entries) > 1:
                    base_dup_candidates += len(base_entries) - 1
                    for entry in base_entries[1:]:
                        dataset_idx = entry.dataset_idx
                        sample_idx = entry.sample_idx
                        if dataset_idx is None or sample_idx is None:
                            continue
                        base_removals.setdefault(int(dataset_idx), set()).add(int(sample_idx))
                if len(reg_entries) > 1:
                    reg_dup_candidates += len(reg_entries) - 1
                    for entry in reg_entries[1:]:
                        dataset_idx = entry.dataset_idx
                        sample_idx = entry.sample_idx
                        if dataset_idx is None or sample_idx is None:
                            continue
                        reg_removals.setdefault(int(dataset_idx), set()).add(int(sample_idx))

            if base_dup_candidates or reg_dup_candidates:
                print(
                    "[Build Dataset] Duplicate sample counts before removal -> "
                    f"base: {base_dup_candidates}, regularization: {reg_dup_candidates}"
                )

            total_removed = 0
            if base_removals:
                total_removed += _apply_removals(base_removals)
            if reg_removals:
                total_removed += _apply_removals(reg_removals)

            if total_removed > 0:
                print(
                    f"[Build Dataset] Removed {total_removed} duplicate samples. "
                    f"New length: {len(dataset_concat)}"
                )
                scan_after_duplicates = _scan_dataset()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        duplicate_count = scan_after_duplicates["duplicate_count"]
        duplicate_base_cnt = scan_after_duplicates.get("duplicate_base_cnt", 0)
        duplicate_reg_cnt = scan_after_duplicates.get("duplicate_reg_cnt", 0)

        print(
            "[Build Dataset] Duplicate (X, y) pairs in concat dataset: "
            f"{duplicate_count} (base: {duplicate_base_cnt}, reg: {duplicate_reg_cnt})"
        )
        if duplicate_base_cnt > 0:
            print(
                "[Build Dataset] Warning: Duplicate base samples found "
                f"(count={duplicate_base_cnt})"
            )
        
        conflict_count = scan_after_duplicates["conflict_count"]
        conflict_base_base_cnt = scan_after_duplicates.get("conflict_base_base_cnt", 0)
        conflict_base_reg_cnt = scan_after_duplicates.get("conflict_base_reg_cnt", 0)
        print(
            "[Build Dataset] Conflicting labels for identical X: "
            f"{conflict_count} (base-base: {conflict_base_base_cnt}, base-reg: {conflict_base_reg_cnt})"
        )

        conflicting_x = scan_after_duplicates["conflicting_x"]
        if conflicting_x:
            removals_per_dataset: Dict[int, Set[int]] = {}
            for x_bytes in conflicting_x:
                locations = scan_after_duplicates["x_metadata"].get(x_bytes, [])
                base_locations = [loc for loc in locations if not loc.is_reg]
                reg_locations = [loc for loc in locations if loc.is_reg]

                if base_locations and reg_locations:
                    # Base-reg conflict: remove only reg samples.
                    targets = reg_locations
                else:
                    # Base-base conflict (or only reg, if any): remove all samples for this X.
                    targets = locations

                for location in targets:
                    dataset_idx = location.dataset_idx
                    sample_idx = location.sample_idx
                    if dataset_idx is None or sample_idx is None:
                        continue
                    removals_per_dataset.setdefault(int(dataset_idx), set()).add(int(sample_idx))

            total_removed = _apply_removals(removals_per_dataset)

            if total_removed > 0:
                print(
                    f"[Build Dataset] Removed {total_removed} samples with conflicting labels. "
                    f"New length: {len(dataset_concat)}"
                )

        _release_scan(scan_initial)
        if scan_after_duplicates is not scan_initial:
            _release_scan(scan_after_duplicates)
        scan_initial = None
        scan_after_duplicates = None
        gc.collect()

    if effective_skip_final_scan:
        print("[Build Dataset] skip_final_scan=True: skipping post-removal scan.")
    else:
        scan_final = _scan_dataset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_duplicate_count = scan_final.get("duplicate_count", 0)
        final_duplicate_base_cnt = scan_final.get("duplicate_base_cnt", 0)
        final_duplicate_reg_cnt = scan_final.get("duplicate_reg_cnt", 0)
        final_conflict_count = scan_final.get("conflict_count", 0)
        final_conflict_base_base_cnt = scan_final.get("conflict_base_base_cnt", 0)
        final_conflict_base_reg_cnt = scan_final.get("conflict_base_reg_cnt", 0)

        print(
            "[Build Dataset] Post-removal scan -> "
            f"duplicates: {final_duplicate_count} (base: {final_duplicate_base_cnt}, "
            f"reg: {final_duplicate_reg_cnt}), "
            f"conflicts: {final_conflict_count} (base-base: {final_conflict_base_base_cnt}, "
            f"base-reg: {final_conflict_base_reg_cnt})"
        )
        _release_scan(scan_final)
        scan_final = None
        gc.collect()

    base_sample_count = 0
    reg_sample_count = 0
    for dataset in datasets_list:
        index_entries = getattr(dataset, "index", [])
        is_reg_attr = getattr(dataset, "is_reg", None)
        for entry in index_entries:
            qid, _resp_id, _cid, _reg_idx = _parse_index_entry(entry)
            if qid is None:
                continue
            try:
                is_reg_value = bool(is_reg_attr[int(qid)]) if is_reg_attr is not None else False
            except Exception:
                is_reg_value = bool(is_reg_attr) if is_reg_attr is not None else False
            if is_reg_value:
                reg_sample_count += 1
            else:
                base_sample_count += 1

    print(
        "[Build Dataset] Sample counts after duplicate/conflict removal -> "
        f"base: {base_sample_count}, regularization: {reg_sample_count}"
    )

    # get augmented cluster embeddings with counts
    aug_cluster_embs: List[Dict[int, Dict[str, object]]] = []

    semantic_counts_by_question: List[Dict[int, int]] = []
    for qid in range(N):
        counts: Dict[int, int] = {}
        unique_cids, cid_counts = np.unique(semantic_ids[qid], return_counts=True)
        for cid_val, cnt in zip(unique_cids, cid_counts):
            cid_int = int(cid_val)
            if cid_int in cluster_embs[qid]:
                counts[cid_int] = int(cnt)
        semantic_counts_by_question.append(counts)

    if MSE_type == "cluster-wise":
        reg_indices_by_question: Dict[int, Set[int]] = {}
        if hasattr(dataset_reg, "index"): # no offset here
            for qid, cid in getattr(dataset_reg, "index", []):
                reg_indices_by_question.setdefault(int(qid), set()).add(int(cid))

        for qid in range(N):
            # Keep all original cluster embeddings for this question with counts
            base_embeddings: Dict[int, Dict[str, object]] = {}
            for cid, emb in cluster_embs[qid].items():
                cid_int = int(cid)
                count_val = semantic_counts_by_question[qid].get(cid_int, 0)
                base_embeddings[cid_int] = {"count": count_val, "emb": emb}

            # Start new IDs after the max existing cid
            existing_ids = list(base_embeddings.keys())
            next_semantic_id = max(existing_ids) + 1 if existing_ids else 0

            reg_cids = sorted(reg_indices_by_question.get(qid, set()))
            augmented_embeddings = dict(base_embeddings)

            for reg_cid in reg_cids:
                metadata_key = (qid + getattr(dataset_reg, "index_offset", 0), reg_cid)
                reg_meta = reg_metadata_map.get(metadata_key, {})
                source_qid = reg_meta.get("source_qid")
                source_cid = reg_meta.get("source_cid")
                if source_qid is None or source_cid is None:
                    continue

                source_embeddings = cluster_embs[int(source_qid)]
                reg_vector = source_embeddings.get(int(source_cid))
                if reg_vector is None:
                    continue

                augmented_embeddings[next_semantic_id] = {"count": 0, "emb": reg_vector}
                next_semantic_id += 1

            aug_cluster_embs.append(augmented_embeddings)
    else:
        for qid in range(N):
            base_embeddings: Dict[int, Dict[str, object]] = {}
            for resp_id, emb in cluster_embs[qid].items():
                resp_idx = int(resp_id)
                count_val = 0
                if 0 <= resp_idx < semantic_ids.shape[1]:
                    cluster_id = int(semantic_ids[qid][resp_idx])
                    count_val = semantic_counts_by_question[qid].get(cluster_id, 0)
                base_embeddings[resp_idx] = {"count": count_val, "emb": emb}
            aug_cluster_embs.append(base_embeddings)


    # distance & similarity statistics for hidden-only and full input features
    sample_stats: Dict[str, Any] = {}
    if isinstance(H, torch.Tensor):
        original_hidden_matrix = H.detach().cpu().numpy().astype(np.float32)
    else:
        original_hidden_matrix = np.asarray(H, dtype=np.float32)
    default_h_dim = int(original_hidden_matrix.shape[1]) if original_hidden_matrix.ndim == 2 else 0
    h_dim = default_h_dim
    if len(dataset_concat) > 0:
        try:
            first_h = dataset_concat[0][0]
            if isinstance(first_h, torch.Tensor):
                h_dim = int(first_h.numel())
            else:
                h_dim = int(np.asarray(first_h).size)
        except Exception:
            h_dim = default_h_dim
    H_q = np.full((N, h_dim), np.nan, dtype=np.float32)

    def _flatten_response_embeddings() -> np.ndarray:
        rows: List[np.ndarray] = []
        if MSE_type == "response-wise" and response_embs is not None:
            for resp_row in response_embs:
                for emb in resp_row:
                    if emb is None:
                        continue
                    rows.append(np.asarray(emb, dtype=np.float32))
        else:
            for emb_dict in cluster_embs:
                for emb in emb_dict.values():
                    rows.append(np.asarray(emb, dtype=np.float32))
        if not rows:
            return np.empty((0, c_dim), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    original_response_embedding_matrix = _flatten_response_embeddings()

    if skip_sample_stats:
        print("[Build Dataset] skip_sample_stats=True: skipping distance/similarity stats.")
        sample_stats = {
            "skipped": True,
            "hidden": {
                "max_cos_per": np.full(N, np.nan, dtype=np.float32),
                "min_dist_per": np.full(N, np.nan, dtype=np.float32),
                "dist_matrix": np.empty((0, 0), dtype=np.float32),
                "cos_matrix": np.empty((0, 0), dtype=np.float32),
            },
            "full": {
                "max_cos_per": np.full(len(dataset_concat), np.nan, dtype=np.float32),
                "min_dist_per": np.full(len(dataset_concat), np.nan, dtype=np.float32),
                "dist_matrix": np.empty((0, 0), dtype=np.float32),
                "cos_matrix": np.empty((0, 0), dtype=np.float32),
                "feature_matrix": np.empty((0, 0), dtype=np.float32),
                "base_reg": {
                    "full_feature_base": np.empty((0, 0), dtype=np.float32),
                    "full_feature_reg": np.empty((0, 0), dtype=np.float32),
                    "cos_matrix": np.empty((0, 0), dtype=np.float32),
                    "dist_matrix": np.empty((0, 0), dtype=np.float32),
                    "cos_violation_count": 0,
                    "cos_violation_rate": float("nan"),
                    "dist_violation_count": 0,
                    "dist_violation_rate": float("nan"),
                    "cos_threshold": 0.8,
                    "dist_threshold": 0.01,
                },
            },
        }
    elif len(dataset_concat) > 0:
        features: List[np.ndarray] = []
        labels: List[float] = []
        sample_details: List[Dict[str, object]] = []

        cumulative_sizes = dataset_concat.cumulative_sizes

        def _locate_dataset_entry(global_idx: int) -> Tuple[int, int, Dataset]:
            dataset_idx = bisect_right(cumulative_sizes, global_idx)
            if dataset_idx >= len(datasets_list):
                dataset_idx = len(datasets_list) - 1
            prev_cum = 0 if dataset_idx == 0 else cumulative_sizes[dataset_idx - 1]
            sample_idx = global_idx - prev_cum
            return dataset_idx, sample_idx, datasets_list[dataset_idx]

        def _extract_sample_details(global_idx: int) -> Dict[str, object]:
            dataset_idx, sample_idx, dataset = _locate_dataset_entry(global_idx)
            qid = resp_id = cid = reg_idx = None
            if hasattr(dataset, "index") and sample_idx < len(dataset.index):
                qid, resp_id, cid, reg_idx = _parse_index_entry(dataset.index[sample_idx])

            is_reg_sample = False
            is_reg_attr = getattr(dataset, "is_reg", None)
            if qid is not None and is_reg_attr is not None:
                if isinstance(is_reg_attr, np.ndarray):
                    if 0 <= int(qid) < len(is_reg_attr):
                        is_reg_sample = bool(is_reg_attr[int(qid)])
                else:
                    try:
                        is_reg_sample = bool(is_reg_attr[int(qid)])
                    except Exception:
                        is_reg_sample = bool(is_reg_attr)

            qid_with_offset: Optional[int] = None
            if qid is not None:
                qid_with_offset = int(qid)
                try:
                    qid_with_offset += int(getattr(dataset, "index_offset", 0))
                except Exception:
                    pass

            source_qid = source_cid = source_resp_id = None
            metadata_map = getattr(dataset, "metadata_map", None)
            if metadata_map and (cid is not None or resp_id is not None):
                source_meta = None
                if resp_id is not None:
                    if reg_idx is not None and qid_with_offset is not None:
                        source_meta = metadata_map.get((qid_with_offset, resp_id, reg_idx))
                    if source_meta is None and reg_idx is not None:
                        source_meta = metadata_map.get((qid, resp_id, reg_idx))
                    if source_meta is None and qid_with_offset is not None:
                        source_meta = metadata_map.get((qid_with_offset, resp_id))
                    if source_meta is None:
                        source_meta = metadata_map.get((qid, resp_id))
                if source_meta is None:
                    if qid_with_offset is not None:
                        source_meta = metadata_map.get((qid_with_offset, cid))
                    if source_meta is None:
                        source_meta = metadata_map.get((qid, cid))
                if isinstance(source_meta, dict):
                    source_qid = source_meta.get("source_qid")
                    source_cid = source_meta.get("source_cid")
                    source_resp_id = source_meta.get("source_resp_id")


            cluster_qid = source_qid if source_qid is not None else qid
            cluster_cid = source_cid if source_cid is not None else cid
            question_text = _get_question_text(qid)
            semantic_pairs = _get_semantic_pairs(cluster_qid, cluster_cid)

            return {
                "global_idx": global_idx,
                "dataset_idx": dataset_idx,
                "sample_idx": sample_idx,
                "raw_qid": qid, 
                "raw_cid": cid, 
                "raw_resp_id": resp_id,
                "raw_reg_idx": reg_idx,
                "question_qid": qid,
                "cluster_qid": cluster_qid,
                "cluster_id": cluster_cid,
                "question_text": question_text,
                "semantic_pairs": semantic_pairs,
                "is_reg": is_reg_sample,
                "source_qid": source_qid,
                "source_cid": source_cid,
                "source_resp_id": source_resp_id,
            }

        def _print_sample_detail(label: str, detail: Dict[str, object]):
            qid_display = detail.get("question_qid")
            cid_display = detail.get("cluster_id")
            is_reg_display = bool(detail.get("is_reg", False))
            raw_qid = detail.get("raw_qid")
            raw_cid = detail.get("raw_cid")
            raw_resp_id = detail.get("raw_resp_id")
            if raw_resp_id is not None:
                print(
                    f"    Sample {label}: qid={qid_display}, resp_id={raw_resp_id}, "
                    f"cid={cid_display}, is_reg={is_reg_display}"
                )
            else:
                print(
                    f"    Sample {label}: qid={qid_display}, cid={cid_display}, is_reg={is_reg_display}"
                )
            if is_reg_display:
                if raw_resp_id is not None:
                    print(f"      (target H from qid={raw_qid}, resp_id={raw_resp_id}, cid={raw_cid})")
                else:
                    print(f"      (target H from qid={raw_qid}, cid={raw_cid})")

            question_text = _get_question_text(qid_display)
            if question_text is not None:
                print(f"      Question: {question_text}")
            else:
                print("      Question: <unavailable>")
            if detail.get("source_qid") is not None or detail.get("source_cid") is not None:
                print(
                    "      Regularization source: "
                    f"qid={detail.get('source_qid')}, "
                    f"resp_id={detail.get('source_resp_id')}, "
                    f"cid={detail.get('source_cid')}"
                )
            semantic_pairs = detail.get("semantic_pairs") or []
            if semantic_pairs:
                print("      Semantic pairs:")
                for resp_text, sid in semantic_pairs:
                    print(f"        - cluster_id={sid}: {resp_text}")
            elif se_resp_data is None:
                print("      Semantic pairs: <unavailable>")
            else:
                print("      Semantic pairs: []")

        # post-removal scan
        for idx in range(len(dataset_concat)):
            h_tensor, c_tensor, y_tensor = dataset_concat[idx][:3]
            h_np = h_tensor.detach().cpu().numpy().ravel().astype(np.float32)
            c_np = c_tensor.detach().cpu().numpy().ravel().astype(np.float32)

            features.append(np.concatenate((h_np, c_np), axis=0))
            labels.append(float(y_tensor.detach().cpu().item()))

            detail = _extract_sample_details(idx)
            sample_details.append(detail)

            qid = detail.get("question_qid")
            is_reg = bool(detail.get("is_reg", False))

            # Only store per-question hidden state for base samples (no offset)
            if qid is not None and not is_reg:
                qid_int = int(qid)
                if 0 <= qid_int < N:
                    H_q[qid_int] = h_np

        is_reg_flags = np.array(
            [bool(detail.get("is_reg", False)) for detail in sample_details],
            dtype=bool,
        )
        base_indices = np.where(~is_reg_flags)[0]
        reg_indices = np.where(is_reg_flags)[0]

        if features:
            full_feature_matrix = np.stack(features, axis=0).astype(np.float32)
        else:
            full_feature_matrix = np.empty((0, h_dim + c_dim), dtype=np.float32)

        full_feature_base = (
            full_feature_matrix[base_indices]
            if base_indices.size > 0
            else np.empty((0, full_feature_matrix.shape[1]), dtype=np.float32)
        )
        full_feature_reg = (
            full_feature_matrix[reg_indices]
            if reg_indices.size > 0
            else np.empty((0, full_feature_matrix.shape[1]), dtype=np.float32)
        )

        base_reg_cos = np.empty((0, 0), dtype=np.float32)
        base_reg_dist = np.empty((0, 0), dtype=np.float32)
        cos_threshold = 0.9
        dist_threshold = 0.01
        cos_violation_count = 0
        dist_violation_count = 0
        cos_violation_rate = float("nan")
        dist_violation_rate = float("nan")

        if full_feature_base.size > 0 and full_feature_reg.size > 0:
            base_sq = np.sum(full_feature_base ** 2, axis=1, keepdims=True)
            reg_sq = np.sum(full_feature_reg ** 2, axis=1, keepdims=True)
            dot = full_feature_base @ full_feature_reg.T
            denom = np.sqrt(base_sq) @ np.sqrt(reg_sq).T + 1e-12
            base_reg_cos = dot / denom
            base_reg_cos = np.clip(base_reg_cos, -1.0, 1.0)

            dist_sq = base_sq + reg_sq.T - 2.0 * dot
            dist_sq = np.maximum(dist_sq, 0.0)
            base_reg_dist = np.sqrt(dist_sq).astype(np.float32)

            cos_violation_mask = (base_reg_cos > cos_threshold).any(axis=1)
            dist_violation_mask = (base_reg_dist < dist_threshold).any(axis=1)
            cos_violation_count = int(np.sum(cos_violation_mask))
            dist_violation_count = int(np.sum(dist_violation_mask))
            cos_violation_rate = cos_violation_count / float(full_feature_base.shape[0])
            dist_violation_rate = dist_violation_count / float(full_feature_base.shape[0])

        X_norms = np.sum(full_feature_matrix ** 2, axis=1, keepdims=True)
        pairwise_sq = X_norms + X_norms.T - 2.0 * (full_feature_matrix @ full_feature_matrix.T)
        pairwise_sq = np.maximum(pairwise_sq, 0.0)
        dist_matrix_full = np.sqrt(pairwise_sq)
        np.fill_diagonal(dist_matrix_full, np.inf)

        norms = np.linalg.norm(full_feature_matrix, axis=1, keepdims=True) + 1e-12
        normalized = full_feature_matrix / norms
        cosine_matrix_full = normalized @ normalized.T
        cosine_matrix_full = np.clip(cosine_matrix_full, -1.0, 1.0, out=cosine_matrix_full)
        np.fill_diagonal(cosine_matrix_full, -np.inf)

        triu_idx = np.triu_indices(len(dataset_concat), k=1)
        distances_upper = dist_matrix_full[triu_idx]

        top_k = min(10, distances_upper.size)
        if top_k > 0:
            top_indices = np.argpartition(distances_upper, kth=top_k - 1)[:top_k]
            top_sorted = top_indices[np.argsort(distances_upper[top_indices])]

            nearest_pairs = []
            diff_label_count = 0
            for order, flat_idx in enumerate(top_sorted, start=1):
                i = triu_idx[0][flat_idx]
                j = triu_idx[1][flat_idx]
                dist_val = float(distances_upper[flat_idx])
                cos_val = float(cosine_matrix_full[i, j])
                label_diff = not np.isclose(labels[i], labels[j])
                diff_label_count += int(label_diff)
                nearest_pairs.append((order, i, j, dist_val, cos_val, label_diff))

            dist_vals = np.array([p[3] for p in nearest_pairs], dtype=np.float32)
            cos_vals = np.array([p[4] for p in nearest_pairs], dtype=np.float32)

            print("[Build Dataset] Top nearest pairs (post-conflict and duplicate removal):")
            for order, i, j, dist_val, cos_val, label_diff in nearest_pairs:
                y_i = labels[i]
                y_j = labels[j]
                print(
                    f"  Pair {order}: indices ({i}, {j}), "
                    f"Euclidean Distance={dist_val:.12f}, Cosine Similarity={cos_val:.12f}, "
                    f"y_i={y_i:.4f}, y_j={y_j:.4f}, Different y={label_diff}"
                )
                _print_sample_detail("A", sample_details[i])
                _print_sample_detail("B", sample_details[j])

            print(
                f"[Build Dataset] Euclidean Distance:  mean: {dist_vals.mean():.12f}, std: {dist_vals.std(ddof=0):.12f}"
            )
            print(
                f"[Build Dataset] Cosine Similarity:  mean: {cos_vals.mean():.12f}, std: {cos_vals.std(ddof=0):.12f}"
            )
            print(
                f"[Build Dataset] Nearest pair count with different y: {diff_label_count} out of {len(nearest_pairs)}"
            )

        # hidden-state-only max distance & max cosine for each sample
        valid_mask = ~np.isnan(H_q).all(axis=1)
        valid_idx = np.where(valid_mask)[0]
        H_valid = H_q[valid_mask]

        hidden_min_dist_per = np.full(N, np.nan, dtype=np.float32)
        hidden_max_cos_per  = np.full(N, np.nan, dtype=np.float32)
        hidden_dist_matrix = np.full((N, N), np.nan, dtype=np.float32)
        hidden_cos_matrix = np.full((N, N), np.nan, dtype=np.float32)

        if H_valid.shape[0] > 0:
            # Euclidean distances (hidden only)
            norms = np.sum(H_valid ** 2, axis=1, keepdims=True)
            pairwise_sq = norms + norms.T - 2.0 * (H_valid @ H_valid.T)
            pairwise_sq = np.maximum(pairwise_sq, 0.0)
            distances_hidden = np.sqrt(pairwise_sq)

            np.fill_diagonal(distances_hidden, np.inf)
            hidden_min_dist_per_valid = distances_hidden.min(axis=1)

            # Cosine similarities (hidden only)
            v_norms = np.linalg.norm(H_valid, axis=1, keepdims=True) + 1e-12
            cos_matrix = (H_valid @ H_valid.T) / (v_norms @ v_norms.T)
            cos_matrix = np.clip(cos_matrix, -1.0, 1.0)
            np.fill_diagonal(cos_matrix, -np.inf)
            hidden_max_cos_per_valid = cos_matrix.max(axis=1)

            hidden_min_dist_per[valid_idx] = hidden_min_dist_per_valid
            hidden_max_cos_per[valid_idx]  = hidden_max_cos_per_valid
            hidden_dist_matrix[np.ix_(valid_idx, valid_idx)] = distances_hidden.astype(np.float32)
            hidden_cos_matrix[np.ix_(valid_idx, valid_idx)] = cos_matrix.astype(np.float32)

        full_min_dist_per = np.full(len(dataset_concat), np.nan, dtype=np.float32)
        full_max_cos_per = np.full(len(dataset_concat), np.nan, dtype=np.float32)
        if base_reg_dist.size > 0 and base_reg_cos.size > 0:
            full_min_dist_per[base_indices] = base_reg_dist.min(axis=1).astype(np.float32)
            full_max_cos_per[base_indices] = base_reg_cos.max(axis=1).astype(np.float32)

        sample_stats = {
            "hidden": {
                "max_cos_per": hidden_max_cos_per.astype(np.float32),
                "min_dist_per": hidden_min_dist_per.astype(np.float32),
                "dist_matrix": hidden_dist_matrix,
                "cos_matrix": hidden_cos_matrix,
            },
            "full": {
                "max_cos_per": full_max_cos_per,
                "min_dist_per": full_min_dist_per,
                "dist_matrix": dist_matrix_full.astype(np.float32),
                "cos_matrix": cosine_matrix_full.astype(np.float32),
                "feature_matrix": full_feature_matrix.astype(np.float32),
                "base_reg": {
                    "full_feature_base": full_feature_base.astype(np.float32),
                    "full_feature_reg": full_feature_reg.astype(np.float32),
                    "cos_matrix": base_reg_cos.astype(np.float32),
                    "dist_matrix": base_reg_dist.astype(np.float32),
                    "cos_violation_count": cos_violation_count,
                    "cos_violation_rate": cos_violation_rate,
                    "dist_violation_count": dist_violation_count,
                    "dist_violation_rate": dist_violation_rate,
                    "cos_threshold": cos_threshold,
                    "dist_threshold": dist_threshold,
                },
            },
        }

        if output_dist:
            finite_mask = np.isfinite(hidden_min_dist_per)
            finite_qids = np.where(finite_mask)[0]

            if finite_qids.size > 0:
                top_k_close = min(100, finite_qids.size)
                sort_order = np.argsort(hidden_min_dist_per[finite_qids])
                top_qids = finite_qids[sort_order[:top_k_close]]

                print("[Build Dataset] Top questions by SMALLEST min hidden-state Euclidean distance to others:")
                for rank, qid_s in enumerate(top_qids, start=1):
                    # find a representative base sample detail for this question: cluster level --> question level
                    detail = next(
                        (
                            d for d in sample_details
                            if d.get("question_qid") == int(qid_s)
                            and not d.get("is_reg", False)
                        ),
                        None,
                    )
                    print(
                        f"  Rank {rank}: qid={qid_s}, "
                        f"min_hidden_dist={hidden_min_dist_per[qid_s]:.8f}, "
                        f"max_hidden_cos={hidden_max_cos_per[qid_s]:.8f}"
                    )
                    if detail is not None:
                        _print_sample_detail("Sample", detail)


    else:
        sample_stats = {
            "hidden": {
                "max_cos_per": np.full(N, np.nan, dtype=np.float32),
                "min_dist_per": np.full(N, np.nan, dtype=np.float32),
                "dist_matrix": np.full((N, N), np.nan, dtype=np.float32),
                "cos_matrix": np.full((N, N), np.nan, dtype=np.float32),
            },
            "full": {
                "max_cos_per": np.full(len(dataset_concat), np.nan, dtype=np.float32),
                "min_dist_per": np.full(len(dataset_concat), np.nan, dtype=np.float32),
                "dist_matrix": np.full((len(dataset_concat), len(dataset_concat)), np.nan, dtype=np.float32),
                "cos_matrix": np.full((len(dataset_concat), len(dataset_concat)), np.nan, dtype=np.float32),
                "feature_matrix": np.empty((0, 0), dtype=np.float32),
                "base_reg": {
                    "full_feature_base": np.empty((0, 0), dtype=np.float32),
                    "full_feature_reg": np.empty((0, 0), dtype=np.float32),
                    "cos_matrix": np.empty((0, 0), dtype=np.float32),
                    "dist_matrix": np.empty((0, 0), dtype=np.float32),
                    "cos_violation_count": 0,
                    "cos_violation_rate": float("nan"),
                    "dist_violation_count": 0,
                    "dist_violation_rate": float("nan"),
                    "cos_threshold": 0.8,
                    "dist_threshold": 0.01,
                },
            },
        }

    # Unique cluster-level statistics (based on cluster embeddings only)
    cluster_entries: List[Tuple[int, int, np.ndarray]] = []
    for qid, emb_dict in enumerate(cluster_embs):
        for cid, emb in emb_dict.items():
            cluster_entries.append((int(qid), int(cid), np.asarray(emb, dtype=np.float32)))

    if cluster_entries:
        cluster_ids = np.array([[qid, cid] for qid, cid, _ in cluster_entries], dtype=np.int64)
        if skip_sample_stats:
            print("[Build Dataset] skip_sample_stats=True: skipping unique cluster-level stats.")
            n_clusters = int(len(cluster_entries))
            cluster_dist_matrix = np.empty((0, 0), dtype=np.float32)
            cluster_cos_matrix = np.empty((0, 0), dtype=np.float32)
            cluster_min_dist = np.full(n_clusters, np.nan, dtype=np.float32)
            cluster_max_cos = np.full(n_clusters, np.nan, dtype=np.float32)
        else:
            cluster_matrix = np.stack([emb for _, _, emb in cluster_entries], axis=0)

            if cluster_matrix.shape[0] > 1:
                c_norms = np.sum(cluster_matrix ** 2, axis=1, keepdims=True)
                c_pairwise_sq = c_norms + c_norms.T - 2.0 * (cluster_matrix @ cluster_matrix.T)
                c_pairwise_sq = np.maximum(c_pairwise_sq, 0.0)
                cluster_dist_matrix = np.sqrt(c_pairwise_sq)
                np.fill_diagonal(cluster_dist_matrix, np.inf)
                cluster_min_dist = cluster_dist_matrix.min(axis=1).astype(np.float32)

                c_norm = np.linalg.norm(cluster_matrix, axis=1, keepdims=True) + 1e-12
                cluster_cos_matrix = (cluster_matrix @ cluster_matrix.T) / (c_norm @ c_norm.T)
                cluster_cos_matrix = np.clip(cluster_cos_matrix, -1.0, 1.0)
                np.fill_diagonal(cluster_cos_matrix, -np.inf)
                cluster_max_cos = cluster_cos_matrix.max(axis=1).astype(np.float32)

                cluster_dist_matrix = cluster_dist_matrix.astype(np.float32)
                cluster_cos_matrix = cluster_cos_matrix.astype(np.float32)
            else:
                cluster_dist_matrix = np.full((1, 1), np.nan, dtype=np.float32)
                cluster_cos_matrix = np.full((1, 1), np.nan, dtype=np.float32)
                cluster_min_dist = np.array([np.nan], dtype=np.float32)
                cluster_max_cos = np.array([np.nan], dtype=np.float32)
    else:
        cluster_ids = np.empty((0, 2), dtype=np.int64)
        cluster_dist_matrix = np.empty((0, 0), dtype=np.float32)
        cluster_cos_matrix = np.empty((0, 0), dtype=np.float32)
        cluster_min_dist = np.array([], dtype=np.float32)
        cluster_max_cos = np.array([], dtype=np.float32)

    sample_stats["clusters"] = {
        "min_dist_per": cluster_min_dist,
        "max_cos_per": cluster_max_cos,
        "dist_matrix": cluster_dist_matrix,
        "cos_matrix": cluster_cos_matrix,
        "ids": cluster_ids,
    }
    sample_stats["min_dist_per_unique_cluster"] = cluster_min_dist
    sample_stats["max_cos_per_unique_cluster"] = cluster_max_cos
    sample_stats["original_hidden_matrix"] = original_hidden_matrix
    sample_stats["original_response_embedding_matrix"] = original_response_embedding_matrix

    if MSE_type == "response-wise":
        sample_stats["response_reg_selection_by_qid"] = response_reg_selection_by_qid

    return dataset_concat, cos_sims_all, euc_dists_all, aug_cluster_embs, sample_stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EUP Training and Evaluation")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", type=str,
                        default="trivia_qa_split")
    parser.add_argument("--judge_type", type=str, default="rouge")
    parser.add_argument("--regularized", type=_bool, default=True)
    parser.add_argument("--loss_type", type=str, default="response-wise")
    parser.add_argument("--norm_cnt", type=_bool, default=False)
    parser.add_argument("--cnt_weighted", type=_bool, default=False)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--valid_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--test_frac", type=count_or_frac, default=0.1)
    parser.add_argument("--n_resp", type=int, default=10)
    parser.add_argument("--n_reg", type=int, default=1)
    # parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--max_cos", type=float, default=0.6)

    parser.add_argument("--sanity_check", type=_bool, default=True)

    args = parser.parse_args()

    resolved_model_name = _resolve_hf_model_name(args.model_name)
    model_id = args.model_name.split("/")[-1]
    dtype = torch.float16 if "llama-2-7b" in args.model_name.lower() else torch.bfloat16
    if _is_gemma_4_e4b_model_name(args.model_name):
        if Gemma4ForCausalLM is None:
            raise ImportError("Gemma4ForCausalLM is unavailable in the installed transformers package.")
        model = Gemma4ForCausalLM.from_pretrained(resolved_model_name, device_map="auto", torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(resolved_model_name, device_map="auto", torch_dtype=dtype)

    load_dir = os.path.join(
        WORKING_DIR, "input_hidden_states",
        f"{model_id}_{args.dataset_name}_{args.n_samples}_{args.judge_type}"
    )

    splits = ["train", "valid", "test"]
    semantic_ids, semantic_cluster_embeddings, se_scores, X, acc, baseline_scores = {}, {}, {}, {}, {}, {}
    for split in splits:
        semantic_ids[split], semantic_cluster_embeddings[split], se_scores[split], X[split], acc[split] = load_split(
            load_dir, split, args.n_resp, args.loss_type
        )
    
    num_layers = model.config.num_hidden_layers if "gemma" not in args.model_name.lower() else model.config.text_config.num_hidden_layers
    k = 5  # last k layers
    layer_range = torch.arange(num_layers - k, num_layers)

    h_dim = int(len(layer_range) * X["train"]["tbg"].shape[-1])
    c_dim = semantic_cluster_embeddings["train"][0][0].shape[0]
    layer_idx = _layer_index_list(layer_range)

    # extract hidden_states from layer_range and flatten input features
    X_input_train = torch.as_tensor(X["train"]["tbg"][layer_idx, :, :], dtype=torch.float32)
    X_flat_train = X_input_train.permute(1, 0, 2).contiguous().view(X_input_train.size(1), -1)
    X_flat_train = torch.nan_to_num(X_flat_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_flat_train = F.normalize(X_flat_train, p=2, dim=1) # normalization on hidden states
    N = X_flat_train.shape[0]

    # X_input_val = torch.as_tensor(X["valid"]["tbg"][layer_range, :, :], dtype=torch.float32)
    # X_flat_val = X_input_val.permute(1, 0, 2).contiguous().view(X_input_val.size(1), -1)
    # X_flat_val = torch.nan_to_num(X_flat_val, nan=0.0, posinf=0.0, neginf=0.0)
    # X_flat_val = F.normalize(X_flat_val, p=2, dim=1) # normalization on hidden states


    # X_input_test = torch.as_tensor(X["test"]["tbg"][layer_range, :, :], dtype=torch.float32)
    # X_flat_test = X_input_test.permute(1, 0, 2).contiguous().view(X_input_test.size(1), -1)
    # X_flat_test = torch.nan_to_num(X_flat_test, nan=0.0, posinf=0.0, neginf=0.0)
    # X_flat_test = F.normalize(X_flat_test, p=2, dim=1) # normalization on hidden states

    assert torch.sum(torch.isnan(X_flat_train)) == 0, "NaN values in X_flat_train"

    dataset_train_reg, cos_sims_all, euc_dists_all, aug_cluster_embeds_train_reg, sample_stats = build_dataset(
        H                    = X_flat_train,
        semantic_ids         = semantic_ids["train"],
        cluster_embs         = semantic_cluster_embeddings["train"],
        c_dim                = c_dim,
        MSE_type             = args.loss_type,
        n_reg                = args.n_reg,
        max_cos              = args.max_cos,
        seed                 = 42,
        regularized          = args.regularized,
        questions_jsonl_path = os.path.join(load_dir, "train", "questions.jsonl"),
        se_resp_jsonl_path   = os.path.join(
            load_dir,
            "train",
            "se_resp_res.jsonl" if args.loss_type == "response-wise" else "se_resp.jsonl",
        ),
        scan_output_path     = None,
        norm                 = args.norm_cnt,
        cnt_weighted         = args.cnt_weighted,
        skip_sample_stats    = False, 
        reduce_scan          = False, 
    )

    if args.sanity_check:
        def _summarize_feature_stats(label: str, stats: Dict[str, Any]) -> None:
            max_cos = np.asarray(stats.get("max_cos_per", []), dtype=np.float32)
            min_dist = np.asarray(stats.get("min_dist_per", []), dtype=np.float32)
            max_cos_finite = max_cos[np.isfinite(max_cos)]
            min_dist_finite = min_dist[np.isfinite(min_dist)]

            if max_cos_finite.size > 0:
                max_cos_max = float(np.max(max_cos_finite))
                max_cos_mean = float(np.mean(max_cos_finite))
            else:
                max_cos_max = float("nan")
                max_cos_mean = float("nan")

            if min_dist_finite.size > 0:
                min_dist_min = float(np.min(min_dist_finite))
                min_dist_mean = float(np.mean(min_dist_finite))
            else:
                min_dist_min = float("nan")
                min_dist_mean = float("nan")

            print(
                f"  {label}: "
                f"max(max_cos_per)={max_cos_max:.4f}, "
                f"mean(max_cos_per)={max_cos_mean:.4f}, "
                f"min(min_dist_per)={min_dist_min:.4f}, "
                f"mean(min_dist_per)={min_dist_mean:.4f}"
            )

        def _compute_preconcat_stats(
            name: str,
            matrix: np.ndarray,
            dist_thresh: float,
            cos_thresh: float,
        ) -> Dict[str, float]:
            n_rows = int(matrix.shape[0])
            if n_rows == 0:
                print(f"[Sanity check] {name}: no samples found.")
                return {
                    "unique_count": 0,
                    "unique_rate": float("nan"),
                    "near_count": 0,
                    "near_rate": float("nan"),
                    "norm_mean": float("nan"),
                    "norm_std": float("nan"),
                }

            matrix = np.asarray(matrix, dtype=np.float32)
            row_view = np.ascontiguousarray(matrix).view(
                np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1]))
            )
            unique_count = int(np.unique(row_view).size)
            unique_rate = unique_count / float(n_rows)

            norms = np.linalg.norm(matrix, axis=1)
            norm_mean = float(np.mean(norms))
            norm_std = float(np.std(norms, ddof=0))
            near_count = 0
            near_rate = 0.0
            if n_rows > 1:
                min_dist = np.full(n_rows, np.inf, dtype=np.float32)
                max_cos = np.full(n_rows, -np.inf, dtype=np.float32)
                norms_sq = norms * norms
                block = min(1024, n_rows)
                for start in range(0, n_rows, block):
                    end = min(start + block, n_rows)
                    block_vecs = matrix[start:end]
                    dot = block_vecs @ matrix.T
                    dist_sq = norms_sq[start:end, None] + norms_sq[None, :] - 2.0 * dot
                    dist_sq = np.maximum(dist_sq, 0.0)
                    diag_idx = np.arange(end - start)
                    dist_sq[diag_idx, start + diag_idx] = np.inf
                    dist_block = np.sqrt(dist_sq, dtype=np.float32)
                    min_dist[start:end] = np.minimum(min_dist[start:end], dist_block.min(axis=1))

                    denom = (norms[start:end, None] * norms[None, :]) + 1e-12
                    cos_block = dot / denom
                    cos_block = np.clip(cos_block, -1.0, 1.0)
                    cos_block[diag_idx, start + diag_idx] = -np.inf
                    max_cos[start:end] = np.maximum(max_cos[start:end], cos_block.max(axis=1))

                near_mask = (min_dist < dist_thresh) | (max_cos > cos_thresh)
                near_count = int(np.sum(near_mask))
                near_rate = near_count / float(n_rows)

            print(
                f"[Sanity check] {name}: unique={unique_count}/{n_rows} "
                f"(rate={unique_rate:.4f}), near={near_count}/{n_rows} "
                f"(rate={near_rate:.4f}), "
                f"norm_mean={norm_mean:.4f}, norm_std={norm_std:.4f}"
            )
            return {
                "unique_count": unique_count,
                "unique_rate": unique_rate,
                "near_count": near_count,
                "near_rate": near_rate,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            }

        hidden_matrix = sample_stats.get("original_hidden_matrix")
        resp_matrix = sample_stats.get("original_response_embedding_matrix")
        dist_thresh = 1e-2
        cos_thresh = 1 - 1e-2
        if isinstance(hidden_matrix, np.ndarray) and isinstance(resp_matrix, np.ndarray):
            print("[Sanity check] Pre-concatenation statistics:")
            pre_concat_stats = {
                "hidden": _compute_preconcat_stats("hidden", hidden_matrix, dist_thresh, cos_thresh),
                "responses": _compute_preconcat_stats("responses", resp_matrix, dist_thresh, cos_thresh),
                "dist_threshold": dist_thresh,
                "cos_threshold": cos_thresh,
            }
            sample_stats["pre_concat_stats"] = pre_concat_stats

        print("[Sanity check] In-concatenation feature similarity summary:")
        summary_targets = [("hidden", "hidden"), ("full", "full")]
        cluster_label = "responses" if args.loss_type == "response-wise" else "clusters"
        summary_targets.append(("clusters", cluster_label))

        for key, label in summary_targets:
            stats = sample_stats.get(key)
            if isinstance(stats, dict):
                _summarize_feature_stats(label, stats)

        base_reg_stats = sample_stats.get("full", {}).get("base_reg", {})
        if isinstance(base_reg_stats, dict):
            base_mat = base_reg_stats.get("full_feature_base")
            reg_mat = base_reg_stats.get("full_feature_reg")
            n_base = int(base_mat.shape[0]) if isinstance(base_mat, np.ndarray) else 0
            n_reg = int(reg_mat.shape[0]) if isinstance(reg_mat, np.ndarray) else 0
            cos_count = int(base_reg_stats.get("cos_violation_count", 0))
            dist_count = int(base_reg_stats.get("dist_violation_count", 0))
            cos_rate = float(base_reg_stats.get("cos_violation_rate", float("nan")))
            dist_rate = float(base_reg_stats.get("dist_violation_rate", float("nan")))
            cos_threshold = float(base_reg_stats.get("cos_threshold", 0.8))
            dist_threshold = float(base_reg_stats.get("dist_threshold", 0.01))

            print(
                "Post-removal summary:\n"
                "[Sanity check] Base-reg cosine violations: "
                f"{cos_count}/{n_base} (rate={cos_rate:.4f}, threshold={cos_threshold:.2f})"
            )
            print(
                "[Sanity check] Base-reg L2 violations: "
                f"{dist_count}/{n_base} (rate={dist_rate:.4f}, threshold={dist_threshold:.4f})"
            )
            print(
                "[Sanity check] Base-reg feature matrices: "
                f"N_base={n_base}, N_reg={n_reg}"
            )

        def _sanity_check_aug_cluster_uniqueness(aug_cluster_embs: List[Dict[int, Dict[str, object]]]) -> None:
            entry_map: Dict[bytes, Dict[str, object]] = {}
            total_rows = 0

            for qid, emb_dict in enumerate(aug_cluster_embs):
                for payload in emb_dict.values():
                    emb = payload.get("emb") if isinstance(payload, dict) else None
                    if emb is None:
                        continue
                    emb_arr = np.asarray(emb, dtype=np.float32)
                    key = emb_arr.tobytes()
                    entry = entry_map.setdefault(key, {"qids": set(), "count": 0})
                    entry["qids"].add(int(qid))
                    entry["count"] += 1
                    total_rows += 1

            if total_rows == 0:
                print("[Sanity check] Augmented cluster embeddings uniqueness: no samples found.")
                return

            unique_rows = 0
            for entry in entry_map.values():
                if len(entry["qids"]) == 1:
                    unique_rows += int(entry["count"])

            rate = float(unique_rows) / float(total_rows)
            print(
                "[Sanity check] Augmented cluster embeddings uniqueness rate (qid-aware): "
                f"{rate:.4f} ({unique_rows}/{total_rows})"
            )

        _sanity_check_aug_cluster_uniqueness(aug_cluster_embeds_train_reg)

    # dataset_val_reg, _, _, aug_cluster_embeds_val_reg, _ = build_dataset(
    #     H                    = X_flat_val,
    #     semantic_ids         = semantic_ids["valid"],
    #     cluster_embs         = semantic_cluster_embeddings["valid"],
    #     c_dim                = c_dim,
    #     MSE_type             = args.loss_type,
    #     n_reg                = args.n_reg,
    #     gamma                = args.gamma if args.regularized else 0.0,
    #     max_cos              = args.max_cos,
    #     seed                 = 42,
    #     regularized          = args.regularized,
    #     questions_jsonl_path = os.path.join(load_dir, "valid", "questions.jsonl"),
    #     se_resp_jsonl_path   = os.path.join(
    #         load_dir,
    #         "valid",
    #         "se_resp_res.jsonl" if args.loss_type == "response-wise" else "se_resp.jsonl",
    #     ),
    #     scan_output_path     = None,
    #     norm                 = args.norm_cnt,
    #     cnt_weighted         = args.cnt_weighted,
    #     skip_sample_stats    = True, 
    # )

    # dataset_test_reg, _, _, aug_cluster_embeds_test_reg, _ = build_dataset(
    #     H                    = X_flat_test,
    #     semantic_ids         = semantic_ids["test"],
    #     cluster_embs         = semantic_cluster_embeddings["test"],
    #     c_dim                = c_dim,
    #     MSE_type             = args.loss_type,
    #     n_reg                = args.n_reg,
    #     gamma                = args.gamma if args.regularized else 0.0,
    #     max_cos              = args.max_cos,
    #     seed                 = 42,
    #     regularized          = args.regularized,
    #     questions_jsonl_path = os.path.join(load_dir, "test", "questions.jsonl"),
    #     se_resp_jsonl_path   = os.path.join(
    #         load_dir,
    #         "test",
    #         "se_resp_res.jsonl" if args.loss_type == "response-wise" else "se_resp.jsonl",
    #     ),
    #     scan_output_path     = None,
    #     norm                 = args.norm_cnt,
    #     cnt_weighted         = args.cnt_weighted,
    #     skip_sample_stats    = True, 
    # )

    pass
