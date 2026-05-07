from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


_BATCH_SIZE = 256
_MAX_EPOCHS = 1000
_TOL = 1e-4
_MLP_HIDDEN_DIM = 512
_LR_LR = 1e-2
_MLP_LR = 1e-3
_LOGIT_CLIP = 60.0
_TEMP_MAX_ITER = 100


class _LRProbe(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(int(input_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _MLPProbe(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = _MLP_HIDDEN_DIM) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchBinaryProbe:
    def __init__(
        self,
        probe_type: str,
        input_dim: int,
        random_state: int = 42,
        batch_size: int = _BATCH_SIZE,
        max_epochs: int = _MAX_EPOCHS,
        tol: float = _TOL,
    ) -> None:
        probe_kind = str(probe_type).strip().lower()
        if probe_kind not in {"lr", "mlp"}:
            raise ValueError(f"Unsupported probe_type: {probe_type!r}.")

        self.probe_type = probe_kind
        self.input_dim = int(input_dim)
        self.random_state = int(random_state)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.tol = float(tol)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _LRProbe(self.input_dim) if self.probe_type == "lr" else _MLPProbe(self.input_dim)
        self.model.to(self.device)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.converged_ = False
        self.n_iter_ = 0
        self._is_fitted = False

    def _seed_everything(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _lr_l2_strength(self, n_samples: int) -> float:
        if self.probe_type != "lr":
            return 0.0
        return 1.0 / float(max(1, int(n_samples)))

    def _make_optimizer(self, n_samples: int) -> torch.optim.Optimizer:
        if self.probe_type == "lr":
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith("bias"):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            param_groups = []
            if decay_params:
                param_groups.append(
                    {
                        "params": decay_params,
                        "weight_decay": self._lr_l2_strength(n_samples),
                    }
                )
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
            return torch.optim.Adam(param_groups, lr=_LR_LR)
        return torch.optim.Adam(self.model.parameters(), lr=_MLP_LR)

    @torch.no_grad()
    def _compute_epoch_loss(
        self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> float:
        self.model.eval()
        n_samples = int(X_tensor.shape[0])
        total_bce = 0.0
        for start in range(0, n_samples, self.batch_size):
            stop = min(start + self.batch_size, n_samples)
            xb = X_tensor[start:stop].to(self.device)
            yb = y_tensor[start:stop].to(self.device)
            logits = self.model(xb)
            batch_loss = criterion(logits, yb)
            total_bce += float(batch_loss.item()) * int(stop - start)

        mean_bce = total_bce / float(max(1, n_samples))
        if self.probe_type != "lr":
            return mean_bce

        l2_penalty = 0.0
        for name, param in self.model.named_parameters():
            if name.endswith("bias"):
                continue
            l2_penalty += float(torch.sum(param.detach() ** 2).item())
        return mean_bce + 0.5 * self._lr_l2_strength(n_samples) * l2_penalty

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchBinaryProbe":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {X_arr.shape}.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Feature/label count mismatch: features={X_arr.shape[0]}, labels={y_arr.shape[0]}."
            )
        if X_arr.shape[0] == 0:
            raise ValueError("TorchBinaryProbe.fit requires at least one training example.")
        if np.unique(y_arr).size < 2:
            raise ValueError("TorchBinaryProbe.fit requires both binary classes.")

        self._seed_everything()
        self.model.train()

        X_tensor = torch.from_numpy(X_arr)
        y_tensor = torch.from_numpy(y_arr.astype(np.float32, copy=False)).unsqueeze(1)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = self._make_optimizer(int(X_tensor.shape[0]))
        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        previous_loss = None
        self.converged_ = False
        self.n_iter_ = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            order = torch.randperm(int(X_tensor.shape[0]), generator=generator)
            for start in range(0, int(X_tensor.shape[0]), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = X_tensor[idx].to(self.device)
                yb = y_tensor[idx].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            epoch_loss = self._compute_epoch_loss(X_tensor, y_tensor, criterion)
            self.n_iter_ = epoch
            if previous_loss is not None:
                improvement = previous_loss - epoch_loss
                if 0.0 <= improvement < self.tol:
                    self.converged_ = True
                    break
            previous_loss = epoch_loss

        self.model.eval()
        self._is_fitted = True
        return self

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("TorchBinaryProbe.predict_logits called before fit().")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {X_arr.shape}.")

        X_tensor = torch.from_numpy(X_arr)
        logits_parts = []
        self.model.eval()
        for start in range(0, int(X_tensor.shape[0]), self.batch_size):
            stop = min(start + self.batch_size, int(X_tensor.shape[0]))
            xb = X_tensor[start:stop].to(self.device)
            logits = self.model(xb).squeeze(1).detach().cpu().numpy().astype(np.float64, copy=False)
            logits_parts.append(logits)

        if logits_parts:
            return np.concatenate(logits_parts).astype(np.float64, copy=False)
        return np.zeros(0, dtype=np.float64)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        pos_prob = positive_probability_from_logits(logits)
        if pos_prob.size > 0:
            pos_prob = pos_prob.astype(np.float64, copy=False)
        else:
            pos_prob = np.zeros(0, dtype=np.float64)
        neg_prob = 1.0 - pos_prob
        return np.stack([neg_prob, pos_prob], axis=1)


def build_probe(
    probe_type: str,
    input_dim: int,
    random_state: int = 42,
) -> TorchBinaryProbe:
    return TorchBinaryProbe(
        probe_type=probe_type,
        input_dim=input_dim,
        random_state=random_state,
    )


def scale_logits_with_temperature(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"Temperature must be positive and finite, got {temperature!r}.")
    return logits_arr / temp


def positive_probability_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    scaled_logits = scale_logits_with_temperature(logits, temperature=temperature)
    finite_mask = np.isfinite(scaled_logits)
    probs = np.full(scaled_logits.shape, np.nan, dtype=np.float64)
    if np.any(finite_mask):
        clipped = np.clip(scaled_logits[finite_mask], -_LOGIT_CLIP, _LOGIT_CLIP)
        probs[finite_mask] = 1.0 / (1.0 + np.exp(-clipped))
    return probs


def fit_temperature_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
    metric_eval_mask: Optional[np.ndarray] = None,
    max_iter: int = _TEMP_MAX_ITER,
) -> Dict[str, object]:
    logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)

    if metric_eval_mask is None:
        mask_arr = np.ones(labels_arr.shape[0], dtype=bool)
    else:
        mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)

    if not (logits_arr.shape[0] == labels_arr.shape[0] == mask_arr.shape[0]):
        raise ValueError(
            "Temperature scaler input length mismatch: "
            f"logits={logits_arr.shape[0]}, labels={labels_arr.shape[0]}, mask={mask_arr.shape[0]}."
        )

    logits_arr = logits_arr[mask_arr]
    labels_arr = labels_arr[mask_arr]

    finite_mask = np.isfinite(logits_arr)
    logits_arr = logits_arr[finite_mask]
    labels_arr = labels_arr[finite_mask]

    payload: Dict[str, object] = {
        "available": False,
        "temperature": 1.0,
        "log_temperature": 0.0,
        "count": int(labels_arr.shape[0]),
        "nll_before": float("nan"),
        "nll_after": float("nan"),
    }
    if labels_arr.size == 0:
        payload["reason"] = "no_finite_logits_after_masking"
        return payload

    logits_tensor = torch.from_numpy(logits_arr.astype(np.float32, copy=False))
    labels_tensor = torch.from_numpy(labels_arr.astype(np.float32, copy=False))
    criterion = torch.nn.BCEWithLogitsLoss()
    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=int(max(1, int(max_iter))),
        line_search_fn="strong_wolfe",
    )

    def current_temperature() -> torch.Tensor:
        return torch.exp(log_temperature)

    with torch.no_grad():
        nll_before = float(criterion(logits_tensor, labels_tensor).item())
    payload["nll_before"] = nll_before

    try:
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits_tensor / current_temperature(), labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            temperature = float(current_temperature().item())
            nll_after = float(criterion(logits_tensor / current_temperature(), labels_tensor).item())
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"learned invalid temperature {temperature!r}")
    except Exception as exc:
        payload["reason"] = str(exc)
        return payload

    payload.update(
        {
            "available": True,
            "temperature": float(temperature),
            "log_temperature": float(log_temperature.detach().item()),
            "nll_after": float(nll_after),
        }
    )
    return payload


def compute_binary_ece(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_eval_mask: np.ndarray,
    split: str,
    metric_name: str,
    bin_edges: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, object]]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    mask_arr = np.asarray(metric_eval_mask, dtype=bool).reshape(-1)

    if not (labels_arr.shape[0] == scores_arr.shape[0] == mask_arr.shape[0]):
        raise ValueError(
            f"{metric_name} ECE length mismatch: "
            f"labels={labels_arr.shape[0]}, scores={scores_arr.shape[0]}, mask={mask_arr.shape[0]}."
        )

    labels_arr = labels_arr[mask_arr]
    scores_arr = scores_arr[mask_arr]

    finite_mask = np.isfinite(scores_arr)
    labels_arr = labels_arr[finite_mask]
    scores_arr = scores_arr[finite_mask]

    if bin_edges is None:
        bin_edges_arr = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    else:
        bin_edges_arr = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if bin_edges_arr.ndim != 1 or bin_edges_arr.size < 2:
        raise ValueError(f"{metric_name} bin_edges must contain at least two values, got {bin_edges_arr}.")

    payload: Dict[str, object] = {
        "ece": float("nan"),
        "total_count": int(labels_arr.shape[0]),
        "bin_edges": bin_edges_arr.astype(float, copy=False).tolist(),
        "bins": [],
    }

    if labels_arr.size == 0:
        print(f"[{metric_name}] Skipping ECE for {split} split (no finite scores after masking).")
        return float("nan"), payload

    scores_arr = np.clip(scores_arr, float(bin_edges_arr[0]), float(bin_edges_arr[-1]))
    n_bins = int(bin_edges_arr.size - 1)
    bin_indices = np.searchsorted(bin_edges_arr, scores_arr, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    total = float(labels_arr.shape[0])
    ece = 0.0
    bin_rows: List[Dict[str, object]] = []
    for idx in range(n_bins):
        in_bin = bin_indices == idx
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        bin_scores = scores_arr[in_bin]
        bin_labels = labels_arr[in_bin]
        mean_pred_prob = float(np.mean(bin_scores))
        observed_rate = float(np.mean(bin_labels))
        abs_gap = abs(mean_pred_prob - observed_rate)
        ece += (count / total) * abs_gap
        bin_rows.append(
            {
                "bin_idx": int(idx),
                "lower": float(bin_edges_arr[idx]),
                "upper": float(bin_edges_arr[idx + 1]),
                "count": count,
                "mean_pred_prob": mean_pred_prob,
                "observed_hallucination_rate": observed_rate,
                "abs_gap": float(abs_gap),
            }
        )

    if not bin_rows:
        print(f"[{metric_name}] Skipping ECE for {split} split (no populated bins).")
        return float("nan"), payload

    payload["ece"] = float(ece)
    payload["bins"] = bin_rows
    return float(ece), payload
