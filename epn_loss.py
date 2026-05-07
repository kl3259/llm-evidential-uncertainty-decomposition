from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet


class UCEforEPN(nn.Module):
    """Uncertainty cross-entropy loss for EPN evidence-density training."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if len(args) == 4:
            return self._forward_prompt_level(*args)
        if len(args) == 6:
            return self._forward_response_level(*args)
        raise TypeError(f"UCEforEPN expected 4 or 6 tensors, got {len(args)}.")

    def _forward_prompt_level(
        self,
        rho: torch.Tensor,
        semantic_probs: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_probs.dim() != 2:
            raise ValueError(
                f"UCEforEPN expects semantic_probs with shape [B, C], got {tuple(semantic_probs.shape)}."
            )
        if targets.shape != semantic_probs.shape:
            raise ValueError(
                "UCEforEPN target shape mismatch: "
                f"targets={tuple(targets.shape)}, semantic_probs={tuple(semantic_probs.shape)}."
            )
        if cluster_mask.shape != semantic_probs.shape:
            raise ValueError(
                "UCEforEPN mask shape mismatch: "
                f"mask={tuple(cluster_mask.shape)}, semantic_probs={tuple(semantic_probs.shape)}."
            )

        rho_arr = rho.reshape(-1).to(dtype=semantic_probs.dtype)
        if rho_arr.numel() != semantic_probs.shape[0]:
            raise ValueError(
                "UCEforEPN rho batch mismatch: "
                f"rho={rho_arr.numel()}, semantic_probs batch={semantic_probs.shape[0]}."
            )

        mask = cluster_mask.to(dtype=semantic_probs.dtype)
        p_hat = torch.clamp(semantic_probs.to(dtype=rho_arr.dtype), min=0.0)
        y_hat = targets.to(dtype=rho_arr.dtype)
        k_count = torch.clamp(mask.sum(dim=1), min=1.0)
        rho_pos = torch.clamp(rho_arr, min=self.eps)

        alpha_0 = k_count * (1.0 + rho_pos)
        alpha_k = k_count.unsqueeze(1) * rho_pos.unsqueeze(1) * p_hat + 1.0
        terms = y_hat * (torch.digamma(alpha_0).unsqueeze(1) - torch.digamma(alpha_k))
        per_question = torch.sum(terms * mask, dim=1)

        if self.reduction == "none":
            return per_question
        if self.reduction == "sum":
            return per_question.sum()
        return per_question.mean()

    def _forward_response_level(
        self,
        response_logits: torch.Tensor,
        potential_evidence: torch.Tensor,
        response_cluster_index: torch.Tensor,
        response_mask: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        if response_logits.dim() != 2:
            raise ValueError(
                "UCEforEPN expects response_logits with shape "
                f"[B, R], got {tuple(response_logits.shape)}."
            )
        if potential_evidence.shape != response_logits.shape:
            raise ValueError(
                "UCEforEPN potential_evidence shape mismatch: "
                f"potential={tuple(potential_evidence.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_cluster_index.shape != response_logits.shape:
            raise ValueError(
                "UCEforEPN response_cluster_index shape mismatch: "
                f"indices={tuple(response_cluster_index.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_mask.shape != response_logits.shape:
            raise ValueError(
                "UCEforEPN response_mask shape mismatch: "
                f"mask={tuple(response_mask.shape)}, logits={tuple(response_logits.shape)}."
            )
        if cluster_mask.dim() != 2:
            raise ValueError(
                "UCEforEPN expects cluster_mask with shape "
                f"[B, C], got {tuple(cluster_mask.shape)}."
            )
        if cluster_mask.shape[0] != response_logits.shape[0]:
            raise ValueError(
                "UCEforEPN batch mismatch between responses and clusters: "
                f"responses={response_logits.shape[0]}, clusters={cluster_mask.shape[0]}."
            )
        expected_target_shape = (cluster_mask.shape[0], cluster_mask.shape[1] + 1)
        if tuple(targets.shape) != expected_target_shape:
            raise ValueError(
                "UCEforEPN target shape mismatch: "
                f"targets={tuple(targets.shape)}, expected={expected_target_shape}."
            )

        dtype = response_logits.dtype
        resp_mask = response_mask.to(dtype=dtype)
        cluster_mask_f = cluster_mask.to(dtype=dtype)
        q_rel = torch.sigmoid(response_logits)
        a_m = torch.clamp(potential_evidence.to(dtype=dtype), min=0.0) * resp_mask
        retained_by_response = a_m * q_rel

        batch_size, n_clusters = cluster_mask.shape
        retained_evidence = response_logits.new_zeros((batch_size, n_clusters))
        scatter_index = torch.clamp(response_cluster_index.to(dtype=torch.long), min=0)
        retained_evidence.scatter_add_(1, scatter_index, retained_by_response)
        retained_evidence = retained_evidence * cluster_mask_f

        alpha_k = 1.0 + torch.clamp(retained_evidence, min=0.0)
        alpha_0 = torch.sum(alpha_k * cluster_mask_f, dim=1)
        y_sem = targets[:, :n_clusters].to(dtype=dtype)
        sem_terms = y_sem * (
            torch.digamma(torch.clamp(alpha_0, min=self.eps)).unsqueeze(1)
            - torch.digamma(torch.clamp(alpha_k, min=self.eps))
        )
        per_question = torch.sum(sem_terms * cluster_mask_f, dim=1)

        if self.reduction == "none":
            return per_question
        if self.reduction == "sum":
            return per_question.sum()
        return per_question.mean()


class ConfAwareUCEforEPN(nn.Module):
    """Confidence-aware UCE for EPN evidence-density training.

    The loss is evaluated over confidence-weighted semantic support W_k:

        sum_k y_k [digamma(K + rho * W) - digamma(rho * W_k + 1)]

    where ``W = sum_k W_k`` is the total confidence-weighted support.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        rho: torch.Tensor,
        semantic_supports: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_supports.dim() != 2:
            raise ValueError(
                "ConfAwareUCEforEPN expects semantic_supports with shape "
                f"[B, C], got {tuple(semantic_supports.shape)}."
            )
        if targets.shape != semantic_supports.shape:
            raise ValueError(
                "ConfAwareUCEforEPN target shape mismatch: "
                f"targets={tuple(targets.shape)}, semantic_supports={tuple(semantic_supports.shape)}."
            )
        if cluster_mask.shape != semantic_supports.shape:
            raise ValueError(
                "ConfAwareUCEforEPN mask shape mismatch: "
                f"mask={tuple(cluster_mask.shape)}, semantic_supports={tuple(semantic_supports.shape)}."
            )

        rho_arr = rho.reshape(-1).to(dtype=semantic_supports.dtype)
        if rho_arr.numel() != semantic_supports.shape[0]:
            raise ValueError(
                "ConfAwareUCEforEPN rho batch mismatch: "
                f"rho={rho_arr.numel()}, semantic_supports batch={semantic_supports.shape[0]}."
            )

        mask = cluster_mask.to(dtype=semantic_supports.dtype)
        support_hat = torch.clamp(semantic_supports.to(dtype=rho_arr.dtype), min=0.0) * mask
        y_hat = targets.to(dtype=rho_arr.dtype)
        k_count = torch.clamp(mask.sum(dim=1), min=1.0)
        support_total = support_hat.sum(dim=1)
        rho_pos = torch.clamp(rho_arr, min=self.eps)

        alpha_0 = k_count + rho_pos * support_total
        alpha_k = rho_pos.unsqueeze(1) * support_hat + 1.0
        terms = y_hat * (torch.digamma(alpha_0).unsqueeze(1) - torch.digamma(alpha_k))
        per_question = torch.sum(terms * mask, dim=1)

        if self.reduction == "none":
            return per_question
        if self.reduction == "sum":
            return per_question.sum()
        return per_question.mean()


class AugmentedResponseUCEforEPNC(nn.Module):
    """Augmented-frame UCE for EPN-C response reliability training.

    EPN-C predicts a response-level reliability score ``q_rel = sigmoid(logit)``.
    Each response contributes potential evidence ``a_m``; ``a_m q_rel_m`` is
    retained by the response semantic class and ``a_m(1 - q_rel_m)`` goes to the
    unsupported class.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        response_logits: torch.Tensor,
        potential_evidence: torch.Tensor,
        response_cluster_index: torch.Tensor,
        response_mask: torch.Tensor,
        augmented_targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        if response_logits.dim() != 2:
            raise ValueError(
                "AugmentedResponseUCEforEPNC expects response_logits with shape "
                f"[B, R], got {tuple(response_logits.shape)}."
            )
        if potential_evidence.shape != response_logits.shape:
            raise ValueError(
                "AugmentedResponseUCEforEPNC potential_evidence shape mismatch: "
                f"potential={tuple(potential_evidence.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_cluster_index.shape != response_logits.shape:
            raise ValueError(
                "AugmentedResponseUCEforEPNC response_cluster_index shape mismatch: "
                f"indices={tuple(response_cluster_index.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_mask.shape != response_logits.shape:
            raise ValueError(
                "AugmentedResponseUCEforEPNC response_mask shape mismatch: "
                f"mask={tuple(response_mask.shape)}, logits={tuple(response_logits.shape)}."
            )
        if cluster_mask.dim() != 2:
            raise ValueError(
                "AugmentedResponseUCEforEPNC expects cluster_mask with shape "
                f"[B, C], got {tuple(cluster_mask.shape)}."
            )
        if cluster_mask.shape[0] != response_logits.shape[0]:
            raise ValueError(
                "AugmentedResponseUCEforEPNC batch mismatch between responses and clusters: "
                f"responses={response_logits.shape[0]}, clusters={cluster_mask.shape[0]}."
            )
        expected_target_shape = (cluster_mask.shape[0], cluster_mask.shape[1] + 1)
        if tuple(augmented_targets.shape) != expected_target_shape:
            raise ValueError(
                "AugmentedResponseUCEforEPNC augmented target shape mismatch: "
                f"targets={tuple(augmented_targets.shape)}, expected={expected_target_shape}."
            )

        dtype = response_logits.dtype
        resp_mask = response_mask.to(dtype=dtype)
        cluster_mask_f = cluster_mask.to(dtype=dtype)
        q_rel = torch.sigmoid(response_logits)
        a_m = torch.clamp(potential_evidence.to(dtype=dtype), min=0.0) * resp_mask
        retained_by_response = a_m * q_rel
        unsupported_evidence = torch.sum(a_m * (1.0 - q_rel), dim=1)

        batch_size, n_clusters = cluster_mask.shape
        retained_evidence = response_logits.new_zeros((batch_size, n_clusters))
        scatter_index = torch.clamp(response_cluster_index.to(dtype=torch.long), min=0)
        retained_evidence.scatter_add_(1, scatter_index, retained_by_response)
        retained_evidence = retained_evidence * cluster_mask_f

        k_count = torch.clamp(cluster_mask_f.sum(dim=1), min=1.0)
        s_aug = torch.sum(a_m, dim=1)
        alpha0_plus = k_count + 1.0 + s_aug
        alpha_k = 1.0 + torch.clamp(retained_evidence, min=0.0)
        alpha_bot = 1.0 + torch.clamp(unsupported_evidence, min=0.0)

        y_sem = augmented_targets[:, :n_clusters].to(dtype=dtype)
        y_bot = augmented_targets[:, n_clusters].to(dtype=dtype)
        sem_terms = y_sem * (
            torch.digamma(torch.clamp(alpha0_plus, min=self.eps)).unsqueeze(1)
            - torch.digamma(torch.clamp(alpha_k, min=self.eps))
        )
        bot_terms = y_bot * (
            torch.digamma(torch.clamp(alpha0_plus, min=self.eps))
            - torch.digamma(torch.clamp(alpha_bot, min=self.eps))
        )
        per_question = torch.sum(sem_terms * cluster_mask_f, dim=1) + bot_terms

        if self.reduction == "none":
            return per_question
        if self.reduction == "sum":
            return per_question.sum()
        return per_question.mean()


class ResponseUCEforEPNC(nn.Module):
    """Semantic-only response UCE for EPN-C reliability training.

    EPN-C predicts a response-level reliability score ``q_rel = sigmoid(logit)``.
    Each response contributes potential evidence ``a_m``. For semantic cluster
    ``k``, retained evidence is
    ``sum_{m: c_m = k} a_m q_rel_m``. This baseline ignores unsupported
    evidence and the augmented unsupported target column.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        response_logits: torch.Tensor,
        potential_evidence: torch.Tensor,
        response_cluster_index: torch.Tensor,
        response_mask: torch.Tensor,
        augmented_targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        if response_logits.dim() != 2:
            raise ValueError(
                "ResponseUCEforEPNC expects response_logits with shape "
                f"[B, R], got {tuple(response_logits.shape)}."
            )
        if potential_evidence.shape != response_logits.shape:
            raise ValueError(
                "ResponseUCEforEPNC potential_evidence shape mismatch: "
                f"potential={tuple(potential_evidence.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_cluster_index.shape != response_logits.shape:
            raise ValueError(
                "ResponseUCEforEPNC response_cluster_index shape mismatch: "
                f"indices={tuple(response_cluster_index.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_mask.shape != response_logits.shape:
            raise ValueError(
                "ResponseUCEforEPNC response_mask shape mismatch: "
                f"mask={tuple(response_mask.shape)}, logits={tuple(response_logits.shape)}."
            )
        if cluster_mask.dim() != 2:
            raise ValueError(
                "ResponseUCEforEPNC expects cluster_mask with shape "
                f"[B, C], got {tuple(cluster_mask.shape)}."
            )
        if cluster_mask.shape[0] != response_logits.shape[0]:
            raise ValueError(
                "ResponseUCEforEPNC batch mismatch between responses and clusters: "
                f"responses={response_logits.shape[0]}, clusters={cluster_mask.shape[0]}."
            )
        expected_target_shape = (cluster_mask.shape[0], cluster_mask.shape[1] + 1)
        if tuple(augmented_targets.shape) != expected_target_shape:
            raise ValueError(
                "ResponseUCEforEPNC augmented target shape mismatch: "
                f"targets={tuple(augmented_targets.shape)}, expected={expected_target_shape}."
            )

        dtype = response_logits.dtype
        resp_mask = response_mask.to(dtype=dtype)
        cluster_mask_f = cluster_mask.to(dtype=dtype)
        q_rel = torch.sigmoid(response_logits)
        a_m = torch.clamp(potential_evidence.to(dtype=dtype), min=0.0) * resp_mask
        retained_by_response = a_m * q_rel

        batch_size, n_clusters = cluster_mask.shape
        retained_evidence = response_logits.new_zeros((batch_size, n_clusters))
        scatter_index = torch.clamp(response_cluster_index.to(dtype=torch.long), min=0)
        retained_evidence.scatter_add_(1, scatter_index, retained_by_response)
        retained_evidence = retained_evidence * cluster_mask_f

        alpha_k = 1.0 + torch.clamp(retained_evidence, min=0.0)
        alpha_0 = torch.sum(alpha_k * cluster_mask_f, dim=1)
        y_sem = augmented_targets[:, :n_clusters].to(dtype=dtype)
        sem_terms = y_sem * (
            torch.digamma(torch.clamp(alpha_0, min=self.eps)).unsqueeze(1)
            - torch.digamma(torch.clamp(alpha_k, min=self.eps))
        )
        per_question = torch.sum(sem_terms * cluster_mask_f, dim=1)

        if self.reduction == "none":
            return per_question
        if self.reduction == "sum":
            return per_question.sum()
        return per_question.mean()


class MagnitudeL2(nn.Module):
    """Magnitude regularization loss for EPN density values."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        self.reduction = reduction

    def forward(self, rho: torch.Tensor, *unused_args: torch.Tensor) -> torch.Tensor:
        per_question = rho.reshape(-1).pow(2)

        if self.reduction == "none":
            return per_question
        if self.reduction == "sum":
            return per_question.sum()
        return per_question.mean()


class Entropy_Dirichlet(nn.Module):
    """Dirichlet entropy over reconstructed EPN alpha vectors."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        self.reduction = reduction

    def forward(self, alpha: torch.Tensor, cluster_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if alpha.dim() != 2:
            raise ValueError(f"Entropy_Dirichlet expects alpha with shape [B, C], got {tuple(alpha.shape)}.")
        if not torch.isfinite(alpha).all().item():
            raise ValueError("Entropy_Dirichlet requires finite alpha values.")
        if torch.any(alpha <= 0.0).item():
            raise ValueError("Entropy_Dirichlet requires strictly positive alpha values.")

        if cluster_mask is None:
            per_question = Dirichlet(alpha).entropy()
        else:
            if cluster_mask.shape != alpha.shape:
                raise ValueError(
                    "Entropy_Dirichlet mask shape mismatch: "
                    f"mask={tuple(cluster_mask.shape)}, alpha={tuple(alpha.shape)}."
                )
            mask = cluster_mask.to(dtype=torch.bool)
            entropy_vals = []
            for row_alpha, row_mask in zip(alpha, mask):
                active_alpha = row_alpha[row_mask]
                if active_alpha.numel() == 0:
                    entropy_vals.append(row_alpha.new_zeros(()))
                else:
                    entropy_vals.append(Dirichlet(active_alpha).entropy())
            per_question = torch.stack(entropy_vals, dim=0)

        if self.reduction == "none":
            return per_question
        if self.reduction == "sum":
            return per_question.sum()
        return per_question.mean()


class PseudoOODRegularizedEPNCLoss(nn.Module):
    """Apply EPN-C augmented UCE to real rows and entropy regularization to pseudo-OOD rows."""

    supports_pseudo_ood = True

    def __init__(
        self,
        base_loss: nn.Module,
        lambda_coef: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.lambda_coef = float(lambda_coef)
        self.eps = float(eps)
        self.entropy_dirichlet = Entropy_Dirichlet(reduction="mean")

    def forward(
        self,
        response_logits: torch.Tensor,
        potential_evidence: torch.Tensor,
        response_cluster_index: torch.Tensor,
        response_mask: torch.Tensor,
        augmented_targets: torch.Tensor,
        cluster_mask: torch.Tensor,
        pseudo_ood_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pseudo_ood_mask is None:
            return self.base_loss(
                response_logits,
                potential_evidence,
                response_cluster_index,
                response_mask,
                augmented_targets,
                cluster_mask,
            )
        if response_logits.dim() != 2:
            raise ValueError(
                "PseudoOODRegularizedEPNCLoss expects response_logits with shape "
                f"[B, R], got {tuple(response_logits.shape)}."
            )
        if potential_evidence.shape != response_logits.shape:
            raise ValueError(
                "PseudoOODRegularizedEPNCLoss potential_evidence shape mismatch: "
                f"potential={tuple(potential_evidence.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_cluster_index.shape != response_logits.shape:
            raise ValueError(
                "PseudoOODRegularizedEPNCLoss response_cluster_index shape mismatch: "
                f"indices={tuple(response_cluster_index.shape)}, logits={tuple(response_logits.shape)}."
            )
        if response_mask.shape != response_logits.shape:
            raise ValueError(
                "PseudoOODRegularizedEPNCLoss response_mask shape mismatch: "
                f"mask={tuple(response_mask.shape)}, logits={tuple(response_logits.shape)}."
            )
        if cluster_mask.dim() != 2 or cluster_mask.shape[0] != response_logits.shape[0]:
            raise ValueError(
                "PseudoOODRegularizedEPNCLoss cluster_mask batch mismatch: "
                f"cluster_mask={tuple(cluster_mask.shape)}, logits={tuple(response_logits.shape)}."
            )

        pseudo_mask = pseudo_ood_mask.reshape(-1).to(device=response_logits.device, dtype=torch.bool)
        if pseudo_mask.numel() != response_logits.shape[0]:
            raise ValueError(
                "PseudoOODRegularizedEPNCLoss pseudo mask batch mismatch: "
                f"pseudo_mask={pseudo_mask.numel()}, logits batch={response_logits.shape[0]}."
            )

        base_mask = ~pseudo_mask
        total_count = int(response_logits.shape[0])
        total_loss = response_logits.new_zeros(())

        if torch.any(base_mask).item():
            base_loss = self.base_loss(
                response_logits[base_mask],
                potential_evidence[base_mask],
                response_cluster_index[base_mask],
                response_mask[base_mask],
                augmented_targets[base_mask],
                cluster_mask[base_mask],
            )
            total_loss = total_loss + base_loss * int(torch.sum(base_mask).item())

        if torch.any(pseudo_mask).item():
            pseudo_logits = response_logits[pseudo_mask]
            pseudo_potential = potential_evidence[pseudo_mask]
            pseudo_cluster_index = response_cluster_index[pseudo_mask]
            pseudo_response_mask = response_mask[pseudo_mask]
            pseudo_cluster_mask = cluster_mask[pseudo_mask]

            dtype = pseudo_logits.dtype
            resp_mask = pseudo_response_mask.to(dtype=dtype)
            cluster_mask_f = pseudo_cluster_mask.to(dtype=dtype)
            q_rel = torch.sigmoid(pseudo_logits)
            retained_by_response = torch.clamp(pseudo_potential.to(dtype=dtype), min=0.0) * resp_mask * q_rel

            batch_size, n_clusters = pseudo_cluster_mask.shape
            retained_evidence = pseudo_logits.new_zeros((batch_size, n_clusters))
            scatter_index = torch.clamp(pseudo_cluster_index.to(dtype=torch.long), min=0)
            retained_evidence.scatter_add_(1, scatter_index, retained_by_response)
            retained_evidence = retained_evidence * cluster_mask_f

            alpha = 1.0 + torch.clamp(retained_evidence, min=0.0)
            entropy_loss = self.entropy_dirichlet(torch.clamp(alpha, min=self.eps), pseudo_cluster_mask)
            pseudo_loss = -self.lambda_coef * entropy_loss
            total_loss = total_loss + pseudo_loss * int(torch.sum(pseudo_mask).item())

        return total_loss / float(max(1, total_count))


class RegUCEforEPN_Mag(nn.Module):
    """UCEforEPN with L2 magnitude regularization on rho."""

    def __init__(
        self,
        reduction: str = "mean",
        lambda_coef: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.lambda_coef = float(lambda_coef)
        self.uce = UCEforEPN(reduction=reduction, eps=eps)
        self.magnitude_l2 = MagnitudeL2(reduction=reduction)

    def forward(
        self,
        rho: torch.Tensor,
        semantic_probs: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        uce_loss = self.uce(rho, semantic_probs, targets, cluster_mask)
        mag_loss = self.magnitude_l2(rho)
        return uce_loss + self.lambda_coef * mag_loss


class RegUCEforEPN_KL(nn.Module):
    """UCEforEPN with entropy maximization on reconstructed Dirichlet alpha."""

    def __init__(
        self,
        reduction: str = "mean",
        lambda_coef: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.lambda_coef = float(lambda_coef)
        self.eps = float(eps)
        self.uce = UCEforEPN(reduction=reduction, eps=eps)
        self.entropy_dirichlet = Entropy_Dirichlet(reduction=reduction)

    def forward(
        self,
        rho: torch.Tensor,
        semantic_probs: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        uce_loss = self.uce(rho, semantic_probs, targets, cluster_mask)
        rho_arr = rho.reshape(-1).to(dtype=semantic_probs.dtype)
        mask = cluster_mask.to(dtype=semantic_probs.dtype)
        p_hat = torch.clamp(semantic_probs.to(dtype=rho_arr.dtype), min=0.0)
        k_count = torch.clamp(mask.sum(dim=1), min=1.0)
        rho_pos = torch.clamp(rho_arr, min=self.eps)
        alpha = k_count.unsqueeze(1) * rho_pos.unsqueeze(1) * p_hat + 1.0
        entropy_loss = self.entropy_dirichlet(alpha, cluster_mask)
        return uce_loss - self.lambda_coef * entropy_loss


class RegConfAwareUCEforEPN_Mag(nn.Module):
    """Confidence-aware UCE with L2 magnitude regularization on rho."""

    def __init__(
        self,
        reduction: str = "mean",
        lambda_coef: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.lambda_coef = float(lambda_coef)
        self.uce = ConfAwareUCEforEPN(reduction=reduction, eps=eps)
        self.magnitude_l2 = MagnitudeL2(reduction=reduction)

    def forward(
        self,
        rho: torch.Tensor,
        semantic_supports: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        uce_loss = self.uce(rho, semantic_supports, targets, cluster_mask)
        mag_loss = self.magnitude_l2(rho)
        return uce_loss + self.lambda_coef * mag_loss


class RegConfAwareUCEforEPN_KL(nn.Module):
    """Confidence-aware UCE with entropy maximization on reconstructed alpha."""

    def __init__(
        self,
        reduction: str = "mean",
        lambda_coef: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.lambda_coef = float(lambda_coef)
        self.eps = float(eps)
        self.uce = ConfAwareUCEforEPN(reduction=reduction, eps=eps)
        self.entropy_dirichlet = Entropy_Dirichlet(reduction=reduction)

    def forward(
        self,
        rho: torch.Tensor,
        semantic_supports: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        uce_loss = self.uce(rho, semantic_supports, targets, cluster_mask)
        rho_arr = rho.reshape(-1).to(dtype=semantic_supports.dtype)
        mask = cluster_mask.to(dtype=semantic_supports.dtype)
        support_hat = torch.clamp(semantic_supports.to(dtype=rho_arr.dtype), min=0.0) * mask
        rho_pos = torch.clamp(rho_arr, min=self.eps)
        alpha = rho_pos.unsqueeze(1) * support_hat + 1.0
        entropy_loss = self.entropy_dirichlet(alpha, cluster_mask)
        return uce_loss - self.lambda_coef * entropy_loss


class PseudoOODRegularizedEPNLoss(nn.Module):
    """Apply the base EPN loss to in-domain rows and entropy regularization to pseudo-OOD rows."""

    supports_pseudo_ood = True

    def __init__(
        self,
        base_loss: nn.Module,
        semantic_prob_source: str,
        lambda_coef: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        source = str(semantic_prob_source or "count").strip().lower()
        if source not in {"count", "norm_seq_prob"}:
            raise ValueError(f"Unsupported semantic_prob_source for pseudo-OOD loss: {semantic_prob_source!r}.")
        self.base_loss = base_loss
        self.semantic_prob_source = source
        self.lambda_coef = float(lambda_coef)
        self.eps = float(eps)
        self.entropy_dirichlet = Entropy_Dirichlet(reduction="mean")

    def forward(
        self,
        rho: torch.Tensor,
        loss_inputs: torch.Tensor,
        targets: torch.Tensor,
        cluster_mask: torch.Tensor,
        pseudo_ood_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pseudo_ood_mask is None:
            return self.base_loss(rho, loss_inputs, targets, cluster_mask)
        if loss_inputs.dim() != 2:
            raise ValueError(
                "PseudoOODRegularizedEPNLoss expects loss_inputs with shape "
                f"[B, C], got {tuple(loss_inputs.shape)}."
            )

        rho_arr = rho.reshape(-1)
        if rho_arr.numel() != loss_inputs.shape[0]:
            raise ValueError(
                "PseudoOODRegularizedEPNLoss rho batch mismatch: "
                f"rho={rho_arr.numel()}, loss_inputs batch={loss_inputs.shape[0]}."
            )
        pseudo_mask = pseudo_ood_mask.reshape(-1).to(device=loss_inputs.device, dtype=torch.bool)
        if pseudo_mask.numel() != loss_inputs.shape[0]:
            raise ValueError(
                "PseudoOODRegularizedEPNLoss pseudo mask batch mismatch: "
                f"pseudo_mask={pseudo_mask.numel()}, loss_inputs batch={loss_inputs.shape[0]}."
            )

        base_mask = ~pseudo_mask
        total_count = int(loss_inputs.shape[0])
        total_loss = loss_inputs.new_zeros(())

        if torch.any(base_mask).item():
            base_loss = self.base_loss(
                rho_arr[base_mask],
                loss_inputs[base_mask],
                targets[base_mask],
                cluster_mask[base_mask],
            )
            total_loss = total_loss + base_loss * int(torch.sum(base_mask).item())

        if torch.any(pseudo_mask).item():
            pseudo_rho = rho_arr[pseudo_mask].to(dtype=loss_inputs.dtype)
            pseudo_inputs = loss_inputs[pseudo_mask]
            pseudo_cluster_mask = cluster_mask[pseudo_mask]
            pseudo_mask_float = pseudo_cluster_mask.to(dtype=pseudo_inputs.dtype)
            rho_pos = torch.clamp(pseudo_rho, min=self.eps)
            if self.semantic_prob_source == "norm_seq_prob":
                support_hat = torch.clamp(pseudo_inputs.to(dtype=pseudo_rho.dtype), min=0.0) * pseudo_mask_float
                alpha = rho_pos.unsqueeze(1) * support_hat + 1.0
            else:
                p_hat = torch.clamp(pseudo_inputs.to(dtype=pseudo_rho.dtype), min=0.0)
                k_count = torch.clamp(pseudo_mask_float.sum(dim=1), min=1.0)
                alpha = k_count.unsqueeze(1) * rho_pos.unsqueeze(1) * p_hat + 1.0
            entropy_loss = self.entropy_dirichlet(alpha, pseudo_cluster_mask)
            pseudo_loss = -self.lambda_coef * entropy_loss
            total_loss = total_loss + pseudo_loss * int(torch.sum(pseudo_mask).item())

        return total_loss / float(max(1, total_count))
