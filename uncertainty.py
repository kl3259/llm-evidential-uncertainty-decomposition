"""
Definition of vacuity and dissonance
"""
import numpy as np
import torch


def normalize_erp_output_param(output_param: str) -> str:
    key = (output_param or "").strip().lower()
    if key in {"mean_concentration", "mu_total", "mu-total"}:
        return "mean_concentration"
    if key == "evidence":
        return "evidence"
    raise ValueError(
        f"Unsupported ERP output parameterization: {output_param}. "
        "Expected one of {'evidence', 'mean_concentration'}."
    )


def erp_output_to_alpha_beta_torch(
    pred_output: torch.Tensor,
    output_param: str = "mean_concentration",
    eps: float = 1e-6,
):
    if pred_output.dim() != 2 or pred_output.size(-1) != 2:
        raise ValueError(
            "ERP conversion expects pred_output with shape [B, 2], "
            f"got {tuple(pred_output.shape)}"
        )
    mode = normalize_erp_output_param(output_param)
    eps_f = float(eps)
    if mode == "mean_concentration":
        mu = torch.clamp(pred_output[:, 0], min=eps_f, max=1.0 - eps_f)
        e_total = torch.clamp(pred_output[:, 1], min=0.0)
        alpha = (1.0 + mu * e_total).clamp_min(1.0)
        beta = (1.0 + (1.0 - mu) * e_total).clamp_min(1.0)
        return alpha, beta

    # Legacy ERP output: evidence for class 1/0.
    e_1 = torch.clamp(pred_output[:, 0], min=0.0)
    e_0 = torch.clamp(pred_output[:, 1], min=0.0)
    alpha = (e_1 + 1.0).clamp_min(eps_f)
    beta = (e_0 + 1.0).clamp_min(eps_f)
    return alpha, beta


def erp_output_to_alpha_beta_np(
    pred_output: np.ndarray,
    output_param: str = "mean_concentration",
    eps: float = 1e-6,
):
    pred_arr = np.asarray(pred_output, dtype=np.float32)
    if pred_arr.ndim != 2 or pred_arr.shape[1] != 2:
        raise ValueError(
            "ERP conversion expects pred_output with shape [B, 2], "
            f"got {tuple(pred_arr.shape)}"
        )
    mode = normalize_erp_output_param(output_param)
    eps_f = float(eps)
    if mode == "mean_concentration":
        mu_arr = np.clip(pred_arr[:, 0], eps_f, 1.0 - eps_f)
        e_total_arr = np.maximum(pred_arr[:, 1], 0.0)
        alpha = np.maximum(1.0 + mu_arr * e_total_arr, 1.0)
        beta = np.maximum(1.0 + (1.0 - mu_arr) * e_total_arr, 1.0)
        return alpha, beta

    # Legacy ERP output: evidence for class 1/0.
    e_1 = np.maximum(pred_arr[:, 0], 0.0)
    e_0 = np.maximum(pred_arr[:, 1], 0.0)
    alpha = np.maximum(e_1 + 1.0, eps_f)
    beta = np.maximum(e_0 + 1.0, eps_f)
    return alpha, beta


def erp_mu_total_to_alpha_beta_torch(
    mu: torch.Tensor,
    e_total: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Convert ERP (mu, e_total) parameterization to beta-binomial (alpha, beta).
      S = e_total + 2
      alpha = 1 + mu * e_total
      beta = 1 + (1 - mu) * e_total
    """
    pred_output = torch.stack([mu, e_total], dim=-1)
    return erp_output_to_alpha_beta_torch(
        pred_output=pred_output,
        output_param="mean_concentration",
        eps=eps,
    )


def erp_mu_total_to_alpha_beta_np(
    mu: np.ndarray,
    e_total: np.ndarray,
    eps: float = 1e-6,
):
    """
    NumPy version of ERP (mu, e_total) -> (alpha, beta) conversion.
    """
    pred_output = np.stack(
        [
            np.asarray(mu, dtype=np.float32).reshape(-1),
            np.asarray(e_total, dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    return erp_output_to_alpha_beta_np(
        pred_output=pred_output,
        output_param="mean_concentration",
        eps=eps,
    )

def vacuity(evidence_dict):
    """
    Vacuity measure of uncertainty per question
    """
    S = sum(evidence_dict.values()) 
    K = len(evidence_dict)
    vacuity = float(K / (S + K + 1e-12)) 
    return vacuity

def Bal(b_i, b_j):
    result = 1 - np.abs(b_i - b_j) / (b_i + b_j + 1e-12)
    return result

def dissonance(evidence_dict):
    """
    Dissonance measure of uncertainty per question
    """
    S = sum(evidence_dict.values()) 
    K = len(evidence_dict)
    b = np.asarray([v / (S + K) for k, v in evidence_dict.items()]) # belief

    dissonance = 0.0
    for i in range(len(b)):
        term_Bal = 0.0
        term_bj = 0.0
        for j in range(len(b)):
            if j != i:
                term_Bal += b[j] * Bal(b[i], b[j])
                term_bj += b[j]
        dissonance += b[i] * (term_Bal / (term_bj + 1e-12))

    return dissonance


def theils_u(target_ids, conditioning_ids, eps: float = 1e-12) -> float:
    """
    Plug-in estimate of Theil's U(target | conditioning) from paired observations.
    """
    target = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    conditioning = np.asarray(conditioning_ids, dtype=np.int64).reshape(-1)
    if target.size == 0 or conditioning.size == 0:
        raise ValueError("theils_u requires non-empty target_ids and conditioning_ids.")
    if target.shape[0] != conditioning.shape[0]:
        raise ValueError(
            "theils_u requires target_ids and conditioning_ids with the same length, "
            f"got {target.shape[0]} and {conditioning.shape[0]}."
        )

    n = float(target.shape[0])
    joint_counts = {}
    conditioning_counts = {}
    target_counts = {}
    for target_id, conditioning_id in zip(target.tolist(), conditioning.tolist()):
        joint_key = (int(conditioning_id), int(target_id))
        joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
        conditioning_counts[int(conditioning_id)] = conditioning_counts.get(int(conditioning_id), 0) + 1
        target_counts[int(target_id)] = target_counts.get(int(target_id), 0) + 1

    target_probs = {
        target_id: float(count) / n
        for target_id, count in target_counts.items()
        if count > 0
    }
    h_target = -sum(
        prob * np.log(max(prob, float(eps)))
        for prob in target_probs.values()
    )
    if h_target <= float(eps):
        return 0.0

    h_target_given_conditioning = 0.0
    for conditioning_id, conditioning_count in conditioning_counts.items():
        p_conditioning = float(conditioning_count) / n
        conditional_entropy = 0.0
        for target_id in target_counts:
            joint_prob = float(joint_counts.get((conditioning_id, target_id), 0)) / n
            if joint_prob <= 0.0:
                continue
            conditional_prob = joint_prob / max(p_conditioning, float(eps))
            conditional_entropy -= conditional_prob * np.log(max(conditional_prob, float(eps)))
        h_target_given_conditioning += p_conditioning * conditional_entropy

    return float((h_target - h_target_given_conditioning) / h_target)


def get_mode_erp(alpha, beta, eps: float = 1e-12):
    """
    Expected semantic probability for ERP:
      mu_p = alpha / (alpha + beta)
    """
    return alpha / (alpha + beta + eps)


def get_pred_count_erp(alpha, beta, n_resp, eps: float = 1e-12):
    """
    Expected count for ERP:
      e_hat = n * mu_p
    """
    mu_p = get_mode_erp(alpha, beta, eps=eps)
    return n_resp * mu_p


def get_epistemic_unc_erp(alpha, beta, eps: float = 1e-12):
    """
    Epistemic uncertainty for ERP:
      u = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    """
    s = alpha + beta
    return (alpha * beta) / ((s * s + eps) * (s + 1.0 + eps))


def get_aleatoric_unc_erp(alpha, beta, eps: float = 1e-12):
    """
    Aleatoric uncertainty for ERP:
      v = (alpha * beta) / ((alpha + beta) * (alpha + beta + 1))
    """
    s = alpha + beta
    return (alpha * beta) / ((s + eps) * (s + 1.0 + eps))


def get_vacuity_unc_erp(alpha, beta, eps: float = 1e-12):
    """
    Vacuity for beta-binomial ERP:
      u_vac = 2 / (alpha + beta)
    """
    s = np.asarray(alpha) + np.asarray(beta)
    s = np.maximum(s, float(eps))
    return 2.0 / s


def get_dissonance_unc_erp(alpha, beta, eps: float = 1e-12):
    """
    Dissonance for beta-binomial ERP using binary belief masses:
      e_1 = max(alpha - 1, 0), e_0 = max(beta - 1, 0)
      b_1 = e_1 / (e_1 + e_0), b_0 = e_0 / (e_1 + e_0)
      u_dis = 2 * min(b_1, b_0)
    """
    e_1 = np.maximum(np.asarray(alpha) - 1.0, 0.0)
    e_0 = np.maximum(np.asarray(beta) - 1.0, 0.0)
    denom = e_1 + e_0
    b_1 = np.divide(
        e_1,
        denom + float(eps),
        out=np.zeros_like(e_1, dtype=np.float32),
        where=denom > 0,
    )
    b_0 = np.divide(
        e_0,
        denom + float(eps),
        out=np.zeros_like(e_0, dtype=np.float32),
        where=denom > 0,
    )
    return 2.0 * np.minimum(b_1, b_0)
