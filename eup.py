"""
Model architecture of the Evidential Uncertainty Probe
"""
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Sequence, Optional

Tensor = torch.Tensor
TensorOrList = Union[Tensor, Sequence[Tensor]]

# -------------------------
# Gradient Reversal Layer
# -------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x, lambd: float):
    return GradReverse.apply(x, lambd)


class EvidentialUncertaintyProbe(nn.Module):
    """
    Probe with optional hidden layer.
      If has_hidden=False: Linear(h_dim+c_dim -> 1) -> Softplus
      If has_hidden=True : Linear(h_dim+c_dim -> hidden_dim) -> ReLU
                           -> Linear(hidden_dim -> 1) -> Softplus
    """
    def __init__(
        self,
        h_dim: int,
        c_dim: int,
        activation: str = "softplus",
        softplus_beta: float = 1.0,
        bias: bool = True,
        rng: Optional[torch.Generator] = None,
        has_hidden: bool = True,
        n_hidden: int = 1,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.h_dim = int(h_dim)
        self.c_dim = int(c_dim)
        in_dim = self.h_dim + self.c_dim
        self.has_hidden = bool(has_hidden)
        self.n_hidden = int(n_hidden) if self.has_hidden else 0

        if self.n_hidden > 0:
            self.fc_hidden_layers = nn.ModuleList()
            prev_dim = in_dim
            for _ in range(self.n_hidden):
                layer = nn.Linear(prev_dim, hidden_dim, bias=bias)
                nn.init.kaiming_uniform_(layer.weight, generator=rng)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    # nn.init.constant_(layer.bias, softplus_beta)
                self.fc_hidden_layers.append(layer)
                prev_dim = hidden_dim            

            self.fc = nn.Linear(hidden_dim, 1, bias=bias)
            nn.init.kaiming_uniform_(self.fc.weight, generator=rng)
            if self.fc.bias is not None: nn.init.zeros_(self.fc.bias)
            # if self.fc.bias is not None: nn.init.constant_(
            #     self.fc.bias, softplus_beta
            # )

            self.hidden_act = nn.ReLU()
        else:
            self.fc = nn.Linear(in_dim, 1, bias=bias)
            nn.init.kaiming_uniform_(self.fc.weight, generator=rng)
            if self.fc.bias is not None: nn.init.zeros_(self.fc.bias)
            # if self.fc.bias is not None: nn.init.constant_(
            #     self.fc.bias, softplus_beta
            # )

        if activation == "softplus":
            self.activation = nn.Softplus(beta=softplus_beta)
        elif activation == "eluplus1":
            self.activation = nn.ELU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, H: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        assert len(H.shape) == 2  # already flattened
        H_flatten = H
        C_flatten = C.reshape(C.shape[0], -1)
        x = torch.cat([H_flatten, C_flatten], dim=-1)
        x = torch.nn.functional.layer_norm(x, x.shape[-1:]) # layer norm

        if self.n_hidden > 0:
            for layer in self.fc_hidden_layers:
                x = self.hidden_act(layer(x))
        x = self.fc(x)

        if self.activation == "eluplus1":
            x = self.activation(x) + 1.0
        else:
            x = self.activation(x)

        return x.squeeze(-1)


class EvidentialRegressionProbe(nn.Module):
    """
    Probe with optional hidden layer.
      If has_hidden=False: Linear(h_dim+c_dim -> 2) -> output transform
      If has_hidden=True : Linear(h_dim+c_dim -> hidden_dim) -> ReLU
                           -> Linear(hidden_dim -> 2) -> output transform

    Returns:
      Tensor of shape [batch_size, 2].
      - output_param="mean_concentration": (mu, e_total)
      - output_param="evidence": (e_1, e_0)
    """
    def __init__(
        self,
        h_dim: int,
        c_dim: int,
        softplus_beta: float = 1.0,
        bias: bool = True,
        rng: Optional[torch.Generator] = None,
        has_hidden: bool = True,
        n_hidden: int = 1,
        hidden_dim: int = 4096,
        output_param: str = "mean_concentration",
    ):
        super().__init__()
        self.h_dim = int(h_dim)
        self.c_dim = int(c_dim)
        in_dim = self.h_dim + self.c_dim
        self.has_hidden = bool(has_hidden)
        self.n_hidden = int(n_hidden) if self.has_hidden else 0

        if self.n_hidden > 0:
            self.fc_hidden_layers = nn.ModuleList()
            prev_dim = in_dim
            for _ in range(self.n_hidden):
                layer = nn.Linear(prev_dim, hidden_dim, bias=bias)
                nn.init.kaiming_uniform_(layer.weight, generator=rng)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                self.fc_hidden_layers.append(layer)
                prev_dim = hidden_dim

            self.fc = nn.Linear(hidden_dim, 2, bias=bias)
            nn.init.kaiming_uniform_(self.fc.weight, generator=rng)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)
            self.hidden_act = nn.ReLU()
        else:
            self.fc = nn.Linear(in_dim, 2, bias=bias)
            nn.init.kaiming_uniform_(self.fc.weight, generator=rng)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

        output_key = (output_param or "").strip().lower()
        if output_key == "mu_total":
            output_key = "mean_concentration"
        if output_key not in {"mean_concentration", "evidence"}:
            raise ValueError(
                f"Unsupported ERP output_param: {output_param}. "
                "Expected 'evidence' or 'mean_concentration'."
            )
        self.output_param = output_key
        self.erp_output_param = output_key

        self.mu_activation = nn.Sigmoid()
        self.evidence_activation = nn.Softplus(beta=softplus_beta)

    def forward(self, H: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        assert len(H.shape) == 2  # already flattened
        H_flatten = H
        C_flatten = C.reshape(C.shape[0], -1)
        x = torch.cat([H_flatten, C_flatten], dim=-1)
        x = torch.nn.functional.layer_norm(x, x.shape[-1:])  # layer norm

        if self.n_hidden > 0:
            for layer in self.fc_hidden_layers:
                x = self.hidden_act(layer(x))
        x = self.fc(x)
        if self.output_param == "evidence":
            # Legacy ERP parameterization: output evidences (e_1, e_0).
            return self.evidence_activation(x)

        mu = self.mu_activation(x[:, 0])
        e_total = self.evidence_activation(x[:, 1])
        return torch.stack([mu, e_total], dim=-1)


class DomainAdversarialEvidentialUnertaintyProbe(nn.Module):
    """
    Domain-adversarial variant of EUP.
      - Feature extractor: same hidden-layer stack as EUP (optional)
      - Regressor head:    Linear(hidden_dim -> 1) + activation
      - Domain head:       Linear(hidden_dim -> domain_hidden_dim) + ReLU + Linear -> num_domains
    """
    def __init__(
        self,
        h_dim: int,
        c_dim: int,
        num_domains: int,
        activation: str = "softplus",
        softplus_beta: float = 1.0,
        bias: bool = True,
        rng: Optional[torch.Generator] = None,
        has_hidden: bool = True,
        n_hidden: int = 1,
        hidden_dim: int = 4096,
        domain_hidden_dim: int = 256,
    ):
        super().__init__()
        self.h_dim = int(h_dim)
        self.c_dim = int(c_dim)
        self.num_domains = int(num_domains)
        in_dim = self.h_dim + self.c_dim
        self.has_hidden = bool(has_hidden)
        self.n_hidden = int(n_hidden) if self.has_hidden else 0

        self.feature_layers = nn.ModuleList()
        prev_dim = in_dim
        if self.n_hidden > 0:
            for _ in range(self.n_hidden):
                layer = nn.Linear(prev_dim, hidden_dim, bias=bias)
                nn.init.kaiming_uniform_(layer.weight, generator=rng)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                self.feature_layers.append(layer)
                prev_dim = hidden_dim
            feature_dim = hidden_dim
        else:
            feature_dim = in_dim

        self.regressor_fc = nn.Linear(feature_dim, 1, bias=bias)
        nn.init.kaiming_uniform_(self.regressor_fc.weight, generator=rng)
        if self.regressor_fc.bias is not None:
            nn.init.zeros_(self.regressor_fc.bias)

        self.domain_fc1 = nn.Linear(feature_dim, domain_hidden_dim, bias=bias)
        nn.init.kaiming_uniform_(self.domain_fc1.weight, generator=rng)
        if self.domain_fc1.bias is not None:
            nn.init.zeros_(self.domain_fc1.bias)

        self.domain_fc2 = nn.Linear(domain_hidden_dim, self.num_domains, bias=bias)
        nn.init.kaiming_uniform_(self.domain_fc2.weight, generator=rng)
        if self.domain_fc2.bias is not None:
            nn.init.zeros_(self.domain_fc2.bias)

        self.hidden_act = nn.ReLU()

        if activation == "softplus":
            self.activation = nn.Softplus(beta=softplus_beta)
        elif activation == "eluplus1":
            self.activation = nn.ELU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(
        self,
        H: torch.Tensor,
        C: torch.Tensor,
        grl_lambda: float = 0.0,
        return_domain_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert len(H.shape) == 2  # already flattened
        H_flatten = H
        C_flatten = C.reshape(C.shape[0], -1)
        x = torch.cat([H_flatten, C_flatten], dim=-1)
        x = torch.nn.functional.layer_norm(x, x.shape[-1:])  # layer norm

        features = x
        if self.n_hidden > 0:
            for layer in self.feature_layers:
                features = self.hidden_act(layer(features))
        pred = self.regressor_fc(features)
        if self.activation == "eluplus1":
            pred = self.activation(pred) + 1.0
        else:
            pred = self.activation(pred)

        if return_domain_logits:
            domain_logits = self.domain_fc2(self.hidden_act(self.domain_fc1(grl(features, grl_lambda))))
            return pred.squeeze(-1), domain_logits

        return pred.squeeze(-1)



