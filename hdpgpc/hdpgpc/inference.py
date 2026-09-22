"""Numerically stable inference primitives used by HDP-GPC.

This module is intentionally free of model mutation.  Keeping the HMM algebra
here makes it possible to test the probabilistic core without constructing GP
models or running the birth/death machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HMMPosterior:
    """Results of a forward-backward pass in log and probability space."""

    log_alpha: torch.Tensor
    log_beta: torch.Tensor
    responsibilities: torch.Tensor
    pairwise_responsibilities: torch.Tensor
    log_evidence: torch.Tensor

    @property
    def filtered_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.log_alpha, dim=-1)

    @property
    def backward_messages(self) -> torch.Tensor:
        return torch.exp(self.log_beta - self.log_beta.amax(dim=-1, keepdim=True))


def _require_floating(name: str, value: torch.Tensor) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype")
    if torch.any(torch.isnan(value)) or torch.any(torch.isposinf(value)):
        raise ValueError(f"{name} contains NaN or +inf")


def normalize_log_scores(
    log_scores: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize log scores while retaining the removed log normalizer.

    Unlike subtracting the maximum, this returns actual normalized log
    probabilities.  Rows containing no finite candidate are rejected because
    they do not define a probability distribution.
    """

    log_scores = torch.as_tensor(log_scores)
    _require_floating("log_scores", log_scores)
    log_normalizer = torch.logsumexp(log_scores, dim=dim, keepdim=True)
    if torch.any(~torch.isfinite(log_normalizer)):
        raise ValueError("each score slice must contain at least one finite value")
    return log_scores - log_normalizer, log_normalizer.squeeze(dim)


def hard_assignments(log_scores: torch.Tensor) -> torch.Tensor:
    """Return one-hot MAP assignments, leaving all-invalid rows empty.

    For a ``(T, K, K)`` pairwise tensor the two state dimensions are treated as
    one categorical dimension.  This matters at ``t=0``, where no transition
    exists and the correct sufficient statistic is an all-zero matrix.
    """

    log_scores = torch.as_tensor(log_scores)
    if log_scores.ndim not in (2, 3):
        raise ValueError("log_scores must have shape (T, K) or (T, K, K)")

    flat = log_scores.reshape(log_scores.shape[0], -1)
    valid = torch.isfinite(flat).any(dim=1)
    result = torch.zeros_like(flat)
    if torch.any(valid):
        indices = flat[valid].argmax(dim=1, keepdim=True)
        result[valid] = result[valid].scatter(1, indices, 1.0)
    return result.reshape_as(log_scores)


def variational_responsibilities(log_scores: torch.Tensor) -> torch.Tensor:
    """Exponentiate normalized log scores, preserving all-invalid slices as zero."""

    log_scores = torch.as_tensor(log_scores)
    if log_scores.ndim not in (2, 3):
        raise ValueError("log_scores must have shape (T, K) or (T, K, K)")
    flat = log_scores.reshape(log_scores.shape[0], -1)
    valid = torch.isfinite(flat).any(dim=1)
    result = torch.zeros_like(flat)
    if torch.any(valid):
        normalized, _ = normalize_log_scores(flat[valid], dim=1)
        result[valid] = torch.exp(normalized)
    return result.reshape_as(log_scores)


def responsibilities_from_log_scores(
    log_scores: torch.Tensor, mode: str = "hard"
) -> torch.Tensor:
    """Convert log scores to either MAP or fractional responsibilities."""

    if mode == "hard":
        return hard_assignments(log_scores)
    if mode == "variational":
        return variational_responsibilities(log_scores)
    raise ValueError("mode must be 'hard' or 'variational'")


def expected_log_dirichlet(concentration: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute ``E[log p]`` for a Dirichlet random variable."""

    concentration = torch.as_tensor(concentration)
    _require_floating("concentration", concentration)
    if torch.any(concentration <= 0):
        raise ValueError("Dirichlet concentrations must be strictly positive")
    return torch.digamma(concentration) - torch.digamma(
        concentration.sum(dim=dim, keepdim=True)
    )


def forward_backward(
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> HMMPosterior:
    """Run an exact, log-domain forward-backward pass.

    Parameters use the convention ``transition[i, j] = p(z_t=j | z_{t-1}=i)``.
    Inputs may be unnormalized log potentials; normalizing constants that are
    common to a categorical slice do not change the posterior.
    """

    log_emission = torch.as_tensor(log_emission)
    device, dtype = log_emission.device, log_emission.dtype
    log_initial = torch.as_tensor(log_initial, device=device, dtype=dtype)
    log_transition = torch.as_tensor(log_transition, device=device, dtype=dtype)

    _require_floating("log_initial", log_initial)
    _require_floating("log_transition", log_transition)
    _require_floating("log_emission", log_emission)

    if log_emission.ndim != 2 or log_emission.shape[0] == 0:
        raise ValueError("log_emission must have non-empty shape (T, K)")
    states = log_emission.shape[1]
    if log_initial.shape != (states,):
        raise ValueError(f"log_initial must have shape ({states},)")
    if log_transition.shape != (states, states):
        raise ValueError(f"log_transition must have shape ({states}, {states})")
    if not torch.isfinite(log_initial).any():
        raise ValueError("log_initial must contain a finite state")
    if torch.any(~torch.isfinite(torch.logsumexp(log_transition, dim=1))):
        raise ValueError("each transition row must contain a finite state")
    if torch.any(~torch.isfinite(torch.logsumexp(log_emission, dim=1))):
        raise ValueError("each observation must have a finite emission score")

    steps = log_emission.shape[0]
    log_alpha = torch.empty_like(log_emission)
    log_alpha[0] = log_initial + log_emission[0]
    for time in range(1, steps):
        log_alpha[time] = log_emission[time] + torch.logsumexp(
            log_alpha[time - 1][:, None] + log_transition, dim=0
        )

    log_evidence = torch.logsumexp(log_alpha[-1], dim=0)
    if not torch.isfinite(log_evidence):
        raise ValueError("the HMM evidence is not finite")

    log_beta = torch.zeros_like(log_emission)
    for time in range(steps - 2, -1, -1):
        log_beta[time] = torch.logsumexp(
            log_transition
            + log_emission[time + 1][None, :]
            + log_beta[time + 1][None, :],
            dim=1,
        )

    log_responsibilities, _ = normalize_log_scores(log_alpha + log_beta, dim=1)
    responsibilities = torch.exp(log_responsibilities)

    pairwise = torch.zeros(
        (steps, states, states), device=device, dtype=dtype
    )
    for time in range(1, steps):
        log_pair = (
            log_alpha[time - 1][:, None]
            + log_transition
            + log_emission[time][None, :]
            + log_beta[time][None, :]
        )
        pairwise[time] = torch.softmax(log_pair.reshape(-1), dim=0).reshape(
            states, states
        )

    return HMMPosterior(
        log_alpha=log_alpha,
        log_beta=log_beta,
        responsibilities=responsibilities,
        pairwise_responsibilities=pairwise,
        log_evidence=log_evidence,
    )
