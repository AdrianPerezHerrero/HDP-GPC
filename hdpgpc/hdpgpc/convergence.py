"""Explicit convergence policy and diagnostics for iterative inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import torch


class StopReason(str, Enum):
    OBJECTIVE_AND_ASSIGNMENTS = "objective_and_assignments_stable"
    MAX_ITERATIONS = "maximum_iterations"
    NON_FINITE_OBJECTIVE = "non_finite_objective"
    TOO_MANY_DECREASES = "too_many_objective_decreases"


@dataclass(frozen=True)
class ConvergenceConfig:
    """Termination policy shared by batch and local inference loops."""

    max_iterations: int = 50
    max_local_iterations: int = 20
    min_iterations: int = 2
    rtol: float = 1e-5
    atol: float = 1e-8
    patience: int = 2
    max_objective_decreases: int = 3

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.max_local_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if self.min_iterations < 1 or self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations must be within max_iterations")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("convergence tolerances cannot be negative")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.max_objective_decreases < 0:
            raise ValueError("max_objective_decreases cannot be negative")


@dataclass(frozen=True)
class ConvergenceStep:
    iteration: int
    objective: float
    delta: float | None
    assignments_stable: bool
    stop_reason: StopReason | None

    @property
    def should_stop(self) -> bool:
        return self.stop_reason is not None


class ConvergenceMonitor:
    """Stateful convergence monitor with an auditable stop reason."""

    def __init__(self, config: ConvergenceConfig | None = None) -> None:
        self.config = config or ConvergenceConfig()
        self.history: list[float] = []
        self._previous_assignments: torch.Tensor | None = None
        self._stable_steps = 0
        self._decreases = 0

    def update(
        self, objective: float | torch.Tensor, assignments: Iterable[int] | torch.Tensor
    ) -> ConvergenceStep:
        value = float(torch.as_tensor(objective).detach().cpu().item())
        labels = torch.as_tensor(assignments).detach().cpu().reshape(-1)
        iteration = len(self.history) + 1

        delta = None if not self.history else value - self.history[-1]
        stable = (
            self._previous_assignments is not None
            and torch.equal(labels, self._previous_assignments)
        )
        self._stable_steps = self._stable_steps + 1 if stable else 0

        reason: StopReason | None = None
        if not math.isfinite(value):
            reason = StopReason.NON_FINITE_OBJECTIVE
        elif delta is not None:
            tolerance = self.config.atol + self.config.rtol * abs(self.history[-1])
            if delta < -tolerance:
                self._decreases += 1
            close = abs(delta) <= tolerance
            if (
                iteration >= self.config.min_iterations
                and close
                and self._stable_steps >= self.config.patience
            ):
                reason = StopReason.OBJECTIVE_AND_ASSIGNMENTS
            elif self._decreases > self.config.max_objective_decreases:
                reason = StopReason.TOO_MANY_DECREASES

        if reason is None and iteration >= self.config.max_iterations:
            reason = StopReason.MAX_ITERATIONS

        self.history.append(value)
        self._previous_assignments = labels.clone()
        return ConvergenceStep(iteration, value, delta, stable, reason)
