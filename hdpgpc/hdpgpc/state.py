"""Immutable snapshots for fitting independent batches on different grids."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


def _cpu_clone(value: object) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu().clone()


@dataclass(frozen=True)
class BatchResult:
    """Final variational state of one rectangular observation batch."""

    x: torch.Tensor
    y: torch.Tensor
    labels: torch.Tensor
    responsibilities: torch.Tensor
    pairwise_responsibilities: torch.Tensor
    emission_scores: torch.Tensor
    latent_scores: torch.Tensor
    snr: torch.Tensor
    objective: float
    convergence_reason: str | None
    split_diagnostics: tuple[dict[str, Any], ...]

    @classmethod
    def capture(cls, model: object, x: object, y: object) -> "BatchResult":
        required = ("resp_last", "respPair_last", "q_last", "q_lat_last", "snr_last")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise RuntimeError("batch inference did not produce: " + ", ".join(missing))
        resp = _cpu_clone(model.resp_last)
        return cls(
            x=_cpu_clone(x),
            y=_cpu_clone(y),
            labels=resp.argmax(dim=1),
            responsibilities=resp,
            pairwise_responsibilities=_cpu_clone(model.respPair_last),
            emission_scores=_cpu_clone(model.q_last),
            latent_scores=_cpu_clone(model.q_lat_last),
            snr=_cpu_clone(model.snr_last),
            objective=float(torch.as_tensor(model.elbo_last).detach().cpu()),
            convergence_reason=getattr(model, "convergence_reason_", None),
            split_diagnostics=tuple(
                copy.deepcopy(getattr(model, "split_diagnostics_", []))
            ),
        )


@dataclass
class BatchHistory:
    """Ragged collection; batches are never concatenated along their time axis."""

    _items: list[BatchResult] = field(default_factory=list)

    def append(self, result: BatchResult) -> None:
        self._items.append(result)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[BatchResult]:
        return iter(self._items)

    def __getitem__(self, index: int) -> BatchResult:
        return self._items[index]

    @property
    def latest(self) -> BatchResult:
        if not self._items:
            raise RuntimeError("the model has no fitted batches")
        return self._items[-1]
