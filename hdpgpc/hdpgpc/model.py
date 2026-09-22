"""Clean public facade over the backwards-compatible HDP-GPC implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import HDPGPCConfig
from .GPI_HDP import GPI_HDP
from .state import BatchHistory, BatchResult


class HDPGPC(GPI_HDP):
    """HDP-GPC model with JSON configuration and ragged batch history."""

    def __init__(
        self,
        x_basis: object,
        *,
        config: HDPGPCConfig | None = None,
        **model_options: Any,
    ) -> None:
        config = config or HDPGPCConfig()
        options = {
            "M": config.initial_states,
            "n_outputs": config.n_outputs,
            "model_type": config.model_type,
            "hdp_hyperparameters": config.hdp,
            "max_models": config.max_models,
            "verbose": config.verbose,
            "convergence": config.convergence,
            "responsibility_mode": config.responsibility_mode,
            "min_responsibility": config.min_responsibility,
            "min_cluster_mass": config.min_cluster_mass,
            "hdp_elbo_scale": config.hdp_elbo_scale,
            "continuous_elbo_scale": config.continuous_elbo_scale,
            "parameter_kl_scale": config.parameter_kl_scale,
            "snr_weight_mode": config.snr_weight_mode,
            "learn_observation_matrix": config.learn_observation_matrix,
            "couple_gp_lds_noise": config.couple_gp_lds_noise,
            **dict(config.model_options),
        }
        options.update(model_options)
        super().__init__(x_basis, **options)
        self.config = config
        self.batch_history_ = BatchHistory()

    @classmethod
    def from_json(
        cls, path: str | Path, *, x_basis: object | None = None, **overrides: Any
    ) -> "HDPGPC":
        config = HDPGPCConfig.from_json(path)
        basis = x_basis if x_basis is not None else config.x_basis
        if basis is None:
            raise ValueError("x_basis must be provided either in JSON or to from_json")
        if x_basis is None:
            basis = np.asarray(basis, dtype=np.float64)
            if basis.ndim == 1:
                basis = basis[:, None]
        return cls(basis, config=config, **overrides)

    def fit(
        self,
        x: object,
        y: object,
        *,
        warp: bool = False,
        max_iterations: int | None = None,
    ) -> "HDPGPC":
        if len(self.batch_history_):
            raise RuntimeError("fit has already been called; use fit_batch for another batch")
        return self.fit_batch(x, y, warp=warp, max_iterations=max_iterations)

    def fit_batch(
        self,
        x: object,
        y: object,
        *,
        warp: bool = False,
        max_iterations: int | None = None,
        carry_latent_state: bool = False,
    ) -> "HDPGPC":
        """Fit one batch while retaining model parameters and a ragged snapshot."""

        if len(self.batch_history_):
            self.start_new_batch(carry_latent_state=carry_latent_state)
        self.include_batch(x, y, it_limit=max_iterations, warp=warp)
        self.batch_history_.append(BatchResult.capture(self, x, y))
        return self

    def partial_fit(self, x: object, y: object, *, warp: bool = False) -> "HDPGPC":
        self.include_sample(x, y, with_warp=warp)
        return self

    def predict(self, x: object, y: object):
        return self.cluster_new_batch(x, y, learning=False)

    @property
    def labels_(self) -> torch.Tensor:
        return self.batch_history_.latest.labels
