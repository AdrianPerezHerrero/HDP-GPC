"""Validated, JSON-serializable configuration for HDP-GPC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from .convergence import ConvergenceConfig


@dataclass(frozen=True)
class HDPHyperparameters:
    gamma: float = 1.0
    transition_alpha: float = 1.0
    start_alpha: float = 0.1
    sticky_kappa: float = 0.0

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.transition_alpha <= 0 or self.start_alpha <= 0:
            raise ValueError("gamma and Dirichlet concentrations must be positive")
        if self.sticky_kappa < 0:
            raise ValueError("sticky_kappa cannot be negative")

    @classmethod
    def from_preset(cls, preset: str) -> "HDPHyperparameters":
        presets = {
            "less": cls(0.01, 0.01, 0.01, 0.0),
            "balanced": cls(1.0, 1.0, 0.1, 0.0),
            "more": cls(10.0, 10.0, 1.0, 0.0),
        }
        try:
            return presets[preset]
        except KeyError as exc:
            choices = ", ".join(sorted(presets))
            raise ValueError(f"unknown HDP preset {preset!r}; choose one of: {choices}") from exc


_MODEL_KEYS = {"initial_states", "n_outputs", "model_type", "max_models", "verbose"}
_INFERENCE_KEYS = {
    "responsibility_mode",
    "min_responsibility",
    "min_cluster_mass",
    "hdp_elbo_scale",
    "continuous_elbo_scale",
    "parameter_kl_scale",
    "snr_weight_mode",
}
_HDP_KEYS = {"preset", "gamma", "transition_alpha", "start_alpha", "sticky_kappa"}
_GP_KEYS = {
    "ini_lengthscale", "bound_lengthscale", "ini_gamma", "ini_sigma",
    "ini_outputscale", "bound_sigma", "bound_gamma", "inducing_points",
    "estimation_limit", "free_deg_MNIV", "share_gp",
}
_WARP_KEYS = {
    "x_basis_warp", "bound_noise_warp", "noise_warp", "recursive_warp",
    "warp_updating", "method_compute_warp", "mode_warp",
}
_ADVANCED_KEYS = {
    "reest_conditions", "annealing", "hmm_switch", "batch", "check_var",
    "bayesian_params", "cuda", "reestimate_initial_params", "n_explore_steps",
    "use_snr", "reduce_outputs", "reduce_outputs_ratio",
}
_LDS_KEYS = {"learn_observation_matrix", "couple_gp_lds_noise"}


def _strict_section(
    source: Mapping[str, Any], name: str, allowed: set[str]
) -> dict[str, Any]:
    section = source.get(name, {})
    if not isinstance(section, Mapping):
        raise TypeError(f"configuration section {name!r} must be an object")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {name!r}: {', '.join(sorted(unknown))}")
    return dict(section)


@dataclass(frozen=True)
class HDPGPCConfig:
    """Complete constructor configuration with a versioned JSON representation."""

    schema_version: int = 1
    x_basis: tuple[Any, ...] | None = None
    initial_states: int = 1
    n_outputs: int = 1
    model_type: str = "dynamic"
    hdp: HDPHyperparameters = field(default_factory=HDPHyperparameters)
    responsibility_mode: str = "hard"
    min_responsibility: float = 1e-4
    min_cluster_mass: float = 1.0
    hdp_elbo_scale: float = 1.0
    continuous_elbo_scale: float = 1.0
    parameter_kl_scale: float = 1.0
    snr_weight_mode: str = "sum"
    learn_observation_matrix: bool = False
    couple_gp_lds_noise: bool = False
    max_models: int | None = None
    verbose: bool = False
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    model_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only HDP-GPC configuration schema_version 1 is supported")
        if self.initial_states < 1 or self.n_outputs < 1:
            raise ValueError("initial_states and n_outputs must be positive")
        if self.model_type not in {"static", "dynamic"}:
            raise ValueError("model_type must be 'static' or 'dynamic'")
        if self.responsibility_mode not in {"hard", "variational"}:
            raise ValueError("responsibility_mode must be 'hard' or 'variational'")
        if not 0.0 <= self.min_responsibility < 1.0:
            raise ValueError("min_responsibility must be in [0, 1)")
        if self.min_cluster_mass <= 0:
            raise ValueError("min_cluster_mass must be positive")
        if self.hdp_elbo_scale <= 0:
            raise ValueError("hdp_elbo_scale must be positive")
        if self.continuous_elbo_scale <= 0:
            raise ValueError("continuous_elbo_scale must be positive")
        if self.parameter_kl_scale < 0:
            raise ValueError("parameter_kl_scale cannot be negative")
        if self.snr_weight_mode not in {"sum", "mean"}:
            raise ValueError("snr_weight_mode must be 'sum' or 'mean'")
        if self.max_models is not None and self.max_models < self.initial_states:
            raise ValueError("max_models cannot be smaller than initial_states")

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> "HDPGPCConfig":
        if not isinstance(source, Mapping):
            raise TypeError("configuration must be a JSON object")
        allowed_top = {
            "schema_version", "x_basis", "model", "inference", "hdp",
            "convergence", "gp", "warping", "advanced",
            "lds",
        }
        unknown = set(source) - allowed_top
        if unknown:
            raise ValueError(f"unknown top-level configuration keys: {', '.join(sorted(unknown))}")

        model = _strict_section(source, "model", _MODEL_KEYS)
        inference = _strict_section(source, "inference", _INFERENCE_KEYS)
        hdp_data = _strict_section(source, "hdp", _HDP_KEYS)
        convergence_data = _strict_section(
            source, "convergence", set(ConvergenceConfig.__dataclass_fields__)
        )
        gp = _strict_section(source, "gp", _GP_KEYS)
        warping = _strict_section(source, "warping", _WARP_KEYS)
        advanced = _strict_section(source, "advanced", _ADVANCED_KEYS)
        lds = _strict_section(source, "lds", _LDS_KEYS)

        preset = hdp_data.pop("preset", None)
        if preset is not None and hdp_data:
            raise ValueError("hdp.preset cannot be combined with explicit HDP parameters")
        hdp = (
            HDPHyperparameters.from_preset(preset)
            if preset is not None
            else HDPHyperparameters(**hdp_data)
        )
        raw_basis = source.get("x_basis")
        x_basis = None if raw_basis is None else tuple(raw_basis)
        return cls(
            schema_version=int(source.get("schema_version", 1)),
            x_basis=x_basis,
            hdp=hdp,
            convergence=ConvergenceConfig(**convergence_data),
            model_options={**gp, **warping, **advanced},
            **lds,
            **model,
            **inference,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "HDPGPCConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> dict[str, Any]:
        options = dict(self.model_options)
        return {
            "schema_version": self.schema_version,
            "x_basis": None if self.x_basis is None else list(self.x_basis),
            "model": {
                "initial_states": self.initial_states,
                "n_outputs": self.n_outputs,
                "model_type": self.model_type,
                "max_models": self.max_models,
                "verbose": self.verbose,
            },
            "inference": {
                "responsibility_mode": self.responsibility_mode,
                "min_responsibility": self.min_responsibility,
                "min_cluster_mass": self.min_cluster_mass,
                "hdp_elbo_scale": self.hdp_elbo_scale,
                "continuous_elbo_scale": self.continuous_elbo_scale,
                "parameter_kl_scale": self.parameter_kl_scale,
                "snr_weight_mode": self.snr_weight_mode,
            },
            "hdp": asdict(self.hdp),
            "convergence": asdict(self.convergence),
            "gp": {key: options.pop(key) for key in list(options) if key in _GP_KEYS},
            "warping": {key: options.pop(key) for key in list(options) if key in _WARP_KEYS},
            "lds": {
                "learn_observation_matrix": self.learn_observation_matrix,
                "couple_gp_lds_noise": self.couple_gp_lds_noise,
            },
            "advanced": options,
        }

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=indent)
            stream.write("\n")
