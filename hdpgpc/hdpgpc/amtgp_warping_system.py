# amtgp_warping_system.py
# Sequential AMTGP-style warping system for HDP–GPC (PyTorch).
#
# Interface compatibility:
#   - Warping_system(...): constructor
#   - compute_warp(x_model, y_model, y_target, theta, noise, visualize, verbose, train_iter)
#       -> x_warp, y_warp, lik_warp, losses
#   - update_warp(x_train, x_warp)
#   - .warp_gp.log_sq_error(x_model, y_target, x_warp, noise)
#
# This implements a monotonic aligned-grid warp g(t) with learnable positive increments.
# The warp is optimized with a MAP objective (data fit + smoothness + amplitude penalty),
# warm-started for sequential use.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch


TensorLike = Union[torch.Tensor, float]


def _as_1d(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


def _as_y(y: torch.Tensor) -> torch.Tensor:
    y = torch.as_tensor(y)
    if y.ndim == 1:
        return y[:, None]  # (T, 1)
    if y.ndim == 2:
        return y
    # flatten all but time dimension
    return y.reshape(y.shape[0], -1)


def _safe_noise(noise: Optional[TensorLike],
                default: float,
                bounds: Tuple[float, float],
                device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
    n = default if noise is None else float(noise)
    lo, hi = bounds
    n = max(lo, min(hi, n))
    return torch.tensor(n, device=device, dtype=dtype)


def _lin_interp_1d(x_grid: torch.Tensor, y_grid: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """
    Differentiable-ish linear interpolation in torch using searchsorted.
    x_grid: (T,), strictly increasing
    y_grid: (T, D)
    xq:     (Tq,)
    returns yq: (Tq, D)
    """
    x_grid = _as_1d(x_grid)
    y_grid = _as_y(y_grid)
    xq = _as_1d(xq)

    # Clamp queries to domain
    xq = torch.clamp(xq, x_grid[0], x_grid[-1])

    idx_hi = torch.searchsorted(x_grid, xq, right=False)
    idx_hi = torch.clamp(idx_hi, 1, x_grid.numel() - 1)
    idx_lo = idx_hi - 1

    x_lo = x_grid[idx_lo]
    x_hi = x_grid[idx_hi]
    y_lo = y_grid[idx_lo]
    y_hi = y_grid[idx_hi]

    denom = (x_hi - x_lo).unsqueeze(-1)  # (Tq, 1)
    w = ((xq - x_lo) / (x_hi - x_lo + 1e-12)).unsqueeze(-1)  # (Tq, 1)
    return (1.0 - w) * y_lo + w * y_hi


@dataclass
class _LossTrace:
    loss: List[float]
    data: List[float]
    smooth: List[float]
    amp: List[float]


class _WarpGPScore:
    """
    Minimal scorer used by HDP–GPC as a replacement for the current warp_gp.
    Provides log_sq_error(...) with the same semantic role: a scalar score based on SSE.
    """
    def __init__(self, x_grid: torch.Tensor):
        self.x_grid = _as_1d(x_grid)

    @torch.no_grad()
    def log_sq_error(self,
                     x_model: torch.Tensor,
                     y_target: torch.Tensor,
                     x_warp: torch.Tensor,
                     noise: TensorLike) -> torch.Tensor:
        """
        Score = -0.5 * SSE/noise  (up to additive constants).
        x_model: (T,)
        y_target: (T,) or (T,D) defined on x_model grid
        x_warp: (T,) offsets so that g = x_model + x_warp
        """
        x_model = _as_1d(x_model)
        x_warp = _as_1d(x_warp)
        y_target = _as_y(y_target)

        g = x_model + x_warp
        y_warp = _lin_interp_1d(x_model, y_target, g)
        # This "baseline" score treats y_warp as prediction of itself (SSE=0) if used wrongly,
        # but in HDP–GPC it is commonly used for relative comparisons and regularization.
        # If you prefer, replace with template-based scoring.
        sse = torch.sum((y_warp - y_target) ** 2)
        n = float(noise)
        return -0.5 * sse / max(n, 1e-12)


class Warping_system:
    """
    Sequential AMTGP-style warper.

    Key design choices for HDP–GPC inner-loop speed:
      - Warp parameterized by positive increments on a *coarse* control grid (n_ctrl),
        expanded to full length T and integrated (cumsum) to enforce monotonicity.
      - MAP objective:
            data_fit(y_target(g(t)) ~ y_model(t))
          + lambda_smooth * ||D2 x_warp||^2
          + lambda_amp * ||x_warp||^2
      - Warm start across calls (per-state object persists in HDP–GPC).

    Parameters like theta=(rho, omega) can be mapped to regularization strengths:
      - rho -> smoothness scale (higher rho => smoother / less penalty)
      - omega -> amplitude scale (higher omega => allow larger warps)
    """

    def __init__(self,
                 x_basis_warp: torch.Tensor,
                 noise_warp: float = 1e-2,
                 bound_noise_warp: Tuple[float, float] = (1e-6, 1e2),
                 recursive: bool = True,
                 cuda: bool = False,
                 bayesian: bool = True,
                 mode: str = "balanced",
                 n_ctrl: int = 32,
                 lr: float = 5e-2,
                 lambda_smooth: float = 1.0,
                 lambda_amp: float = 1e-3):
        self.device = torch.device("cuda") if (cuda and torch.cuda.is_available()) else torch.device("cpu")
        self.dtype = torch.float64  # match your codebase default
        self.x_basis = _as_1d(torch.as_tensor(x_basis_warp, device=self.device, dtype=self.dtype))
        self.T = self.x_basis.numel()

        self.noise_warp_default = float(noise_warp)
        self.noise_bounds = bound_noise_warp

        self.recursive = bool(recursive)
        self.bayesian = bool(bayesian)
        self.mode = str(mode)

        self.n_ctrl = int(max(4, min(n_ctrl, self.T)))
        self.lr = float(lr)

        # Base regularizers (can be overridden by theta)
        self.lambda_smooth_base = float(lambda_smooth)
        self.lambda_amp_base = float(lambda_amp)

        # Warm-start parameters (control increments, unconstrained)
        self._u_ctrl_prev: Optional[torch.Tensor] = None

        # Expose a scorer as expected by the HDP–GPC code
        self.warp_gp = _WarpGPScore(self.x_basis)

    # ---------- internal helpers ----------

    def _expand_ctrl_to_T(self, u_ctrl: torch.Tensor) -> torch.Tensor:
        """
        Expand control vector u_ctrl (n_ctrl,) to length T by linear interpolation
        over control locations in x.
        """
        u_ctrl = _as_1d(u_ctrl)
        # Control x positions uniformly across the domain
        x0 = self.x_basis[0]
        x1 = self.x_basis[-1]
        x_ctrl = torch.linspace(float(x0), float(x1), steps=self.n_ctrl, device=self.device, dtype=self.dtype)
        uT = _lin_interp_1d(x_ctrl, u_ctrl[:, None], self.x_basis)[:, 0]  # (T,)
        return uT

    def _monotonic_grid(self, u_ctrl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build monotonic aligned grid g(t) from unconstrained control params u_ctrl.
        Returns:
            g: (T,) aligned inputs in [x_min, x_max]
            x_warp: (T,) offsets g - x
        """
        uT = self._expand_ctrl_to_T(u_ctrl)  # (T,)
        inc = torch.nn.functional.softplus(uT) + 1e-6  # positive increments
        g_raw = torch.cumsum(inc, dim=0)
        # Normalize to [x_min, x_max]
        x_min = self.x_basis[0]
        x_max = self.x_basis[-1]
        g = (g_raw - g_raw[0]) / (g_raw[-1] - g_raw[0] + 1e-12)
        g = x_min + (x_max - x_min) * g
        x_warp = g - self.x_basis
        return g, x_warp

    def _second_diff_penalty(self, x_warp: torch.Tensor) -> torch.Tensor:
        """
        Smoothness penalty: ||D2 x_warp||^2
        """
        # second finite difference
        d2 = x_warp[:-2] - 2.0 * x_warp[1:-1] + x_warp[2:]
        return torch.sum(d2 * d2)

    def _theta_to_lambdas(self, theta: Any) -> Tuple[float, float]:
        """
        Map theta (commonly (rho, omega)) to lambda_smooth and lambda_amp.
        If theta is None or malformed, fallback to base values.
        """
        lam_s = self.lambda_smooth_base
        lam_a = self.lambda_amp_base

        if theta is None:
            return lam_s, lam_a

        # Accept (rho, omega) or dict-like.
        try:
            if isinstance(theta, (tuple, list)) and len(theta) >= 2:
                rho = float(theta[0])
                omg = float(theta[1])
                # Heuristic mapping:
                #   larger rho => smoother allowed => smaller penalty
                #   larger omega => larger amplitude allowed => smaller penalty
                lam_s = self.lambda_smooth_base / (rho * rho + 1e-12)
                lam_a = self.lambda_amp_base / (omg * omg + 1e-12)
            elif isinstance(theta, dict):
                rho = float(theta.get("rho", 1.0))
                omg = float(theta.get("omega", 1.0))
                lam_s = self.lambda_smooth_base / (rho * rho + 1e-12)
                lam_a = self.lambda_amp_base / (omg * omg + 1e-12)
        except Exception:
            # keep defaults
            pass

        return lam_s, lam_a

    # ---------- public API expected by HDP–GPC ----------

    def compute_warp(self,
                     x_model: torch.Tensor,
                     y_model: torch.Tensor,
                     y_target: torch.Tensor,
                     theta: Any = None,
                     noise: Optional[TensorLike] = None,
                     visualize: bool = False,
                     verbose: bool = False,
                     train_iter: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        """
        Fit warp aligning y_target to y_model on grid x_model.

        Expected by HDP–GPC:
            x_warp: (T,) offsets so that g = x_model + x_warp
            y_warp: (T,) warped target signal on model grid (same shape as y_target)
            lik_warp: scalar (torch.Tensor)
            losses: dict of traces

        Notes:
          - y_model is the "template" (e.g., GP posterior mean of state m at x_model).
          - y_target is the observed signal segment on x_model.
        """
        x_model = _as_1d(torch.as_tensor(x_model, device=self.device, dtype=self.dtype))
        y_model = _as_y(torch.as_tensor(y_model, device=self.device, dtype=self.dtype))
        y_target = _as_y(torch.as_tensor(y_target, device=self.device, dtype=self.dtype))

        if x_model.numel() != self.T:
            # Allow using a different grid; update basis for this object
            self.x_basis = x_model
            self.T = x_model.numel()
            self.warp_gp = _WarpGPScore(self.x_basis)
            self.n_ctrl = int(max(4, min(self.n_ctrl, self.T)))

        # Noise and regularization
        n = _safe_noise(noise, self.noise_warp_default, self.noise_bounds, self.device, self.dtype)
        lam_s, lam_a = self._theta_to_lambdas(theta)

        # Initialize control parameters (warm-start if recursive)
        if self.recursive and (self._u_ctrl_prev is not None) and (self._u_ctrl_prev.numel() == self.n_ctrl):
            u_ctrl = self._u_ctrl_prev.clone().detach().to(self.device, self.dtype).requires_grad_(True)
        else:
            # near-identity warp: increments almost constant
            u_ctrl = torch.zeros(self.n_ctrl, device=self.device, dtype=self.dtype, requires_grad=True)

        opt = torch.optim.Adam([u_ctrl], lr=self.lr)

        trace = _LossTrace(loss=[], data=[], smooth=[], amp=[])

        # Optimization loop
        for it in range(int(train_iter)):
            opt.zero_grad(set_to_none=True)

            g, x_warp = self._monotonic_grid(u_ctrl)  # (T,), (T,)

            # Warp target: y_target(g(t))
            y_warp = _lin_interp_1d(x_model, y_target, g)  # (T, D)

            # Data fit to template y_model(t)
            resid = (y_warp - y_model)
            sse = torch.sum(resid * resid)

            # Regularizers
            smooth_pen = self._second_diff_penalty(x_warp)
            amp_pen = torch.sum(x_warp * x_warp)

            # Objective
            # We scale by noise to keep consistent units.
            data_term = 0.5 * sse / (n + 1e-12)
            loss = data_term + lam_s * smooth_pen + lam_a * amp_pen

            loss.backward()
            opt.step()

            # Trace
            trace.loss.append(float(loss.detach().cpu()))
            trace.data.append(float(data_term.detach().cpu()))
            trace.smooth.append(float((lam_s * smooth_pen).detach().cpu()))
            trace.amp.append(float((lam_a * amp_pen).detach().cpu()))

            if verbose and (it % max(1, train_iter // 10) == 0 or it == train_iter - 1):
                print(f"[warp] it={it:03d} loss={trace.loss[-1]:.4e} "
                      f"data={trace.data[-1]:.4e} smooth={trace.smooth[-1]:.4e} amp={trace.amp[-1]:.4e}")

        # Final warp
        with torch.no_grad():
            g, x_warp = self._monotonic_grid(u_ctrl.detach())
            y_warp = _lin_interp_1d(x_model, y_target, g)

            # "lik_warp" to use inside HDP–GPC:
            # Use Gaussian log-likelihood of warped target under the *template* (up to constants)
            # plus optional penalty terms (MAP).
            # If your HDP–GPC expects pure "data likelihood", remove the reg terms.
            sse = torch.sum((y_warp - y_model) ** 2)
            T_eff = float(y_warp.numel())
            ll = -0.5 * (sse / (n + 1e-12) + T_eff * torch.log(2.0 * torch.tensor(math.pi, device=self.device, dtype=self.dtype) * (n + 1e-12)))

            if self.bayesian:
                # include penalties as log-priors (negative of penalties)
                ll = ll - (lam_s * self._second_diff_penalty(x_warp) + lam_a * torch.sum(x_warp * x_warp))

            lik_warp = ll

        # Warm-start update for sequential use
        if self.recursive:
            self._u_ctrl_prev = u_ctrl.detach().clone()

        losses = {
            "loss": trace.loss,
            "data": trace.data,
            "smooth": trace.smooth,
            "amp": trace.amp,
        }

        # Return shapes compatible with your current code:
        # x_warp: (T,) ; y_warp: (T,) if D==1 else (T,D)
        y_warp_out = y_warp[:, 0] if y_warp.shape[1] == 1 else y_warp
        return x_warp.detach(), y_warp_out.detach(), lik_warp.detach(), losses

    def update_warp(self, x_train: torch.Tensor, x_warp: torch.Tensor) -> None:
        """
        Optional hook called by HDP–GPC to update warp memory.
        In this sequential version, we simply keep the last fitted parameters (already done in compute_warp).
        This method exists to match the expected interface.

        If you want stronger recursion, you can:
          - blend previous and new params
          - maintain an EMA of u_ctrl
          - increase/decrease regularization based on stability
        """
        if not self.recursive:
            return
        # Nothing else required: warm-start is already handled.
        # You may store last x_warp for debugging:
        self._last_x_warp = _as_1d(torch.as_tensor(x_warp, device=self.device, dtype=self.dtype)).detach()

    def reset(self) -> None:
        """Clear sequential state."""
        self._u_ctrl_prev = None
        if hasattr(self, "_last_x_warp"):
            delattr(self, "_last_x_warp")