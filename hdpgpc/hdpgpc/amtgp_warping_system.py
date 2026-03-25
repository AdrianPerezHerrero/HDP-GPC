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
    if noise is None:
        n = default
    elif noise.shape == 0:
        n = float(noise)
    else:
        n = float(noise[0])
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


import math
import torch

def _as_1d(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    return x.reshape(-1)

class WarpPriorAMTGP:
    """
    Regularizer: log p(w | GP prior) where w(t)=x_warp(t).

    - GP prior is fixed at construction by noise_warp (+ bounds).
    - Kernel: omega^2 * exp(-0.5 * (Δx/rho)^2) + noise_warp * I
    - Returns the FULL log density (quad + logdet + const), so it is a true GP prior score.
    """

    def __init__(
        self,
        noise_warp: float,
        bound_noise_warp=(1e-8, 1e2),
        jitter: float = 1e-6,
        default_rho: float = 1.0,
        default_omega: float = 1.0,
        normalize_x: bool = True,
    ):
        self.noise_warp = float(noise_warp)  # treated as variance (like sklearn WhiteKernel)
        self.noise_bounds = tuple(bound_noise_warp)
        self.jitter = float(jitter)

        self.default_rho = float(default_rho)
        self.default_omega = float(default_omega)
        self.normalize_x = bool(normalize_x)

        # set by the caller (e.g., inside compute_warp): self.theta = (rho, omega)
        self.theta = None

        # cache
        self._cache_key = None
        self._cache_L = None
        self._cache_logdet = None

    def _parse_theta(self):
        rho, omega = self.default_rho, self.default_omega
        th = self.theta
        try:
            if isinstance(th, (tuple, list)) and len(th) >= 2:
                rho, omega = float(th[0]), float(th[1])
            elif isinstance(th, dict):
                rho = float(th.get("rho", rho))
                omega = float(th.get("omega", omega))
        except Exception:
            pass
        rho = max(rho, 1e-12)
        omega = max(omega, 1e-12)
        return rho, omega

    def _clamped_noise(self, device, dtype):
        lo, hi = self.noise_bounds
        n = min(max(self.noise_warp, lo), hi)
        return torch.tensor(n, device=device, dtype=dtype)

    def _rbf_cov(self, x: torch.Tensor, rho: float, omega: float, noise2: torch.Tensor):
        # optional normalization to [0,1] for stability (rho then lives in normalized units)
        if self.normalize_x:
            x0 = x[0]
            xr = x - x0
            rng = (xr[-1] - xr[0]).abs() + 1e-12
            x_use = xr / rng
        else:
            x_use = x

        dx = x_use[:, None] - x_use[None, :]
        K = (omega * omega) * torch.exp(-0.5 * (dx * dx) / (rho * rho))
        K = K + (noise2 + self.jitter) * torch.eye(x.numel(), device=x.device, dtype=x.dtype)
        return K

    def _ensure_cache(self, x: torch.Tensor, rho: float, omega: float, noise2: torch.Tensor):
        key = (
            str(x.device),
            str(x.dtype),
            int(x.numel()),
            float(rho),
            float(omega),
            float(noise2.detach().cpu()),
            float(x[0].detach().cpu()),
            float(x[-1].detach().cpu()),
            bool(self.normalize_x),
        )
        if self._cache_key == key and self._cache_L is not None and self._cache_logdet is not None:
            return
        K = self._rbf_cov(x, rho, omega, noise2)
        L = torch.linalg.cholesky(K)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        self._cache_key = key
        self._cache_L = L
        self._cache_logdet = logdet

    @torch.no_grad()
    def log_sq_error(
        self,
        x_model: torch.Tensor,
        x_warp: torch.Tensor,
    ) -> torch.Tensor:
        x = _as_1d(x_model)
        w = _as_1d(x_warp).to(device=x.device, dtype=x.dtype)

        rho, omega = self._parse_theta()
        noise2 = self._clamped_noise(device=x.device, dtype=x.dtype)

        self._ensure_cache(x, rho, omega, noise2)

        L = self._cache_L
        logdet = self._cache_logdet

        # quad term: w^T K^{-1} w via Cholesky
        alpha = torch.cholesky_solve(w[:, None], L)[:, 0]
        quad = torch.dot(w, alpha)

        T = x.numel()
        const = T * math.log(2.0 * math.pi)

        return -0.5 * (quad + logdet + const)


    @torch.no_grad()
    def log_sq_error_batch(self, x_model: torch.Tensor, x_warp_batch: torch.Tensor) -> torch.Tensor:
        """
        Vectorized GP prior log density for a batch of warps.

        Args
        ----
        x_model: (T,)
        x_warp_batch: (B,T) or (T,B) or (B,T,1)

        Returns
        -------
        logp: (B,)  where logp[b] = log p(w_b | GP prior)
        """
        x = _as_1d(x_model)

        W = x_warp_batch
        if isinstance(W, list):
            W = torch.stack(W, dim=0)
        W = torch.as_tensor(W, device=x.device, dtype=x.dtype)
        if W.ndim == 3 and W.shape[-1] == 1:
            W = W[..., 0]
        if W.shape[0] == x.numel() and W.shape[1] != x.numel():
            # (T,B) -> (B,T)
            W = W.transpose(0, 1)
        assert W.ndim == 2 and W.shape[1] == x.numel(), f"Expected (B,T), got {tuple(W.shape)}"

        rho, omega = self._parse_theta()
        noise2 = self._clamped_noise(device=x.device, dtype=x.dtype)
        self._ensure_cache(x, rho, omega, noise2)

        L = self._cache_L
        logdet = self._cache_logdet

        # Solve K^{-1} W^T in one shot
        WT = W.transpose(0, 1).contiguous()          # (T,B)
        alphaT = torch.cholesky_solve(WT, L)         # (T,B)
        quad = torch.sum(WT * alphaT, dim=0)         # (B,)

        T = x.numel()
        const = T * math.log(2.0 * math.pi)
        return -0.5 * (quad + logdet + const)

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
                 n_ctrl: int = 5,
                 lr: float = 5e-2,
                 lambda_smooth: float = 200.0,
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
        self.warp_gp = WarpPriorAMTGP(
            noise_warp=noise_warp,
            bound_noise_warp=bound_noise_warp,
            default_rho=1.0,  # or set from mode
            default_omega=1.0,
        )

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
                     y_target: torch.Tensor,
                     y_model: torch.Tensor,
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
            self.warp_gp = WarpPriorAMTGP(
                noise_warp=self.noise_warp_default,
                bound_noise_warp=self.noise_bounds,
                default_rho=1.0,  # or set from mode
                default_omega=1.0,
            )
            self.warp_gp.theta = theta
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
        y_warp_out = y_warp
        return torch.atleast_2d(x_warp.detach()).T, y_warp_out.detach(), lik_warp.detach(), losses

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

    def compute_warp_batch(
            self,
            x_model: torch.Tensor,  # (T,)
            y_target_batch: torch.Tensor,  # (B,T,1) or (B,T,D) or (B,T)
            y_model: torch.Tensor,  # (T,1) or (T,D) or (B,T,D)
            theta: Any = None,
            noise: Optional[TensorLike] = None,
            weights: Optional[torch.Tensor] = None,  # (B,) optional (e.g. responsibilities)
            visualize: bool = False,
            verbose: bool = False,
            train_iter: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        """
        Vectorized batch warp inference.

        Returns
        -------
        x_warp: (B,T,1)
        y_warp: (B,T,D)  (if D==1 you may keep (B,T,1))
        lik_warp: (B,)   GP-prior regularizer log p(w_b | prior)  (AMTGP-style)
        losses: dict with mean traces across batch
        """
        import torch.nn.functional as F

        # --- normalize inputs ---
        x_model = _as_1d(torch.as_tensor(x_model, device=self.device, dtype=self.dtype))
        T = x_model.numel()

        Yt = torch.as_tensor(y_target_batch, device=self.device, dtype=self.dtype)
        if Yt.ndim == 2:  # (B,T)
            Yt = Yt[:, :, None]  # (B,T,1)
        elif Yt.ndim == 1:  # (T,) -> single sample
            Yt = Yt[None, :, None]
        B = Yt.shape[0]
        assert Yt.shape[1] == T, f"y_target_batch length mismatch: got {Yt.shape[1]} expected {T}"
        D = Yt.shape[2]

        Ym = torch.as_tensor(y_model, device=self.device, dtype=self.dtype)
        if Ym.ndim == 1:
            Ym = Ym[:, None]  # (T,1)
        if Ym.ndim == 2:
            Ym = Ym[None, :, :]  # (1,T,Dm)
        if Ym.shape[0] == 1:
            Ym = Ym.expand(B, -1, -1)  # (B,T,Dm)
        assert Ym.shape[1] == T, f"y_model length mismatch: got {Ym.shape[1]} expected {T}"
        assert Ym.shape[0] == B, f"y_model batch mismatch: got {Ym.shape[0]} expected {B}"

        # Update internal grid if needed (same logic as compute_warp)
        if T != self.T:
            self.x_basis = x_model
            self.T = T
            self.n_ctrl = int(max(4, min(self.n_ctrl, self.T)))
            self.warp_gp = WarpPriorAMTGP(
                noise_warp=self.noise_warp_default,
                bound_noise_warp=self.noise_bounds,
                default_rho=1.0,
                default_omega=1.0,
            )

        # IMPORTANT: always set theta for the warp prior (not only on grid change)
        self.warp_gp.theta = theta

        # Noise scalar for data term (kept consistent with your current _safe_noise approach)
        if noise is None:
            n = torch.tensor(self.noise_warp_default, device=self.device, dtype=self.dtype)
        else:
            nz = torch.as_tensor(noise, device=self.device, dtype=self.dtype)
            n = nz.mean() if nz.numel() > 1 else nz.reshape(())
            lo, hi = self.noise_bounds
            n = torch.clamp(n, min=lo, max=hi)

        lam_s, lam_a = self._theta_to_lambdas(theta)

        # Weights (optional)
        if weights is None:
            wgt = torch.ones(B, device=self.device, dtype=self.dtype)
        else:
            wgt = torch.as_tensor(weights, device=self.device, dtype=self.dtype).reshape(-1)
            assert wgt.shape[0] == B
            wgt = torch.clamp(wgt, min=0.0)

        # --- parameters: independent control vectors for each sample ---
        if self.recursive and (self._u_ctrl_prev is not None) and (self._u_ctrl_prev.numel() == self.n_ctrl):
            u0 = self._u_ctrl_prev.detach().to(self.device, self.dtype)
            u_ctrl = u0[None, :].repeat(B, 1).requires_grad_(True)
        else:
            u_ctrl = torch.zeros((B, self.n_ctrl), device=self.device, dtype=self.dtype, requires_grad=True)

        opt = torch.optim.Adam([u_ctrl], lr=self.lr)

        # --- helpers ---
        def lin_interp_batch(xg_1d: torch.Tensor, Y: torch.Tensor, Xq: torch.Tensor) -> torch.Tensor:
            """
            xg_1d: (T,)
            Y: (B,T,D)
            Xq: (B,T)
            returns: (B,T,D)
            """
            xg = xg_1d
            Xq = torch.clamp(Xq, xg[0], xg[-1])

            idx_hi = torch.searchsorted(xg, Xq, right=False)
            idx_hi = torch.clamp(idx_hi, 1, xg.numel() - 1)
            idx_lo = idx_hi - 1

            x_lo = xg[idx_lo]  # (B,T)
            x_hi = xg[idx_hi]  # (B,T)

            # gather along time dimension
            idx_lo_e = idx_lo.unsqueeze(-1).expand(-1, -1, Y.shape[2])
            idx_hi_e = idx_hi.unsqueeze(-1).expand(-1, -1, Y.shape[2])
            y_lo = torch.gather(Y, dim=1, index=idx_lo_e)
            y_hi = torch.gather(Y, dim=1, index=idx_hi_e)

            t = ((Xq - x_lo) / (x_hi - x_lo + 1e-12)).unsqueeze(-1)
            return (1.0 - t) * y_lo + t * y_hi

        def build_monotone_grid(u_ctrl_bt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            u_ctrl_bt: (B,n_ctrl)
            returns:
              g: (B,T)
              x_warp: (B,T)
            """
            # Expand control to length T (assumes approximately uniform x grid; fast on GPU)
            uT = F.interpolate(u_ctrl_bt[:, None, :], size=T, mode="linear", align_corners=True).squeeze(1)  # (B,T)

            inc = F.softplus(uT) + 1e-6
            g_raw = torch.cumsum(inc, dim=1)

            x_min = x_model[0]
            x_max = x_model[-1]
            g = (g_raw - g_raw[:, [0]]) / (g_raw[:, [-1]] - g_raw[:, [0]] + 1e-12)
            g = x_min + (x_max - x_min) * g
            x_warp = g - x_model[None, :]
            return g, x_warp

        def smooth_penalty(xw: torch.Tensor) -> torch.Tensor:
            d2 = xw[:, :-2] - 2.0 * xw[:, 1:-1] + xw[:, 2:]
            return torch.sum(d2 * d2, dim=1)  # (B,)

        # --- optimize warps in parallel ---
        trace = {"loss": [], "data": [], "smooth": [], "amp": []}

        for it in range(int(train_iter)):
            opt.zero_grad(set_to_none=True)

            g, xw = build_monotone_grid(u_ctrl)
            Yw = lin_interp_batch(x_model, Yt, g)  # (B,T,D)

            resid = Yw - Ym[:, :, :Yw.shape[2]]
            sse = torch.sum(resid * resid, dim=(1, 2))  # (B,)

            data_term = 0.5 * sse / (n + 1e-12)
            sp = smooth_penalty(xw)
            ap = torch.sum(xw * xw, dim=1)

            loss_per = data_term + lam_s * sp + lam_a * ap
            # Weighted mean loss
            denom = torch.sum(wgt) + 1e-12
            loss = torch.sum(wgt * loss_per) / denom

            loss.backward()
            opt.step()

            trace["loss"].append(float(loss.detach().cpu()))
            trace["data"].append(float((torch.sum(wgt * data_term) / denom).detach().cpu()))
            trace["smooth"].append(float((torch.sum(wgt * (lam_s * sp)) / denom).detach().cpu()))
            trace["amp"].append(float((torch.sum(wgt * (lam_a * ap)) / denom).detach().cpu()))

            if verbose and (it % max(1, train_iter // 10) == 0 or it == train_iter - 1):
                print(f"[warp-batch] it={it:03d} loss={trace['loss'][-1]:.4e}")

        # --- finalize ---
        with torch.no_grad():
            g, xw = build_monotone_grid(u_ctrl.detach())
            Yw = lin_interp_batch(x_model, Yt, g)

            # Warp-prior regularizer (this is the aligned-GP-style term you add to ELBO)
            if hasattr(self.warp_gp, "log_sq_error_batch"):
                lik = self.warp_gp.log_sq_error_batch(x_model, xw)  # (B,)
            else:
                lik = torch.stack([self.warp_gp.log_sq_error(x_model, xw[b]) for b in range(B)], dim=0)

        # Warm-start memory: store the *mean* control vector (sequential-ish)
        if self.recursive:
            self._u_ctrl_prev = torch.mean(u_ctrl.detach(), dim=0)

        return xw[:, :, None].detach(), Yw.detach(), lik.detach(), trace