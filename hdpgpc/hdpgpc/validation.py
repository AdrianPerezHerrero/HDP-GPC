"""Input validation kept separate from stateful model code."""

from __future__ import annotations

import numpy as np
import torch


def validate_batch(x: object, y: object, n_outputs: int) -> None:
    x_array = np.asarray(x) if not torch.is_tensor(x) else x.detach().cpu().numpy()
    y_array = np.asarray(y) if not torch.is_tensor(y) else y.detach().cpu().numpy()
    if x_array.ndim != 3 or x_array.shape[-1] != 1:
        raise ValueError("x must have shape (n_samples, n_timepoints, 1)")
    if y_array.ndim != 3:
        raise ValueError("y must have shape (n_samples, n_timepoints, n_outputs)")
    if x_array.shape[:2] != y_array.shape[:2]:
        raise ValueError("x and y must agree on sample and time dimensions")
    if y_array.shape[2] != n_outputs:
        raise ValueError(
            f"y has {y_array.shape[2]} outputs but the model expects {n_outputs}"
        )
    if x_array.shape[0] == 0 or x_array.shape[1] == 0:
        raise ValueError("training batches cannot be empty")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("x and y must contain only finite values")
