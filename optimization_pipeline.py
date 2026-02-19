"""Biomechanical helpers used by the Streamlit app."""

from __future__ import annotations

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def biomechanical_model(params, n_steps: int = 200):
    """
    Compute knee stress trajectory from parameters.

    params columns: [angle_init, raideur, damping, frequence]
    Returns stress trajectory with shape [batch, n_steps].
    """
    if TORCH_AVAILABLE and isinstance(params, torch.Tensor):
        if params.ndim == 1:
            params = params.unsqueeze(0)
        angle_init, raideur, damping, frequence = params.unbind(dim=1)
        time = torch.linspace(0.0, 1.0, n_steps, dtype=params.dtype, device=params.device)
        stress = (
            raideur.unsqueeze(1) * angle_init.unsqueeze(1)
            + damping.unsqueeze(1)
            * torch.sin(2.0 * torch.pi * frequence.unsqueeze(1) * time.unsqueeze(0))
        )
        return stress

    params_np = np.asarray(params, dtype=float)
    if params_np.ndim == 1:
        params_np = params_np.reshape(1, -1)
    angle_init = params_np[:, 0]
    raideur = params_np[:, 1]
    damping = params_np[:, 2]
    frequence = params_np[:, 3]
    time = np.linspace(0.0, 1.0, n_steps)
    stress = (
        raideur[:, None] * angle_init[:, None]
        + damping[:, None] * np.sin(2.0 * np.pi * frequence[:, None] * time[None, :])
    )
    return stress


def pain_from_stress(stress, stress_target: float = 3.0):
    """
    Convert stress trajectory to pain trajectory.

    Pain increases when stress deviates from a comfort target.
    """
    if TORCH_AVAILABLE and isinstance(stress, torch.Tensor):
        delta = torch.abs(stress - stress_target)
        return 0.6 * delta + 0.4 * torch.exp(0.35 * delta)
    stress_np = np.asarray(stress, dtype=float)
    delta = np.abs(stress_np - stress_target)
    return 0.6 * delta + 0.4 * np.exp(0.35 * delta)


def total_variation_1d(signal):
    """
    TV seminorm along the last axis.

    For shape [n], returns a scalar.
    For shape [..., n], returns shape [...].
    """
    if TORCH_AVAILABLE and isinstance(signal, torch.Tensor):
        if signal.ndim < 1:
            raise ValueError("signal must have at least one dimension")
        diff = signal[..., 1:] - signal[..., :-1]
        tv = torch.sum(torch.abs(diff), dim=-1)
        return tv

    signal_np = np.asarray(signal, dtype=float)
    if signal_np.ndim < 1:
        raise ValueError("signal must have at least one dimension")
    diff = signal_np[..., 1:] - signal_np[..., :-1]
    tv = np.sum(np.abs(diff), axis=-1)
    return tv


def chambolle_tv_denoise_1d(
    signal,
    weight: float = 0.15,
    n_iter: int = 80,
    tau: float = 0.125,
) -> np.ndarray:
    """
    1D TV denoising using Chambolle's projection algorithm.

    Minimizes: 0.5 * ||u - signal||_2^2 + weight * TV(u)
    """
    y = np.asarray(signal, dtype=float).reshape(-1)
    if y.size <= 1 or weight <= 0:
        return y.copy()

    p = np.zeros(y.size - 1, dtype=float)

    for _ in range(max(1, int(n_iter))):
        div_p = np.empty_like(y)
        div_p[0] = -p[0]
        div_p[1:-1] = p[:-1] - p[1:]
        div_p[-1] = p[-1]

        u = y - weight * div_p
        grad_u = np.diff(u)

        p = p + (tau / weight) * grad_u
        p = p / np.maximum(1.0, np.abs(p))

    div_p = np.empty_like(y)
    div_p[0] = -p[0]
    div_p[1:-1] = p[:-1] - p[1:]
    div_p[-1] = p[-1]
    return y - weight * div_p
