"""Continuous-time knee dynamics modeled with a Neural ODE."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore[assignment]

try:
    from torchdiffeq import odeint, odeint_adjoint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None  # type: ignore[assignment]
    odeint_adjoint = None  # type: ignore[assignment]


if TORCH_AVAILABLE:
    class KneeDynamicsODE(nn.Module):
        """
        Neural ODE for knee temporal dynamics.

        State vector:
            [position, velocity, force]
        """

        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2),  # [delta_accel, dforce_dt]
            )

            # Start close to the physical model: neutral neural correction.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

            # Learnable physiological parameters.
            self.mass = nn.Parameter(torch.tensor(0.1))
            self.stiffness = nn.Parameter(torch.tensor(10.0))
            self.damping = nn.Parameter(torch.tensor(0.5))

        def forward(self, t, state):
            pos = state[..., 0:1]
            vel = state[..., 1:2]
            force = state[..., 2:3]

            mass = torch.clamp(self.mass, min=1e-3)
            physics_accel = (-self.stiffness * pos - self.damping * vel + force) / mass

            neural_correction = self.net(torch.cat([pos, vel, force], dim=-1))

            dpos_dt = vel
            dvel_dt = physics_accel + neural_correction[..., 0:1]
            dforce_dt = neural_correction[..., 1:2]
            return torch.cat([dpos_dt, dvel_dt, dforce_dt], dim=-1)
else:
    class KneeDynamicsODE:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to use KneeDynamicsODE.")


def integrate_knee_dynamics(
    model: "KneeDynamicsODE",
    initial_state: "torch.Tensor",
    time_steps: "torch.Tensor",
    method: str = "dopri5",
    use_adjoint: bool = True,
):
    """Integrate the Neural ODE trajectory."""
    if not TORCHDIFFEQ_AVAILABLE:
        raise RuntimeError("torchdiffeq is required to integrate KneeDynamicsODE.")
    solver = odeint_adjoint if use_adjoint else odeint
    return solver(model, initial_state, time_steps, method=method)
