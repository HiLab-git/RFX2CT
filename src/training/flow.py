"""Rectified flow and Self-Rollout with noise at t=0 and CT at t=1."""
from __future__ import annotations

from typing import Callable, Tuple

import torch


def view_timestep(t: torch.Tensor, ndim: int) -> torch.Tensor:
  if t.ndim != 1:
    raise ValueError("Timesteps must have shape [batch]")
  return t.reshape(t.shape[0], *([1] * (ndim - 1)))


def interpolate(noise: torch.Tensor, endpoint: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  if noise.shape != endpoint.shape or t.shape != (noise.shape[0],):
    raise ValueError("Noise, endpoint and timestep batch shapes must agree")
  fraction = view_timestep(t, noise.ndim)
  return (1.0 - fraction) * noise + fraction * endpoint, endpoint - noise


def sample_rf_batch(real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  noise = torch.randn_like(real)
  t = torch.rand(real.shape[0], device=real.device, dtype=real.dtype)
  state, target = interpolate(noise, real, t)
  return state, t, target


@torch.no_grad()
def euler_solve(velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                noise: torch.Tensor, steps: int,
                end_t: float | torch.Tensor = 1.0) -> torch.Tensor:
  """Integrate from zero to each sample's end_t using positive Euler steps."""
  if steps <= 0:
    raise ValueError("Euler steps must be positive")
  if torch.is_tensor(end_t):
    terminal = end_t.to(device=noise.device, dtype=noise.dtype)
  else:
    terminal = noise.new_full((noise.shape[0],), float(end_t))
  if terminal.shape != (noise.shape[0],) or not torch.all(
      (terminal >= 0.0) & (terminal <= 1.0)):
    raise ValueError("end_t must lie in [0, 1], one value per sample")
  dt = terminal / int(steps)
  state = noise.detach().clone()
  for index in range(int(steps)):
    t = dt * index
    state = state + view_timestep(dt, state.ndim) * velocity(state, t)
  return state.detach()


def rollout_target_velocity(real: torch.Tensor, state: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
  if state.shape != real.shape or t.shape != (real.shape[0],):
    raise ValueError("Rollout state, endpoint and timestep batch shapes must agree")
  if not torch.all((t >= 0.0) & (t < 1.0)):
    raise ValueError("Rollout target requires 0 <= t < 1")
  return (real - state) / view_timestep(1.0 - t, real.ndim)


@torch.no_grad()
def sample_self_rollout_batch(
    real: torch.Tensor,
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int = 2, min_t: float = 0.2, max_t: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if not 0.0 <= min_t < max_t < 1.0:
    raise ValueError("Self-Rollout requires 0 <= min_t < max_t < 1")
  noise = torch.randn_like(real)
  t = torch.empty(real.shape[0], device=real.device, dtype=real.dtype).uniform_(
    float(min_t), float(max_t))
  state = euler_solve(velocity, noise, steps=steps, end_t=t)
  return state, t, rollout_target_velocity(real, state, t)
