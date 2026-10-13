"""Batched post-IK actuator-residual mapping for the G6 screw controller.

The mapper is deliberately independent of the phase supervisor and learner.
Its caller supplies the executable-channel mask after all phase and measured
gate decisions have been made.  All persistent state is held in Torch tensors,
so one instance can service every MJWarp world without host-side loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


ACTUATOR_RESIDUAL_DIM = 16
LINEAR_ACTUATOR_INDICES = (0, 3, 6, 9)
ROTARY_ACTUATOR_INDICES = tuple(
    index
    for index in range(ACTUATOR_RESIDUAL_DIM)
    if index not in LINEAR_ACTUATOR_INDICES
)

DEFAULT_BLEND = 0.35
DEFAULT_ROTARY_LIMIT_RAD = 0.005
DEFAULT_ROTARY_SLEW_RAD_PER_TICK = 0.0001
DEFAULT_LINEAR_LIMIT_M = 0.0005
DEFAULT_LINEAR_SLEW_M_PER_TICK = 0.00002

_STATE_SCHEMA = "screw-post-ik-actuator-residual-v1"


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class ActuatorResidualStep:
    """Outputs and diagnostics from one batched residual-control tick."""

    applied_ctrl: Any
    requested_residual: Any
    commanded_residual: Any
    realised_residual: Any
    executable_mask: Any
    action_clipped_mask: Any
    ctrl_saturation_mask: Any


class BatchedActuatorResidualMapper:
    """Map normalised TQC actions to bounded G6 actuator corrections.

    The mapping order is action clipping, unit-aware scaling, exponential
    blending, per-tick slew limiting, absolute residual limiting, actuator
    ctrlrange clipping, and anti-windup.  A false executable mask immediately
    removes the corresponding residual authority and clears its carried state.

    Parameters are expressed in native actuator units: metres for the four
    linear slides and radians for the twelve rotary actuators.
    """

    action_dim = ACTUATOR_RESIDUAL_DIM
    linear_indices = LINEAR_ACTUATOR_INDICES
    rotary_indices = ROTARY_ACTUATOR_INDICES

    def __init__(
        self,
        worlds: int,
        ctrl_low: Iterable[float],
        ctrl_high: Iterable[float],
        *,
        device: Any = "cpu",
        dtype: Any = None,
        blend: float = DEFAULT_BLEND,
        rotary_limit: float = DEFAULT_ROTARY_LIMIT_RAD,
        rotary_slew: float = DEFAULT_ROTARY_SLEW_RAD_PER_TICK,
        linear_limit: float = DEFAULT_LINEAR_LIMIT_M,
        linear_slew: float = DEFAULT_LINEAR_SLEW_M_PER_TICK,
    ) -> None:
        torch = _torch()
        if isinstance(worlds, bool) or int(worlds) != worlds or worlds <= 0:
            raise ValueError("worlds must be a positive integer")
        self.worlds = int(worlds)
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.float32
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be a floating-point Torch dtype")

        self.ctrl_low = torch.as_tensor(
            ctrl_low, device=self.device, dtype=self.dtype
        ).reshape(-1)
        self.ctrl_high = torch.as_tensor(
            ctrl_high, device=self.device, dtype=self.dtype
        ).reshape(-1)
        if self.ctrl_low.shape != (self.action_dim,):
            raise ValueError(
                f"ctrl_low must have shape ({self.action_dim},), "
                f"got {tuple(self.ctrl_low.shape)}"
            )
        if self.ctrl_high.shape != (self.action_dim,):
            raise ValueError(
                f"ctrl_high must have shape ({self.action_dim},), "
                f"got {tuple(self.ctrl_high.shape)}"
            )
        self._require_finite("control ranges", self.ctrl_low, self.ctrl_high)
        if bool((self.ctrl_low >= self.ctrl_high).any()):
            raise ValueError("every ctrl_low value must be below ctrl_high")

        scalars = {
            "blend": blend,
            "rotary_limit": rotary_limit,
            "rotary_slew": rotary_slew,
            "linear_limit": linear_limit,
            "linear_slew": linear_slew,
        }
        scalar_tensor = torch.as_tensor(
            tuple(scalars.values()), device=self.device, dtype=self.dtype
        )
        self._require_finite("residual settings", scalar_tensor)
        if not 0.0 < float(blend) <= 1.0:
            raise ValueError("blend must be in (0, 1]")
        if any(float(scalars[name]) <= 0.0 for name in scalars if name != "blend"):
            raise ValueError("residual limits and slew limits must be positive")

        self.blend = float(blend)
        self.limits = torch.full(
            (self.action_dim,),
            float(rotary_limit),
            device=self.device,
            dtype=self.dtype,
        )
        self.rates = torch.full(
            (self.action_dim,),
            float(rotary_slew),
            device=self.device,
            dtype=self.dtype,
        )
        linear = torch.as_tensor(
            self.linear_indices, device=self.device, dtype=torch.long
        )
        self.limits[linear] = float(linear_limit)
        self.rates[linear] = float(linear_slew)
        self.residual = torch.zeros(
            (self.worlds, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _require_finite(label: str, *values: Any) -> None:
        torch = _torch()
        if not values:
            return
        finite = torch.ones((), device=values[0].device, dtype=torch.bool)
        for value in values:
            finite &= torch.isfinite(value).all()
        if not bool(finite):
            raise FloatingPointError(f"{label} contains NaN or infinity")

    def _batched_float(self, value: Any, *, label: str):
        torch = _torch()
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        expected = (self.worlds, self.action_dim)
        if tensor.shape != expected:
            raise ValueError(
                f"{label} must have shape {expected}, got {tuple(tensor.shape)}"
            )
        return tensor

    def _batched_mask(self, value: Any, *, label: str):
        torch = _torch()
        tensor = torch.as_tensor(value, device=self.device)
        expected = (self.worlds, self.action_dim)
        if tensor.shape != expected:
            raise ValueError(
                f"{label} must have shape {expected}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bool:
            raise TypeError(f"{label} must have boolean dtype")
        return tensor

    def reset(self, world_mask: Any = None) -> None:
        """Clear carried residuals for all worlds or a boolean world subset."""

        torch = _torch()
        with torch.no_grad():
            if world_mask is None:
                self.residual.zero_()
                return
            mask = torch.as_tensor(world_mask, device=self.device)
            if mask.shape != (self.worlds,):
                raise ValueError(
                    "world_mask must have shape "
                    f"({self.worlds},), got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise TypeError("world_mask must have boolean dtype")
            self.residual.masked_fill_(mask[:, None], 0.0)

    def apply(
        self,
        nominal_ctrl: Any,
        normalized_action: Any,
        executable_mask: Any,
    ) -> ActuatorResidualStep:
        """Apply one residual tick to every world.

        ``nominal_ctrl`` is the certified post-IK command.  The caller-provided
        boolean ``executable_mask`` is the single authority source for phase and
        measured-gate restrictions.  Inputs and outputs have shape
        ``(worlds, 16)``.
        """

        torch = _torch()
        nominal = self._batched_float(nominal_ctrl, label="nominal_ctrl")
        action = self._batched_float(
            normalized_action, label="normalized_action"
        )
        mask = self._batched_mask(executable_mask, label="executable_mask")
        self._require_finite("residual mapper inputs", nominal, action)

        with torch.no_grad():
            clipped_action = action.clamp(-1.0, 1.0)
            action_clipped = action != clipped_action
            requested = clipped_action * self.limits
            requested = torch.where(mask, requested, torch.zeros_like(requested))

            # A disabled channel has no residual authority.  Clear its memory
            # before blending so phase/gate transitions cannot release a stale
            # correction when the channel is enabled again.
            previous = torch.where(
                mask, self.residual, torch.zeros_like(self.residual)
            )
            blended = (1.0 - self.blend) * previous + self.blend * requested
            delta = (blended - previous).clamp(-self.rates, self.rates)
            commanded = (previous + delta).clamp(-self.limits, self.limits)
            commanded = torch.where(
                mask, commanded, torch.zeros_like(commanded)
            )

            unclipped_ctrl = nominal + commanded
            applied = torch.maximum(
                torch.minimum(unclipped_ctrl, self.ctrl_high), self.ctrl_low
            )
            saturation = applied != unclipped_ctrl

            # Anti-windup carries only the correction that reached an enabled
            # actuator after ctrlrange clipping.  Masked channels stay exactly
            # zero even if a malformed nominal command lies outside ctrlrange.
            realised = torch.where(
                mask, applied - nominal, torch.zeros_like(commanded)
            )
            self.residual.copy_(realised)
            self._require_finite(
                "residual mapper outputs", applied, commanded, realised
            )

        return ActuatorResidualStep(
            applied_ctrl=applied,
            requested_residual=requested,
            commanded_residual=commanded,
            realised_residual=realised,
            executable_mask=mask,
            action_clipped_mask=action_clipped,
            ctrl_saturation_mask=saturation,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return the complete mapper contract and per-world dynamic state."""

        return {
            "schema": _STATE_SCHEMA,
            "worlds": self.worlds,
            "action_dim": self.action_dim,
            "linear_indices": self.linear_indices,
            "blend": self.blend,
            "ctrl_low": self.ctrl_low.detach().clone(),
            "ctrl_high": self.ctrl_high.detach().clone(),
            "limits": self.limits.detach().clone(),
            "rates": self.rates.detach().clone(),
            "residual": self.residual.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state only when its actuator and world contract matches."""

        required = {
            "schema",
            "worlds",
            "action_dim",
            "linear_indices",
            "blend",
            "ctrl_low",
            "ctrl_high",
            "limits",
            "rates",
            "residual",
        }
        missing = required.difference(state)
        if missing:
            raise KeyError(f"residual mapper state is missing {sorted(missing)}")
        if state["schema"] != _STATE_SCHEMA:
            raise ValueError("residual mapper state schema does not match")
        if int(state["worlds"]) != self.worlds:
            raise ValueError("residual mapper world count does not match")
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError("residual mapper action dimension does not match")
        if tuple(state["linear_indices"]) != self.linear_indices:
            raise ValueError("residual mapper actuator ordering does not match")
        if float(state["blend"]) != self.blend:
            raise ValueError("residual mapper blend does not match")

        torch = _torch()
        ctrl_low = torch.as_tensor(
            state["ctrl_low"], device=self.device, dtype=self.dtype
        )
        ctrl_high = torch.as_tensor(
            state["ctrl_high"], device=self.device, dtype=self.dtype
        )
        limits = torch.as_tensor(
            state["limits"], device=self.device, dtype=self.dtype
        )
        rates = torch.as_tensor(
            state["rates"], device=self.device, dtype=self.dtype
        )
        residual = torch.as_tensor(
            state["residual"], device=self.device, dtype=self.dtype
        )
        if ctrl_low.shape != (self.action_dim,):
            raise ValueError("state ctrl_low shape does not match")
        if ctrl_high.shape != (self.action_dim,):
            raise ValueError("state ctrl_high shape does not match")
        if limits.shape != (self.action_dim,):
            raise ValueError("state limits shape does not match")
        if rates.shape != (self.action_dim,):
            raise ValueError("state rates shape does not match")
        if residual.shape != (self.worlds, self.action_dim):
            raise ValueError("state residual shape does not match")
        self._require_finite(
            "residual mapper state",
            ctrl_low,
            ctrl_high,
            limits,
            rates,
            residual,
        )
        if not torch.equal(ctrl_low, self.ctrl_low):
            raise ValueError("state ctrl_low does not match")
        if not torch.equal(ctrl_high, self.ctrl_high):
            raise ValueError("state ctrl_high does not match")
        if not torch.equal(limits, self.limits):
            raise ValueError("state residual limits do not match")
        if not torch.equal(rates, self.rates):
            raise ValueError("state residual slew limits do not match")
        if bool((residual.abs() > self.limits + 1.0e-7).any()):
            raise ValueError("state residual exceeds configured limits")
        with torch.no_grad():
            self.residual.copy_(residual)


__all__ = [
    "ACTUATOR_RESIDUAL_DIM",
    "ActuatorResidualStep",
    "BatchedActuatorResidualMapper",
    "DEFAULT_BLEND",
    "DEFAULT_LINEAR_LIMIT_M",
    "DEFAULT_LINEAR_SLEW_M_PER_TICK",
    "DEFAULT_ROTARY_LIMIT_RAD",
    "DEFAULT_ROTARY_SLEW_RAD_PER_TICK",
    "LINEAR_ACTUATOR_INDICES",
    "ROTARY_ACTUATOR_INDICES",
]
