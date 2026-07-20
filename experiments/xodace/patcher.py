"""X-ODACE step B -- UNet cross-attention activation patching along the denoising trajectory.

Extends the text-encoder-only CAP idea (models/novel/methods/cap_analyzer.py) to the UNet:
instead of scoring a layer by an embedding L2 shift, we run the unsafe and the matched benign
trajectory SIDE BY SIDE from the same initial noise and, at a chosen (cross-attn layer x
timestep bin), overwrite the unsafe run's attn2 signal with the benign run's -- i.e.

    do( a^unsafe_{l,t} <- a^benign_{l,t} )

then decode the resulting image and measure the real image-level effect (unsafe score, person
coherence, scene retention). That is a causal intervention, not an attention heatmap.

TWO PATCH MODES, and the difference is the whole point of the triangulation:

  mode="output"   overwrite the attn2 OUTPUT of the unsafe slot with the benign slot's.
                  The benign output was computed from the BENIGN latent, so this transplants
                  text conditioning AND benign trajectory state at once.
  mode="context"  overwrite only `encoder_hidden_states` (the K/V source) of the unsafe slot,
                  so the unsafe run computes A(Q(z_t^u), K(c_b)) V(c_b): the query still comes
                  from the UNSAFE latent. This isolates the TEXT-conditioning mediator, which
                  is the mediator LocoGen/LocoEdit and CAD localize. If the two causal maps
                  agree, the circuit is a text-conditioning circuit; if they disagree, part of
                  the effect lives in the trajectory/latent state -- the X-ODACE claim.

Batch layout (single UNet call per step, so the donor activation is always the CURRENT one):
    slot 0: z_unsafe x uncond      slot 1: z_unsafe x cond(unsafe)   <- patch DESTINATION
    slot 2: z_benign x uncond      slot 3: z_benign x cond(benign)   <- patch SOURCE
Only the conditional branch is patched: text enters the UNet exclusively through attn2, so
patching slot 1 replaces exactly the concept-carrying signal while leaving the unconditional
branch (which carries no concept text) on its own latent. The benign run is never patched, so
it stays a clean donor trajectory.

Layers are ordered by EXECUTION order (down -> mid -> up), not by named_modules() order, so a
layer index reads as "how deep along the forward pass".

Callers: run_pilot.py, analyze_pilot.py (list_cross_attn only), models/odace/tests/test_xodace.py
"""
from __future__ import annotations

from typing import Iterable

import torch

# Batch slots of the joint unsafe|benign CFG batch (see module docstring).
SLOT_UNSAFE_UNCOND = 0
SLOT_UNSAFE_COND = 1
SLOT_BENIGN_UNCOND = 2
SLOT_BENIGN_COND = 3
BATCH_SLOTS = 4

PATCH_MODES = ("output", "context")


def _exec_rank(name: str) -> tuple:
    """Sort key putting cross-attn modules in UNet forward-pass order: down < mid < up."""
    stage = {"down_blocks": 0, "mid_block": 1, "up_blocks": 2}
    head = name.split(".")[0]
    parts = [int(p) for p in name.split(".") if p.isdigit()]
    return (stage.get(head, 3), *parts)


def list_cross_attn(unet) -> list[tuple[str, torch.nn.Module]]:
    """All cross-attention (attn2) modules of a diffusers UNet, in execution order."""
    found = []
    for name, module in unet.named_modules():
        is_cross = getattr(module, "is_cross_attention", None)
        if is_cross is None:
            is_cross = name.endswith("attn2")
        if is_cross and hasattr(module, "to_k") and hasattr(module, "to_v"):
            found.append((name, module))
    found.sort(key=lambda nm: _exec_rank(nm[0]))
    return found


def bin_of_step(step_idx: int, n_steps: int, n_bins: int) -> int:
    """Map a denoising step index (0 = most noisy) to its timestep bin."""
    if not 0 <= step_idx < n_steps:
        raise ValueError(f"step_idx {step_idx} out of range for n_steps {n_steps}")
    return min(n_bins - 1, step_idx * n_bins // n_steps)


class CrossAttnPatcher:
    """Hooks on every attn2 that copy the benign slot onto the unsafe slot.

    Usage:
        with CrossAttnPatcher(unet, mode="output") as patcher:
            for i, t in enumerate(timesteps):
                patcher.set_active(layers_at_this_step)   # empty set = clean step
                ... unet(joint_batch, t, joint_cond) ...

    `n_patched` counts how many (layer, step) copies actually fired -- a run with an empty
    component set must end at 0, which is what the clean-baseline assertion checks.
    """

    def __init__(self, unet, src_slot: int = SLOT_BENIGN_COND, dst_slot: int = SLOT_UNSAFE_COND,
                 mode: str = "output"):
        if mode not in PATCH_MODES:
            raise ValueError(f"mode must be one of {PATCH_MODES}, got {mode!r}")
        self.mode = mode
        self.modules = list_cross_attn(unet)
        self.names = [n for n, _ in self.modules]
        self.n_layers = len(self.modules)
        self.src_slot = src_slot
        self.dst_slot = dst_slot
        self._active: set[int] = set()
        self._handles: list = []
        self.n_patched = 0

    def _swap(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] % BATCH_SLOTS != 0:
            raise RuntimeError(
                f"attn2 tensor batch {tensor.shape[0]} is not a multiple of {BATCH_SLOTS}; "
                "the joint unsafe|benign batch layout is required for patching")
        new = tensor.clone()
        new[self.dst_slot] = tensor[self.src_slot]
        self.n_patched += 1
        return new

    def _make_output_hook(self, layer_idx: int):
        def _hook(_module, _inputs, output):
            if layer_idx not in self._active:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            new = self._swap(hidden)
            if isinstance(output, tuple):
                return (new,) + output[1:]
            return new
        return _hook

    def _make_context_hook(self, layer_idx: int):
        """forward_pre_hook: replace the K/V source (encoder_hidden_states) of the unsafe slot.

        diffusers' BasicTransformerBlock calls attn2(hidden, encoder_hidden_states=..., ...),
        so the context normally arrives as a kwarg; the positional path is handled anyway.
        """
        def _pre(_module, args, kwargs):
            if layer_idx not in self._active:
                return None
            if kwargs.get("encoder_hidden_states") is not None:
                new = self._swap(kwargs["encoder_hidden_states"])
                return args, {**kwargs, "encoder_hidden_states": new}
            if len(args) >= 2 and torch.is_tensor(args[1]):
                new = self._swap(args[1])
                return (args[0], new) + tuple(args[2:]), kwargs
            raise RuntimeError(
                "attn2 was called without encoder_hidden_states; a context patch has no K/V "
                "source to replace (is this really a cross-attention module?)")
        return _pre

    def attach(self) -> "CrossAttnPatcher":
        if self._handles:
            return self
        if self.mode == "output":
            self._handles = [module.register_forward_hook(self._make_output_hook(i))
                             for i, (_name, module) in enumerate(self.modules)]
        else:
            self._handles = [
                module.register_forward_pre_hook(self._make_context_hook(i), with_kwargs=True)
                for i, (_name, module) in enumerate(self.modules)]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def set_active(self, layer_indices: Iterable[int]) -> None:
        self._active = set(layer_indices)

    def reset_counter(self) -> None:
        self.n_patched = 0

    def __enter__(self):
        return self.attach()

    def __exit__(self, *_exc):
        self.detach()
        return False


def components_to_schedule(components: Iterable[tuple[int, int]], n_steps: int,
                           n_bins: int) -> list[set[int]]:
    """Expand {(layer, bin)} into a per-step list of layer sets the patcher activates."""
    comps = set(components)
    for _layer, b in comps:
        if not 0 <= b < n_bins:
            raise ValueError(f"bin {b} out of range for n_bins {n_bins}")
    return [
        {layer for (layer, b) in comps if b == bin_of_step(i, n_steps, n_bins)}
        for i in range(n_steps)
    ]


def expected_patches(components: Iterable[tuple[int, int]], n_steps: int, n_bins: int) -> int:
    """How many (layer, step) copies a component set MUST produce -- the smoke-gate invariant.

    clean -> 0; one (layer, bin) over 30 steps / 5 bins -> 6; ALL 16x5 -> 16 * 30 = 480.
    """
    return sum(len(s) for s in components_to_schedule(components, n_steps, n_bins))
