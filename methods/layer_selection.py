"""Single source of truth for SD1.x UNet cross-attention (attn2) bands + trainable selection.

Band = attn2 module-name prefix set on the SD1.x UNet, keyed by the latent feature-map
resolution at 512px generation:

    res64: down_blocks.0 + up_blocks.3   (320 ch, 64x64)
    res32: down_blocks.1 + up_blocks.2   (640 ch, 32x32)
    res16: down_blocks.2 + up_blocks.1   (1280 ch, 16x16)  -> 5 attn2 modules on SD1.x
    res8 : mid_block                     (1280 ch,  8x8)

The names are validated against the live ``unet.named_modules()`` at selection time -- a
scope that matches an unexpected number of attn2 modules raises instead of silently
training the wrong layers. On SD1.4 the full cross-attn scope is 16 attn2 modules /
43,962,560 q+k+v+out parameters; res16 is 5 attn2 / 26,220,800 (59.6% -- NOT lightweight).

Consumers (do not redefine bands anywhere else):
  - core/trainer.py                       (training-time trainable scope)
  - methods/unet_edit.py                  (legacy set_trainable_cross_attn_kv wrapper)
  - experiments/xodace/weight_transplant.py (delta band split)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

BANDS_SD1: Dict[str, Tuple[str, ...]] = {
    "res64": ("down_blocks.0", "up_blocks.3"),
    "res32": ("down_blocks.1", "up_blocks.2"),
    "res16": ("down_blocks.2", "up_blocks.1"),
    "res8": ("mid_block",),
}

SCOPES: Tuple[str, ...] = ("all_xattn", "res64", "res32", "res16", "res8", "explicit")
PROJECTIONS: Tuple[str, ...] = ("q", "k", "v", "out")
_PROJ_ATTR = {"q": "to_q", "k": "to_k", "v": "to_v", "out": "to_out"}

# attn2 module counts on the SD1.x UNet (layers_per_block=2; 3 cross-attn down blocks,
# 3 cross-attn up blocks, 1 mid block). Checked when strict=True.
EXPECTED_ATTN2_SD1: Dict[str, int] = {
    "all_xattn": 16, "res64": 5, "res32": 5, "res16": 5, "res8": 1,
}


class LayerSelectionError(ValueError):
    """Raised when a scope/projection request does not match the live UNet."""


def band_of_module(name: str) -> Optional[str]:
    """Band of an attn2 module (or parameter) name, else None."""
    for band, prefixes in BANDS_SD1.items():
        if any(name.startswith(p) for p in prefixes):
            return band
    return None


def iter_cross_attn_modules(unet) -> List[Tuple[str, object]]:
    """(name, module) for every cross-attention (attn2) module, in named_modules() order.

    Mirrors the legacy detection in unet_edit.set_trainable_cross_attn_kv: a module is
    cross-attention iff its ``is_cross_attention`` attribute is truthy, falling back to a
    name ending in ``attn2``; it must expose to_k and to_v.
    """
    out = []
    for name, module in unet.named_modules():
        is_cross = getattr(module, "is_cross_attention", None)
        if is_cross is None:
            is_cross = name.endswith("attn2")
        if is_cross and hasattr(module, "to_k") and hasattr(module, "to_v"):
            out.append((name, module))
    return out


@dataclass
class LayerSelection:
    """Result of select_trainable_cross_attention -- everything a run manifest needs."""

    scope: str
    projections: Tuple[str, ...]
    module_names: List[str]
    parameter_names: List[str]
    trainable_parameter_count: int
    total_cross_attention_parameter_count: int
    band_of_modules: Dict[str, Optional[str]] = field(default_factory=dict)
    architecture_fingerprint: str = ""

    def to_manifest(self) -> dict:
        return {
            "scope": self.scope,
            "projections": list(self.projections),
            "module_names": self.module_names,
            "parameter_names": self.parameter_names,
            "trainable_parameter_count": self.trainable_parameter_count,
            "total_cross_attention_parameter_count":
                self.total_cross_attention_parameter_count,
            "band_of_modules": self.band_of_modules,
            "architecture_fingerprint": self.architecture_fingerprint,
        }


def _projection_modules(attn_module, projections: Sequence[str], strict: bool):
    mods = []
    for proj in projections:
        sub = getattr(attn_module, _PROJ_ATTR[proj], None)
        if sub is None:
            if strict:
                raise LayerSelectionError(
                    f"attn2 module lacks projection '{proj}' ({_PROJ_ATTR[proj]})")
            continue
        mods.append((proj, sub))
    return mods


def select_trainable_cross_attention(
    unet,
    scope: str = "all_xattn",
    projections: Sequence[str] = ("q", "k", "v", "out"),
    explicit_layers: Optional[Sequence[str]] = None,
    strict: bool = True,
) -> LayerSelection:
    """Freeze the whole UNet, then unfreeze the requested attn2 projections.

    scope: 'all_xattn', a band name from BANDS_SD1, or 'explicit' (+explicit_layers of
    attn2 module names). strict=True additionally asserts the SD1.x expected attn2 count
    for the scope, so a renamed/architecturally-different UNet fails loudly.
    Returns a LayerSelection (module/parameter names, counts, fingerprint).
    """
    if scope not in SCOPES:
        raise LayerSelectionError(f"unknown scope '{scope}'; expected one of {SCOPES}")
    projections = tuple(projections)
    if not projections:
        raise LayerSelectionError("projections must be non-empty")
    for proj in projections:
        if proj not in PROJECTIONS:
            raise LayerSelectionError(
                f"unknown projection '{proj}'; expected subset of {PROJECTIONS}")
    if scope == "explicit" and not explicit_layers:
        raise LayerSelectionError("scope='explicit' requires explicit_layers")
    if scope != "explicit" and explicit_layers:
        raise LayerSelectionError("explicit_layers only allowed with scope='explicit'")

    all_attn2 = iter_cross_attn_modules(unet)
    if not all_attn2:
        raise LayerSelectionError("no cross-attention (attn2) modules found on this UNet")

    if scope == "all_xattn":
        selected = all_attn2
    elif scope == "explicit":
        by_name = dict(all_attn2)
        missing = [n for n in explicit_layers if n not in by_name]
        if missing:
            raise LayerSelectionError(f"explicit_layers not found on UNet: {missing}")
        selected = [(n, by_name[n]) for n in explicit_layers]
    else:
        selected = [(n, m) for n, m in all_attn2 if band_of_module(n) == scope]
        if not selected:
            raise LayerSelectionError(f"band '{scope}' matched no attn2 modules")

    if strict and scope in EXPECTED_ATTN2_SD1:
        expected = EXPECTED_ATTN2_SD1[scope]
        if len(selected) != expected:
            raise LayerSelectionError(
                f"scope '{scope}' matched {len(selected)} attn2 modules, expected "
                f"{expected} on SD1.x -- refusing to train an unverified layer set")

    for p in unet.parameters():
        p.requires_grad_(False)

    named_params = dict(unet.named_parameters())
    param_names: List[str] = []
    n_train = 0
    for mod_name, module in selected:
        for proj, sub in _projection_modules(module, projections, strict):
            for local_name, p in sub.named_parameters():
                p.requires_grad_(True)
                n_train += p.numel()
                param_names.append(f"{mod_name}.{_PROJ_ATTR[proj]}.{local_name}")
    if n_train == 0:
        raise LayerSelectionError("selection produced zero trainable parameters")

    # sanity: recorded names must exist in named_parameters (catches naming drift)
    unknown = [n for n in param_names if n not in named_params]
    if unknown:
        raise LayerSelectionError(f"selected parameter names not found on UNet: {unknown[:5]}")

    total_xattn = 0
    for mod_name, module in all_attn2:
        for proj, sub in _projection_modules(module, PROJECTIONS, strict=False):
            total_xattn += sum(p.numel() for p in sub.parameters())

    fingerprint_src = json.dumps(
        [(n, [tuple(p.shape) for p in m.parameters()]) for n, m in all_attn2],
        default=str)
    fingerprint = hashlib.sha256(fingerprint_src.encode("utf-8")).hexdigest()

    return LayerSelection(
        scope=scope,
        projections=projections,
        module_names=[n for n, _ in selected],
        parameter_names=param_names,
        trainable_parameter_count=n_train,
        total_cross_attention_parameter_count=total_xattn,
        band_of_modules={n: band_of_module(n) for n, _ in all_attn2},
        architecture_fingerprint=fingerprint,
    )
