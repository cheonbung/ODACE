"""Layer-selection invariants on a tiny SD1.x-shaped UNet: res16 == exactly 5 attn2 on
down_blocks.2/up_blocks.1, projection filtering, manifest completeness, legacy-wrapper
equivalence, error paths."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.layer_selection import (BANDS_SD1, EXPECTED_ATTN2_SD1, LayerSelectionError,
                                     band_of_module, iter_cross_attn_modules,
                                     select_trainable_cross_attention)
from methods.unet_edit import set_trainable_cross_attn_kv


def test_tiny_unet_has_sd1_attn2_layout(tiny_unet):
    mods = iter_cross_attn_modules(tiny_unet)
    assert len(mods) == EXPECTED_ATTN2_SD1["all_xattn"] == 16
    by_band = {}
    for name, _ in mods:
        by_band.setdefault(band_of_module(name), []).append(name)
    assert len(by_band["res16"]) == 5
    assert len(by_band["res64"]) == 5
    assert len(by_band["res32"]) == 5
    assert len(by_band["res8"]) == 1


def test_res16_selects_exactly_five_attn2(tiny_unet):
    sel = select_trainable_cross_attention(tiny_unet, scope="res16")
    assert len(sel.module_names) == 5
    assert all(n.startswith(BANDS_SD1["res16"]) for n in sel.module_names)
    assert all(".attn2" in n for n in sel.module_names)
    # every selected parameter really requires grad; count matches
    named = dict(tiny_unet.named_parameters())
    n = sum(named[p].numel() for p in sel.parameter_names)
    assert n == sel.trainable_parameter_count > 0
    for pname, p in named.items():
        assert p.requires_grad == (pname in set(sel.parameter_names))


def test_non_selected_params_frozen(tiny_unet):
    sel = select_trainable_cross_attention(tiny_unet, scope="res8")
    selected = set(sel.parameter_names)
    for pname, p in tiny_unet.named_parameters():
        if pname not in selected:
            assert not p.requires_grad


def test_kv_only_excludes_q_and_out(tiny_unet):
    sel = select_trainable_cross_attention(tiny_unet, scope="all_xattn",
                                           projections=("k", "v"))
    assert all((".to_k." in p or ".to_v." in p) for p in sel.parameter_names)
    full = select_trainable_cross_attention(tiny_unet, scope="all_xattn")
    assert sel.trainable_parameter_count < full.trainable_parameter_count


def test_explicit_scope_and_errors(tiny_unet):
    mods = [n for n, _ in iter_cross_attn_modules(tiny_unet)][:2]
    sel = select_trainable_cross_attention(tiny_unet, scope="explicit",
                                           explicit_layers=mods)
    assert sel.module_names == mods
    with pytest.raises(LayerSelectionError):
        select_trainable_cross_attention(tiny_unet, scope="nope")
    with pytest.raises(LayerSelectionError):
        select_trainable_cross_attention(tiny_unet, scope="explicit")
    with pytest.raises(LayerSelectionError):
        select_trainable_cross_attention(tiny_unet, scope="res16",
                                         explicit_layers=mods)
    with pytest.raises(LayerSelectionError):
        select_trainable_cross_attention(tiny_unet, projections=("z",))
    with pytest.raises(LayerSelectionError):
        select_trainable_cross_attention(tiny_unet, projections=())


def test_manifest_fields(tiny_unet):
    sel = select_trainable_cross_attention(tiny_unet, scope="res16")
    m = sel.to_manifest()
    for key in ("scope", "projections", "module_names", "parameter_names",
                "trainable_parameter_count", "total_cross_attention_parameter_count",
                "band_of_modules", "architecture_fingerprint"):
        assert key in m
    assert m["scope"] == "res16"
    assert len(m["architecture_fingerprint"]) == 64


def test_legacy_wrapper_matches_selection(tiny_unet_pair):
    a, b = tiny_unet_pair
    n_wrap = set_trainable_cross_attn_kv(a, include_q_out=True)
    sel = select_trainable_cross_attention(b, scope="all_xattn",
                                           projections=("q", "k", "v", "out"))
    assert n_wrap == sel.trainable_parameter_count
    ra = {n for n, p in a.named_parameters() if p.requires_grad}
    rb = {n for n, p in b.named_parameters() if p.requires_grad}
    assert ra == rb


def test_legacy_wrapper_kv_matches_selection(tiny_unet_pair):
    a, b = tiny_unet_pair
    n_wrap = set_trainable_cross_attn_kv(a, include_q_out=False)
    sel = select_trainable_cross_attention(b, scope="all_xattn", projections=("k", "v"))
    assert n_wrap == sel.trainable_parameter_count


def test_strict_count_check_fires():
    import torch.nn as nn

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_cross_attention = True
            self.to_q = nn.Linear(8, 8); self.to_k = nn.Linear(8, 8)
            self.to_v = nn.Linear(8, 8); self.to_out = nn.ModuleList([nn.Linear(8, 8)])

    class _NotSD1(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn2 = _Attn()

    with pytest.raises(LayerSelectionError):
        select_trainable_cross_attention(_NotSD1(), scope="all_xattn")   # 1 != 16
    sel = select_trainable_cross_attention(_NotSD1(), scope="all_xattn", strict=False)
    assert sel.trainable_parameter_count > 0
