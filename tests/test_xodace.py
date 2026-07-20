"""CPU unit tests for the X-ODACE causal-patching pilot (no GPU, no SD weights).

The things that MUST be right before burning GPU hours:
  * the intervention itself -- if the hook copied the wrong batch slot, every causal score would
    be noise and the pilot would "disprove" the hypothesis for a plumbing reason;
  * the metric -- the NudeNet label set must match the repo canonical one
    (eval/eval_label_decomp.py:38-42), and CE must be driven by the EXPOSED-only score, or a
    clothed-but-covered image would count as erasure;
  * the pair selection -- eligibility and the discovery/held-out split decide whether Stage 2's
    top-k test is honest or circular.

Run (WSL conda env lsse):
  pytest models/odace/tests/test_xodace.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

XODACE = Path(__file__).resolve().parents[1] / "experiments" / "xodace"
sys.path.insert(0, str(XODACE))

from make_pairs import STRATA, build_pairs, read_pairs, write_pairs        # noqa: E402
from patcher import (BATCH_SLOTS, SLOT_BENIGN_COND, SLOT_UNSAFE_COND,      # noqa: E402
                     CrossAttnPatcher, bin_of_step, components_to_schedule,
                     expected_patches, list_cross_attn)
from scorer import FCF4, OUR8, causal_effect                               # noqa: E402
from select_pairs import is_eligible, stratified_split                     # noqa: E402


class FakeAttn(torch.nn.Module):
    """Minimal stand-in for a diffusers cross-attention block (attn2)."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.is_cross_attention = True
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.seen_context: torch.Tensor | None = None

    def forward(self, x, encoder_hidden_states=None):
        # record the K/V source the module was actually handed -> lets the context-patch test
        # observe the pre-hook's effect without needing real attention math
        self.seen_context = None if encoder_hidden_states is None else encoder_hidden_states.clone()
        return x


class FakeUNet(torch.nn.Module):
    """down -> mid -> up blocks whose named_modules() order is deliberately NOT exec order."""

    def __init__(self):
        super().__init__()
        self.up_blocks = torch.nn.ModuleList([FakeAttn(), FakeAttn()])
        self.down_blocks = torch.nn.ModuleList([FakeAttn(), FakeAttn()])
        self.mid_block = FakeAttn()

    def blocks(self):
        return list(self.down_blocks) + [self.mid_block] + list(self.up_blocks)

    def forward(self, x, encoder_hidden_states=None):
        for m in self.blocks():
            x = m(x, encoder_hidden_states=encoder_hidden_states)
        return x


# ---------- metric: label set + CE definition ----------

def test_label_sets_match_the_repo_canonical_definition():
    # eval/eval_label_decomp.py:38-42 -- the single source of truth for what "nudity" means here
    assert FCF4 == {"ANUS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
                    "MALE_GENITALIA_EXPOSED"}
    assert OUR8 == FCF4 | {"BUTTOCKS_EXPOSED", "FEMALE_BREAST_COVERED",
                           "FEMALE_GENITALIA_COVERED", "BUTTOCKS_COVERED"}
    assert len(OUR8) == 8
    assert "MALE_BREAST_EXPOSED" not in OUR8       # not a nudity label in this repo


def test_causal_effect_is_driven_by_the_exposed_only_score():
    # a covered-but-clothed image moves unsafe8 without moving unsafe4: that is NOT erasure
    base = {"unsafe4": 0.9, "unsafe8": 0.9, "person": 0.95, "scene": 0.30}
    erased = {"unsafe4": 0.1, "unsafe8": 0.1, "person": 0.95, "scene": 0.30}
    dressed = {"unsafe4": 0.9, "unsafe8": 0.1, "person": 0.95, "scene": 0.30}
    collapsed = {"unsafe4": 0.1, "unsafe8": 0.1, "person": 0.10, "scene": 0.05}

    assert causal_effect(base, erased) == pytest.approx(0.8)
    assert causal_effect(base, dressed) == pytest.approx(0.0)          # 8-label drop ignored
    assert causal_effect(base, dressed, primary="unsafe8") == pytest.approx(0.8)
    assert causal_effect(base, collapsed) < causal_effect(base, erased)
    assert causal_effect(base, base) == pytest.approx(0.0)


# ---------- the intervention ----------

def test_cross_attn_listed_in_execution_order():
    names = [n for n, _ in list_cross_attn(FakeUNet())]
    assert names == ["down_blocks.0", "down_blocks.1", "mid_block", "up_blocks.0", "up_blocks.1"]


def test_output_patch_copies_benign_cond_onto_unsafe_cond_only():
    unet = FakeUNet()
    patcher = CrossAttnPatcher(unet, mode="output").attach()
    x = torch.arange(BATCH_SLOTS * 3 * 4, dtype=torch.float32).reshape(BATCH_SLOTS, 3, 4)
    before = x.clone()

    patcher.set_active({0})                       # patch the first cross-attn layer only
    out = unet(x)

    assert torch.equal(out[SLOT_UNSAFE_COND], before[SLOT_BENIGN_COND]), "donor slot not copied"
    for slot in (0, 2, SLOT_BENIGN_COND):
        assert torch.equal(out[slot], before[slot]), f"slot {slot} must be untouched"
    assert patcher.n_patched == 1                 # exactly one (layer, step) copy fired
    patcher.detach()


def test_context_patch_replaces_only_the_kv_source_of_the_unsafe_slot():
    """A(Q(z_u), K(c_b))V(c_b): the query stays on the unsafe latent, only the text changes."""
    unet = FakeUNet()
    patcher = CrossAttnPatcher(unet, mode="context").attach()
    x = torch.randn(BATCH_SLOTS, 3, 4)
    ctx = torch.arange(BATCH_SLOTS * 2 * 4, dtype=torch.float32).reshape(BATCH_SLOTS, 2, 4)

    patcher.set_active({0})
    out = unet(x, encoder_hidden_states=ctx)

    patched = unet.blocks()[0].seen_context       # what layer 0 actually received
    clean = unet.blocks()[1].seen_context         # layer 1 was not active
    assert torch.equal(patched[SLOT_UNSAFE_COND], ctx[SLOT_BENIGN_COND])
    for slot in (0, 2, SLOT_BENIGN_COND):
        assert torch.equal(patched[slot], ctx[slot]), f"slot {slot} context must be untouched"
    assert torch.equal(clean, ctx), "an inactive layer must see the original context"
    assert torch.equal(out, x), "the hidden states themselves must not be touched"
    assert patcher.n_patched == 1
    patcher.detach()


def test_inactive_patcher_is_a_no_op():
    for mode in ("output", "context"):
        unet = FakeUNet()
        patcher = CrossAttnPatcher(unet, mode=mode).attach()
        x = torch.randn(BATCH_SLOTS, 3, 4)
        patcher.set_active(())                    # clean run
        assert torch.equal(unet(x, encoder_hidden_states=torch.randn(BATCH_SLOTS, 2, 4)), x)
        assert patcher.n_patched == 0             # the clean baseline must patch nothing
        patcher.detach()


def test_detach_removes_hooks():
    unet = FakeUNet()
    patcher = CrossAttnPatcher(unet).attach()
    patcher.set_active({0, 1})
    patcher.detach()
    x = torch.randn(BATCH_SLOTS, 3, 4)
    assert torch.equal(unet(x), x)
    assert patcher.n_patched == 0


def test_patch_rejects_a_batch_that_is_not_the_joint_layout():
    unet = FakeUNet()
    with CrossAttnPatcher(unet) as patcher:
        patcher.set_active({0})
        with pytest.raises(RuntimeError, match="joint unsafe"):
            unet(torch.randn(3, 3, 4))            # 3 is not a multiple of 4 -> unpatchable


def test_unknown_patch_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        CrossAttnPatcher(FakeUNet(), mode="heatmap")


# ---------- schedule + the smoke-gate invariants ----------

def test_bin_of_step_partitions_the_trajectory():
    n_steps, n_bins = 30, 5
    bins = [bin_of_step(i, n_steps, n_bins) for i in range(n_steps)]
    assert bins[0] == 0 and bins[-1] == n_bins - 1
    assert bins == sorted(bins)                                    # monotone in step index
    assert [bins.count(b) for b in range(n_bins)] == [6] * n_bins  # even split
    with pytest.raises(ValueError):
        bin_of_step(n_steps, n_steps, n_bins)


def test_schedule_activates_a_component_only_inside_its_bin():
    n_steps, n_bins = 30, 5
    sched = components_to_schedule([(3, 2)], n_steps, n_bins)
    fired = [i for i, s in enumerate(sched) if s]
    assert all(bin_of_step(i, n_steps, n_bins) == 2 for i in fired)
    assert len(fired) == n_steps // n_bins
    assert all(sched[i] == {3} for i in fired)


def test_schedule_rejects_an_out_of_range_bin():
    with pytest.raises(ValueError):
        components_to_schedule([(0, 9)], 30, 5)


def test_expected_patches_pins_the_smoke_gate_numbers():
    n_steps, n_bins, n_layers = 30, 5, 16
    all_comps = [(lyr, b) for lyr in range(n_layers) for b in range(n_bins)]
    assert expected_patches([], n_steps, n_bins) == 0               # clean
    assert expected_patches([(0, 0)], n_steps, n_bins) == 6         # one layer x one bin
    assert expected_patches(all_comps, n_steps, n_bins) == n_layers * n_steps   # ALL = 480


# ---------- pair construction and selection ----------

def test_pairs_differ_only_in_the_concept_phrase(tmp_path):
    rows = build_pairs(n=12, n_paraphrase=4, seed=42)
    base = [r for r in rows if r["paraphrase_of"] is None]
    assert len(base) == 12
    assert len({r["id"] for r in rows}) == len(rows)              # ids unique

    for r in rows:
        tail_unsafe = r["unsafe"].split(r["concept_phrase"], 1)[1]
        tail_benign = r["benign"].split(r["benign_phrase"], 1)[1]
        assert tail_unsafe == tail_benign, "scene text drifts between unsafe and benign"
        assert r["concept_phrase"] not in r["scene"]              # scene is concept-neutral
        assert r["scene"].endswith(tail_unsafe)
        for k in STRATA:                                          # strata fields for the split
            assert r[k] and r[k] in r["scene"]

    variants = [r for r in rows if r["paraphrase_of"] is not None]
    assert len(variants) == 4
    by_id = {r["id"]: r for r in base}
    for v in variants:
        assert v["scene"] == by_id[v["paraphrase_of"]]["scene"]   # same scene, other wording
        assert v["unsafe"] != by_id[v["paraphrase_of"]]["unsafe"]

    out = write_pairs(rows, tmp_path / "pairs.jsonl")
    assert read_pairs(out) == rows                                # round-trip


def test_eligibility_needs_an_exposed_clean_run_and_a_clothed_benign_run():
    exposed = {"unsafe4": 0.85, "unsafe8": 0.9}
    covered_only = {"unsafe4": 0.0, "unsafe8": 0.63}   # clothed body: unsafe8 fires, unsafe4 not
    clothed = {"unsafe4": 0.02, "unsafe8": 0.4}

    assert is_eligible(exposed, clothed)[0] is True
    assert is_eligible(covered_only, clothed) == (False, "clean_not_exposed")
    assert is_eligible(exposed, exposed) == (False, "benign_still_exposed")
    assert is_eligible(None, clothed) == (False, "no_clean_row")
    assert is_eligible(exposed, None) == (False, "no_benign_row")


def test_split_is_disjoint_stratified_and_not_score_ranked():
    rows = build_pairs(n=40, n_paraphrase=0, seed=7)
    disc, held = stratified_split(rows, n_discovery=10, n_heldout=10, seed=1234)

    ids_d = {r["id"] for r in disc}
    ids_h = {r["id"] for r in held}
    assert len(disc) == len(held) == 10
    assert not (ids_d & ids_h), "discovery and held-out must be disjoint"
    # stratified, not "the 10 first / most extreme": both splits see several settings
    assert len({r["setting"] for r in disc}) >= 5
    assert len({r["setting"] for r in held}) >= 5
    # deterministic given the seed
    assert [r["id"] for r in stratified_split(rows, 10, 10, seed=1234)[0]] == [r["id"] for r in disc]
