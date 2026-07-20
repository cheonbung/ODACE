"""CPU regression parity: frozen legacy snapshot vs refactored trainer (legacy_exact) on
identical tiny components + identical global RNG. One optimizer step: the trajectory
index, latent, benign-neg target, both losses and the updated trainable weights must all
match. (The full-checkpoint GPU version of this gate is
experiments/regression_compare.py.)"""
import copy
import random
import sys
from pathlib import Path

import torch
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusers import DDIMScheduler

from tests.conftest import StubTeacher, fake_encode, legacy_cfg_dict, \
    make_tiny_sd1_unet, make_trainer
from core.legacy_trainer_snapshot import LegacyODACETrainer, set_trainable_cross_attn_kv

SEED = 42
FORGET = "a nude figure on a beach"
RETAIN = "a person on a beach"


def _build_legacy_snapshot(unet, teacher):
    """LegacyODACETrainer over tiny components, bypassing from_pretrained (__new__)."""
    leg = LegacyODACETrainer.__new__(LegacyODACETrainer)
    leg.device = torch.device("cpu")
    leg.alpha, leg.beta, leg.eta = 1.0, 1.0, 3.0
    leg.ddim_steps = 6
    leg.sample_guidance = 3.0
    leg.max_length = 77
    leg.erase_mode = "benign_neg"
    leg.benign_prompt = "a fully clothed person, photograph"
    leg.benign_neg_lambda = 1.0
    leg._c_benign = None
    leg._uncond = None
    leg.tokenizer = None
    leg.text_encoder = None
    leg.unet = unet
    leg.unet.enable_gradient_checkpointing()
    set_trainable_cross_attn_kv(leg.unet, include_q_out=True)
    leg.unet.train()
    leg.unet_frozen = teacher
    leg.ddim = DDIMScheduler()
    leg.optimizer = Adam([p for p in leg.unet.parameters() if p.requires_grad], lr=1e-3)
    leg.lat_ch = 4
    leg._encode = fake_encode
    leg.capture_debug = True
    return leg


def test_legacy_exact_reproduces_snapshot_one_step():
    base = make_tiny_sd1_unet(seed=5)
    teacher = StubTeacher()

    # --- frozen snapshot -------------------------------------------------------
    leg = _build_legacy_snapshot(copy.deepcopy(base), teacher)
    random.seed(SEED)
    torch.manual_seed(SEED)
    s_leg = leg._train_step([FORGET], [RETAIN])
    dbg_leg = leg.last_debug

    # --- refactored trainer, legacy_exact mode ---------------------------------
    tr = make_trainer(legacy_cfg_dict(learning_rate=1e-3, seed=SEED),
                      unet=copy.deepcopy(base), teacher=teacher)
    # make_trainer -> ODACETrainer.__init__ already reseeded the global RNG to SEED
    tr.capture_debug = True
    s_new = tr._train_step([FORGET], [RETAIN])
    dbg_new = tr.last_debug

    # trajectory draw + timestep pairing identical (incl. the legacy off-by-one)
    assert dbg_leg["t_enc_idx"] == dbg_new["t_enc_idx"]
    assert dbg_leg["t"] == dbg_new["t"]
    # latent + benign-neg target numerically identical
    assert torch.equal(dbg_leg["z"], dbg_new["z"]), "trajectory latent diverged"
    assert torch.allclose(dbg_leg["target"], dbg_new["target"], atol=0.0), \
        "legacy benign-neg 1-step target differs from refactor"
    # losses identical
    for k in ("L_forget", "L_retain", "L_total"):
        assert abs(s_leg[k] - s_new[k]) <= 1e-9, f"{k}: {s_leg[k]} vs {s_new[k]}"
    # updated trainable weights identical after the Adam step
    pa = dict(leg.unet.named_parameters())
    for name, p in tr.unet.named_parameters():
        if p.requires_grad:
            assert torch.allclose(pa[name], p, atol=1e-7), f"{name} diverged post-step"


def test_two_steps_history_parity():
    base = make_tiny_sd1_unet(seed=6)
    teacher = StubTeacher()

    leg = _build_legacy_snapshot(copy.deepcopy(base), teacher)
    random.seed(SEED)
    torch.manual_seed(SEED)
    prompts_f = [FORGET, "nude portrait painting"]
    prompts_r = [RETAIN, "portrait painting"]
    hist_leg = [leg._train_step([prompts_f[i]], [prompts_r[i]]) for i in range(2)]

    tr = make_trainer(legacy_cfg_dict(learning_rate=1e-3, seed=SEED),
                      unet=copy.deepcopy(base), teacher=teacher)
    hist_new = [tr._train_step([prompts_f[i]], [prompts_r[i]]) for i in range(2)]

    for a, b in zip(hist_leg, hist_new):
        for k in ("L_forget", "L_retain", "L_total"):
            assert abs(a[k] - b[k]) <= 1e-9
