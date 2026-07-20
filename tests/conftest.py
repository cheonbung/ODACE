"""Shared fixtures for ODACE unit tests: a tiny SD1.x-SHAPED UNet (same block layout as
SD1.4 -- 3 cross-attn down blocks + DownBlock2D, mid, UpBlock2D + 3 cross-attn up blocks,
layers_per_block=2 => the same 16 attn2 modules and the same band membership, just tiny
channels), a deterministic stub teacher, and a deterministic prompt encoder. Everything
runs on CPU in fp32 except the fp16 initial latent, matching the trainer's dtype flow.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # models/odace

REPO_ROOT = Path(__file__).resolve().parents[3]

CROSS_DIM = 32
MAX_LEN = 77


def make_tiny_sd1_unet(seed: int = 0):
    """SD1.x-shaped UNet2DConditionModel with tiny channels (16 attn2, res16 = 5)."""
    from diffusers import UNet2DConditionModel
    torch.manual_seed(seed)
    return UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(32, 32, 64, 64),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D"),
        cross_attention_dim=CROSS_DIM, attention_head_dim=4, norm_num_groups=8)


class StubTeacher(torch.nn.Module):
    """Deterministic frozen-teacher stand-in: fp32 sample, a smooth function of inputs
    (so different conditionings give different outputs), no RNG consumption."""

    def forward(self, z, t, encoder_hidden_states=None):
        s = z.float().mean() + encoder_hidden_states.float().mean() + float(t) / 1000.0
        return SimpleNamespace(sample=torch.tanh(z.float() * 0.1 + s))


class ZeroTeacher(torch.nn.Module):
    """eps=0 teacher: makes the DDIM trajectory a deterministic function of z0 alone."""

    def forward(self, z, t, encoder_hidden_states=None):
        return SimpleNamespace(sample=torch.zeros(z.shape, dtype=torch.float32))


def fake_encode(texts):
    """Deterministic per-prompt embedding (sha256 -> seeded randn), shape (1, 77, 32)."""
    seed = int.from_bytes(hashlib.sha256(texts[0].encode()).digest()[:4], "little")
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, MAX_LEN, CROSS_DIM, generator=gen)


def paper_cfg_dict(**over):
    d = {
        "experiment_name": "unit", "execution_mode": "paper_aligned",
        "sd_model_id": "tiny/sd1-shaped", "target_mode": "anchor_contrastive",
        "target_lambda": 1.0, "anchor_prompt": "a fully clothed person, photograph",
        "trainable_scope": "all_xattn", "trainable_projections": ["q", "k", "v", "out"],
        "timestep_policy": "legacy_trajectory_index_uniform",
        "ddim_steps": 6, "trajectory_index_min": 1, "trajectory_index_max": 5,
        "learning_rate": 1e-3, "num_steps": 2, "batch_size": 1,
        "forget_prompts_file": "unused.txt", "retain_prompts_file": "unused.txt",
        "output_dir": "outputs/_unit", "seed": 42,
    }
    d.update(over)
    return d


def legacy_cfg_dict(**over):
    d = {
        "experiment_name": "unit_legacy", "sd_model_id": "tiny/sd1-shaped",
        "erase_mode": "benign_neg", "benign_neg_lambda": 1.0,
        "benign_prompt": "a fully clothed person, photograph",
        "xattn_full": True, "eta": 3.0, "ddim_steps": 6, "sample_guidance": 3.0,
        "learning_rate": 1e-3, "num_steps": 2, "batch_size": 4, "seed": 42,
        "t_min": 5, "t_max": 950,
        "forget_prompts_file": "unused.txt", "retain_prompts_file": "unused.txt",
        "output_dir": "outputs/_unit_legacy",
    }
    d.update(over)
    return d


def make_trainer(cfg_dict, unet=None, teacher=None):
    """Refactored ODACETrainer over injected tiny components + deterministic encoder."""
    from diffusers import DDIMScheduler
    from core.config_schema import resolve_config
    from core.trainer import ODACETrainer
    cfg = resolve_config(cfg_dict, source_path="<unit>")
    components = {
        "tokenizer": None, "text_encoder": None,
        "unet": unet if unet is not None else make_tiny_sd1_unet(),
        "unet_frozen": teacher if teacher is not None else StubTeacher(),
        "scheduler": DDIMScheduler(),
    }
    tr = ODACETrainer(cfg, torch.device("cpu"), components=components)
    tr._encode = fake_encode
    return tr


@pytest.fixture
def tiny_unet():
    return make_tiny_sd1_unet()


@pytest.fixture
def tiny_unet_pair():
    """Two bit-identical tiny UNets (independent copies) for A/B comparisons."""
    a = make_tiny_sd1_unet(seed=1)
    return a, copy.deepcopy(a)
