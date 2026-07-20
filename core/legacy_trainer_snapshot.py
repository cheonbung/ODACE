"""FROZEN verbatim snapshot of the pre-refactor ODACE trainer (2026-07-16). DO NOT EDIT.

This file exists ONLY as the numerical reference for the P1 regression gate: the
refactored core/trainer.py in execution_mode='legacy_exact' must reproduce this
implementation's targets/losses/updates within tolerance (see
experiments/regression_compare.py and tests/test_regression_cpu.py). It must never be
used for real training runs.

Documented deviations from the original core/trainer.py (numerics untouched):
  1. class renamed ODACETrainer -> LegacyODACETrainer (avoids import collision),
  2. set_trainable_cross_attn_kv is INLINED verbatim from the pre-refactor
     methods/unet_edit.py so this snapshot cannot drift with future refactors,
  3. an opt-in debug capture block in _train_step (guarded by self.capture_debug,
     default False) records the target tensor/losses for comparison,
  4. the unused `from core.dataset import neutralize` import was dropped.
Everything else (RNG order, fp16 teacher, DDIM loop, loss order, Adam) is verbatim.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Dict, List

import torch
from torch.optim import Adam

logger = logging.getLogger(__name__)


def set_trainable_cross_attn_kv(unet, include_q_out: bool = False) -> int:
    """Verbatim pre-refactor copy (methods/unet_edit.py). Freeze all, unfreeze attn2 proj."""
    for p in unet.parameters():
        p.requires_grad_(False)
    n = 0
    for name, module in unet.named_modules():
        is_cross = getattr(module, "is_cross_attention", None)
        if is_cross is None:
            is_cross = name.endswith("attn2")
        if is_cross and hasattr(module, "to_k") and hasattr(module, "to_v"):
            mods = [module.to_k, module.to_v]
            if include_q_out:
                if hasattr(module, "to_q"):
                    mods.append(module.to_q)
                if getattr(module, "to_out", None) is not None:
                    mods.append(module.to_out)
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(True); n += p.numel()
    return n


class LegacyODACETrainer:
    def __init__(self, sd_model_id: str, device, learning_rate: float = 1e-5,
                 alpha: float = 1.0, beta: float = 1.0, eta: float = 1.0,
                 ddim_steps: int = 30, sample_guidance: float = 3.0,
                 xattn_full: bool = False, erase_mode: str = "negguide",
                 benign_prompt: str = "a fully clothed person, photograph",
                 benign_neg_lambda: float = 1.0,
                 batch_size: int = 1, max_length: int = 77, seed: int = 42, **_ignore):
        from diffusers import UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device
        self.alpha, self.beta, self.eta = alpha, beta, eta
        self.ddim_steps = ddim_steps
        self.sample_guidance = sample_guidance
        self.max_length = max_length
        self.erase_mode = erase_mode
        self.benign_prompt = benign_prompt
        self.benign_neg_lambda = benign_neg_lambda
        self._c_benign = None
        random.seed(seed); torch.manual_seed(seed)

        self.tokenizer = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_model_id, subfolder="text_encoder").to(device)
        self.text_encoder.requires_grad_(False); self.text_encoder.eval()

        self.unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet").to(device)
        try:
            self.unet.enable_gradient_checkpointing()
        except Exception as exc:
            logger.warning(f"grad checkpointing unavailable: {exc}")
        n_train = set_trainable_cross_attn_kv(self.unet, include_q_out=xattn_full)
        self.unet.train()

        self.unet_frozen = UNet2DConditionModel.from_pretrained(
            sd_model_id, subfolder="unet").to(device=device, dtype=torch.float16)
        self.unet_frozen.requires_grad_(False); self.unet_frozen.eval()

        self.ddim = DDIMScheduler.from_pretrained(sd_model_id, subfolder="scheduler")
        self.optimizer = Adam([p for p in self.unet.parameters() if p.requires_grad], lr=learning_rate)
        self.lat_ch = self.unet.config.in_channels
        self._uncond = None
        logger.info(f"[ODACE-legacy-snapshot] trainable cross-attn params: {n_train:,} "
                    f"(full={xattn_full}) | eta={eta} guid={sample_guidance} lr={learning_rate}")

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding="max_length", max_length=self.max_length,
                             truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder(tok.input_ids)[0]

    @torch.no_grad()
    def _uncond_emb(self):
        if self._uncond is None:
            self._uncond = self._encode([""])
        return self._uncond

    @torch.no_grad()
    def _benign_emb(self):
        if self._c_benign is None:
            self._c_benign = self._encode([self.benign_prompt])
        return self._c_benign

    @torch.no_grad()
    def _sample_until(self, c_forget, t_enc_idx):
        """DDIM-denoise from noise with frozen UNET + forget prompt (CFG) for t_enc_idx steps.
        Returns (z_fp16, t_current) on the concept trajectory where text conditioning matters."""
        self.ddim.set_timesteps(self.ddim_steps, device=self.device)
        z = torch.randn(1, self.lat_ch, 64, 64, device=self.device, dtype=torch.float16)
        z = z * self.ddim.init_noise_sigma
        cond = torch.cat([self._uncond_emb().half(), c_forget.half()])
        tlist = self.ddim.timesteps
        t_cur = tlist[0]
        for i, t in enumerate(tlist):
            if i >= t_enc_idx:
                break
            t_cur = t
            zin = self.ddim.scale_model_input(torch.cat([z] * 2), t)
            npred = self.unet_frozen(zin, t, encoder_hidden_states=cond).sample
            nu, nc = npred.chunk(2)
            npred = nu + self.sample_guidance * (nc - nu)
            z = self.ddim.step(npred, t, z).prev_sample
        return z, t_cur

    def _train_step(self, forget: List[str], retain: List[str]) -> Dict[str, float]:
        c_f = self._encode(forget[:1])
        c_un = self._uncond_emb()
        c_r = self._encode(retain[:1])
        t_enc = random.randint(1, self.ddim_steps - 1)
        z, t_cur = self._sample_until(c_f, t_enc)        # fp16 latent on concept trajectory

        with torch.no_grad():
            e_0 = self.unet_frozen(z, t_cur, encoder_hidden_states=c_un.half()).sample.float()
            e_p = self.unet_frozen(z, t_cur, encoder_hidden_states=c_f.half()).sample.float()
            e_r_fz = self.unet_frozen(z, t_cur, encoder_hidden_states=c_r.half()).sample.float()
            if self.erase_mode in ("benign_anchor", "benign_neg"):
                c_b = self._benign_emb()
                e_b = self.unet_frozen(z, t_cur, encoder_hidden_states=c_b.half()).sample.float()
        if self.erase_mode == "benign_anchor":
            target = e_b.detach()
        elif self.erase_mode == "benign_neg":
            target = (e_b - self.benign_neg_lambda * (e_p - e_b)).detach()
        else:
            target = (e_0 - self.eta * (e_p - e_0)).detach()  # negative-guidance: erase concept

        zf = z.float()
        self.optimizer.zero_grad()
        e_n = self.unet(zf, t_cur, encoder_hidden_states=c_f).sample
        e_r = self.unet(zf, t_cur, encoder_hidden_states=c_r).sample
        L_forget = torch.nn.functional.mse_loss(e_n, target)
        L_retain = torch.nn.functional.mse_loss(e_r, e_r_fz.detach())
        L = self.alpha * L_forget + self.beta * L_retain
        # --- snapshot-only debug capture (numerics untouched; opt-in) ---
        if getattr(self, "capture_debug", False):
            self.last_debug = {
                "target": target.detach().float().cpu(),
                "z": zf.detach().cpu(),
                "t": int(t_cur),
                "t_enc_idx": int(t_enc),
                "forget_prompt": forget[0],
                "retain_prompt": retain[0],
            }
        # --- end snapshot-only block ---
        L.backward()
        self.optimizer.step()
        return {"L_forget": L_forget.item(), "L_retain": L_retain.item(), "L_total": L.item()}

    def train(self, dataset, num_steps: int = 400, log_every: int = 25) -> List[Dict[str, float]]:
        forget_p, retain_p = dataset.forget_prompts, dataset.retain_prompts
        history = []
        for step in range(1, num_steps + 1):
            bf = [random.choice(forget_p)]
            br = [random.choice(retain_p)]
            s = self._train_step(bf, br); s["step"] = step
            history.append(s)
            if step % log_every == 0 or step == 1:
                logger.info(f"  step {step}/{num_steps} | Lf={s['L_forget']:.4f} "
                            f"Lr={s['L_retain']:.4f} Ltot={s['L_total']:.4f}")
        return history

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.unet.save_pretrained(save_dir)
        logger.info(f"modified UNET saved -> {save_dir}")
