"""ODACETrainer -- output-direction-anchored concept erasure on SD UNet cross-attention.

Refactored (plan P1, 2026-07): target math lives in core/targets.py, layer selection in
methods/layer_selection.py, config validation in core/config_schema.py. The trainer takes
a ResolvedODACEConfig and honors its execution_mode:

  legacy_exact   -- numerically reproduces the pre-refactor trainer (reference frozen in
                    core/legacy_trainer_snapshot.py): global random/torch RNG seeded once,
                    ONE example per optimizer step (the legacy loop ignored batch_size),
                    and the legacy trajectory convention where _sample_until returns the
                    latent AFTER t_enc_idx DDIM steps paired with the timestep of the step
                    it just took (off-by-one; the UNet is queried one scheduler index
                    behind the latent state).
  paper_aligned  -- batch_size is the real number of examples accumulated per optimizer
                    step (micro-loop, losses divided by the effective batch), the
                    trajectory latent/timestep pair is index-aligned (z_i returned with
                    tlist[i], the timestep whose step has NOT yet been taken), and local
                    RNGs (random.Random / torch.Generator) are used so seeds are
                    separable and recorded.

Teacher = frozen fp16 UNet on the unsafe-prompt DDIM trajectory; student gradient flows
only through the projections chosen by select_trainable_cross_attention. Timestep policy
is explicit; only 'legacy_trajectory_index_uniform' (uniform over DDIM trajectory
indices) is implemented, and config_schema rejects anything else.

Callers: train_odace.py, experiments/regression_compare.py, tests/*.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Optional

import torch
from torch.optim import Adam

from methods.layer_selection import LayerSelection, select_trainable_cross_attention
from core.config_schema import ResolvedODACEConfig
from core.targets import compute_target

logger = logging.getLogger(__name__)


class ODACETrainer:
    def __init__(self, cfg: ResolvedODACEConfig, device,
                 components: Optional[dict] = None):
        """components (tests only): dict with tokenizer/text_encoder/unet/unet_frozen/
        scheduler already built; skips from_pretrained. Production passes None."""
        self.cfg = cfg
        self.device = device
        self._c_anchor = None
        self._uncond = None
        self.capture_debug = False
        self.last_debug: Optional[dict] = None
        self.counters = {"teacher_forward": 0, "student_forward": 0, "backward": 0,
                         "examples_forget": 0, "examples_retain": 0, "optimizer_steps": 0}

        # RNG setup -- legacy_exact keeps the legacy GLOBAL seeding so the draw sequence
        # (choice, choice, randint, randn) is bit-identical to the snapshot; paper_aligned
        # uses local, separately-recorded generators.
        if cfg.execution_mode == "legacy_exact":
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            self._prompt_rng = random          # the module itself: legacy global RNG
            self._latent_generator = None      # None -> torch global RNG (legacy)
            self.rng_record = {"rng_style": "global_legacy", "train_seed": cfg.seed,
                               "python_prompt_seed": cfg.seed,
                               "torch_global_seed": cfg.seed}
        else:
            torch.manual_seed(cfg.seed)        # library internals only; latents use gen
            self._prompt_rng = random.Random(cfg.seed)
            gen_device = device if (isinstance(device, torch.device)
                                    and device.type == "cuda") else "cpu"
            self._latent_generator = torch.Generator(device=gen_device)
            self._latent_generator.manual_seed(cfg.seed + 1)
            self.rng_record = {"rng_style": "local", "train_seed": cfg.seed,
                               "python_prompt_seed": cfg.seed,
                               "torch_latent_seed": cfg.seed + 1,
                               "torch_global_seed": cfg.seed}

        if components is not None:
            self.tokenizer = components["tokenizer"]
            self.text_encoder = components["text_encoder"]
            self.unet = components["unet"]
            self.unet_frozen = components["unet_frozen"]
            self.ddim = components["scheduler"]
        else:
            from diffusers import UNet2DConditionModel, DDIMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(
                cfg.sd_model_id, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                cfg.sd_model_id, subfolder="text_encoder").to(device)
            self.unet = UNet2DConditionModel.from_pretrained(
                cfg.sd_model_id, subfolder="unet").to(device)
            self.unet_frozen = UNet2DConditionModel.from_pretrained(
                cfg.sd_model_id, subfolder="unet").to(device=device, dtype=torch.float16)
            self.ddim = DDIMScheduler.from_pretrained(cfg.sd_model_id,
                                                      subfolder="scheduler")

        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        try:
            self.unet.enable_gradient_checkpointing()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"grad checkpointing unavailable: {exc}")

        self.selection: LayerSelection = select_trainable_cross_attention(
            self.unet, scope=cfg.trainable_scope,
            projections=cfg.trainable_projections,
            explicit_layers=cfg.explicit_layers)
        self.unet.train()
        self.unet_frozen.requires_grad_(False)
        self.unet_frozen.eval()

        self.optimizer = Adam([p for p in self.unet.parameters() if p.requires_grad],
                              lr=cfg.learning_rate)
        self.lat_ch = self.unet.config.in_channels
        logger.info(
            f"[ODACE] mode={cfg.execution_mode} target={cfg.target_mode} "
            f"scope={cfg.trainable_scope} proj={list(cfg.trainable_projections)} "
            f"trainable={self.selection.trainable_parameter_count:,}/"
            f"{self.selection.total_cross_attention_parameter_count:,} xattn params | "
            f"batch={cfg.batch_size}x{cfg.gradient_accumulation_steps} lr={cfg.learning_rate}")

    # ---------------------------------------------------------------- embeddings
    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding="max_length", max_length=self.cfg.max_length,
                             truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder(tok.input_ids)[0]

    @torch.no_grad()
    def _uncond_emb(self):
        if self._uncond is None:
            self._uncond = self._encode([""])
        return self._uncond

    @torch.no_grad()
    def _anchor_emb(self):
        if self._c_anchor is None:
            self._c_anchor = self._encode([self.cfg.anchor_prompt])
        return self._c_anchor

    # ---------------------------------------------------------------- trajectory
    @torch.no_grad()
    def _sample_until(self, c_forget: torch.Tensor, t_enc_idx: int):
        """DDIM-denoise from noise with the frozen UNet + forget prompt (CFG) for
        t_enc_idx steps; return (z_fp16, timestep).

        legacy_exact: returns (z after t_enc_idx steps, tlist[t_enc_idx - 1]) -- the
        legacy off-by-one pairing (latent is one scheduler index AHEAD of the timestep).
        paper_aligned: returns (same z, tlist[t_enc_idx]) -- z is exactly the input the
        scheduler expects for the step at that timestep (index-aligned convention 2 of
        plan section 6.2).
        """
        self.ddim.set_timesteps(self.cfg.ddim_steps, device=self.device)
        if self._latent_generator is None:
            z = torch.randn(1, self.lat_ch, 64, 64, device=self.device,
                            dtype=torch.float16)
        else:
            z = torch.randn(1, self.lat_ch, 64, 64, device=self.device,
                            dtype=torch.float16, generator=self._latent_generator)
        z = z * self.ddim.init_noise_sigma
        cond = torch.cat([self._uncond_emb().half(), c_forget.half()])
        tlist = self.ddim.timesteps
        for i, t in enumerate(tlist):
            if i >= t_enc_idx:
                break
            zin = self.ddim.scale_model_input(torch.cat([z] * 2), t)
            npred = self.unet_frozen(zin, t, encoder_hidden_states=cond).sample
            self.counters["teacher_forward"] += 1
            nu, nc = npred.chunk(2)
            npred = nu + self.cfg.sample_guidance * (nc - nu)
            z = self.ddim.step(npred, t, z).prev_sample
        if self.cfg.execution_mode == "legacy_exact":
            return z, tlist[max(t_enc_idx - 1, 0)]
        return z, tlist[t_enc_idx]

    # ---------------------------------------------------------------- one example
    def _example_losses(self, forget_prompt: str, retain_prompt: str):
        """Forget/retain losses for ONE (forget, retain) prompt pair.

        RNG consumption order matches the legacy step exactly: encode (no RNG) ->
        randint(trajectory index) -> randn(initial latent). Teacher forwards consume no
        RNG, so skipping teacher outputs the target mode does not need is
        numerics-neutral vs the legacy trainer (which always computed e_0/e_p/e_r).
        """
        cfg = self.cfg
        c_f = self._encode([forget_prompt])
        c_r = self._encode([retain_prompt])
        c_un = self._uncond_emb()
        t_enc = self._prompt_rng.randint(cfg.trajectory_index_min,
                                         cfg.trajectory_index_max)
        z, t_cur = self._sample_until(c_f, t_enc)

        with torch.no_grad():
            e_c = self.unet_frozen(z, t_cur, encoder_hidden_states=c_f.half()
                                   ).sample.float()
            e_r_fz = self.unet_frozen(z, t_cur, encoder_hidden_states=c_r.half()
                                      ).sample.float()
            self.counters["teacher_forward"] += 2
            e_u = e_b = None
            if cfg.target_mode == "push":
                e_u = self.unet_frozen(z, t_cur, encoder_hidden_states=c_un.half()
                                       ).sample.float()
                self.counters["teacher_forward"] += 1
            else:
                c_b = self._anchor_emb()
                e_b = self.unet_frozen(z, t_cur, encoder_hidden_states=c_b.half()
                                       ).sample.float()
                self.counters["teacher_forward"] += 1

        target, _ = compute_target(cfg.target_mode, eps_uncond=e_u, eps_benign=e_b,
                                   eps_concept=e_c, eta=cfg.eta, lam=cfg.target_lambda)

        zf = z.float()
        e_n = self.unet(zf, t_cur, encoder_hidden_states=c_f).sample
        e_r = self.unet(zf, t_cur, encoder_hidden_states=c_r).sample
        self.counters["student_forward"] += 2
        L_forget = torch.nn.functional.mse_loss(e_n, target)
        L_retain = torch.nn.functional.mse_loss(e_r, e_r_fz.detach())
        self.counters["examples_forget"] += 1
        self.counters["examples_retain"] += 1

        if self.capture_debug:
            self.last_debug = {
                "target": target.detach().float().cpu(),
                "z": zf.detach().cpu(),
                "t": int(t_cur),
                "t_enc_idx": int(t_enc),
                "forget_prompt": forget_prompt,
                "retain_prompt": retain_prompt,
            }
        return L_forget, L_retain

    # ---------------------------------------------------------------- optimizer step
    def _train_step(self, forget_pairs: List[str], retain_pairs: List[str]
                    ) -> Dict[str, float]:
        """One optimizer step. legacy_exact consumes exactly one pair; paper_aligned
        accumulates cfg.effective_batch_size pairs with losses scaled by 1/effective."""
        cfg = self.cfg
        self.optimizer.zero_grad()
        if cfg.execution_mode == "legacy_exact":
            L_forget, L_retain = self._example_losses(forget_pairs[0], retain_pairs[0])
            L = cfg.alpha * L_forget + cfg.beta * L_retain
            L.backward()
            self.counters["backward"] += 1
            lf_sum, lr_sum, lt_sum = L_forget.item(), L_retain.item(), L.item()
            n = 1
        else:
            n = cfg.effective_batch_size
            assert len(forget_pairs) == len(retain_pairs) == n
            lf_sum = lr_sum = lt_sum = 0.0
            for fp, rp in zip(forget_pairs, retain_pairs):
                L_forget, L_retain = self._example_losses(fp, rp)
                L = (cfg.alpha * L_forget + cfg.beta * L_retain) / n
                L.backward()
                self.counters["backward"] += 1
                lf_sum += L_forget.item()
                lr_sum += L_retain.item()
                lt_sum += L.item() * n
        self.optimizer.step()
        self.counters["optimizer_steps"] += 1
        return {"L_forget": lf_sum / n, "L_retain": lr_sum / n, "L_total": lt_sum / n}

    # ---------------------------------------------------------------- loop
    def train(self, dataset, num_steps: Optional[int] = None,
              log_every: Optional[int] = None) -> List[Dict[str, float]]:
        cfg = self.cfg
        num_steps = num_steps or cfg.num_optimizer_steps
        log_every = log_every or cfg.log_every
        forget_p, retain_p = dataset.forget_prompts, dataset.retain_prompts
        per_step = 1 if cfg.execution_mode == "legacy_exact" else cfg.effective_batch_size
        history = []
        for step in range(1, num_steps + 1):
            bf = [self._prompt_rng.choice(forget_p) for _ in range(per_step)]
            br = [self._prompt_rng.choice(retain_p) for _ in range(per_step)]
            s = self._train_step(bf, br)
            s["step"] = step
            history.append(s)
            if step % log_every == 0 or step == 1:
                logger.info(f"  step {step}/{num_steps} | Lf={s['L_forget']:.4f} "
                            f"Lr={s['L_retain']:.4f} Ltot={s['L_total']:.4f}")
        return history

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.unet.save_pretrained(save_dir)
        logger.info(f"modified UNET saved -> {save_dir}")
