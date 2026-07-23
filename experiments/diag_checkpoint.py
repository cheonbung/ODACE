"""Free (CPU-only) diagnosis of the ODACE v2 checkpoint.

(A) Loss trajectory: did training converge / move at all?
(B) Weight delta: did the saved cross-attn K/V actually diverge from raw SD-v1-4?

If losses are flat AND weight delta ~0  -> training bug (no learning).
If weights moved a lot but ASR ~ raw    -> edit doesn't transfer to generation.
Run:  python experiments/diag_checkpoint.py
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel

HERE = Path(__file__).resolve().parents[1]
CKPT = HERE / "outputs/odace_nudity/final/diffusion_pytorch_model.safetensors"
HIST = HERE / "outputs/odace_nudity/history.json"
SD_ID = "CompVis/stable-diffusion-v1-4"


def loss_summary():
    h = json.loads(HIST.read_text())
    lf = [r["L_forget"] for r in h]
    lr = [r["L_retain"] for r in h]
    n = len(h)
    first = sum(lf[:20]) / min(20, n)
    last = sum(lf[-20:]) / min(20, n)
    print(f"[A] steps={n}")
    print(f"    L_forget: first20={first:.5f}  last20={last:.5f}  "
          f"min={min(lf):.5f}  max={max(lf):.5f}")
    print(f"    L_retain: first20={sum(lr[:20])/min(20,n):.6f}  "
          f"last20={sum(lr[-20:])/min(20,n):.6f}  max={max(lr):.6f}")
    return first, last


def weight_delta():
    print(f"[B] loading trained ckpt: {CKPT.name}")
    trained = load_file(str(CKPT))
    print("    loading raw SD-v1-4 UNET (CPU, from HF cache)...")
    raw = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet")
    raw_sd = raw.state_dict()

    # cross-attn K/V keys (attn2.to_k / attn2.to_v)
    kv_keys = [k for k in trained if ".attn2.to_k." in k or ".attn2.to_v." in k]
    other_keys = [k for k in trained if k in raw_sd and k not in kv_keys]

    def rel_l2(keys):
        num = den = 0.0
        moved = 0
        for k in keys:
            if k not in raw_sd:
                continue
            d = (trained[k].float() - raw_sd[k].float())
            dn = d.norm().item()
            num += dn ** 2
            den += raw_sd[k].float().norm().item() ** 2
            if dn > 1e-6:
                moved += 1
        return (num ** 0.5) / (den ** 0.5 + 1e-12), moved, len(keys)

    kv_rel, kv_moved, kv_tot = rel_l2(kv_keys)
    oth_rel, oth_moved, oth_tot = rel_l2(other_keys[:200])  # sample non-trainable
    print(f"    cross-attn K/V : rel_L2={kv_rel:.4e}  moved={kv_moved}/{kv_tot}")
    print(f"    other (frozen) : rel_L2={oth_rel:.4e}  moved={oth_moved}/{len(other_keys[:200])}")
    return kv_rel, kv_moved


if __name__ == "__main__":
    f, l = loss_summary()
    print()
    kv_rel, kv_moved = weight_delta()
    print()
    print("=== VERDICT ===")
    if kv_moved == 0 or kv_rel < 1e-4:
        print("BUG: cross-attn K/V essentially unchanged -> save/train did not take effect.")
    elif l >= f * 0.9:
        print("BUG: loss did not decrease -> training signal too weak (lr/steps/target).")
    else:
        print("Weights moved & loss dropped, but ASR ~ raw -> edit fails to transfer to generation.")
