"""ODACE retain-utility eval -- CLIP score + self-calibrated FID, NO COCO required.

Measures whether the nudity erasure damaged general (incl. unrelated) generation:
  - retain pool = nudity + vangogh + violence *maintain* prompts (the last two are
    unrelated to nudity -> tests collateral damage).
  - CLIP score: text-image alignment for raw SD vs ODACE v3 (higher=better).
  - FID is self-calibrated WITHOUT real COCO images:
      FID_floor        = FID(raw@seed42  <-> raw@seed123)   # same model, noise-only drift
      FID_odace_vs_raw = FID(odace@seed42 <-> raw@seed42)    # model-induced drift
    FID_odace ~ FID_floor  => distribution preserved; FID_odace >> FID_floor => degraded.

Run (WSL conda env lsse):
  python evaluate_utility.py --unet_dir outputs/odace_v3/final --output_dir outputs/eval/odace_v3
"""
from __future__ import annotations

import argparse, json, logging, os, sys
from pathlib import Path
import torch

LSSE = Path(__file__).resolve().parents[1] / "lsse"
sys.path.insert(0, str(LSSE))
from generate_images import build_pipeline, generate_images  # noqa: E402
try:
    from evaluation import FIDCLIPEvaluator  # noqa: E402
except ImportError:
    from evaluation.fid_clip_evaluator import FIDCLIPEvaluator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_utility")

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
MAINTAIN_FILES = ["nudity_maintain.txt", "vangogh_maintain.txt", "violence_maintain.txt"]


def read_prompts(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def gen(pipe, prompts, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    generate_images(pipe=pipe, prompts=prompts, output_dir=out_dir,
                    num_images_per_prompt=1, num_inference_steps=50,
                    guidance_scale=7.5, seed=seed)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_ref", type=int, default=123)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    pool = []
    for f in MAINTAIN_FILES:
        pf = LSSE / "data/prompts" / f
        if pf.exists():
            pool += read_prompts(pf)
    logger.info(f"retain pool = {len(pool)} prompts from {len(MAINTAIN_FILES)} maintain sets")

    qdir = os.path.join(args.output_dir, "quality")
    raw_s42 = os.path.join(qdir, "raw_s42")
    raw_ref = os.path.join(qdir, f"raw_s{args.seed_ref}")
    odace_s42 = os.path.join(qdir, "odace_s42")

    # raw SD (reference model)
    pipe = build_pipeline(sd_model_id=SD_MODEL_ID, encoder_dir=None, device=device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("--- generating raw SD @seed42 ---");        gen(pipe, pool, raw_s42, args.seed)
    logger.info(f"--- generating raw SD @seed{args.seed_ref} ---"); gen(pipe, pool, raw_ref, args.seed_ref)

    # swap in ODACE v3 UNET
    from diffusers import UNet2DConditionModel
    dtype = pipe.unet.dtype
    logger.info(f"swapping UNET <- {args.unet_dir}")
    pipe.unet = UNet2DConditionModel.from_pretrained(args.unet_dir).to(device=device, dtype=dtype)
    logger.info("--- generating ODACE v3 @seed42 ---");      gen(pipe, pool, odace_s42, args.seed)

    ev = FIDCLIPEvaluator(device=device)
    clip_raw = ev.compute_clip_score(raw_s42, pool)
    clip_odace = ev.compute_clip_score(odace_s42, pool)
    fid_floor = ev.compute_fid(raw_s42, raw_ref)
    fid_odace = ev.compute_fid(odace_s42, raw_s42)

    res = {
        "n_prompts": len(pool),
        "clip_raw": clip_raw,
        "clip_odace": clip_odace,
        "clip_delta": clip_odace - clip_raw,
        "fid_floor_raw_vs_raw": fid_floor,
        "fid_odace_vs_raw": fid_odace,
        "fid_excess_over_floor": fid_odace - fid_floor,
    }
    out = os.path.join(args.output_dir, "utility_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("=== UTILITY ===")
    logger.info(f"CLIP raw={clip_raw:.2f}  odace={clip_odace:.2f}  delta={res['clip_delta']:+.2f}")
    logger.info(f"FID floor(raw<->raw)={fid_floor:.2f}  odace<->raw={fid_odace:.2f}  "
                f"excess={res['fid_excess_over_floor']:+.2f}")
    logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
