"""ODACE ASR eval -- reuses the shared lsse harness but swaps pipe.unet (the modified UNET).

Keeps generation/scoring byte-identical to the unified comparison (build_pipeline +
generate_images + ASREvaluator on the same attack prompts), only replacing the UNET.

Run (WSL conda env lsse):
  python evaluate_odace.py --unet_dir outputs/odace_nudity/final \
      --output_dir outputs/eval/odace --num_images 50 [--attacks I2P,Ring-A-Bell]
"""
from __future__ import annotations

import argparse, json, logging, os, sys
from pathlib import Path
import torch

LSSE = Path(__file__).resolve().parents[1] / "lsse"
sys.path.insert(0, str(LSSE))
from generate_images import build_pipeline, generate_images, load_prompts  # noqa: E402
from evaluation import ASREvaluator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_odace")

SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
ATTACK_FILES = {
    "I2P": "i2p_nudity.txt", "Ring-A-Bell": "ring_a_bell_nudity.txt",
    "Ring-A-Bell(Re)": "ring_a_bell_re_nudity.txt", "P4D": "p4d_nudity.txt",
    "UnlearnDiffAtk": "unlearnDiffAtk_nudity.txt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_images", type=int, default=50)
    ap.add_argument("--attacks", default="all", help="comma list or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    attacks = list(ATTACK_FILES) if args.attacks == "all" else args.attacks.split(",")

    pipe = build_pipeline(sd_model_id=SD_MODEL_ID, encoder_dir=None, device=device)
    from diffusers import UNet2DConditionModel
    dtype = pipe.unet.dtype
    logger.info(f"swapping UNET <- {args.unet_dir} (dtype={dtype})")
    pipe.unet = UNet2DConditionModel.from_pretrained(args.unet_dir).to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    evaluator = ASREvaluator(concept_type="nudity")
    base = LSSE
    results = {}
    for label in attacks:
        pf = base / "data/eval" / ATTACK_FILES[label]
        if not pf.exists():
            logger.warning(f"missing {pf}"); continue
        prompts = load_prompts(str(pf))[:args.num_images]
        img_dir = os.path.join(args.output_dir, "images", "fcf_nudity",
                               label.lower().replace("-", "_").replace("(", "").replace(")", ""))
        os.makedirs(img_dir, exist_ok=True)
        logger.info(f"--- generating {label} ({len(prompts)}) ---")
        generate_images(pipe=pipe, prompts=prompts, output_dir=img_dir,
                        num_images_per_prompt=1, num_inference_steps=50,
                        guidance_scale=7.5, seed=args.seed)
        results[label] = evaluator.evaluate_directory(img_dir)
        logger.info(f"  {label}: ASR={results[label].get('asr')}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump({"asr": results}, f, indent=2)
    vals = [v["asr"] for v in results.values() if v.get("asr") is not None]
    logger.info(f"mean ASR = {sum(vals)/len(vals)*100:.1f}  over {len(vals)} attacks")


if __name__ == "__main__":
    main()
