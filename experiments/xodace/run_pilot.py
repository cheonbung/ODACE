"""X-ODACE feasibility pilot -- does a MINIMAL causal circuit for nudity even exist?

For every matched pair (unsafe vs benign counterfactual, same scene) we run both trajectories
from the SAME initial noise on the FROZEN SD1.4 and, one component at a time, replace the
unsafe run's cross-attn signal with the benign run's:

    clean      : no patch                       -> the baseline unsafe image
    benign     : the donor run (free, same batch)
    L<l>_B<b>  : patch cross-attn layer l during timestep bin b   (16 x 5 = 80 components)
    ALL        : every component at once. This is a PLUMBING check, not a scientific result:
                 with every conditional cross-attn signal replaced, the unsafe run must become
                 pixel-identical to the benign run. If it does not, some cross-attn path is not
                 hooked and every causal score below is under-counted.

Each run is scored on three axes (scorer.py): unsafe (NudeNet, exposed-only primary), person
coherence and scene retention. analyze_pilot.py turns those into CE_c and answers the go/no-go
questions: concentration, top-20% effect, random control, seed/paraphrase stability, ODACE-dW
overlap.

Nothing is trained here and no existing checkpoint is touched -- this is measurement on the
frozen raw SD1.4. Modes 'scan' (single components) and 'setpatch' (named component sets, for
the insertion/deletion curve and the random control) share one driver. --patch_mode picks the
mediator: 'output' (text + trajectory state) or 'context' (K/V source only, text-conditioning
mediator alone) -- see patcher.py.

Resumable: rows are appended to the output JSONL and a re-run skips (pair, seed, comp,
patch_mode) keys that are already present, matching eval/xeval.py's convention. Every run also
drops a <out>.manifest.json (code hashes + config + pair-file hash) so a scan can be traced
back to the exact code that produced it.

Run (WSL conda env lsse):
  # smoke: 1 pair, 2 components + clean + ALL, with the plumbing invariants asserted
  python models/odace/experiments/xodace/run_pilot.py --limit 1 --limit_components 2 \
      --sanity --assert_invariants
  # stage 1 (coarse scan over the discovery split)
  python models/odace/experiments/xodace/run_pilot.py --mode scan \
      --pairs models/odace/experiments/xodace/data/matched_pairs_nudity_screened.jsonl \
      --split discovery --n_pairs 20 --seeds 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_pairs import read_pairs                                      # noqa: E402
from patcher import (PATCH_MODES, CrossAttnPatcher,                    # noqa: E402
                     components_to_schedule, expected_patches)
from scorer import Scorer                                              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xodace.pilot")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:  # noqa: BLE001
    _HAS_TQDM = False

SD_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_PAIRS = HERE / "data" / "matched_pairs_nudity.jsonl"
DEFAULT_OUT = HERE / "outputs" / "pilot_scan.jsonl"
DEFAULT_IMG = HERE / "outputs" / "images"
LATENT_HW = 64          # 512px / VAE factor 8
VAE_SCALE = 0.18215
CODE_FILES = ("run_pilot.py", "patcher.py", "scorer.py", "make_pairs.py")


class Ctx:
    """Frozen SD1.4 pieces + the attn2 patcher, built once and reused across every run."""

    def __init__(self, device, sd_id: str = SD_ID, patch_mode: str = "output"):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device
        self.dtype = torch.float16
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_id, subfolder="text_encoder").to(device=device, dtype=self.dtype).eval()
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_id, subfolder="unet").to(device=device, dtype=self.dtype).eval()
        self.vae = AutoencoderKL.from_pretrained(
            sd_id, subfolder="vae").to(device=device, dtype=self.dtype).eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.sched_id = sd_id
        self.scheduler_cls = DDIMScheduler
        self.patcher = CrossAttnPatcher(self.unet, mode=patch_mode).attach()
        self.n_layers = self.patcher.n_layers
        logger.info(f"[ctx] SD={sd_id} cross-attn layers={self.n_layers} patch_mode={patch_mode}")

    def new_scheduler(self, n_steps: int):
        s = self.scheduler_cls.from_pretrained(self.sched_id, subfolder="scheduler")
        s.set_timesteps(n_steps, device=self.device)
        return s

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        tok = self.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                             return_tensors="pt").to(self.device)
        return self.text_encoder(tok.input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor):
        from PIL import Image
        img = self.vae.decode(z / VAE_SCALE).sample
        img = (img / 2 + 0.5).clamp(0, 1).float().cpu().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray((img * 255).round().astype("uint8"))


@torch.no_grad()
def run_trajectories(ctx: Ctx, unsafe: str, benign: str, seed: int, components,
                     n_steps: int, n_bins: int, guidance: float):
    """Denoise the unsafe and benign runs jointly, patching `components` into the unsafe run.

    Returns (unsafe_image, benign_image, n_patched). Two scheduler instances so no state can
    leak between the two trajectories.
    """
    sched_u = ctx.new_scheduler(n_steps)
    sched_b = ctx.new_scheduler(n_steps)
    g = torch.Generator(device=ctx.device).manual_seed(seed)
    z0 = torch.randn((1, ctx.unet.config.in_channels, LATENT_HW, LATENT_HW),
                     generator=g, device=ctx.device, dtype=ctx.dtype)
    z_u = z0 * sched_u.init_noise_sigma
    z_b = z0.clone() * sched_b.init_noise_sigma

    uncond = ctx.encode("")
    cond = torch.cat([uncond, ctx.encode(unsafe), uncond, ctx.encode(benign)])  # slots 0..3
    schedule = components_to_schedule(components, n_steps, n_bins)

    ctx.patcher.reset_counter()
    for i, t in enumerate(sched_u.timesteps):
        ctx.patcher.set_active(schedule[i])
        lat = torch.cat([z_u, z_u, z_b, z_b])
        lat = sched_u.scale_model_input(lat, t)
        eps = ctx.unet(lat, t, encoder_hidden_states=cond).sample
        e_uu, e_uc, e_bu, e_bc = eps.chunk(4)
        eps_u = e_uu + guidance * (e_uc - e_uu)
        eps_b = e_bu + guidance * (e_bc - e_bu)
        z_u = sched_u.step(eps_u, t, z_u).prev_sample
        z_b = sched_b.step(eps_b, t, z_b).prev_sample
    ctx.patcher.set_active(())
    return ctx.decode(z_u), ctx.decode(z_b), ctx.patcher.n_patched


def check_invariants(comp_key: str, comps: list, n_patched: int, img_u, img_b,
                     n_steps: int, n_bins: int) -> None:
    """Smoke gate: the patch plumbing must be provably complete BEFORE any GPU hours are spent.

    clean -> 0 copies; a single (layer, bin) -> one copy per step in that bin (6 @ 30/5);
    ALL -> every layer at every step (480 @ 16x30) AND the patched unsafe image must be
    pixel-identical to the benign donor. The image check is the one that proves NO conditional
    cross-attn path was missed; it says nothing about the concept itself (ALL transplants the
    entire benign conditioning, not only the nudity part).
    """
    want = expected_patches(comps, n_steps, n_bins)
    if n_patched != want:
        raise AssertionError(f"[invariant] {comp_key}: n_patched={n_patched}, expected {want}")
    if comp_key == "ALL" and img_u.tobytes() != img_b.tobytes():
        raise AssertionError(
            "[invariant] ALL-patch image differs from the benign donor -> some conditional "
            "cross-attn path is NOT hooked; every causal score would be under-counted")


def load_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["pair"], r["seed"], r["comp"], r.get("patch_mode", "output")))
            except Exception:  # noqa: BLE001
                logger.warning("skipping malformed result line")
    return done


def component_jobs(mode: str, n_layers: int, n_bins: int, sets_json: Path | None,
                   limit_components: int, sanity: bool) -> list[tuple[str, list]]:
    """(comp_key, components) jobs. 'clean' must stay first: it is the CE baseline."""
    jobs: list[tuple[str, list]] = [("clean", [])]
    if sanity:
        jobs.append(("ALL", [(lyr, b) for lyr in range(n_layers) for b in range(n_bins)]))
    if mode == "screen":
        # Baseline pass only: a pair whose CLEAN unsafe image is not actually exposed has
        # nothing to erase, so CE against it is noise. select_pairs.py keeps the ones that fire.
        return jobs
    if mode == "scan":
        scan = [(f"L{lyr}_B{b}", [(lyr, b)]) for lyr in range(n_layers) for b in range(n_bins)]
        if limit_components:
            scan = scan[:limit_components]
        jobs += scan
    elif mode == "setpatch":
        if sets_json is None:
            raise SystemExit("--mode setpatch requires --sets_json")
        sets = json.loads(sets_json.read_text(encoding="utf-8"))
        jobs += [(name, [tuple(c) for c in comps]) for name, comps in sets.items()
                 if not name.startswith("_")]
    else:
        raise SystemExit(f"unknown mode {mode}")
    return jobs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_manifest(out: Path, args, rows: list[dict], n_layers: int) -> Path:
    """Provenance for the scan: which code, which config, which pairs produced these rows."""
    path = out.with_suffix(".manifest.json")
    manifest = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": sys.argv,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "code_sha256": {f: _sha(HERE / f) for f in CODE_FILES},
        "pairs_sha256": _sha(args.pairs),
        "pair_ids": [r["id"] for r in rows],
        "sd_id": SD_ID,
        "n_cross_attn_layers": n_layers,
        "torch": torch.__version__,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--n_pairs", type=int, default=20, help="base pairs to use")
    ap.add_argument("--split", default="", choices=["", "discovery", "heldout"],
                    help="restrict to one split of the screened pair file (guards Stage 2A)")
    ap.add_argument("--include_paraphrase", action="store_true",
                    help="also run the wording-variant pairs (Q4 stability)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap total pairs")
    ap.add_argument("--seeds", default="42", help="comma list of generation seeds")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--mode", default="scan", choices=["screen", "scan", "setpatch"])
    ap.add_argument("--patch_mode", default="output", choices=list(PATCH_MODES),
                    help="output = attn2 output (text + trajectory state); "
                         "context = encoder_hidden_states only (text-conditioning mediator)")
    ap.add_argument("--sets_json", type=Path, default=None)
    ap.add_argument("--limit_components", type=int, default=0, help="smoke: cap scan components")
    ap.add_argument("--sanity", action="store_true", help="add the patch-everything control run")
    ap.add_argument("--assert_invariants", action="store_true",
                    help="hard-fail if n_patched or the ALL-patch image break the plumbing rules")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--images_dir", type=Path, default=DEFAULT_IMG)
    ap.add_argument("--save_images", action="store_true",
                    help="keep intervention PNGs (clean/benign are always kept)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = read_pairs(args.pairs)
    if args.split:
        pairs = [p for p in pairs if p.get("split") == args.split]
        if not pairs:
            raise SystemExit(f"no pairs with split={args.split} in {args.pairs}")
    base = [p for p in pairs if p["paraphrase_of"] is None][:args.n_pairs]
    keep_ids = {p["id"] for p in base}
    rows = base + ([p for p in pairs if p["paraphrase_of"] in keep_ids]
                   if args.include_paraphrase else [])
    if args.limit:
        rows = rows[:args.limit]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    ctx = Ctx(device, patch_mode=args.patch_mode)
    scorer = Scorer(device)
    jobs = component_jobs(args.mode, ctx.n_layers, args.bins, args.sets_json,
                          args.limit_components, args.sanity)
    done = load_done(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"manifest -> {write_manifest(args.out, args, rows, ctx.n_layers)}")

    total = len(rows) * len(seeds) * len(jobs)
    logger.info(f"pairs={len(rows)} split={args.split or 'all'} seeds={seeds} "
                f"jobs/pair={len(jobs)} -> {total} runs ({len(done)} already recorded) | "
                f"steps={args.steps} bins={args.bins} patch_mode={args.patch_mode}")

    it = [(p, s, j) for p in rows for s in seeds for j in jobs]
    bar = tqdm(it, unit="run") if _HAS_TQDM else it
    t0 = time.time()
    n_new = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for pair, seed, (comp_key, comps) in bar:
            if (pair["id"], seed, comp_key, args.patch_mode) in done:
                continue
            img_dir = args.images_dir / pair["id"] / f"seed{seed}"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_u, img_b, n_patched = run_trajectories(
                ctx, pair["unsafe"], pair["benign"], seed, comps,
                args.steps, args.bins, args.guidance)
            if args.assert_invariants:
                check_invariants(comp_key, comps, n_patched, img_u, img_b,
                                 args.steps, args.bins)

            png_u = img_dir / f"{comp_key}.png"
            img_u.save(png_u)
            sc_u = scorer.score(img_u, str(png_u), pair["scene"])
            common = {"pair": pair["id"], "seed": seed, "steps": args.steps, "bins": args.bins,
                      "guidance": args.guidance, "paraphrase_of": pair["paraphrase_of"],
                      "split": pair.get("split"), "patch_mode": args.patch_mode}
            fh.write(json.dumps({**common, "comp": comp_key, "n_patched": n_patched,
                                 "layer": comps[0][0] if len(comps) == 1 else None,
                                 "bin": comps[0][1] if len(comps) == 1 else None,
                                 **sc_u.as_dict()}) + "\n")

            if comp_key == "clean":       # the donor run is identical for every job -> score once
                png_b = img_dir / "benign.png"
                img_b.save(png_b)
                sc_b = scorer.score(img_b, str(png_b), pair["scene"])
                fh.write(json.dumps({**common, "comp": "benign", "n_patched": 0,
                                     "layer": None, "bin": None, **sc_b.as_dict()}) + "\n")
            elif not args.save_images:
                png_u.unlink(missing_ok=True)
            fh.flush()
            n_new += 1

    dt = time.time() - t0
    per = dt / n_new if n_new else 0.0
    logger.info(f"done: {n_new} new runs in {dt/60:.1f} min ({per:.1f} s/run) -> {args.out}")


if __name__ == "__main__":
    main()
