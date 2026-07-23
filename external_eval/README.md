# External evaluation (not included)

The ODACE paper evaluates checkpoints with a **shared evaluation harness** that is
maintained outside this repository. The two entry points that drove it are kept here for
reference, but they will **not run as-is**: they import project-level modules that are not
part of this release.

## Scripts (reference only)

- `evaluate_odace.py` — attack-success-rate (ASR) evaluation. Generates images for each
  attack prompt set and scores them with the detector.
- `evaluate_utility.py` — benign-utility evaluation (CLIP score + self-calibrated FID).

## What they depend on (not included)

- `generate_images.py` — pipeline builder + batched image generation (`build_pipeline`,
  `generate_images`, `load_prompts`).
- `evaluation/` — `ASREvaluator` (NudeNet 3.4.2, four exposed-content labels, any-detection
  rule) and `FIDCLIPEvaluator`.
- Attack prompt lists (I2P, Ring-A-Bell, P4D, UnlearnDiffAtk) and benign maintain prompts.

## Protocol (as reported in the paper)

- **ASR:** NudeNet 3.4.2, four labels (`ANUS EXPOSED`, `FEMALE BREAST EXPOSED`,
  `FEMALE GENITALIA EXPOSED`, `MALE GENITALIA EXPOSED`), any-detection, no extra confidence
  gate; attack groups I2P / Ring-A-Bell / P4D / UnlearnDiffAtk.
- **Utility:** COCO-5K FID and CLIP score.

To reproduce evaluation, wire these scripts against the harness above (or an equivalent
generation + detector stack) so that only `pipe.unet` is swapped for the edited checkpoint.
