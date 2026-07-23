<!--
Standalone code release of the ODACE method, extracted from a larger research
project. Notes for this repo:

* Layout: the package that lived at `models/odace/` in the source tree is placed
  at the repository root here.
* External dependencies not included: `evaluate_odace.py`, `evaluate_utility.py`,
  and `train_odace.py` import project-level modules (`evaluation/`,
  `eval/cost_utils`) from the larger repository, and some configs reference an
  external OOD-augmentation prompt file (`ood_aug_file`, e.g. `../lsse/...`).
  The core training/editing code under `core/`, `methods/`, and `targets.py` is
  self-contained.
* Not included by design: model checkpoints/weights, generated images, and run
  outputs (`outputs/`). See `.gitignore`.
-->

# ODACE



**ODACE** (Output-grounded Denoising-Anchored Contrastive Erasure) is a post-hoc
concept-erasure method for text-to-image diffusion models. It edits the Stable
Diffusion **UNet cross-attention** so the model stops producing a target concept
(here, `nudity`) while preserving unrelated generation. This repository
accompanies the ODACE paper and contains the training/editing code only.

## What Is Trained

```text
prompt -> CLIP text encoder -> text embedding -> SD UNet -> image
                                               ^ edited here (cross-attention)
```

| Component | Setting |
|---|---|
| Base model | `CompVis/stable-diffusion-v1-4` |
| Text encoder | frozen |
| Trainable module | UNet cross-attention `to_q`, `to_k`, `to_v`, `to_out` (`xattn_full: true`) |
| Target concept | `nudity` |

## Core Idea

Text-encoder erasure optimizes a proxy ("does the text embedding look erased?").
ODACE instead optimizes the quantity that actually drives the generated image:
the **UNet noise prediction** at a diffusion timestep.

At a latent sampled along a frozen-model denoising trajectory, ODACE builds an
output-grounded target from frozen-teacher noise predictions evaluated at the
**same** latent/timestep:

```text
e_c = teacher output for the concept (forget) prompt
e_b = teacher output for a benign anchor prompt ("a fully clothed person, photograph")

target = e_b - lambda * (e_c - e_b) = (1 + lambda) * e_b - lambda * e_c
```

The benign term `e_b` **anchors** a coherent, safe redirection; the contrastive
term repels the concept output `e_c`. Two objectives train the editable UNet:

```text
L_forget = MSE(edited output on concept prompt, target)
L_retain = MSE(edited output on retain prompt, frozen output on retain prompt)
L_total  = alpha * L_forget + beta * L_retain
```

`lambda = 0` reduces to a pure benign anchor (`erase_mode: benign_anchor`);
`lambda > 0` adds contrastive repulsion (`erase_mode: benign_neg`). Redirecting
to a coherent benign output — rather than only pushing away from the concept —
is what keeps generation from collapsing under adversarial prompts.

Target math is in `core/targets.py`; trainable-module selection in
`methods/unet_edit.py` and `methods/layer_selection.py`; the training loop in
`core/trainer.py`.

## Configuration

Final ODACE (SD v1.4, nudity):

| Config | `erase_mode` | `lambda` |
|---|---|---|
| `configs/nudity_odace_benign_n1.yaml` | `benign_neg` | 1.0 |
| `configs/nudity_odace_benign.yaml` | `benign_anchor` | — (pure anchor) |

Key hyperparameters:

| Parameter | Value |
|---|---:|
| `learning_rate` | `1.0e-4` |
| `num_steps` | `1500` |
| `batch_size` | `4` |
| `benign_prompt` | `"a fully clothed person, photograph"` |
| `sample_guidance` | `3.0` |
| `ddim_steps` | `30` |
| `xattn_full` | `true` |

## Run

From the repository root:

```bash
python train_odace.py --config configs/nudity_odace_benign_n1.yaml
```

## Evaluate

```bash
python evaluate_odace.py \
  --unet_dir outputs/odace_benign_n1/final \
  --output_dir outputs/eval/odace \
  --num_images 50

python evaluate_utility.py \
  --unet_dir outputs/odace_benign_n1/final \
  --output_dir outputs/eval/odace
```

The evaluation entry points import project-level modules (`evaluation/`,
`eval/cost_utils`) from the larger research repository; see the note at the top.

## Tests

```bash
python -m pytest tests -q
```
