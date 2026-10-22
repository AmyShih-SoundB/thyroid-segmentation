# Thyroid nodule segmentation — baselines and probes

Objective 1: establish reproducible segmentation baselines on public thyroid ultrasound, and determine which model and which training signal are worth building on.

Public data only (TN3K, TG3K, DDTI). No clinical or patient data is in this
repository, by design.

## Layout

| Path | Contents |
|---|---|
| `checks/` | Standalone dataset and metric audits (01–10) |
| `harness/` | `predict.py`, `score.py`, `compare.py`, `train_pseudo.py`, scoring drivers |
| `splits/` | Frozen split definitions — `v1.json` |
| `patches/` | Diffs against the vendored upstream repo, pinned by `UPSTREAM_COMMIT.txt` |
| `models/` | `resnet18_unet.py` — the one new architecture |
| `notes/` | Every log, per-case score CSV and `.meta` provenance record |
| `run_0*.sh` | Training drivers, one per run |
| `docs/` | Weekly progress reports and the W4 results log |

Large artefacts are gitignored: `repos/`, `runs/`, `checkpoints/`, `preds/`,
`pseudo/`, `venv/`. Checkpoint MD5s are recorded in
`notes/27_checkpoint_manifest.txt`.

The upstream repo (TRFE-Net-for-thyroid-nodule-segmentation) is not vendored here.
`patches/` plus `patches/UPSTREAM_COMMIT.txt` reproduce every modification made to it.

## Reproducing

```bash
source setup.sh              # venv lives on the network volume
./run_008.sh                 # or any run_0*.sh
./harness/run_score_XXX.sh   # scoring is always a separate step
```

Every `score.py` invocation writes a `.meta` recording the date, working
directory, git commit, full command line and all resolved arguments.
`train.py` is seeded and bit-deterministic — verified twice by exact reproduction
of a lost fold-0 checkpoint.

## Week index

| Week | Question | Scripts | Evidence |
|---|---|---|---|
| W1–2 | What is actually in these datasets? | `checks/01`–`04`, `make_splits.py` | `notes/01`–`05`, `splits/v1.json` |
| W2 | Can the published U-Net be reproduced? | — | run-004, `docs/W3-progress.docx` |
| W3 | Does the scorer give trustworthy numbers? | `checks/09`, `harness/score.py` | `notes/09_dummy_validation.log` |
| W3 | Is the published HD95 really Hausdorff distance? | `checks/08` | `notes/08_hd_definition*.log` |
| W3 | Any near-duplicates across splits? | `checks/10` | `notes/10_duplicate_*` |
| W3 | U-Net baseline, 5 folds | `run_008.sh` | `notes/06_*`, `15_*`, `26_*` |
| W4 | Q2 Does gland supervision help nodule segmentation? | `run_009.sh` | `notes/14_*`, `19_ladder_nodule.txt` |
| W4 | Q3 Does the TRFE+ architecture earn its complexity? | `run_010.sh` | `notes/17_*`, `19b_ladder_rung3.txt` |
| W4 | Q1 Which model should be the working baseline? | `harness/compare.py` | `notes/19`, `19b`, `20` |
| W4 | Q4 Are cross-dataset pseudo-labels plausible? | `run_011.sh` | `notes/21`, `22`, `23`, `28` |
| W5 | Q4.1 Do pseudo-labels work as training labels? | `run_012.sh` | `notes/30`–`38` |
| W5 | 3.1 Re-run without blank frames | `run_013.sh` | `notes/44`, `45` |
| W5 | 1.1 Is MTNet's advantage capacity? | `run_014.sh` | `notes/46_*` |
| W5 | 1.2 Do USCL weights help as initialisation? | `run_015.sh` | `notes/47_*` |

## Run index

| Run | Model | Params | Train data | Purpose |
|---|---|---|---|---|
| 008 | U-Net | 31.04 M | TN3K, 5 folds | baseline, rung 1 |
| 009 | MTNet | 43.2 M | TATN, 5 folds | rung 2 — adds gland supervision |
| 010 | TRFE+ | 45.4 M | TATN, 5 folds | rung 3 — adds explicit gland prior |
| 011 | U-Net | 31.04 M | TG3K | gland teacher for pseudo-labels |
| 012 | U-Net ×3 | 31.04 M | pseudo-labels | students B (control), C, G |
| 013 | U-Net | 31.04 M | pseudo, blanks removed | student C2 |
| 014 | U-Net, width 76 | 43.77 M | TN3K, 5 folds | capacity control vs MTNet |
| 015 | ResNet18-UNet ×3 | 14.34 M | TN3K, fold 0 | random / ImageNet / USCL init |

## Headline results

Nodule Dice on the TN3K test set (614 cases), per-case mean, SD across folds.
All rows scored through the identical `predict.py` → `score.py` path.

| Model | Params | Dice | IoU | HD95 |
|---|---|---|---|---|
| U-Net (008) | 31.04 M | 0.7805 ± 0.0067 | 0.6824 | 25.51 |
| U-Net widened (014) | 43.77 M | 0.7830 ± 0.0040 | 0.6853 | — |
| MTNet (009) | 43.2 M | 0.7991 ± 0.0079 | 0.7078 | 21.18 |
| TRFE+ (010) | 45.4 M | 0.8035 ± 0.0106 | 0.7077 | 21.59 |

Decision threshold **0.014 Dice**, fixed from the U-Net fold-to-fold spread before
any multi-task number was seen.

- **Gland supervision helps**: +0.0186 Dice, 5/5 folds, t = 12.56. Confounded with
  +39% parameters.
- **TRFE+'s explicit gland prior does not**: +0.0044, 3/5 folds, IoU flat, HD95
  slightly worse — three metrics disagreeing in sign, the signature of noise.
- **Capacity is not the explanation**: widening the U-Net by 41% (31.04 M → 43.77 M,
  matched to MTNet) moved Dice +0.0025, inside the fold spread, closing an eighth
  of the 0.0186 gap.
- **Pseudo-labelling fails in both directions** and is closed. Gland → TN3K
  over-fires; nodule → TG3K abstains on 56% of frames and misfires on the rest.
  A student trained on nodule pseudo-labels collapsed to all-zero predictions;
  removing the 26.1% blank targets recovered it to only 0.2060 against a control's
  0.7918, so the labels themselves carry little signal.
- **Initialisation is the lever.** Same 14.34 M ResNet18-UNet, TN3K fold 0:

  | Encoder starts from | Dice | IoU | HD95 | Empty preds |
  |---|---|---|---|---|
  | nothing (random) | 0.7550 | 0.6569 | 22.88 | 13 |
  | ImageNet photographs | 0.8134 | 0.7196 | 19.23 | 2 |
  | USCL (US-4 ultrasound) | 0.7372 | 0.6384 | 25.16 | 20 |

  USCL initialisation is worse than no pretraining (−0.0178). ImageNet gains
  +0.0584 at the same learning rate, so lr is not the explanation — the weights
  are. Most likely domain mismatch: US-4 contains no thyroid.

## Provenance notes

- `predict.py` thresholds **after** resizing to original size and writes PNG; the
  upstream `eval.py` rounded **before** resizing and wrote JPEG. This is a
  characterised 0.0005–0.0007 Dice difference, not a bug. Comparisons must stay
  within one pipeline: `notes/15_*` is the `predict.py` path for run-008,
  `notes/06_*_at-input` is the `eval.py` path.
- `.gitignore` excludes `*.png`; the three evidence images
  (`notes/22_gland_on_tn3k.png`, `notes/23_nodule_on_tg3k.png`,
  `notes/43_learning_curves.png`) are committed with `git add -f`.
- Git identity is set per-pod by `setup.sh` — `/root/.gitconfig` is container disk
  and does not survive restarts.
- `run_011.sh` and `run_013.sh` are reconstructions: both experiments were
  originally run as one-off commands. `run_011` predates
  `patches/w5_savedir.patch` and is what overwrote the run-008 fold-0 checkpoint;
  the incident and the verified recovery are documented in
  `docs/W4-results-log.md`.
- The gland teacher's 0.7287 in-domain Dice is **selection-biased upward**:
  `train.py` picks `*_best.pth` on the same TG3K validation frames it is scored
  on. It is not a clean held-out figure.

## Known limitations

See `docs/W4-results-log.md` for the full list. The ones that most affect
interpretation:

- TG3K is roughly 16 videos, not 3,585 independent images — effective sample size
  is far below the nominal count, and gland Dice has effective n ≈ 2 videos.
- The shipped TG3K split cuts through video 14, leaking 12.3% of the gland
  validation set. Regenerated on video boundaries here.
