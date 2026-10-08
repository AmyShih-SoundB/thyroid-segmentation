#!/usr/bin/env bash
# run-009 — MTNet on TATN (TN3K nodule + TG3K gland), 5 folds.
# Settings matched to run-008 where the repo allows it.
# NOTE: train.py forces batch_size=8 for trfeplus/trfe* regardless of -batch_size.
# Split: /workspace/amy/splits/v1.json (TG3K regrouped on video boundaries).
set -euo pipefail

REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
NOTES=/workspace/amy/notes
cd "$REPO"

for f in 0 1 2 3 4; do
  echo "=== trfeplus fold $f  $(date -Is) ==="
  python -u train.py \
    -model_name trfeplus \
    -dataset TATN \
    -fold "$f" \
    -lr 0.01 \
    -nepochs 50 \
    2>&1 | tee "$NOTES/run010-trfeplus-fold$f.log"
done
