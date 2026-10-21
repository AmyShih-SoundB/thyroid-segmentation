#!/usr/bin/env bash
# Score run-014 (widened U-Net, width 76) on TN3K test, 5 folds.
set -euo pipefail
REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
H=/workspace/amy/harness
export UNET_WIDTH=76
cd "$REPO"
for f in 0 1 2 3 4; do
  echo "===== scoring fold $f  $(date) ====="
  python "$H/predict.py" --model unet \
    --checkpoint /workspace/amy/runs/run-014-unet-w76-fold$f.pth \
    --images data/tn3k/test-image --out /workspace/amy/preds/run014-w76-fold$f \
    --split-key tn3k:test
  python "$H/score.py" --pred /workspace/amy/preds/run014-w76-fold$f \
    --gt data/tn3k/test-mask --split-key tn3k:test --at input \
    --out /workspace/amy/notes/46_score_run014_fold$f.csv
done
echo "===== done $(date) ====="
