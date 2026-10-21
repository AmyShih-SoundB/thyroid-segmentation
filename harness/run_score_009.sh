#!/usr/bin/env bash
# Score run-009 (MTNet, TATN) nodule head on the TN3K test set, 5 folds.
set -euo pipefail
REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
H=/workspace/amy/harness
cd "$REPO"
for f in 0 1 2 3 4; do
  echo "===== mtnet fold $f ====="
  python "$H/predict.py" --model mtnet --head nodule \
    --checkpoint "$REPO/run/mtnet/fold$f/mtnet_best.pth" \
    --images data/tn3k/test-image \
    --out /workspace/amy/preds/run009-mtnet-fold$f \
    --split-key tn3k:test
  python "$H/score.py" \
    --pred /workspace/amy/preds/run009-mtnet-fold$f \
    --gt   data/tn3k/test-mask \
    --split-key tn3k:test --at input \
    --out  /workspace/amy/notes/14_score_run009_fold$f.csv
done
