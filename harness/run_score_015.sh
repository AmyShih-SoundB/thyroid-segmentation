#!/usr/bin/env bash
# Score run-015 (ResNet18-UNet, three inits) on the TN3K test set, fold 0.
set -euo pipefail
REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
H=/workspace/amy/harness
cd "$REPO"
for init in random imagenet uscl; do
  echo "===== scoring $init  $(date) ====="
  python "$H/predict.py" --model resnet18unet \
    --checkpoint /workspace/amy/runs/run-015-r18unet-$init-fold0.pth \
    --images data/tn3k/test-image \
    --out /workspace/amy/preds/run015-$init-fold0 \
    --split-key tn3k:test
  python "$H/score.py" \
    --pred /workspace/amy/preds/run015-$init-fold0 \
    --gt   data/tn3k/test-mask \
    --split-key tn3k:test --at input \
    --out  /workspace/amy/notes/47_score_run015_$init.csv
done
echo "===== done $(date) ====="
