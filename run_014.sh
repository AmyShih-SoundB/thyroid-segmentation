#!/bin/bash
# run-014: capacity control for Q1. U-Net widened to 43.77 M (MTNet is 43.2 M), TN3K only.
# Identical to run-008 except -width 76. Checkpoints land in run/unet-TN3K-w76/ thanks to
# the save_dir patch, so run-008's artifacts are untouched.
set -euo pipefail
cd /workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
mkdir -p /workspace/amy/runs

for f in 0 1 2 3 4; do
  echo "===== widened unet fold $f  $(date) ====="
  python -u train.py -model_name unet -dataset TN3K -fold $f -gpu 0 -width 76 \
    -lr 1e-2 -batch_size 16 -nepochs 50 -input_size 224 -criterion Dice \
    2>&1 | tee /workspace/amy/notes/run014-unet-w76-fold$f.log
  cp run/unet-TN3K-w76/fold$f/unet_best.pth /workspace/amy/runs/run-014-unet-w76-fold$f.pth
done
echo "===== done $(date) ====="
