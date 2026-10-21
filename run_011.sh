#!/usr/bin/env bash
# run-011 — gland teacher. U-Net trained on TG3K (gland masks) to generate
# pseudo gland labels on TN3K. Reconstructed from notes/run011-glandteacher.log
# (num_iter_tr 189 x batch 16 = 3,024 ~ 3,026 frames) and the W4 results log.
#
# HISTORICAL NOTE: the original run predated patches/w5_savedir.patch, so
# train.py built its save path from model_name + fold only and wrote to
# run/unet/fold0/ — silently overwriting the run-008 fold-0 checkpoint.
# See "Incident" in docs/W4-results-log.md. With the patch applied this script
# writes to run/unet-TG3K/fold0/ and is safe to re-run.
#
# CAVEAT ON THE NUMBER: train.py selects *_best.pth on the TG3K validation set,
# which is the same 559 frames the teacher is scored on. Its 0.7287 Dice is
# selection-biased upward and is NOT a clean held-out figure.
set -euo pipefail
cd /workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
mkdir -p /workspace/amy/runs

python -u train.py -model_name unet -dataset TG3K -fold 0 -gpu 0 \
  -lr 1e-2 -batch_size 16 -nepochs 50 -input_size 224 -criterion Dice \
  2>&1 | tee /workspace/amy/notes/run011-glandteacher.log

cp run/unet-TG3K/fold0/unet_best.pth /workspace/amy/runs/run-011-glandteacher.pth
