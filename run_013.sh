#!/usr/bin/env bash
# run-013 — Student C2. Identical to run-012's arm C in every respect except
# that the 791 blank pseudo-label masks are excluded (--drop-blank).
#
# WHY: arm C collapsed to all-zero predictions (Dice 0.0000, 614/614 empty).
# 26.1% of its targets were blank, and SoftDiceLoss scores an empty prediction
# on a blank target at -1 while a perfect prediction on a real target scores ~0,
# so silence was the optimum. C2 separates "the loss rewards silence" from
# "the pseudo-labels carry no signal". Result: 0.2060 — the loss explains the
# collapse, the labels are still unusable. Requested by Brice (feedback 3.1).
set -euo pipefail
REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
H=/workspace/amy/harness
N=/workspace/amy/notes
cd "$REPO"

echo "=== C2: nodule student, TG3K pseudo-labels, blanks removed  $(date -Is) ==="
python -u "$H/train_pseudo.py" \
  --images data/tg3k/thyroid-image \
  --masks  /workspace/amy/pseudo/nodule-on-tg3k-train \
  --split-key tg3k:train \
  --drop-blank \
  --out /workspace/amy/runs/student-C2-nonblank.pth \
  2>&1 | tee "$N/44_student_C2.log"
