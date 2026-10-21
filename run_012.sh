#!/usr/bin/env bash
# run-012 — students trained on pseudo-labels (cross-domain distillation probe)
set -euo pipefail
REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
H=/workspace/amy/harness
N=/workspace/amy/notes
cd "$REPO"

echo "=== B: nodule student, TN3K pseudo-labels (control)  $(date -Is) ==="
python -u "$H/train_pseudo.py" \
  --images data/tn3k/trainval-image \
  --masks  /workspace/amy/pseudo/nodule-on-tn3k-trainval \
  --split-key tn3k:trainval \
  --out /workspace/amy/runs/student-B-nodule-from-tn3k.pth 2>&1 | tee "$N/30_student_B.log"

echo "=== C: nodule student, TG3K pseudo-labels (cross-domain)  $(date -Is) ==="
python -u "$H/train_pseudo.py" \
  --images data/tg3k/thyroid-image \
  --masks  /workspace/amy/pseudo/nodule-on-tg3k-train \
  --split-key tg3k:train \
  --out /workspace/amy/runs/student-C-nodule-from-tg3k.pth 2>&1 | tee "$N/31_student_C.log"

echo "=== G: gland student, TN3K pseudo-labels  $(date -Is) ==="
python -u "$H/train_pseudo.py" \
  --images data/tn3k/trainval-image \
  --masks  /workspace/amy/pseudo/gland-on-tn3k-trainval \
  --split-key tn3k:trainval \
  --out /workspace/amy/runs/student-G-gland-from-tn3k.pth 2>&1 | tee "$N/32_student_G.log"
