#!/usr/bin/env bash
# Score the three distillation students (run-012).
# B, C -> TN3K test (real nodule GT).  G -> TG3K val (real gland GT).
set -euo pipefail
REPO=/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
H=/workspace/amy/harness
R=/workspace/amy/runs
GT_GLAND=data/tg3k/thyroid-mask
cd "$REPO"
[ -d "$GT_GLAND" ] || { echo "ERROR: $GT_GLAND not found -- fix GT_GLAND"; exit 1; }

score () {  # $1=tag  $2=ckpt  $3=images  $4=gt  $5=splitkey  $6=csvnum
  echo "===== $1 ====="
  python "$H/predict.py" --model unet \
    --checkpoint "$2" --images "$3" \
    --out "/workspace/amy/preds/run012-$1" --split-key "$5"
  python "$H/score.py" \
    --pred "/workspace/amy/preds/run012-$1" --gt "$4" \
    --split-key "$5" --at input \
    --out "/workspace/amy/notes/${6}_score_student_$1.csv"
}

score B "$R/student-B-nodule-from-tn3k.pth" data/tn3k/test-image data/tn3k/test-mask tn3k:test 33
score C "$R/student-C-nodule-from-tg3k.pth" data/tn3k/test-image data/tn3k/test-mask tn3k:test 34
score G "$R/student-G-gland-from-tn3k.pth" data/tg3k/thyroid-image "$GT_GLAND" tg3k:val 35
