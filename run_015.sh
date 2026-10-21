#!/bin/bash
# run-015: USCL screen. ResNet18-UNet (14.34 M), three inits, fold 0 only, TN3K.
# Decision rule: uscl must beat random by >0.014 Dice to earn a 5-fold run.
set -euo pipefail
cd /workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation
mkdir -p /workspace/amy/runs /workspace/amy/notes
for init in random imagenet uscl; do
  echo "===== resnet18unet init=$init fold 0  $(date) ====="
  python -u train.py -model_name resnet18unet -init $init -dataset TN3K -fold 0 -gpu 0 \
    -lr 1e-2 -batch_size 16 -nepochs 50 -input_size 224 -criterion Dice \
    2>&1 | tee /workspace/amy/notes/run015-r18unet-$init-fold0.log
  cp run/resnet18unet-TN3K-$init/fold0/*_best.pth \
     /workspace/amy/runs/run-015-r18unet-$init-fold0.pth
done
echo "===== done $(date) ====="
