#!/usr/bin/env python
"""Run a trained checkpoint over a folder of images and save binary masks.

Preprocessing is built from the repo's own transforms, identical to eval.py,
so there is no chance of a mismatch with training.

  python predict.py --model unet --checkpoint runs/run-008-unet-fold0.pth \
      --images data/tn3k/test-image --out preds/run008-fold0 \
      --split-key tn3k:test
"""
import argparse, glob, json, os, sys, subprocess, datetime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.getcwd())
from dataloaders import custom_transforms as trforms


def build_model(name, device):
    if name == 'unet':
        from model.unet import Unet
        import os
        _w = int(os.environ.get('UNET_WIDTH', 64))
        print(f'  unet width={_w}')
        return Unet(3, 1, width=_w).to(device)
    if name == 'resnet18unet':
        from model.resnet18_unet import ResNet18Unet
        print('  resnet18-unet (encoder weights come from the checkpoint)')
        return ResNet18Unet(3, 1, init='random').to(device)
    if name == 'mtnet':
        from model.mtnet import MTNet
        return MTNet(3, 1).to(device)
    if name == 'trfeplus':
        from model.trfeplus import TRFEPLUS
        return TRFEPLUS(3, 1).to(device)
    sys.exit(f'unknown model: {name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['unet', 'mtnet', 'trfeplus', 'resnet18unet'])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--images', required=True, help='directory of input images')
    ap.add_argument('--out', required=True, help='directory to write masks into')
    ap.add_argument('--head', default='nodule', choices=['nodule', 'gland'],
                    help='which output of a multi-task model (index 0 / 1)')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='probability threshold applied after sigmoid')
    ap.add_argument('--split', default='/workspace/amy/splits/v1.json')
    ap.add_argument('--split-key', default=None,
                    help='restrict to this list, e.g. tn3k:test or tg3k:val')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(args.model, dev)
    state = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # exactly eval.py lines 101-104
    composed = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    files = sorted(f for f in glob.glob(f'{args.images}/*')
                   if os.path.isfile(f) and '.ipynb_checkpoints' not in f)

    if args.split_key:
        ds, key = args.split_key.split(':')
        allow = {f'{i:04d}' for i in json.load(open(args.split))[ds][key]}
        before = len(files)
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in allow]
        print(f'split filter {args.split_key}: {before} -> {len(files)} images')
        if not files:
            sys.exit('the split filter removed every image')

    os.makedirs(args.out, exist_ok=True)
    head_idx = 0 if args.head == 'nodule' else 1
    empty = 0

    with torch.no_grad():
        for n, f in enumerate(files, 1):
            img = Image.open(f).convert('RGB')
            w, h = img.size
            sample = composed({'image': img, 'label': Image.new('L', img.size)})
            x = sample['image'].unsqueeze(0).to(dev)

            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[head_idx]

            prob = torch.sigmoid(out)
            # mirror eval.py: interpolate the PROBABILITY to original size,
            # then threshold - not the other way round
            prob = F.interpolate(prob, size=(h, w), mode='bilinear',
                                 align_corners=False)
            mask = (prob.squeeze().cpu().numpy() > args.threshold)
            if mask.sum() == 0:
                empty += 1

            stem = os.path.splitext(os.path.basename(f))[0]
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                os.path.join(args.out, stem + '.png'))

            if n % 100 == 0:
                print(f'  {n}/{len(files)}', flush=True)

    print(f'\nwrote {len(files)} masks to {args.out}')
    print(f'  model={args.model}  head={args.head}  '
          f'size={args.input_size}  threshold={args.threshold}')
    print(f'  checkpoint={args.checkpoint}')
    print(f'  empty predictions: {empty}')

    meta = os.path.join(args.out, '_predict.meta')
    try:
        commit = subprocess.check_output(
            ['git', '-C', '/workspace/amy', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = 'no-git'
    with open(meta, 'w') as fh:
        fh.write(f'date    : {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n')
        fh.write(f'cwd     : {os.getcwd()}\n')
        fh.write(f'commit  : {commit}\n')
        fh.write(f'command : python {" ".join(sys.argv)}\n')
        fh.write(f'images  : {len(files)}\n')
        fh.write(f'empty   : {empty}\n')
        for k, v in sorted(vars(args).items()):
            fh.write(f'  {k:<12}= {v}\n')
    print(f'wrote {meta}')


if __name__ == '__main__':
    main()
