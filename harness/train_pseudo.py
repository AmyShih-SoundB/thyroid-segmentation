"""Train a student U-Net on pseudo-labels. Standalone: train.py is not touched.

Recipe matched to run-008: SGD lr 0.01, momentum 0.9, weight decay 5e-4,
poly decay (1 - epoch/epochs)^0.9, 50 epochs, batch 16, 224x224,
soft_dice loss (the repo's own, from model/utils.py -- it applies sigmoid
internally, so raw logits are passed).

Two deliberate deviations from train.py, both reported in the .meta file:
  1. No validation-based checkpoint selection. The final epoch is saved,
     because selecting on pseudo-labels only selects for agreeing with the
     teacher.
  2. Masks are thresholded at 127 rather than divided by label.max(). A blank
     pseudo-label has max 0, and the repo's division would yield NaN.

Note on the loss: SoftDiceLoss smooths with +1 top and bottom, so on a blank
target it returns 2/(pred_sum+1) and the loss reaches -1 when the model predicts
nothing -- a larger reward than the ~0 earned by a perfect prediction on a real
mask. Where a large fraction of targets are blank, this loss pays the model to
go silent. The blank fraction is printed at startup for this reason.

Usage:
  python harness/train_pseudo.py \
    --images data/tg3k/thyroid-image \
    --masks  /workspace/amy/pseudo/nodule-on-tg3k-train \
    --split-key tg3k:train \
    --out /workspace/amy/runs/student-nodule-from-tg3k.pth
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.getcwd())
from model.unet import Unet
from model.utils import soft_dice
from dataloaders import custom_transforms as trforms


class PseudoSet(Dataset):
    """Images from one directory, pseudo-labels from another, paired by stem."""

    def __init__(self, image_dir, mask_dir, names, transform, drop_blank=False):
        def index(d):
            out = {}
            for q in sorted(glob.glob(os.path.join(d, '*'))):
                if '.ipynb_checkpoints' in q or q.endswith('.meta'):
                    continue
                stem = os.path.splitext(os.path.basename(q))[0]
                out.setdefault(stem, q)
                try:
                    out.setdefault(str(int(stem)), q)   # unpadded form too
                except ValueError:
                    pass
            return out

        ims, mks = index(image_dir), index(mask_dir)
        self.pairs, self.missing = [], []
        for n in names:
            im, mk = ims.get(n), mks.get(n)
            if im and mk:
                self.pairs.append((im, mk))
            else:
                self.missing.append(n)
        self.dropped = 0
        if drop_blank:
            kept = [(i, m) for i, m in self.pairs
                    if (np.array(Image.open(m).convert('L')) > 127).sum() > 0]
            self.dropped = len(self.pairs) - len(kept)
            self.pairs = kept
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        image = Image.open(ip).convert('RGB')
        lab = (np.array(Image.open(mp).convert('L')) > 127).astype(np.uint8)
        return self.transform({'image': image, 'label': Image.fromarray(lab)})


def build_transform(input_size):
    tf = [trforms.FixedResize(size=(input_size, input_size))]
    if hasattr(trforms, 'RandomHorizontalFlip'):
        tf.append(trforms.RandomHorizontalFlip())
    else:
        print('WARNING: RandomHorizontalFlip not found in custom_transforms; '
              'training without it (train.py applies it)')
    tf += [trforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
           trforms.ToTensor()]
    return transforms.Compose(tf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True, help='directory of input images')
    p.add_argument('--masks', required=True, help='directory of pseudo-labels')
    p.add_argument('--split', default='/workspace/amy/splits/v1.json')
    p.add_argument('--split-key', required=True, help='e.g. tg3k:train')
    p.add_argument('--out', required=True, help='checkpoint path to write')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--drop-blank', action='store_true',
                   help='exclude pairs whose mask has no foreground')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds, key = args.split_key.split(':')
    names = [str(n) for n in json.load(open(args.split))[ds][key]]

    data = PseudoSet(args.images, args.masks, names, build_transform(args.input_size),
                     drop_blank=args.drop_blank)
    if getattr(data, 'dropped', 0):
        print(f'dropped: {data.dropped} blank-mask pairs')
    print(f'split {args.split_key}: {len(names)} names -> {len(data)} pairs')
    if data.missing:
        print(f'  unmatched: {len(data.missing)} (first few: {data.missing[:5]})')
    if len(data) == 0:
        sys.exit('ERROR: no image/mask pairs matched -- check --images and --masks')

    blank = sum(1 for _, m in data.pairs
                if (np.array(Image.open(m).convert('L')) > 127).sum() == 0)
    pct = 100.0 * blank / len(data)
    print(f'blank targets: {blank}/{len(data)} ({pct:.1f}%)')
    if pct > 25:
        print('  NOTE: SoftDiceLoss rewards an empty prediction on a blank target '
              '(-1) more than a perfect prediction on a real one (~0). Expect the '
              'student to be pulled toward predicting nothing.')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Unet(in_ch=3, out_ch=1).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=True)
    print(f'device={dev}  iters/epoch={len(loader)}  params='
          f'{sum(q.numel() for q in net.parameters())/1e6:.2f} M', flush=True)

    t0 = time.time()
    for ep in range(args.epochs):
        lr = args.lr * (1 - ep / args.epochs) ** 0.9
        for g in opt.param_groups:
            g['lr'] = lr
        net.train()
        tot = n = 0
        for b in loader:
            x = b['image'].to(dev)
            y = b['label'].to(dev).float()
            out = net(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            loss = soft_dice(out, y)          # applies sigmoid internally
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f'epoch: {ep} lr: {lr:.5f} trainloss: {tot/max(n,1):.4f} '
              f'timecost:{time.time()-t0:.1f} secs', flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(net.state_dict(), args.out)
    with open(args.out + '.meta', 'w') as f:
        f.write(f'date:        {time.strftime("%Y-%m-%dT%H:%M:%S")}\n'
                f'cwd:         {os.getcwd()}\n'
                f'command:     {" ".join(sys.argv)}\n'
                f'images:      {args.images}\n'
                f'masks:       {args.masks}\n'
                f'split:       {args.split} [{args.split_key}]\n'
                f'pairs:       {len(data)}  unmatched: {len(data.missing)}\n'
                f'blank:       {blank} ({pct:.1f}%)\n'
                f'loss:        soft_dice (model/utils.py SoftDiceLoss)\n'
                f'optimizer:   SGD lr={args.lr} momentum={args.momentum} '
                f'wd={args.weight_decay}, poly (1-ep/N)^0.9\n'
                f'epochs:      {args.epochs}  batch: {args.batch_size}  '
                f'input: {args.input_size}  seed: {args.seed}\n'
                f'selection:   final epoch, no validation-based selection\n'
                f'mask decode: threshold > 127 (not label/label.max())\n')
    print(f'wrote {args.out}')
    print(f'wrote {args.out}.meta')


if __name__ == '__main__':
    main()
