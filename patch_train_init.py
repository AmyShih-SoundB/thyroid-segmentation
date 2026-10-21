p = '/workspace/amy/repos/TRFE-Net-for-thyroid-nodule-segmentation/train.py'
s = open(p).read()

edits = [
(
'''    parser.add_argument('-model_name', type=str, default='unet')''',
'''    parser.add_argument('-model_name', type=str, default='unet')
    parser.add_argument('-init', type=str, default='none',
                        choices=['none', 'random', 'imagenet', 'uscl'],
                        help='encoder init for resnet18unet; ignored by other models')'''
),
(
'''    elif 'unet' == args.model_name:
        net = Unet(in_ch=3, out_ch=1, width=args.width)''',
'''    elif 'resnet18unet' == args.model_name:
        assert args.init != 'none', '-init must be random|imagenet|uscl'
        from model.resnet18_unet import ResNet18Unet
        net = ResNet18Unet(in_ch=3, out_ch=1, init=args.init)
    elif 'unet' == args.model_name:
        net = Unet(in_ch=3, out_ch=1, width=args.width)'''
),
(
'''                            + (f"-w{args.width}" if args.width != 64 else ""),''',
'''                            + (f"-w{args.width}" if args.width != 64 else "")
                            + (f"-{args.init}" if args.init != 'none' else ""),'''
),
]

for old, new in edits:
    n = s.count(old)
    assert n == 1, f'anchor appears {n}x, expected 1:\n{old[:90]}'
    s = s.replace(old, new)

import ast; ast.parse(s)
open(p, 'w').write(s)
print('train.py patched OK')
