p = '/workspace/amy/harness/predict.py'
s = open(p).read()

edits = [
(
'''    if name == 'mtnet':
        from model.mtnet import MTNet
        return MTNet(3, 1).to(device)''',
'''    if name == 'resnet18unet':
        from model.resnet18_unet import ResNet18Unet
        print('  resnet18-unet (encoder weights come from the checkpoint)')
        return ResNet18Unet(3, 1, init='random').to(device)
    if name == 'mtnet':
        from model.mtnet import MTNet
        return MTNet(3, 1).to(device)'''
),
(
'''    ap.add_argument('--model', required=True, choices=['unet', 'mtnet', 'trfeplus'])''',
'''    ap.add_argument('--model', required=True,
                    choices=['unet', 'mtnet', 'trfeplus', 'resnet18unet'])'''
),
]

for old, new in edits:
    n = s.count(old)
    assert n == 1, f'anchor appears {n}x, expected 1:\n{old[:90]}'
    s = s.replace(old, new)

import ast; ast.parse(s)
open(p, 'w').write(s)
print('predict.py patched OK')
