import os
import json
import logging
import shutil
from torchvision import transforms

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def get_dynamic_transform(metadata, resize_to=32):
    is_grayscale = metadata.get("is_grayscale", False)

    if is_grayscale:
        to_tensor = transforms.Lambda(lambda img: transforms.functional.to_tensor(img.convert("L")))
        normalize = transforms.Normalize((0.5,), (0.5,))
    else:
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        to_tensor,
        normalize
    ])
