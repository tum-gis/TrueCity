import torch
from typing import Optional

from octformerseg import OctFormerSeg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Model(c: int = 3, k: int = 13, **kwargs) -> torch.nn.Module:
    # Use OctFormer defaults for best accuracy unless user overrides
    # (dwconv requires installing https://github.com/octree-nn/dwconv)
    return OctFormerSeg(in_channels=c, out_channels=k, **kwargs) 