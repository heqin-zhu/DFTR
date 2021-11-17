from functools import partial

import torch

from .sod_loss import SODLoss


def get_loss(s, **kargs):
    return {
            'sodloss': partial(SODLoss, **kargs),
           }[s.lower()]


def get_optimizer(s):
    return {'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
           }[s.lower()]


def get_scheduler(s):
    dic =  {'cycliclr': torch.optim.lr_scheduler.CyclicLR,
            'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
           }
    s = s.lower()
    if s in dic:
        return dic[s]
