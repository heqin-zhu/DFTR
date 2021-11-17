import os
from collections.abc import Iterable

import numpy as np


def mkdir(path):
    def _mkdir(p):
        if p and not os.path.exists(p):
            _mkdir(p[:p.rfind(os.path.sep)])
            os.mkdir(p)
    _mkdir(os.path.abspath(path))


def np2py(obj):
    if isinstance(obj, dict):
        return {k: np2py(v) for k,v in obj.items()}
    if isinstance(obj, Iterable):
        return [np2py(i) for i in obj]
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj


def countParam(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
