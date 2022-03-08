import torch
import numpy as np

def cf(batch):
    zipped = tuple(zip(*batch))
    return [torch.stack(a, 0) for a in zipped]

def dn(x):
    return x.detach().cpu().numpy()

def nut(x, t=0.5):
    return ((x-x.min())/(x.max()-x.min()) > t).astype(np.float32)

def c2e(x):
    return x.transpose(1,2,0)