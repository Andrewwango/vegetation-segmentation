import torch
import numpy as np

def cf(batch):
    zipped = tuple(zip(*batch))
    return [torch.stack(a, 0) for a in zipped]

def dn(x):
    return x.detach().cpu().numpy()

def nut(x):
    return ((x-x.min())/(x.max()-x.min()) > 0.5).astype(np.float32)