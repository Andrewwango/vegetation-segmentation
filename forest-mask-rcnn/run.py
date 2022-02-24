from datasets import FreiburgDataset
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import copy, csv
from deeplabmodel import DeepLabv3Model
from sklearn.metrics import f1_score, roc_auc_score

fd_train = FreiburgDataset("data/freiburg/train")
fd_test = FreiburgDataset("data/freiburg/test")

batch_size=8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def my_collate(batch):
    imgs = [s[0] for s in batch]
    masks = [s[1] for s in batch]
    return torch.stack(imgs, 0), torch.stack(masks, 0)
#for indices in batch_sampler:
    #yield collate_fn([dataset[i] for i in indices])

def cf(batch):
    zipped = tuple(zip(*batch))
    return [torch.stack(a, 0) for a in zipped]

train_loader = torch.utils.data.DataLoader(fd_train, batch_size=batch_size, shuffle=True, collate_fn=cf)
test_loader  = torch.utils.data.DataLoader(fd_test , batch_size=batch_size, shuffle=True, collate_fn=cf)

n_classes = fd_train[0][1].shape[0]
print(n_classes)

model = DeepLabv3Model(batch_size=batch_size,
                       device=device,
                       n_classes=n_classes,
                       criterion=torch.nn.CrossEntropyLoss(reduction='mean'),
                       epochs=10,
                       lr=1e-4)


model.train(train_loader, test_loader,
            metrics={'f1_score': f1_score, 'auroc': roc_auc_score},
            logdir='results')