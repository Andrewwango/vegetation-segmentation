import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

class DeepLabv3Model:
    def __init__(self, batch_size, device, n_classes, criterion, epochs, lr=1e-4):
        self.batch_size = batch_size
        self.device = device
        self.resnet_penultimate_layer_sizes=[512,512,2048,2048] #for 18,34,50,101
        self.n_classes = n_classes
        self.criterion=criterion
        self.epochs=epochs
        self.lr = lr
        
        self.model = self.build(out_channels = self.n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        

    def build(self, out_channels=6):
        model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                        progress=True)
        model.classifier = DeepLabHead(2048, out_channels)
        model.train()
        model.to(self.device)
        return model
    

    def train(self, train_loader, test_loader, metrics, logdir):
        since = time.time()
        best_weights = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        fieldnames = ['epoch', 'train_loss', 'test_loss'] + \
                     [f'train_{m}' for m in metrics.keys()] + \
                     [f'test_{m}' for m in metrics.keys()]

        with open(os.path.join(logdir, 'log.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        for epoch in range(1, self.epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.epochs))
            print('-' * 10)
            log = {a: [0] for a in fieldnames}
            log['epoch'] = epoch

            #TRAIN
            self.model.train()

            for imgs, masks in tqdm(train_loader):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            log[f'train_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            log[f'train_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    loss.backward()
                    self.optimizer.step()

            train_loss = loss.item()

            #VALIDATION
            self.model.eval()

            for imgs,masks in tqdm(test_loader):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                with torch.set_grad_enabled(False):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            log[f'test_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            log[f'test_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

            test_loss = loss.item()


            log[f'train_loss'] = train_loss
            log[f'test_loss'] = test_loss
            print('Train: {:.4f}'.format(train_loss))
            print('Test: {:.4f}'.format(test_loss))        

            for field in fieldnames[3:]:
                log[field] = np.mean(log[field])

            print(log)

            with open(os.path.join(logdir, 'log.csv'), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(log)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
            
            torch.save(self.model,              os.path.join(logdir, 'deeplabv3_model.pt'))
            torch.save(self.model.state_dict(), os.path.join(logdir, 'deeplabv3_weights.pt'))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Lowest Loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_weights)
        
        torch.save(self.model,              os.path.join(logdir, 'deeplabv3_model.pt'))
        torch.save(self.model.state_dict(), os.path.join(logdir, 'deeplabv3_weights.pt'))
