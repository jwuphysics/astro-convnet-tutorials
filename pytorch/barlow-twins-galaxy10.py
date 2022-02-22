"""John F Wu

Example Pytorch code for running Barlow Twins for self-supervised learning.

Uses Galaxy10_DECaLS images and a resnet18 backbone. Can train 30 epochs in about 
1.5 hrs on an NVidia T4, using a max of ~8 GB of RAM.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet
import torch.nn.functional as F

from torch.cuda import amp

from tqdm.notebook import tqdm

# Galaxy10_DECals image stats
image_stats = (
    np.array([41.2858088 , 40.11242127, 39.1236794 ]),
    np.array([30.53698744, 27.71269393, 25.75387572])
)

rng = np.random.RandomState(42)

class GalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)

class PairTransform:
    def __init__(
        self, 
        train_transform=True, 
        pair_transform=True,
        image_size=224,
        image_stats=image_stats
    ):
        if train_transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], 
                #     p=0.8
                # ),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(180),
                transforms.ToTensor(),
                transforms.Normalize(*image_stats)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*image_stats)
            ])
        self.pair_transform = pair_transform
        
    def __call__(self, x):
        if self.pair_transform:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

# See https://github.com/yaohungt/Barlow-Twins-HSIC/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# train for one epoch to learn unique features
def train_one_epoch(net, data_loader, train_optimizer, epoch, n_epochs, batch_size, lmbda, feature_dim, scaler=None):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_tuple in train_bar:
        
        if scaler is None:
            (pos_1, pos_2), _ = data_tuple
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            # Barlow Twins

            # normalize the representations along the batch dimension
            out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
            out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)

            # cross-correlation matrix
            c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

            # loss
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

            off_diag = off_diagonal(c).pow_(2).sum()
            # off_diag = off_diagonal(c).add_(1).pow_(2).sum() # alternative where off-diagonals should be -1
            loss = on_diag + lmbda * off_diag
            
        else:
            with amp.autocast():
                (pos_1, pos_2), _ = data_tuple
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
                feature_1, out_1 = net(pos_1)
                feature_2, out_2 = net(pos_2)
                # Barlow Twins

                # normalize the representations along the batch dimension
                out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
                out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)

                # cross-correlation matrix
                c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

                # loss
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

                off_diag = off_diagonal(c).pow_(2).sum()
                # off_diag = off_diagonal(c).add_(1).pow_(2).sum() # alternative where off-diagonals should be -1
                loss = on_diag + lmbda * off_diag

        train_optimizer.zero_grad()
        
        if scaler is None:
            loss.backward()
            train_optimizer.step() 
        else:
            scaler.scale(loss).backward()
            scaler.step(train_optimizer)
            scaler.update()
            
        total_loss += loss.item() * batch_size   
        total_num += batch_size

    
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} lmbda:{:.4f} bs:{} f_dim:{}'.format(\
                                epoch, n_epochs, total_loss / total_num, lmbda, batch_size, feature_dim))
    return total_loss / total_num

# Based on https://github.com/yaohungt/Barlow-Twins-HSIC/blob/main/model.py
class Model(nn.Module):
    def __init__(self, feature_dim=32):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet.resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 128, bias=False), nn.BatchNorm1d(128),
                               nn.ReLU(inplace=True), nn.Linear(128, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

if __name__ == "__main__":

    # load Galaxy10_DECaLS (https://astronn.readthedocs.io/en/latest/galaxy10.html)
    with h5py.File('Galaxy10_DECals.h5', 'r') as f:
        images = np.array(f['images'])
        labels = np.array(f['ans'])    
    
    N = len(images)
    indices = rng.permutation(N)
    train_idxs = indices[:int(0.8*N)]
    valid_idxs = indices[int(0.8*N):]

    train_ds = GalaxyDataset(images[train_idxs], labels[train_idxs], transform=PairTransform(train_transform=True))
    valid_ds = GalaxyDataset(images[valid_idxs], labels[valid_idxs], transform=PairTransform(train_transform=False))

    # hyperparams
    lmbda = 0.005
    feature_dim = 32
    n_epochs = 30
    bs = 64
    lr = 1e-3
    weight_decay = 1e-3

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=bs,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # initialize resnet18-like model
    model = Model(feature_dim=feature_dim).cuda()
    train_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = amp.GradScaler() # for automatic mixed precision, otherwise do `scaler = None`

    losses = []
    for epoch in range(1, 1+n_epochs):
        loss = train_one_epoch(
            model, 
            train_loader, 
            train_opt, 
            epoch=epoch, 
            n_epochs=n_epochs, 
            batch_size=bs, 
            feature_dim=feature_dim, 
            lmbda=lmbda, 
            scaler=scaler
        )

        losses.append(loss)
