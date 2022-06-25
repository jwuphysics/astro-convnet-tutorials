"""John F Wu

Example Pytorch code for running VICReg on a directory of JPG image cutouts 
from DESI Legacy Imaging Surveys (grz band).

This example uses the hybrid deconvolution CNN from Wu & Peek (2020) but this 
can easily be substituted with any other encoder model (e.g. from timm). It 
also takes takes inspiration by using similar transforms as to Hayat et al. (2021).

Note -- this example served as my introduction to the new torch DataPipes.

Using an g5.4xlarge instance on AWS (NVidia A10G 24GB + 16 core machine), we can 
train 5M images using 512 images/batch in about 100 minutes per epoch 
(1.6 batches per second).
"""

import torch
import torchvision 
import torchdata.datapipes.iter as pipes

import kornia.augmentation as K

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from fastai.vision.all import ranger, Mish
import PIL.Image as PILImage

from pathlib import Path

PATH = Path("../").resolve()
images_path = Path(PATH / 'data/unlabeled-images/')

# might need to clone the code at https://github.com/jwuphysics/predicting-spectra-from-images/
import sys
sys.path.append(f"{PATH}/src")
from xresnet_hybrid import xresnet34_hybrid

num_workers = 16

size = 112
image_stats = [np.array([0.14814416, 0.14217226, 0.13984123]), np.array([0.0881476 , 0.07823102, 0.07676626])]

# training (data_transforms) and validation (base_transforms) -- note we don't use the latter here
data_transforms = torch.nn.Sequential(
    K.RandomRotation(360),
    K.RandomCrop((size, size)),
    K.Resize((size, size)),
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),
    PS1Reddening(redden=True, ebv_max=0.5),
    K.Normalize(*image_stats)
)

base_transforms = torch.nn.Sequential(
    K.CenterCrop((size, size)),
    # K.Resize((size, size)),
    K.Normalize(*image_stats)
)

def PS1_transforms(x):
    return (base_transforms(x), data_transforms(x))

def open_image(fp): return np.array(PILImage.open(fp))

def off_diagonal(x):
   """Return flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    # assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def VICReg_loss(z1, z2, eps=1e-4, lmbda=0.005, mu=25, nu=1):
    """ VICReg Loss as defined by paper (https://arxiv.org/pdf/2105.04906.pdf)
    
    Hyperparams lmbda, mu, and nu are set to the ImageNet optimal values, 
    as is the minimum scatter eps.
    
    Note N is batchsize and D is dimensionality of projector head.
    """
    N, D = z1.shape
    # assert z1.shape == z2.shape
    
    # invariance loss
    sim_loss = torch.nn.functional.mse_loss(z1, z2)

    # variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(torch.nn.functional.relu(1 - std_z1)) + torch.mean(torch.nn.functional.relu(1 - std_z2))
    
    # covariance loss
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)
    cov_loss = off_diagonal(cov_z1).pow_(2).sum() / D + off_diagonal(cov_z2).pow_(2).sum() / D
    
    # total loss
    loss = lmbda * sim_loss + mu * std_loss + nu * cov_loss
    
    return loss

# based on https://github.com/MustafaMustafa/ssl-sky-surveys/blob/main/utils/sdss_dr12_galactic_reddening.py
# assume no dereddening (only excess reddening applied as data augmentation)
class PS1Reddening(torch.nn.Module):
    def __init__(self, redden=True, ebv_max=0.5):
        self.R = torch.tensor([3.237, 2.176, 1.217])  # see Schlafly & Finkbeiner 2011, Table 6: Galactic RV=3.1 // DES grz
        self.redden = redden # apply reddening augmentation
        self.ebv_max = ebv_max

    def __call__(self, data):

        if type(data)==list:
            image = data[0]
        else:
            image = data

        shape = data.shape
        

        if self.redden:
            new_ebv = torch.rand(1) * self.ebv_max
            reddening_factor = 10.**(-self.R * new_ebv / 2.5)

            # broadcast and multiply in place
            image.multiply_(reddening_factor[None,...,None, None])

        return image

def lr_warmup(opt, max_lr, n_epoch, n_iter, n_iter_per_epoch, max_epoch=3):
        """Linearly increases learning rate of `opt` from 0 to `max_lr` 
        based on the current iter (batch) and epoch number.
        """
        opt.set_hyper('lr', (max_lr * max((n_epoch * n_iter_per_epoch + n_iter) / (max_epoch * n_iter_per_epoch), 1)))
        return opt

if __name__ == "__main__":

    N_epochs = 20
    batch_size = 512

    logfile = "vicreg-pretrain.log"
    modelfile = "pretrained-model.pth"

    dataset = (
        pipes.FileLister([images_path], masks="*.jpg")
        .map(open_image)
        .map(torchvision.transforms.ToTensor())
        .map(PS1_transforms) 
    )

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    N_images = len(list(images_path.rglob("*.jpg")))
    N_batch_per_epoch = np.ceil(N_images / batch_size).astype(int)

    print(N_images)

    # based on paper; we warm up over 3 epochs rather than 10
    lr = 1e-2
    weight_decay = 1e-6



    # model
    model = xresnet34_hybrid(sa=True, act_cls=Mish, n_out=512)

    # + projection head
    model = torch.nn.Sequential(
        *model,
        torch.nn.Linear(512, 2048, bias=True),
        torch.nn.Linear(2048, 2048, bias=True),
        torch.nn.Linear(2048, 2048, bias=True)
    ).cuda()

    opt = ranger(model.parameters(), lr=lr, wd=weight_decay)

    train_losses = []

    with open(logfile, 'a') as out_file:
        print(f"Epoch,Step,Loss", file=out_file)
        for n_ep in tqdm(range(N_epochs)):
            train_losses.append([])

            for n_iter, (b1, b2) in tqdm(enumerate(dataloader), total=N_batch_per_epoch):

                b1 = b1.squeeze_().to('cuda')
                b2 = b2.squeeze_().to('cuda')

                z1 = model(b1)
                z2 = model(b2)
                
                opt = lr_warmup(opt, max_lr=lr, n_epoch=n_ep, n_iter=n_iter, n_iter_per_epoch=N_batch_per_epoch, max_epoch=3) 
                
                opt.zero_grad()
                
                loss = VICReg_loss(z1, z2).to('cuda')  

                loss.backward()
                opt.step()

                train_losses[-1].append(loss)

                print(f"{n_ep+1},{n_iter},{loss:.10f}", file=out_file, flush=True)
            torch.save(model.state_dict(), modelfile)