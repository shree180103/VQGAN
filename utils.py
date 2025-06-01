import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from PIL import Image
import numpy as np
import os


def Data_pipeline(args):
    transform=transforms.Compose(transforms=[
                                            transforms.ToTensor(),
                                                transforms.Resize((256, 256)),
                                            ])
    dataset=datasets.ImageFolder(root=args.path,transform=transform)
    data_loader=DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=True)

    return data_loader



def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') !=-1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    
    elif classname.find('InstanceNorm') !=-1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)




import torch
from torchvision.utils import make_grid
from PIL import Image

def save_image(tensor: torch.Tensor,
               filepath: str,
               nrow: int = 8,
               normalize: bool = True,
               value_range: tuple = None) -> None:
    """
    Save a batch of images as a grid to disk.

    Args:
        tensor (torch.Tensor): 4D mini-batch Tensor of shape (B, C, H, W).
        filepath (str): where to save the resulting image (including filename.png).
        nrow (int): number of images per row in the grid.
        normalize (bool): if True, shift the image to the range (0, 1) by min/max.
        value_range (tuple or None): tuple (min, max) for normalization; if None uses tensor min/max.
    """
    # make a grid of images
    grid = make_grid(tensor, nrow=nrow, normalize=normalize, range=value_range)
    # convert to a CPU uint8 numpy array (H, W, C)
    ndarr = (grid.mul(255)
                 .add(0.5)
                 .clamp(0, 255)
                 .permute(1, 2, 0)
                 .to('cpu', torch.uint8)
                 .numpy())
    # build and save PIL image
    im = Image.fromarray(ndarr)
    im.save(filepath)

