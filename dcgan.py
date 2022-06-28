import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as tv_dset
import torchvision.transforms as tv_transforms
import torchvision.utils as tv_utils
import numpy as np
import matplotlib.pyplot as plt

import config

def init_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    print(f'Set seed:{seed}')


def set_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(config.DEVICE)


class CelebGanData(object):
    """ Class for loading the Celeb image datset
    """
    def __init__(self, dataset_dir: str, image_size: int):
        self._dataset_dir = dataset_dir
        self._image_size = image_size

        self._dataset = tv_dset.ImageFolder(
            root=self._dataset_dir,
            transform=tv_transforms.Compose([
                tv_transforms.Resize(self._image_size),
                tv_transforms.CenterCrop(self._image_size),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
    
    def get_batch_loader(self, batch_size:int, num_workers:int=2, shuffle=True):
        return torch.utils.data.DataLoader(
            self._dataset,
            batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )    


def main():
    init_seed()


if __name__ == '__main__':
    main()