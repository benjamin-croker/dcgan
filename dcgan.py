import os
import sys

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.utils as tv_utils

import numpy as np
import matplotlib.pyplot as plt

import config
from modules import CelebGan, CelebGanOptimiser


def save_celebgan(celebgan: CelebGan):
    torch.save(
        {'celebgan': celebgan},
        os.path.join(config.MODEL_DIR, f'{celebgan.id}.gan')
    )


def load_celebgan(model_id: str) -> CelebGan:
    celebgan_state = torch.load(
        os.path.join(config.MODEL_DIR, f'{model_id}.gan')
    )
    return celebgan_state['celebgan']


def show_generated_images(model_id: str):
    celebgan = load_celebgan(model_id)

    gen_img = celebgan.generate(64)
    gen_img = tv_utils.make_grid(gen_img, padding=2, normalize=True)

    # Plot the fake images from the last epoch
    plt.subplot(1, 1, 1)
    plt.axis("off")
    plt.title("Generated Celebrities")
    plt.imshow(np.transpose(gen_img, (1, 2, 0)))
    plt.show()


def train_celebgan(model_id: str):
    celeb_gan_optimiser = CelebGanOptimiser(
        model_id, config.DATASET_PARAMS, config.MODEL_PARAMS, config.OPTIMIZER_PARAMS
    )
    celebgan = celeb_gan_optimiser.train()
    save_celebgan(celebgan)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ['train', 'generate']:
        print("usage: python dcgan.py {train|generate} <model id>")
        sys.exit(1)

    if sys.argv[1] == 'train':
        train_celebgan(sys.argv[2])
    elif sys.argv[1] == 'generate':
        show_generated_images(sys.argv[2])


if __name__ == '__main__':
    main()
