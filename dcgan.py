import os

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


def train_celebgan():
    celeb_gan_optimiser = CelebGanOptimiser(
        'CelebGAN',
        config.DATASET_PARAMS, config.MODEL_PARAMS, config.OPTIMIZER_PARAMS
    )
    celebgan = celeb_gan_optimiser.train()
    save_celebgan(celebgan)


def main():
    # train_celebgan()
    show_generated_images('CelebGAN')


if __name__ == '__main__':
    main()
