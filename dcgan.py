import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as tv_dset
import torchvision.transforms as tv_transforms
import torchvision.utils as tv_utils
import numpy as np
import matplotlib.pyplot as plt

import config
import modules


class CelebGanData(object):
    """ Class for loading the Celeb image datset
    """
    def __init__(self, dataset_params):
        print(f"Initialising CelebGan: {dataset_params}")

        self._dataset_dir = dataset_params['dataset_dir']
        self._image_size = dataset_params['image_size']

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


class CelebGan(object):
    """ Class for combining the modules into a single GAN
    """
    def __init__(self, model_params):
        print(f"Initialising CelebGan: {model_params}")
        self.generator = modules.Generator(**model_params)
        self.discriminator = modules.Discriminator(**model_params)
        self.loss = nn.BCELoss()
        self._n_latent = model_params['n_latent']
        
    def use_device(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        self._device = device

    def train(self):
        # setup training mode for component models
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        # setup evaluation mode for component models
        self.generator.eval()
        self.discriminator.eval()

    def generate(self, batch_size):
        """ Generate fake images from the latent space
        """
        self.generator(
            torch.randn(batch_size, self._n_latent, 1, 1, device=self._device)
        ).detach()


class CelebGanOptimiser(object):
    """ Class for training a GAN model. Initialises and returns the actual GAN model.
    """
    def __init__(self, dataset_params, model_params, opt_params):
        print(f"Initialising CelebGanOptimiser: {opt_params}")

        self._batch_size = opt_params['batch_size']
        self._workers = opt_params['workers']
        self._n_epochs = opt_params['n_epochs']
        self._learning_rate = opt_params['learning_rate']
        self._beta1 = opt_params['beta1']
        self._device = torch.device(opt_params['device'])
        random.seed(opt_params['seed'])
        torch.manual_seed(opt_params['seed'])
                
        self._dataset = CelebGanData(dataset_params)
        self._gan = CelebGan(model_params)

        self._gen_optimser = torch.optim.Adam(
            self._gan.generator.parameters(),
            lr=self._learning_rate, betas=(self._beta1, 0.999)
        )
        self._dis_optimser = torch.optim.Adam(
            self._gan.discriminator.parameters(),
            lr=self._learning_rate, betas=(self._beta1, 0.999)
        )
        # pre-set real and fake labels for generated or loaded data
        self._real_labels = torch.full(
            (self._batch_size,), 1.0, dtype=torch.float, device=self._device
        )
        self._fake_label = torch.full(
            (self._batch_size,), 0.0, dtype=torch.float, device=self._device
        )

    def _step_discriminator(self, data_real, data_fake):
        # Maximize log(D(x)) + log(1 - D(G(z)))
        # Gradients can be calculated separately as pytorch will sum gradients
        # from .backwards() steps
        self._dis_optimser.zero_grad()

        # real images
        dis_output = self._gan.discriminator(data_real).view(-1)
        loss_dis_real = self._gan.loss(dis_output, self._real_labels)
        loss_dis_real.backward()
        # percentage of real reported as real
        u_pred_real = dis_output.mean().item()

        # fake images
        dis_output = self._gan.discriminator(data_fake).view(-1)
        loss_dis_fake = self._gan.loss(dis_output, self._fake_labels)
        loss_dis_fake.backward()
        # percentage of fake reported as real
        u_pred_fake = dis_output.mean().item()

        self._dis_optimser.step()

        # total err
        loss_dis = loss_dis_real + loss_dis_fake
        return u_pred_real, u_pred_fake, loss_dis.item()

    def _step_generator(self, data_fake):
        # Maximize log(D(G(z)))
        self._gen_optimser.zero_grad()

        # fake images - aiming to get discriminator to classify as real
        dis_output = self._gan.discriminator(data_fake).view(-1)
        loss_gen = self._gan.loss(dis_output, self._real_labels)
        loss_gen.backward()
        # percentage of fake reported as real
        u_pred_fake = dis_output.mean().item()

        self._gen_optimser.step()

        # total err
        return u_pred_fake, loss_gen
    
    def _report(self,
                i_epoch, n_epoch, i_batch, n_batch,
                loss_dis, loss_gen,
                u_pred_real, u_pred_fake_pre, u_pred_fake_post):
        msg = f'[{i_epoch+1}/{n_epoch}][{i_batch+1}/{n_batch}]\t'
        msg += f'LossD={loss_dis:.4f}\tLossG={loss_gen:.4f}\t'
        msg += f'D(real)={u_pred_real:.4f}\tD(G(z))={u_pred_fake_pre:.4f}->{u_pred_fake_post:.4f}\t'
        print(msg)


    def train(self):
        print(f"Starting training on {self._device}")
        # set the model for training mode
        self._gan.use_device(self._device)
        self._gan.train()

        for i_epoch in range(self._n_epochs):
            loader = self._dataset.get_batch_loader(self._batch_size, self._workers)
            
            # the _ is for the unused labels returned by the file loader
            for i_batch, (data_real, _) in enumerate(loader):
                data_real = data_real.to(self._device)
                data_fake = self._gan.generate(self._batch_size)

                u_pred_real, u_pred_fake_pre, loss_dis = \
                    self._step_discriminator(data_real, data_fake)
                u_pred_fake_post, loss_gen = \
                    self._step_generator(data_fake)
            
            if i_batch % 50 == 0:
                self._report(
                    i_epoch, self._n_epochs, i_batch, len(loader),
                    loss_dis, loss_gen, u_pred_real, u_pred_fake_pre, u_pred_fake_post
                )


def main():
    celeb_gan_optimiser = CelebGanOptimiser(
        config.DATASET_PARAMS, config.MODEL_PARAMS, config.OPTIMIZER_PARAMS
    )
    celeb_gan_optimiser.train()


if __name__ == '__main__':
    main()