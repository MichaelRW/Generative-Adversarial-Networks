# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:58:01 2019

@author: micha
"""

import os
# import functools  # Not Used
# import pickle  # Not Used
from collections import namedtuple
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
# from torch.nn import init  # Not Used
import torch.nn.functional as F

import numpy as np
import imageio
# import scipy  # Not Used



## Utils


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_data_loader(data_type, opts):
    """Creates training and test data loaders.
    """
    transform = transforms.Compose([
                    transforms.Scale(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_path = os.path.join(opts.data_dir, "train", data_type)
    test_path = os.path.join(opts.data_dir, "test", data_type)

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader




# Models

def transposedConv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out
    
    
    
##@title CycleGAN Generator Solution {display-mode: "form"}
class CycleGenerator(nn.Module):
    """
    # Defines the architecture of the generator networks
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        self.conv1 = conv(opts.num_channels, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        self.resnet_block = ResnetBlock(conv_dim * 2)

        self.transposedConv1 = transposedConv(conv_dim * 2, conv_dim, 4)
        self.transposedConv2 = transposedConv(conv_dim, 3, 4)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        out = F.relu(self.transposedConv1(out))
        out = F.tanh(self.transposedConv2(out))

        return out
    
    
    
##@title CycleGAN Discriminator Solution {display-mode: "form"}
class DCDiscriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()

        self.conv1 = conv(opts.num_channels, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)
        return out
    
    
    
class GANLoss(nn.Module):
    """
    Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        # return target_tensor.expand_as(prediction)
        return target_tensor.expand_as(prediction).to(torch.device('cuda'))

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            # prediction is 2D prediction map vector from Discriminator
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
    
    
# apples2oranges => options are apple & orange
# emojis => options are Apple & Windows

opt = {
    'image_size': 32, # type=int, help='The side length N to convert images to NxN.'
    'g_conv_dim': 32, #type=int
    'd_conv_dim': 32, # type=int
    'use_cycle_consistency_loss': False, # type=bool, help='Choose whether to include the cycle consistency term in the loss.'
    'init_zero_weights': False, # type=bool, help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).'

    # Training hyper-parameters
    'train_iters': 60000, # type=int, help='The number of training iterations to run.'
    'batch_size': 16, # type=int, help='The number of images in a batch.'
    'num_workers': 0, # type=int, help='The number of threads to use for the DataLoader.'
    'num_channels': 3,
    'lr': 0.00003, # type=float, help='The learning rate (default 0.0003)'
    'beta1': 0.5, # type=float, help='Adam parameter: exponential decay rate for the first moment estimates'
    'beta2': 0.999, # type=float, help='exponential decay rate for the second-moment estimates'

    # Data sources
    'data_dir': '/content/drive/My Drive/gans_workshop/emojis', # type=str, help='location of train and test data'
    'X': 'Apple', # type=str, help='Choose the type of images for domain X.'
    'Y': 'Windows', # type=str, help='Choose the type of images for domain Y.'

    # Saving directories and checkpoint/sample iterations
    'checkpoint_dir': '/content/drive/My Drive/gans_workshop/workshop_training/cyclegan/checkpoints', # type=str, help='location of model checkpoints'
    'sample_dir': '/content/drive/My Drive/gans_workshop/workshop_training/cyclegan/samples', # type=str, help='location of model samples'
    'load': None, # type=str, help='model checkpoint to load parameters from'
    'log_step': 10, # type=int, help='how often to log losses to console
    'sample_every': 1000, # type=int, help='how often to generate and save samples
    'checkpoint_every': 1000, # type=int, help='how often to save model checkpoint
}

opts = namedtuple("opt", opt.keys())(*opt.values())



def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """
    Prints model information for the generators and discriminators.
    """
    print("                 G_XtoY                ")
    print("---------------------------------------")
    print(G_XtoY)
    print("---------------------------------------")

    print("                 G_YtoX                ")
    print("---------------------------------------")
    print(G_YtoX)
    print("---------------------------------------")

    print("                  D_X                  ")
    print("---------------------------------------")
    print(D_X)
    print("---------------------------------------")

    print("                  D_Y                  ")
    print("---------------------------------------")
    print(D_Y)
    print("---------------------------------------")


def create_model(opts):
    """
    Initialize the generators and discriminators.
    Place them on GPU if desired.
    """
    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_checkpoint(opts):
    """
    Loads the generator and discriminator models from checkpoints.
    """
    G_XtoY_path = os.path.join(opts.load, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.load, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.load, 'D_X.pkl')
    D_Y_path = os.path.join(opts.load, 'D_Y.pkl')

    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage))
    G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def merge_images(sources, targets, recons, opts, k=10):
    """
    Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row*h, row*w*3])
    for idx, (s, t, r) in enumerate(zip(sources, targets, recons)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*3)*h:(j*3+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*3+1)*h:(j*3+2)*h] = t
        merged[:, i*h:(i+1)*h, (j*3+2)*h:(j*3+3)*h] = r
    return merged.transpose(1, 2, 0)

from PIL import Image
import numpy as np

def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """
    Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    recon_X = G_YtoX(fake_Y)
    recon_Y = G_XtoY(fake_X)

    X, fake_X, recon_X = to_data(fixed_X), to_data(fake_X), to_data(recon_X)
    Y, fake_Y, recon_Y = to_data(fixed_Y), to_data(fake_Y), to_data(recon_Y)

    merged = merge_images(X, fake_Y, recon_X, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X, recon_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))
    
    
    
##@title Training loop solution {display-mode: "form"}
SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):

    # Create generators and discriminators
    if opts.load:
        G_XtoY, G_YtoX, D_X, D_Y = load_checkpoint(opts)
    else:
        G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = to_var(test_iter_X.next()[0])
    fixed_Y = to_var(test_iter_Y.next()[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    criterionGAN = GANLoss(gan_mode='lsgan')
    criterionCycle = nn.L1Loss()

    lambda_X = 1.0  # weight for cycle loss (A -> B -> A^)
    lambda_Y = 1.0  # weight for cycle loss (B -> A -> B^)

    for iteration in range(1, opts.train_iters+1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, labels_X = iter_X.next()
        images_X, labels_X = to_var(images_X), to_var(labels_X).long().squeeze()

        images_Y, labels_Y = iter_Y.next()
        images_Y, labels_Y = to_var(images_Y), to_var(labels_Y).long().squeeze()


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        #########################################
        ##             FILL THIS IN            ##
        #########################################

        d_optimizer.zero_grad()

        # loss_real
        pred_real = D_X(images_Y)
        loss_D_x_real = criterionGAN(pred_real, True)

        # loss_fake
        fake_y = G_XtoY(images_X)         # Y -> X'
        pred_fake = D_X(fake_y)
        loss_D_x_fake = criterionGAN(pred_fake, False)

        # loss and backward
        loss_D_x = (loss_D_x_real + loss_D_x_fake) 

        loss_D_x.backward()
        d_optimizer.step()


        ######################
        # netD_y
        ######################

        d_optimizer.zero_grad()

        # loss_real
        pred_real = D_Y(images_X)
        loss_D_y_real = criterionGAN(pred_real, True)

        # loss_fake
        fake_x = G_YtoX(images_Y)  
        pred_fake = D_Y(fake_x)
        loss_D_y_fake = criterionGAN(pred_fake, False)

        # loss and backward
        loss_D_y = (loss_D_y_real + loss_D_y_fake) 

        loss_D_y.backward()
        d_optimizer.step()



        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================


        #########################################
        ##    FILL THIS IN: Y--X-->Y CYCLE     ##
        #########################################
        g_optimizer.zero_grad()

        fake_y = G_XtoY(images_X)         # Y -> X'
        prediction = D_X(fake_y)
        loss_G_X = criterionGAN(prediction, True)

        x_hat = G_YtoX(fake_y)          # Y -> X' -> Y^

        # Forward cycle loss y^ = || G_x(G_y(real_y)) ||
        loss_cycle_X = criterionCycle(x_hat, images_X) * lambda_Y

        loss_X = loss_G_X + loss_cycle_X 
        loss_X.backward()
        g_optimizer.step()

        #########################################
        ##    FILL THIS IN: X--Y-->X CYCLE     ##
        #########################################

        g_optimizer.zero_grad() # set g_x and g_y gradients to zero

        fake_x = G_YtoX(images_Y)         # Y -> X'
        prediction = D_Y(fake_x)
        loss_G_Y = criterionGAN(prediction, True)
        # print(f'loss_G_Y = {round(float(loss_G_Y), 3)}')

        y_hat = G_XtoY(fake_x)          # Y -> X' -> Y^

        # Forward cycle loss y^ = || G_x(G_y(real_y)) ||
        loss_cycle_Y = criterionCycle(y_hat, images_Y) * lambda_Y


        loss_Y = loss_G_Y + loss_cycle_Y 
        loss_Y.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | loss_Y: {:6.4f} | loss_X: {:6.4f} | loss_D_y: {:6.4f} | '
                  'loss_D_x: {:6.4f}'.format(
                    iteration, opts.train_iters, loss_Y.item(), loss_X.item(),
                    loss_D_y.item(), loss_D_x.item()))

        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)


        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)
            
            
            
def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_data_loader(opts.X, opts=opts)
    dataloader_Y, test_dataloader_Y = get_data_loader(opts.Y, opts=opts)

    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts)




main(opts)



