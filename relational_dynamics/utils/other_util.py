import numpy as np
import torch
import torch.nn as nn 
import os

from relational_dynamics.utils import torch_util
from relational_dynamics.utils.data_utils import get_norm, get_activation, scale_min_max

def rotate_2d(x, theta):
    """ Rotate x by theta degrees (counter clockwise)

        @param x: a 2D vector (numpy array with shape 2)
        @param theta: a scalar representing radians
    """

    rot_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=np.float32)

    return rot_mat.dot(x)

def create_log_dirs(config):
    args = config.args
    # Create logger directory if required
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(config.get_logger_dir()):
        os.makedirs(config.get_logger_dir())
    if not os.path.exists(config.get_model_checkpoint_dir()):
        os.makedirs(config.get_model_checkpoint_dir())

class LinearBlock(nn.Module):

    def __init__(self, in_size, out_size, activation=None, norm=None):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, out_size))
        if activation:
            self.layers.append(get_activation(activation))
        if norm is not None:
            self.layers.append(get_norm(norm, out_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(nn.Module):

    def __init__(self, in_size, out_size, hidden_sizes=[], activation='relu',
                 norm=None, out_activation=None, out_norm=None, vae=False):
        """
        Multi-layer perception module. Stacks linear layers with customizable sizes, 
        activations, and normalization.

        Args:
            in_size (int): Size of input tensor
            out_size (int): Size of output tensor
            hidden_sizes (List[int]): Output sizes of each hidden layer
            activation (str): Activations to apply to each hidden layer
            layer_norms (bool): Apply layer norm to hidden layers if true
            out_activation (str): Activation to apply to output layer (default is none)
            out_layer_norm (bool): Apply layer norm to output layer if true
            vae (bool): Sample output as VAE
        Returns:
            Output of last layer; if vae=True, returns sample of output as well as mean and std

        TODO right now only layer norm supported, can make more general
        """
        super(MLP, self).__init__()

        self.vae = vae
        if self.vae:
            out_size *= 2
        
        self.layers = nn.ModuleList()
        prev_size = in_size
        for size in hidden_sizes:
            self.layers.append(LinearBlock(prev_size, size, activation, norm))
            prev_size = size
        self.layers.append(LinearBlock(prev_size, out_size, out_activation, out_norm))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if self.vae:
            mu, log_std = torch.chunk(x, 2, dim=-1)
            std = torch.exp(log_std)
            noise = torch.randn_like(mu)
            if self.training:
                return mu + std * noise
            else:
                return mu
        else:
            return x
