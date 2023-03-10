# AI for Breakout

# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Initializing and setting the variance of a tensor of weights

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))   # var(out) = std^2

    return out


# Initializing the weights of the neural network in an optimal way for the learning

def weights_init(m):
    classname = m.__class__.__name__
    weight_shape = list(m.weight.data.size())

    if classname.find('Conv') != -1:
        fan_in = np.prod(weight_shape[1:4])     # dim1 * dimi2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]  # dim0 * dim2 * dim3
    elif classname.find('Linear') != -1:
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
    else:
        return

    w_bound = np.sqrt(6. / fan_in + fan_out)
    m.weight.data.uniform_(-w_bound, w_bound)
    m.bias.data.fill_(0)


# Making the A3C brain

