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


# Making the A3C brain

