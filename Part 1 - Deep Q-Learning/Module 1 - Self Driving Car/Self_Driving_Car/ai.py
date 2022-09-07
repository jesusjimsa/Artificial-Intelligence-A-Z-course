# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Creating the architecture of the Neural Network
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))    # Hidden neurons
        q_values = self.fc2(x)

        return q_values


# Implementing Expirience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity    # Maximum number of transitions in our memory
        self.memory = []

    def push(self, event):
        self.memory.append(event)

        # If memory exceeds capacity after adding the new event, we'll delete the oldest event in the list
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))

        return map(lambda x: Variable(torch.cat(x, 0)), samples)
