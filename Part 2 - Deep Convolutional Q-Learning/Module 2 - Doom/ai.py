# AI for Doom

# Importing the libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay
import image_preprocessing


# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):  # Convolutional Neural Network

    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)

    def propagate_signals_convolutional(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        return x

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = self.propagate_signals_convolutional(x)

        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = self.propagate_signals_convolutional(x)
        x = x.view(x.size(0), -1)   # Flatten convolutional layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Making the body

class SoftmaxBody(nn.Module):

    def __init__(self, Temperature):
        super(SoftmaxBody, self).__init__()
        self.Temperature = Temperature  # The higher the temperature, the less of the other actions we will do

    def forward(self, outputs):
        probs = F.softmax(outputs * self.Temperature)
        actions = probs.multinomial()

        return actions


# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.brain(input)
        actions = self.body(output)

        return actions.data.numpy()

# Part 2 - Training the AI with Deep Convolutional Q-Learning
