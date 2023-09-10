import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


'''
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
'''


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(210, 160, stride=4)
        self.conv2 = nn.Conv2d(100, 50, stride=2)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DQN():

    def __init__(self, epsilon=0.75):
        super().__init__()
        self.network = NeuralNetwork().to(device)
        self.target_network = (self.network).to(device)
        self.epsilon = epsilon
        self.prev_action = None
        self.prev_state = None
        self.prev_q = None

    def getEpsilon(self):
        return self.epsilon
    
    def setEpsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def generate_action(self, state):
       # forward propogation through network
        s = torch.tensor(state, dtype=torch.float32, device=device)
        q_values = self.network.forward(s)

        #store state, action, and associated q value
        self.prev_state = s
        rand = random.uniform(0, 1)
        if rand > self.epsilon:
            self.prev_action = random.randint(0, 5)
        else:
            self.prev_action = q_values.argmax()
        self.prev_q = s
        return self.prev_action

    def train(self, state, reward):
        # forward propogation through target network
        s = torch.tensor(state, dtype=torch.float32, device=device)
        q_values = self.target_network.forward(s)

        # backward propogation
        

    def updateTarget(self):
        self.target_network = self.network

    def save(self, episode):
        pass

    def load(self, episode):
        pass

    def print(self):
        pass


