import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count
import copy

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
        self.conv1 = nn.Conv2d(210, 160, 4,  stride=4)
        self.conv2 = nn.Conv2d(100, 50, 2, stride=2)
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

    def __init__(self, epsilon=0.75, lr=0.0001):
        super().__init__()
        self.policy_network = NeuralNetwork().to(device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.lr = lr
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.prev_action = None
        self.prev_state = None
        self.prev_q = None

    def getNetwork(self):
        return self.policy_network

    def getEpsilon(self):
        return self.epsilon
    
    def setEpsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def generate_action(self, state):
        # convert data to torch tensors
        s = torch.tensor(state, dtype=torch.float32, device=device)
        self.prev_state = s

        # forward propogation through policy net
        q_values = self.policy_network.forward(s)
        self.prev_q = q_values

        # epilson-greedy action selection
        rand = random.uniform(0, 1)
        if rand > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.prev_action = q_values.max(1)[1].view(1, 1)
        else:
            self.prev_action = torch.tensor(random.randint(0, 5), device=device, dtype=torch.int)
        return self.prev_action

    def train(self, state, reward):
        # convert data to torch tensors
        r = torch.tensor(reward, dtype=torch.float32, device=device)
        s = torch.tensor(state, dtype=torch.float32, device=device)

        # forward propogation through target network
        q_values = self.target_network.forward(s)

        # compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(self.prev_q, 

        # backward propogation
        self.optimizer.
    
    def updateTarget(self):
        self.target_network = copy.deepcopy(self.network)

    def save(self, episode):
        pass

    def load(self, episode):
        pass

    def print(self):
        pass


