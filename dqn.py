import random
from collections import namedtuple, deque
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


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state')
)


class ReplayMemory(object):

    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):

    def __init__(self, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.fc1 = nn.Linear(11934, 100)
        self.fc2 = nn.Linear(100, output_size)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 4)
        x = x.view(-1, torch.numel(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class DQN():

    def __init__(self, epsilon=0.75, lr=0.0001, gamma = 0.25, update_freq=20, num_actions=1, batch_size=10):
        super().__init__()
        self.policy_network = NeuralNetwork(output_size=num_actions).to(device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.update_freq = update_freq

        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        
        self.replay = ReplayMemory(capacity=500, batch_size=batch_size)
        self.num_actions = num_actions

    def select_action(self, state):
        # convert data to torch tensors
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # forward propogation through policy net
        q_values = self.policy_network.forward(s)

        # epilson-greedy action selection
        rand = random.uniform(0, 1)
        action = None
        if rand < self.epsilon:
            with torch.no_grad():
                action = q_values.max(1)[1][0]
        else:
            action = random.randint(0, (self.num_actions-1))
            action = torch.tensor(action, device=device)

        # save the selected action q value
        self.predicted_q = q_values[0][action]

        return action

    def train(self, state, reward, epoch):
        # convert data to torch tensors
        r = torch.tensor(reward, dtype=torch.float32, device=device)
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # forward propogation of current state through target network
        q_values = self.target_network.forward(s)

        # update q value with bellman equation
        q_value = r + (self.gamma * q_values.max(1)[0])

        # compute loss
        loss = self.criterion(self.predicted_q, q_value[0])
                         
        # backward propogation
        loss.backward()
        self.optimizer.step()   
        self.optimizer.zero_grad()

        # update target network
        if (epoch % self.update_freq) == 0:
            self.__updateTarget()

    def __save_transition(self, state, action, reward, next_state):
        self.replay.push(state, action, reward, next_state)

    def getLearningRate(self):
        return self.lr
    
    def setLearningRate(self, lr):
        self.lr = lr

    def getEpsilon(self):
        return self.epsilon
    
    def setEpsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def getGamma(self):
        return self.gamma
    
    def setGamma(self, new_gamma):
        self.gamma = new_gamma

    def __updateTarget(self):
        self.target_network = copy.copy(self.policy_network)

    def getUpdateFreq(self):
        return self.update_freq
    
    def setUpdateFreq(self, new_freq):
        self.update_freq = new_freq

    def save(self, file_name):
        torch.save(self.policy_network, file_name)

    def load(self):
        return torch.load("dqn.pth")
    
    def print(self):
        print(self.policy_network.fc1.weight)