import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.dqn_stack = nn.Sequential(
            nn.Conv2d
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


class DQN():

    def __init__(self, epsilon=0.75):
        super().__init__()
        self.network = NeuralNetwork()
        self.target_network = self.network
        self.epsilon = epsilon
        self.prev_action = None
        self.prev_state = None
        self.prev_q = None

    def new_network(self):
        # instantiate layers
        L0 = layers.InputLayer(np.zeros((210, 160)))
        L1 = layers.ConvolutionalLayer(kernel_shape=(4, 4), eta=0.0001)
        L2 = layers.MaxPoolLayer(window_shape=(4, 4), stride=4)
        L3 = layers.FlatteningLayer()
        L4 = layers.FullyConnectedLayer(sizeIn=1989, sizeOut=50, eta=0.0001)
        L5 = layers.ReLuLayer()
        L6 = layers.FullyConnectedLayer(sizeIn=50, sizeOut=6, eta=0.0001)
        L7 = layers.SquaredTemporalDifferenceError(gamma=0.2)
        # assemble network
        network = [L0, L1, L2, L3, L4, L5, L6, L7]
        return network

    def action(self, state):
       # forward propogation through network
        t = state
        for k in range(len(self.network)-1):
            t = self.network[k].forward(t) 

        #store state, action, and associated q value
        self.prev_state = state
        rand = random.uniform(0, 1)
        if rand > self.epsilon:
            self.prev_action = random.randint(0, 5)
        else:
            self.prev_action = t.argmax()
        self.prev_q = t
        return self.prev_action

    def train(self, state, reward):

        # forward propogation through target network
        h = state
        for i in range(len(self.target_network)-1):
            h = self.target_network[i].forward(h)

        # backward propogation
        grad = self.network[-1].gradient(self.prev_action, self.prev_q, h, reward)
        for z in range(len(self.network)-2, 0, -1):
            newgrad = self.network[z].backward(grad)
            if(isinstance(self.network[z], layers.FullyConnectedLayer)):
                self.network[z].updateWeights(np.array(grad))
            if(isinstance(self.network[z], layers.ConvolutionalLayer)):
                self.network[z].updateWeights(np.array(grad))
            grad = newgrad

    def updateTarget(self):
        self.target_network = self.network

    def getEpsilon(self):
        return self.epsilon
    
    def setEpsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def save(self, episode):
        pass

    def load(self, episode):
        pass

    def print(self):
        pass
