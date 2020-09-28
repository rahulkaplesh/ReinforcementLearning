import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, env, h_size=32):
        super(Agent, self).__init__()
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.n
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)
        self.env = env

    def set_weights(self, weights):
        fc1_W = torch.from_numpy(weights['fc1_W'].reshape(self.s_size, self.h_size))
        fc1_b = torch.from_numpy(weights['fc1_b'])
        fc2_W = torch.from_numpy(weights['fc2_W'].reshape(self.h_size, self.a_size))
        fc2_b = torch.from_numpy(weights['fc2_b'])

        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def getfc1_W_size(self):
        return self.s_size * self.h_size

    def getfc1_b_size(self):
        return self.h_size

    def getfc2_W_size(self):
        return self.h_size * self.a_size

    def getfc2_b_size(self):
        return self.a_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data

    def evaluate(self, weights, gamma=1.0, max_t = 5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(np.argmax(action.numpy()))
            episode_return += reward + math.pow(gamma, t)
            if done:
                break
        return episode_return