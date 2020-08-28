import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Critic(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer
            
        Return:
            value output of network 
    """
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 200)
        self.l4 = nn.Linear(200, 1)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 400)
        self.l6 = nn.Linear(400, 300)
        self.l7 = nn.Linear(300, 200)
        self.l8 = nn.Linear(200, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = F.relu(self.l3(x1))
        x1 = self.l4(x1)

        x2 = F.relu(self.l5(xu))
        x2 = F.relu(self.l6(x2))
        x2 = F.relu(self.l7(x2))
        x2 = self.l8(x2)
        return x1, x2


    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = F.relu(self.l3(x1))
        x1 = self.l4(x1)
        return x1