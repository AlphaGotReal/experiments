import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('..')
from animate import *

x = torch.tensor(np.linspace(-3.14, 3.14, 1000), dtype=torch.float32)
x = x.reshape(x.shape + (1,))

y = torch.nn.ReLU()(torch.sin(x)) + torch.nn.ReLU()(-torch.cos(x))

class activate(nn.Module):
    def __init__(self):
        super(activate, self).__init__()
    
    def forward(self, x):
        return torch.sin(x) * x

activation = nn.Tanh()#nn.LeakyReLU(-1.2)

class network(nn.Module):

    def __init__(self):

        super(network, self).__init__()

        self.seq = nn.Sequential(

            nn.Linear(1, 4, bias=True),
            activation,

            nn.Linear(4, 16, bias=True),
            activation,

            nn.Linear(16, 64, bias=True),
            activation,

            nn.Linear(64, 16, bias=True),
            activation,

            nn.Linear(16, 4, bias=True),
            activation,

            nn.Linear(4, 1, bias=True)
        )

    def forward(self, x):

        return self.seq(x)

model = network()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

screen = Screen(1900, 1000, "test", 60, (0, 10))

def render_callback():
    global pred_y, x, y, epoch
    plot(x, y, (1, 0, 0))
    plot(x, pred_y, (1, 0, 1))
    rendertext(f"epoch = {epoch}", (600, 800))

for epoch in range(4000):

    print(f"epoch = {epoch+1}", end='\r')

    pred_y = model(x)
    loss = loss_function(pred_y, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    screen.mainloop(render_callback)

