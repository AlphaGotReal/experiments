import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from animate import *

X = torch.tensor(np.linspace(-3.14, 3.14, 1000), dtype=torch.float32)
X = X.reshape(X.shape + (1,))

y = torch.nn.ReLU()(torch.sin(X)) + torch.nn.ReLU()(-torch.cos(X))

x = X

class activate(nn.Module):
    def __init__(self):
        super(activate, self).__init__()
    
    def forward(self, x):
        return torch.sin(torch.nn.ReLU()(x))

activation = nn.LeakyReLU(-1.2)

class fourier_encoder(nn.Module):

    def __init__(self, n):

        super(fourier_encoder, self).__init__()
        self.n = n

    def forward(self, x):
        out = torch.zeros((x.shape[0], self.n*2+1))
        out[:, 0] = torch.ones(x.shape[0])
        for r in range(1, self.n+1):
            out[:, 2*r-1] = torch.sin(r*x[:, 0])
            out[:, 2*r] = torch.cos(r*x[:, 0])
        return out

class fourier_decoder(nn.Module):

    def __init__(self, features):

        super(fourier_decoder, self).__init__()

        self.features = features

    def forward(self, coeff):

        out = torch.sum(coeff*self.features, dim=1)
        return out.reshape(out.shape + (1, ))

class fourier(nn.Module):

    def __init__(self, n, sequential):

        super(fourier, self).__init__()
        self.n = n
        self.user_seq = sequential

    def forward(self, x):
        features = fourier_encoder(self.n)(x)
        coeff = self.user_seq(features)

        return fourier_decoder(features)(coeff)

n = 3 #features

seq = nn.Sequential(
    nn.Linear(n*2+1, 16, bias=True),
    activation,

    nn.Linear(16, 64, bias=True),
    activation,

    nn.Linear(64, 16, bias=True),
    activation,

    nn.Linear(16, n*2+1, bias=True)
)

model = fourier(n, seq)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

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



