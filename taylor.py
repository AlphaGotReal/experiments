import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('..')
from animate import *

X = torch.tensor(np.linspace(-3.14, 3.14, 1000), dtype=torch.float32)
X = X.reshape(X.shape + (1,))

y = torch.nn.ReLU()(torch.sin(X)) + torch.nn.ReLU()(-torch.cos(X))

x = X/3.14#normalize

class activate(nn.Module):
    def __init__(self):
        super(activate, self).__init__()
    
    def forward(self, x):
        return torch.sin(x) * x

activation = nn.LeakyReLU(-1.2)

class taylor_encoder(nn.Module):

    def __init__(self, n):

        super(taylor_encoder, self).__init__()
        self.exponents = torch.arange(n)

    def forward(self, x):
        return torch.pow(x, self.exponents)

class taylor_decoder(nn.Module):

    def __init__(self, x):

        super(taylor_decoder, self).__init__()
        self.inputs = x

    def forward(self, a):

        out = torch.sum(a*self.inputs, dim=1)
        return out.reshape(out.shape + (1, ))


class taylor(nn.Module):

    def __init__(self, n, sequential):

        super(taylor, self).__init__()

        self.user_seq = sequential
        self.n = n

    def raw_forward(self, x):

        features = taylor_encoder(self.n)(x)
        coeff = self.user_seq(features)

        return coeff

    def forward(self, x):

        features = taylor_encoder(self.n)(x)
        coeff = self.user_seq(features)
        
        return taylor_decoder(features)(coeff)

n = 10 #features

seq = nn.Sequential(
    nn.Linear(n, 16, bias=True),
    nn.LeakyReLU(-1.2),

    nn.Linear(16, 64, bias=True),
    nn.LeakyReLU(-1.2),

    nn.Linear(64, 16, bias=True),
    nn.LeakyReLU(-1.2),

    nn.Linear(16, n, bias=True)
)

model = taylor(n, seq)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)

screen = Screen(1900, 1000, "test", 60, (0, 10))

def render_callback():
    global pred_y, X, y, epoch
    plot(X, y, (1, 0, 0))
    plot(X, pred_y, (1, 0, 1))
    rendertext(f"epoch = {epoch}", (600, 800))

for epoch in range(6000):

    print(f"epoch = {epoch+1}", end='\r')

    pred_y = model(x)
    loss = loss_function(pred_y, y)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    screen.mainloop(render_callback)




