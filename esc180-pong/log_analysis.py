import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



def fit_nn(X, Y):
    # data
    class MyDataset(Dataset):

        def __init__(self):
            super().__init__()

        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            return (X[i], Y[i])

    dataloader = DataLoader(
        MyDataset(), batch_size=256, shuffle=True)
    dataloader = [d for d in dataloader]
    test_data = dataloader[:10]
    train_data = dataloader[10:]

    # model

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(X.shape[1], 12, bias=True),
                nn.ReLU(),
                nn.Linear(12, 8, bias=True),
                nn.ReLU(),
                nn.Linear(8, Y.shape[1], bias=True),
                nn.Tanh()
            )
            with torch.no_grad():
                e = 1e-2
                self.main[0].weight = nn.Parameter(
                    e*torch.randn_like(self.main[0].weight))
                self.main[0].weight[0, X.shape[1]-1] = 1.0
                self.main[0].bias = nn.Parameter(
                    e*torch.randn_like(self.main[0].bias))
                self.main[2].weight = nn.Parameter(
                    e*torch.randn_like(self.main[2].weight))
                self.main[2].weight[0, 0] = 1.0
                self.main[2].bias = nn.Parameter(
                    e*torch.randn_like(self.main[2].bias))
                self.main[4].weight = nn.Parameter(
                    e*torch.randn_like(self.main[4].weight))
                self.main[4].weight[0, 0] = 1.0
                self.main[4].bias = nn.Parameter(
                    e*torch.randn_like(self.main[4].bias))

        def forward(self, x):
            return self.main(x)

    net = Net()


    # optimize
    def loss_bin(x, y, detach=False):
        p = 0.5+0.4999999*net(x)
        v = -torch.mean(y*torch.log(p) + (1.0-y)*torch.log(1.0-p))
        return v.detach().numpy() if detach else v
    def loss_mse(x, y, detach=False):
        p = 0.5+0.4999999*net(x)
        v = torch.mean((p-y)**2)**0.5
        return v.detach().numpy() if detach else v
    X, Y = torch.tensor(X), torch.tensor(Y)
    print("train (before):", np.mean([loss_mse(*d, True) for d in train_data]))
    print("test (before):", np.mean([loss_mse(*d, True) for d in test_data]))

    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(1):
        for x, y in train_data:
            net.zero_grad()
            err = loss_bin(x, y)
            err.backward()
            optimizer.step()
        print(f"test ({epoch}):", np.mean([loss_mse(*d, True) for d in test_data]))
    print("train (after):", np.mean([loss_mse(*d, True) for d in train_data]))
    print("test (after):", np.mean([loss_mse(*d, True) for d in test_data]))

    # export
    ws = []
    for param in net.parameters():
        data = param.data
        ws.append(data.numpy())
    for i in range(0, len(ws), 2):
        w = np.concatenate([ws[i], ws[i+1].reshape(len(ws[i]), 1)], axis=1)
        k = 1e6
        w = np.round(w.astype(np.float64)*k)/k
        print(w.tolist())





if __name__ == "__main__":
    data_files = [
        'logs/1670291656.raw',
        'logs/1670291665.raw',
        'logs/1670292031.raw',
        'logs/1670292053.raw',
        'logs/1670292104.raw',
        'logs/1670292148.raw',
        'logs/1670292297.raw',
        'logs/1670292404.raw'
    ]
    if True:
        import os
        files = os.listdir('logs/')
        data_files = ['logs/'+f for f in files]
    data = [np.fromfile(file, dtype=np.float32) for file in data_files]
    data = np.concatenate(data)
    data = data.reshape((len(data)//34, 34))

    xs = data[:, :-1]
    ys = 0.5 + 0.5 * data[:, -2] * data[:, -1]
    ys = ys.reshape(len(ys), 1)
    print(xs.shape, ys.shape)

    fit_nn(xs, ys)

