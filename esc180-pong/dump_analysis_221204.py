import numpy as np
import matplotlib.pyplot as plt

def load_data(files):
    data = [np.fromfile(file, dtype=np.float32) for file in files]
    data = np.concatenate(data)
    return data.reshape((len(data)//13, 13))


def fit_linear(params, preds, reals):
    dtype = np.float64
    X = np.concatenate([
        params, preds, np.ones((len(params), 1))], axis=1).astype(dtype)
    Y = np.array(reals, dtype=dtype)
    C = np.concatenate([
        np.zeros((reals.shape[1], params.shape[1])),
        np.eye(reals.shape[1]),
        np.zeros((reals.shape[1], 1))], axis=1).astype(dtype)
    print(np.mean((C@X.T-Y.T)**2, axis=1)**0.5)
    for i in range(len(C)):
        C[i] = np.linalg.solve(X.T@X,X.T@Y.T[i])
    print(np.mean((C@X.T-Y.T)**2, axis=1)**0.5)
    print(C.tolist())
    plt.matshow(np.log10(abs(C)))
    plt.colorbar()
    plt.show()


def fit_nn(params, preds, reals):
    # data
    dtype = np.float32
    X = np.concatenate([params, preds], axis=1).astype(dtype)
    Y = np.array(reals, dtype=dtype)
    W = 1.0/np.linalg.norm(np.concatenate([X,Y],axis=1), axis=1)
    W = np.array([0.1*W, W, W], dtype=np.float32).T

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class MyDataset(Dataset):

        def __init__(self):
            super().__init__()

        def __len__(self):
            return len(X)

        def __getitem__(self, i):
            return (X[i], Y[i], W[i])

    dataloader = DataLoader(
        MyDataset(), batch_size=64, shuffle=True)
    dataloader = [d for d in dataloader]
    test_data = dataloader[:10]
    train_data = dataloader[10:]

    # model

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(X.shape[1], 8, bias=True),
                nn.ReLU(),
                nn.Linear(8, 8, bias=True),
                nn.ReLU(),
                nn.Linear(8, Y.shape[1], bias=True)
            )

        def forward(self, x):
            return self.main(x)

    net = Net()
    print(net)

    # optimize

    def loss(x, y, w, detach=False):
        v = torch.mean((w*(y-net(x)))**2)
        return v.detach().numpy() if detach else v
    X, Y, W = torch.tensor(X), torch.tensor(Y), torch.tensor(W)
    print("train (before):", np.mean([loss(*d, True) for d in train_data]))
    print("test (before):", np.mean([loss(*d, True) for d in test_data]))

    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(1):
        for x, y, w in train_data:
            net.zero_grad()
            err = loss(x, y, w)
            err.backward()
            optimizer.step()
        print(f"test ({epoch}):", np.mean([loss(*d, True) for d in test_data]))
    print("train (after):", np.mean([loss(*d, True) for d in train_data]))
    print("test (after):", np.mean([loss(*d, True) for d in test_data]))

    # export
    ws = []
    for param in net.parameters():
        data = param.data
        ws.append(data.numpy())
    for i in range(0, len(ws), 2):
        w = np.concatenate([ws[i], ws[i+1].reshape(len(ws[i]), 1)], axis=1)
        print(w.tolist())





if __name__ == "__main__":
    data = load_data([
        'dump/dump_2212041857.dat',
        'dump/dump_2212041903.dat',
        'dump/dump_2212041906.dat',
        'dump/dump_2212041907.dat',
        'dump/dump_2212041912.dat',
        'dump/dump_2212041914.dat',
    ])

    # split into parts by velocity
    parts = []
    prev_v = np.empty(2, dtype=np.float32)*np.nan
    for row in data:
        v = row[8:10]
        d = np.linalg.norm(v-prev_v)
        if not d < 1e-4:
            parts.append([])
        parts[-1].append(row)
        prev_v = v

    # split into paths by position
    paths = []
    prev_x = np.empty(2, dtype=np.float32)*np.nan
    for part in parts:
        x0 = part[0][6:8]
        v0 = part[0][8:10]
        d1 = np.linalg.norm(x0-prev_x)
        d2 = np.linalg.norm((x0-prev_x)-v0)
        if not (d1 < 40 and d2 < 1e-4):
            paths.append([])
        paths[-1].append(part)
        prev_x = part[-1][6:8]
    paths = [p for p in paths if len(p) > 2]

    def plot_path(path):
        for p in path:
            p = np.array(p, dtype=np.float32)
            plt.scatter(p[:,6], p[:,7])
        plt.show()
        plt.plot([len(p) for p in path], 'o')
        plt.yscale('log')
        plt.show()

    for i in range(5, 5):
        plot_path(paths[i])

    # real vs. predicted
    params = []
    preds = []
    reals = []
    for path in paths:
        path = [np.array(p) for p in path if len(p) > 1]
        for path1, path2 in zip(path[:-1], path[1:]):
            p1 = path1[0][6:8]
            v1 = (path1[-1][6:8]-p1)/(len(path1)-1)
            v1_ = np.mean(path1[:, 8:10], axis=0)
            assert np.linalg.norm(v1_-v1) < 1e-4
            p2 = path2[0][6:8]
            v2 = (path2[-1][6:8]-p2)/(len(path2)-1)
            # p1 + v1 t1 = p2 + v2 t2
            # [v1 -v2] [t1 t2]' = p2-p1
            t1, t2 = np.linalg.solve(np.transpose([v1, -v2]), p2-p1)
            for i in range(len(path1)):
                if not np.isfinite(path1[i][10:13]).all():
                    continue
                params.append(path1[i][0:10])
                preds.append(path1[i][10:13])
                reals.append([t1-i, v2[0], v2[1]])
    params = np.array(params, dtype=np.float32)
    preds = np.array(preds, dtype=np.float32)
    reals = np.array(reals, dtype=np.float32)
    #plt.scatter(preds[:, 0], reals[:, 0])
    #plt.show()

    #fit_linear(params, preds, reals)
    fit_nn(params, preds, reals)

    
