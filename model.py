import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
min_ = train_df[['distance', 'speed', 'route_distance_km']].min()
max_ = train_df[['distance', 'speed', 'route_distance_km']].max()
train_df[['distance', 'speed', 'route_distance_km']] = (train_df[['distance', 'speed', 'route_distance_km']] - min_) / (
        max_ - min_)

train_df['speed'].fillna(train_df['speed'].mean(), inplace=True)

node_emb_size = train_df['node_finish'].max() + 1
node_emb_dim = 256+128

hour_emb_size = train_df['day_time'].max() + 1
hour_emb_dim = 50

dropuot_c = 0.5


class MyData(Dataset):
    def __init__(self, X):
        super().__init__()

        self.X = list(X)
        self.n = len(self.X)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        X = self.X[item][1].to_numpy(dtype='float32')
        x = X[:, 1:-1]
        y = X[0, -1:]

        x = np.pad(x, ((0, max(272 - x.shape[0], 0)), (0, 0)), constant_values=0)[:272]
        return x, y


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_emb = nn.Embedding(node_emb_size, node_emb_dim)
        self.hour_emb = nn.Sequential(
            nn.Embedding(hour_emb_size, hour_emb_dim),
            nn.Linear(hour_emb_dim, node_emb_dim),
            nn.Tanh(),
            nn.Dropout(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3, 256),
            nn.SiLU(),

            nn.Linear(256, 128),
            nn.SiLU(),
            nn.BatchNorm1d(272),

            nn.Linear(128, node_emb_dim),
            nn.SiLU(),
            nn.BatchNorm1d(272),

        )

        self.fc2 = nn.Sequential(
            nn.Linear(node_emb_dim * 2, 256),
            nn.SiLU(),
            nn.BatchNorm1d(272),

            nn.Linear(256, 256),
            nn.SiLU(),
            nn.BatchNorm1d(272),

            nn.Linear(256, 256),
            # nn.SiLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(node_emb_dim * 2, 256),
            nn.SiLU(),
            nn.BatchNorm1d(272),

            nn.Linear(256, 256),
            nn.SiLU(),
            nn.BatchNorm1d(272),

            nn.Linear(256, 256),
            # nn.SiLU(),
        )

    def forward(self, x):
        x1 = self.node_emb(x[:, :, :2].to(torch.int32))
        x2 = self.hour_emb(x[:, :, 2].to(torch.int32))
        node_x1 = (x1[:, :, 0, :] + x2)
        node_x2 = (x1[:, :, 1, :] + x2)

        x3 = self.fc1(x[:, :, 3:])

        f = self.fc2(torch.concat([node_x1 + x3, x2], axis=2))
        g = self.fc3(torch.concat([node_x2 + x3, x2], axis=2))
        return ((f + g)).sum(axis=[1, 2])[:, None]


groups = train_df.groupby('Id')
train_, test_ = train_test_split(list(groups), test_size=0.1)

train_dataloader = DataLoader(
    MyData(train_),
    batch_size=100,
    shuffle=True,
)

test_dataloader = DataLoader(
    MyData(test_),
    batch_size=500,
    shuffle=False,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

errors = []


def test(model, loss, dl):
    model = model.eval()
    X, y = next(iter(dl))
    X = X.to(device)
    y = y.to(device)
    pred = model(X)
    l = loss(y, pred)
    print(f'{torch.sqrt(l).item()}'.center(60, '*'))
    return torch.sqrt(l).item()


def train(epochs, model, optim, loss, dl):
    for epoch in range(epochs):
        model = model.train()
        print(f'Epoch {epoch + 1}'.center(50, '='))
        for batch, (X, y) in enumerate(dl):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            l = loss(y, pred)

            optim.zero_grad()
            l.backward()
            optim.step()

            if batch % 10 == 0:
                print(
                    f'Batch: {batch * 64}\n'
                    f'Loss: {l.item()}'
                )

        errors.append(test(model, nn.MSELoss(), test_dataloader))


# net = Model().to(device)
# opt = Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999))

# train(35, net, opt, nn.HuberLoss(), train_dataloader)
# train(35, net, opt, nn.HuberLoss(delta=0.7), train_dataloader)

# torch.save(net.to('cpu'), 'model2')

net = torch.load('model2').to(device)

df = pd.read_csv('test.csv')

df['delta_time'] = 0
groups = df.groupby('Id')
loader = DataLoader(
    MyData(groups),
    batch_size=len(list(groups))//4,
    shuffle=False
)

net = net.eval()
l = iter(loader)
X = next(l)[0].to(device)
y1 = net(X).to('cpu')
del X
X = next(l)[0].to(device)
y2 = net(X).to('cpu')
del X
X = next(l)[0].to(device)
y3 = net(X).to('cpu')
del X
X = next(l)[0].to(device)
y4 = net(X).to('cpu')
y = torch.stack([y1, y2, y3, y4])
print(y.detach())


plt.plot(errors)
plt.show()



