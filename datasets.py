import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def gaussian_mixture_dataset(n=8000,modes=2):
    xs = np.linspace(0,10,modes)
    ys = [i/modes for i in range(modes)]
    means = np.array([(x, y) for x in xs for y in xs])
    print(f'means {means}')
    covariances = np.eye(2)
    X = np.concatenate([np.random.multivariate_normal(mean, covariances, int(n/modes)) for mean in means])
    #return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))





import random
class DataSet(torch.utils.data.Dataset):
    def __init__(self,modes=3, total_len = 10000,max=20,idx_to_remove=3,remove=True):
        """self.device='cuda'
        self.modes = modes
        self.dist1_mean, self.dist1_var = dist1[0], dist1[1]
        self.dist2_mean, self.dist2_var = dist2[0], dist2[1]
        xs = np.linspace(2, 10, self.modes)
        ys = [i / self.modes for i in range(self.modes)]
        self.means = np.array([(x, y) for x in xs for y in xs])
        self.means = np.array([(3,3),(-3,-3)])
        self.covariances = np.eye(2)
        print(self.means)
        self.X = np.concatenate([np.random.multivariate_normal(mean, self.covariances, total_len//len(self.means)) for mean in self.means])
        self.data = torch.from_numpy(self.X.astype(np.float32)).to(self.device)
        self.shape = shape
        self.total_len = total_len"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.shape=(2,)
        self.dist1_var = 1.0
        self.total_len = total_len
        self.data = torch.zeros((total_len, 2))
        self.modes = modes
        xs = np.linspace(2, max, self.modes)
        self.means = np.array([(x, y) for x in xs for y in xs])
        uniform = list(range(len(self.means)))
        #Remove and store 3 random means
        self.removed = []
        if remove:
            for i in range(idx_to_remove):
                idx = random.choice(uniform)
                self.removed.append(idx)
                uniform.remove(idx)
        for i in range(total_len):
            mean_idx = random.choice(uniform)
            mean = self.means[mean_idx]
            self.data[i] = torch.tensor(mean) + torch.randn(self.shape) * self.dist1_var
            self.data[i] = (self.data[i] - 10)/10

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        data = self.data[idx]
        return data.to(self.device)



    def plot(self,save_path='data.png'):
        data = self.data.cpu().detach().numpy()
        plt.figure()
        plt.scatter(data[:,0],data[:,1])
        plt.savefig(save_path)
        plt.show()

def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "gaussian_mixture":
        return gaussian_mixture_dataset(n=n)
    else:
        raise ValueError(f"Unknown dataset: {name}")



