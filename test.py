import datasets
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



plt.figure(figsize=(12, 8))
losses = np.load('exps/base/loss.npy')
plt.plot(losses)
plt.show()
plt.savefig('loss.png')