import numpy as np
import torch

data = torch.from_numpy(np.loadtxt('embedding/cora.txt', delimiter=' ', dtype='int64'))

print(data)