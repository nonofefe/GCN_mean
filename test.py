import numpy as np
import torch

#data = torch.from_numpy(np.loadtxt('embedding/cora.txt', delimiter=' ', dtype='int64'))

a = 0
b = 0
c = 1
for i in range(30):
  aa = (a+b)/2
  bb = (a+b+c)/3
  a = aa
  b = bb
  print(a, end=" ")
  print(b, end=" ")
  print(c)