import numpy as np
data = np.loadtxt("embedding/deepwalk.txt")

import torch

k = 6 # 特徴量の次元

top_num = 3

size = 0
for line in data:
    size += 1
x = np.zeros((size,k))
score = torch.zeros((size,size))

cnt = 0
for line in data:
    x[cnt] = np.array(line)
    cnt += 1
print(x)

for i in range(size):
    for j in range(size):
        score[i,j] = -np.linalg.norm(x[i] - x[j],ord=2)
print(score)

for i in range(size):
    score[i,i] = -100000

values, indices = torch.topk(score, top_num)
print(values)
print(indices)

f = open('embedding/topk.txt', 'w')
for i in range(size):
  vector = indices[i]
  topk_i = ""
  for j in range(top_num):
    topk_i += str(vector[j].item()) + " "
  f.write("{}\n".format(topk_i))