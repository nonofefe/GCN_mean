import numpy as np
import torch
import time


#data = torch.from_numpy(np.loadtxt('embedding/cora.txt', delimiter=' ', dtype='int64'))


# MAX1 = 5
# MAX2 = 10
# x = np.ones((MAX1,MAX2))
# y = np.ones((MAX1,1))
# y *= 2
# print(x)
# print(y)
# start = time.time()
# # for i in range(MAX1):
# #   x[i] /= y[i]
# x = np.dot(x,y)
# elapsed_time = time.time() - start
# print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# print(x)


# k = 0
# a = 0
# b = 0
# c = 1
# for i in range(100):
#   kk = (k+a)/2
#   aa = (a+b+k)/3
#   bb = (a+b+c)/3
#   a = aa
#   b = bb
#   k = kk
#   print(k, end=" ")
#   print(a, end=" ")
#   print(b, end=" ")
#   print(c)

a = torch.ones((2,3))
b = torch.ones((2,3))
b += 1
mask = torch.tensor([[True, False, False],[False,False,True]])
mask_int = mask.to(torch.int)
c = mask_int * a + (1 - mask_int) * b
print(c)


def apply_neighbor_mean_recursive(features, mask, miss_struct, adj, epoch=30):
  n_adj = adj.size()[0]
  n_feat = features.size()[1]
  n_edge = adj._indices().size()[1]

  apply_zero(features, mask)

  ind_arr = adj._indices()

  degree = miss_struct.degree
  mask_int = mask.to(torch.int)
  for _ in range(epoch):
    X = torch.zeros_like(features)
    for i in range(n_edge):
      node1 = ind_arr[0,i].item()
      node2 = ind_arr[1,i].item()
      X[node2] += features[node1]
      X[node1] += features[node2]
    for i in range(X.shape[0]):
      X[i] /= 2 # エッジが倍存在するので
      X[i] /= degree[i]
    Y = mask_int * X + (1 - mask_int) * features
    features = Y
    print(features)
      # if mask[i,0] == True:
      #   features[i] = X[i]