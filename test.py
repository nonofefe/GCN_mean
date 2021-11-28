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

# MAXSIZE = 2000
# ite = 10000

# arr = np.zeros(MAXSIZE)
# arr_next = np.zeros(MAXSIZE)
# arr[0] = 1
# for _ in range(ite):
#   for i in range(MAXSIZE):
#     if i == 0:
#       arr_next[i] = arr[i]
#     elif i == MAXSIZE-1:
#       arr_next[i] = (arr[i] + arr[i-1]) / 2
#     else:
#       arr_next[i] = (arr[i-1] + arr[i] + arr[i+1]) / 3
#   arr = arr_next
# print(arr)

# MAXSIZE = 2000
# ite = 20000

# arr = np.zeros(3)
# arr_next = np.zeros(3)
# arr[0] = 1
# for _d in range(ite):
#   arr_next[0] = arr[0]
#   arr_next[1] = (arr[0] + arr[1] + (MAXSIZE-2) * arr[2]) / MAXSIZE
#   arr_next[2] = (arr[1] + arr[2]) / 2
#   arr = arr_next
#   print(arr)

import matplotlib.pyplot as plt
import numpy as np
N = 300
xmin = 0.01
xmax = 3
x = np.linspace(xmin, xmax, N)
y = 1.40 / pow(x, 0.24)
#plt.title("Learning Curve")
plt.xlabel("N")
plt.ylabel("RT")
plt.plot(x, y, label="Learning Curve")
plt.legend()
plt.savefig("learningCurve.png")
