import numpy as np

# numpy.linalg.normを使う
x = np.array([1,2,3,4,5])
x_l2_norm = np.linalg.norm(x,ord=2)
x_l2_normalized = x / x_l2_norm

print(x_l2_norm)