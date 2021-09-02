# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
 
# left = np.array([1, 2, 4, 8, 16,32,64,128,512,1024])
# height = np.array([
# 0.5794943820224718,
# 0.6883550488599349,
# 0.7818584070796459,
# 0.7683060109289617,
# 0.7418181818181818,
# 0.865,
# 0.93,
# 0.888888888888889,
# 0.78125,
# 0.9833333333333332])
# label = ['0~1','2~4','4~6','6~8','8~10','10~12','12~14','14~16','16~18','18~20']
# plt.xscale("log",base=2)
# plt.bar(left, height, tick_label=label, align="center")
# #plt.bar(left, height, align="center")
 
# #グラフを保存するときは描画させない
# #plt.show()
# plt.savefig("results/degree.jpg")

import pandas as pd
import matplotlib.pyplot as plt
#cora
#y = [70.513,72.118,74.71,75.405,76.352,76.378,76.964,77.153,77.348,77.968,77.789]
#citeseer
#y = [54.552,56.392,56.033,57.608,58.093,59.694,60.105,60.379,60.181,59.671,61.559]
#amaphoto
#y = [90.62,91.10,91.11,91.24,90.92,90.89,90.83,90.81,90.85,90.85,90.93]
#amacomp
#y = []

x = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

fig = plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.plot(x, y)
plt.grid()

plt.ylim([85,95])
plt.semilogx(base=2)
plt.xlabel("the number of recursion", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.title("Amacomp",fontsize=25)
#plt.show()
plt.savefig("results/amacomp.jpg")