import numpy as np
import matplotlib.pyplot as plt
 
left = np.array([1, 2, 3, 4, 5,6,7,8,9,10])
height = np.array([
0.5794943820224718,
0.6883550488599349,
0.7818584070796459,
0.7683060109289617,
0.7418181818181818,
0.865,
0.93,
0.888888888888889,
0.78125,
0.9833333333333332])
label = ['0~2','2~4','4~6','6~8','8~10','10~12','12~14','14~16','16~18','18~20']
plt.bar(left, height, tick_label=label, align="center")
 
#グラフを保存するときは描画させない
#plt.show()
plt.savefig("results/degree.jpg")