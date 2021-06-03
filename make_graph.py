import numpy as np
import matplotlib.pyplot as plt
 
left = np.array([1, 2, 3, 4, 5])
height = np.array([0.7387086765954792,0.7653107881874744,0.7464517501051066,0.7495408759326402,0.6078021075799261])
label = ['0~0.2','0.2~0.4','0.4~0.6','0.6~0.8','0.8~1.0']
plt.bar(left, height, tick_label=label, align="center")
 
#グラフを保存するときは描画させない
#plt.show()
plt.savefig("results/0.5.jpg")