import pandas as pd
import matplotlib.pyplot as plt


dataset = "cora"
dataset = "citeseer"
dataset = "amaphoto"
dataset = "amacomp"

# uniform randomly missing
# if dataset == "cora":
#   y1 = [70.709,72.336,74.186,75.348,76.68,76.763,77.355,77.4,77.548,77.812,77.6355]
#   y2 = [78.542,79.093,79.116,79.603,79.814,79.571,79.6,79.395,79.299,79.92,79.538]
#   y3 =[81.437,81.506,81.075,81.251,81.103,81.265,81.227,81.683,80.966,81.426,81.67]
# elif dataset == "citeseer":
#   y1 = [54.8165,56.3105,56.4705,57.743,58.834,59.4665,60.376,60.4445,60.0785,60.0085,60.4905] # 0.8
#   y2 = [65.338,64.927,65.766,65.873,65.072,66.104,65.674,65.391,65.571,65.121,65.489] # 0.5
#   y3 = [69.068,68.762,68.804,68.9,69.042,68.248,68.376,68.546,68.471,69.191,68.172] #0.2
# elif dataset == "amaphoto":
#   y1 = [90.62,91.10,91.11,91.24,90.92,90.89,90.83,90.81,90.85,90.85,90.93]
#   y2 = [92.27,92.16,92.12,92.04,92.00,91.86,91.89,91.96,92.00,92.08,92.01]
#   y3 = [92.63,92.50,92.46,92.42,92.34,92.42,92.50,92.45,92.43,92.54,92.45]
# elif dataset == "amacomp":
#   y1 = [82.79,84.03,84.01,84.10,83.77,83.64,83.40,83.33,83.60,83.56,83.50]
#   y2 = [85.11,85.02,84.94,85.01,84.87,84.87,85.01,84.70,84.76,84.97,85.09]
#   y3 = [85.62,84.93,85.42,84.93,85.42,85.43,85.64,85.51,85.46,85.29,85.41]

if dataset == "cora":
  y1 = [62.969,65.561,70.403,71.245,74.681,75.774,74.566,75.696,75.618,76.253,76.089]
  y2 = [76.737,78.331,77.664,79.139,79.274,78.876,79.482,79.137,78.649,78.775,78.991]
  y3 = [81.474,81.325,81.39,81.502,81.53,81.741,81.498,81.369,81.198,81.373,81.217]
elif dataset == "citeseer":
  y1 = [49.379,52.796,53.35,55.138,56.121,58.639,58.354,57.633,59.168,57.347,58.487] # 0.8
  y2 = [63.062,63.414,63.2295,64.266,64.2535,64.656,64.775,63.7625,64.7375,64.496,64.7045] # 0.5
  y3 = [68.306,68.156,68.475,68.497,68.79,68.144,68.231,68.548,68.479,67.717,68.051] #0.2
elif dataset == "amaphoto":
  y1 = [88.51,90.018,90.26,90.43,90.16,90.19,90.18,90.24,90.33,89.73,90.09]
  y2 = [91.52,91.57,91.51,91.42,91.46,90.74,91.52,91.63,91.53,91.34,91.61]
  y3 = [92.25,92.14,92.23,92.21,92.33,92.24,92.17,92.15,92.24,92.29,92.19]
elif dataset == "amacomp":
  y1 = [80.30,81.47,82.26,82.22,80.96,80.40,79.31,78.58,79.86,79.13,79.97]
  y2 = [84.78,84.95,84.88,84.46,83.07,84.21,83.48,84.17,84.65,83.53,84.40]
  y3 = [85.53,84.71,85.53,85.46,85.49,85.42,85.04,85.60,85.57,85.50,85.46101774]

x = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

fig = plt.figure(figsize=(8, 6))
plt.scatter(x, y1)
plt.scatter(x, y2)
plt.scatter(x, y3)
plt.plot(x, y1, label="missing rate = 0.8")
plt.plot(x, y2, label="missing rate = 0.5")
plt.plot(x, y3, label="missing rate = 0.2")
plt.grid()


plt.semilogx(base=2)
plt.xlabel("the number of recursion", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.title(dataset, fontsize=25)
plt.legend()
if dataset == "cora":
  plt.ylim([65,85])
elif dataset == "citeseer":
  plt.ylim([50,70])
elif dataset == "amaphoto":
  plt.ylim([75,95])
elif dataset == "amacomp":
  plt.ylim([70,90])
#plt.show()
plt.savefig("results/" + dataset + ".jpg")