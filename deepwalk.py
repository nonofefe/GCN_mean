# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import random
from gensim.models import Word2Vec as word2vec

from utils import NodeClsData

# Random Walk を生成するメソッド
# input:
# G Graph(networkx)
# num_of_walk それぞれの頂点から何本のwalkを作るか
# length_of_wake それぞれのwalkの長さ
# output: random walk
def make_random_walks(G, num_of_walk, length_of_walk):
  walks = list()
  for i in range(num_of_walk):
    node_list = list(G.nodes())
    for node in node_list:
      now_node = node
      walk = list()
      walk.append(str(node))
      for j in range(length_of_walk):
        next_node = random.choice(list(G.neighbors(now_node)))
        walk.append(str(next_node))
        now_node = node
      walks.append(walk)
  return walks


dataset = "cora"
vector_size = 6
data = NodeClsData(dataset)



# 今回はKarate Clubというグラフを使用
G = data.G
# ランダムウォークを生成
walks = make_random_walks(G, 20, 20)
# gensim の Word2Vecを使った学習部分
model = word2vec(walks, min_count=0, vector_size=vector_size, window=5, workers=1)

# 可視化部分
# Karate Culubのそれぞれの頂点は2つのグループに分けられるので色分けして表示
x = list()
y = list()
node_list = list()
colors = list()
fig, ax = plt.subplots()



f = open('embedding/deepwalk.txt', 'w')
for node in G.nodes():
  vector = model.wv[str(node)]
  print("%s:"%(str(node)), end="")
  print(vector)
  vec_n = ""
  for i in range(vector_size):
    vec_n += str(vector[i]) + " "
  f.write("{}\n".format(vec_n))
  x.append(vector[0])
  y.append(vector[1])
  ax.annotate(str(node), (vector[0], vector[1]))
  colors.append("r")
  # if G.nodes[node]["club"] == "Officer":
  #   colors.append("r")
  # else:
  #   colors.append("b")
f.close()
#図示するためのもの
# for i in range(len(x)):
#   ax.scatter(x[i], y[i], c=colors[i])
# plt.show() #2次元までの表現の出力