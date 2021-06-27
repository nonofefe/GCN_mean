import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import torch
from gensim.models import Word2Vec as word2vec

from utils import NodeClsData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_emb_deepwalk(G, vector_size):
  # ランダムウォークを生成
  walks = make_random_walks(G, 20, 20)
  # gensim の Word2Vecを使った学習部分
  model = word2vec(walks, min_count=0, vector_size=vector_size, window=5, workers=1)
  size = G.number_of_nodes()
  x = np.zeros((size,vector_size))
  cnt = 0
  for node in G.nodes():
    x[cnt] = model.wv[str(node)]
    cnt += 1
  return x

def get_topk(G, vector_size):
  x = get_emb_deepwalk(G, vector_size)
  size = G.number_of_nodes()
  score = torch.zeros((size,size))
  for i in range(size):
    for j in range(size):
      score[i,j] = -np.linalg.norm(x[i] - x[j],ord=2)
  top_num = size
  values, indices = torch.topk(score, top_num)
  return indices

def write_topk(G, vector_size, dataset):
  indices = get_topk(G, vector_size)
  size = G.number_of_nodes()
  indices = indices.numpy()
  np.savetxt('embedding/' + dataset + '.txt', indices, fmt='%d')

if __name__ == "__main__":
  dataset = "amacomp"
  data = NodeClsData(dataset)
  G = data.G
  vector_size = 16
  # write_topk(G, vector_size, dataset)

  plt.figure(figsize=(15,15))
  pos = nx.spring_layout(G)
  nx.draw_networkx(G,pos)

  plt.axis("off")
  plt.savefig("default.png")
  plt.show()