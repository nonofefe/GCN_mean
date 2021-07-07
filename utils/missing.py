import torch


def generate_mask(features, missing_rate, missing_type):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float
    missing_type : string

    Returns
    -------

    """
    if missing_type == 'uniform':
        return generate_uniform_mask(features, missing_rate)
    if missing_type == 'bias':
        return generate_bias_mask(features, missing_rate)
    if missing_type == 'struct':
        return generate_struct_mask(features, missing_rate)
    raise ValueError("Missing type {0} is not defined".format(missing_type))


def generate_uniform_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask


def generate_bias_mask(features, ratio, high=0.9, low=0.1):
    """
    Parameters
    ----------
    features: torch.Tensor
    ratio: float
    high: float
    low: float

    Returns
    -------
    mask: torch.Tensor
    """
    node_ratio = (ratio - low) / (high - low)
    feat_mask = torch.rand(size=(1, features.size(1)))
    high, low = torch.tensor(high), torch.tensor(low)
    feat_threshold = torch.where(feat_mask < node_ratio, high, low)
    mask = torch.rand_like(features) < feat_threshold
    return mask


def generate_struct_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    node_mask = torch.rand(size=(features.size(0), 1))
    mask = (node_mask <= missing_rate).repeat(1, features.size(1))
    return mask


def apply_mask(features, mask):
    """

    Parameters
    ----------
    features : torch.tensor
    mask : torch.tensor

    """
    features[mask] = float('nan')

# 0埋めするためのモデル
def apply_zero(features, mask):
    """

    Parameters
    ----------
    features : torch.tensor
    mask : torch.tensor

    """
    features[mask] = 0

# 欠損値のない近隣ノードの平均で特徴量を埋める関数。
# そのような近隣ノードがない場合は0埋め
def apply_neighbor_mean(features, mask, miss_struct, adj):
    n_adj = adj.size()[0]
    n_feat = features.size()[1]
    n_edge = adj._indices().size()[1]

    X = torch.zeros_like(features)
    ind_arr = adj._indices()
    dig_miss = torch.zeros(adj.shape[0])
    for i in range(n_edge):
        node1 = ind_arr[0,i].item()
        node2 = ind_arr[1,i].item()
        if mask[node1,0] == False:
            X[node2] += features[node1]
            dig_miss[node2] += 1
        if mask[node2,0] == False:
            X[node1] += features[node2]
            dig_miss[node1] += 1
    
    for i in range(n_adj):
        if dig_miss[i] == 0:
            dig_miss[i] = 1
    dig_miss = dig_miss.repeat(n_feat,1).T
    X = torch.div(X, dig_miss)
    
    for i in range(X.shape[0]):
        if mask[i,0] == True:
            features[i] = X[i]

def apply_neighbor_mean_recursive(features, mask, miss_struct, adj, epoch=30):
  n_adj = adj.size()[0]
  n_feat = features.size()[1]
  n_edge = adj._indices().size()[1]

  apply_zero(features, mask)

  ind_arr = adj._indices()

  degree = miss_struct.degree
  #print(degree)
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
      if mask[i,0] == True:
        features[i] = X[i]

def apply_mean_each(features, mask, miss_struct, adj, epoch=30):
  n_adj = adj.size()[0]
  n_feat = features.size()[1]
  n_edge = adj._indices().size()[1]

  X = torch.zeros_like(features)
  ind_arr = adj._indices()
  dig_miss = torch.zeros(adj.shape[0])
  for i in range(n_edge):
      node1 = ind_arr[0,i].item()
      node2 = ind_arr[1,i].item()
      if mask[node1,0] == False:
          X[node2] += features[node1]
          dig_miss[node2] += 1
      if mask[node2,0] == False:
          X[node1] += features[node2]
          dig_miss[node1] += 1
  
  for i in range(n_adj):
      if dig_miss[i] == 0:
          dig_miss[i] = 1
  dig_miss = dig_miss.repeat(n_feat,1).T
  X = torch.div(X, dig_miss)
  
  for i in range(X.shape[0]):
      if mask[i,0] == True:
          features[i] = X[i]