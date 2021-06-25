from .data_utils import NodeClsData, LinkPredData, get_degree, preprocess_features
from .missing import generate_mask, apply_mask, apply_zero, apply_neighbor_mean
from .embedding import apply_embedding_mean