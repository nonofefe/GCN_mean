import argparse

from models import GCNmf, GCN
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask, apply_zero, apply_neighbor_mean, apply_neighbor_mean_recursive, apply_embedding_mean, preprocess_features
from miss_struct import MissStruct
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=16, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
parser.add_argument('--epoch', default=10000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--model',
                    default='GCNmf',
                    choices=['GCNmf', 'GCN'],
                    help='model name')
parser.add_argument('--split', default=1, type=int, help='the number of split units')
args = parser.parse_args()

if __name__ == '__main__':
    data = NodeClsData(args.dataset)
    #print(data.features.sum(axis=0))
    mask = generate_mask(data.features, args.rate, args.type)
    miss_struct = MissStruct(mask, data.adj, args.split)
    #print(data.features)
    apply_mask(data.features, mask)
    #print(data.features)
    params = {
        'lr': args.lr,
        'weight_decay': args.wd,
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    model = 0
    if args.model == 'GCNmf':
      model = GCNmf(data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)
    elif args.model == 'GCN':
      if args.type == "struct":
        #print("apply_neighbor_mean!!")  
        #apply_neighbor_mean(data.features, mask, miss_struct, data.adj)
        print("apply_neighbor_mean_recursive!!")  
        apply_neighbor_mean_recursive(data.features, mask, miss_struct, data.adj)       
        #print("apply_embedding_mean!!")
        #apply_embedding_mean(data.features, mask, args.dataset)
      else:
        print("apply_mean_each!!")
        apply_mean_each(data.features, mask, miss_struct, data.adj)
        #apply_zero(data.features, mask)

      #data.features = preprocess_features(data.features)

      model = GCN(data, nhid=args.nhid, dropout=args.dropout)

    trainer = NodeClsTrainer(data, model, params, niter=20, verbose=args.verbose)
    trainer.run()
