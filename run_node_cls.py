import argparse

from models import GCNmf, GCN
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask, apply_zero
from miss_struct import MissStruct
import numpy as np


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
parser.add_argument('--split', default=2, type=int, help='the number of split units')
args = parser.parse_args()

if __name__ == '__main__':
    data = NodeClsData(args.dataset)
    mask = generate_mask(data.features, args.rate, args.type)
    miss_struct = MissStruct(mask, data.adj, args.split)

    miss_struct.degree /= 20
    # len1 = len(miss_struct.degree)
    # ma = 0
    # for i in range(len1):
    #     if miss_struct.degree[i] > ma:
    #         ma = miss_struct.degree[i]
    #     print(miss_struct.degree[i])
    # print(ma)

    # for i in range(data.adj.size()[0]):
    #     print(miss_struct.mask_neighbor[i]) #特徴量

    apply_mask(data.features, mask)

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
        apply_zero(data.features, mask)
        model = GCN(data, nhid=args.nhid, dropout=args.dropout)
    # model = GCNmf(data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)

    trainer = NodeClsTrainer(data, model, params, niter=20, verbose=args.verbose)
    trainer.run(miss_struct)
