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
    # mask_per = sum(mask.t(),0) / float(mask.shape[1])
    # mask_per_neighbor = mask_per * 0
    # #print(mask_per)
    # deg_list = np.zeros(mask_per.shape[0])
    # adj_values = data.adj._values()
    # ind_arr = data.adj._indices()
    # n = data.adj._indices().size()[1]
    # print(ind_arr[0,0].item())
    # for i in range(n):
    #     node1 = ind_arr[0,i].item()
    #     node2 = ind_arr[1,i].item()
    #     deg_list[node1] += 1
    #     deg_list[node2] += 1
    #     mask_per_neighbor[node1] += mask_per[node2]
    #     # if node1 == 0:
    #     #     print(mask_per[node2])
    #     #     print(adj_values[i])
            
    # for i in range(mask_per.shape[0]):
    #     deg_list[i] /= 2
    
    # for i in range(data.adj.size()[0]):
    #     mask_per_neighbor[i] /= deg_list[i]
    #     # print(mask_per_neighbor[i]) #特徴量
    # ko.a = mask_per_neighbor
    # ko.b = mask_per
    miss_struct = MissStruct(mask, data.adj, args.split)

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
