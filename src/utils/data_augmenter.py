import torch
from tqdm import tqdm
import networkx as nx
import argparse
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from functools import partial
import sys
sys.path.append("..")
import  matplotlib.pyplot as plt
from graph_augmentation import graph_augment
import time

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='REDDIT-BINARY', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=2, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=1, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.7,0.8,0.9])
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args

class data_augmenter(object):
    def __init__(self, args):
        self.args = args

    def line_genor(self, lines):
        for line in lines:
            yield line
    def gen_graph(self, f, i, label_dict, feat_dict, deg_as_tag):
        row = next(f).strip().split()
        n, label = [int(w) for w in row]
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        g = nx.Graph()
        g.add_nodes_from(list(range(1,n+1)))
        node_tags = []
        for j in range(n):
            row = next(f).strip().split()
            tmp = int(row[1]) + 2
            row = [int(w) for w in row[:tmp]]
            if row[0] not in feat_dict:
                feat_dict[row[0]] = len(feat_dict)
            for k in range(2, len(row)):
                if j < row[k]:
                #if j!=row[k]:
                    g.add_edge(j+1, row[k]+1)
            if len(row) > 2:
                node_tags.append(feat_dict[row[0]])
        g.label = label
        g.remove_nodes_from(list(nx.isolates(g)))
        if deg_as_tag:
            g.node_tags = list(dict(g.degree).values())
        else:
            g.node_tags = node_tags
        return g
    def augment_data(self):
        args = self.args
        print('loading data ...')
        g_list = []
        label_dict = {}
        feat_dict = {}
        dataset = 'REDDITBINARY'
        sum_dif=0
        cnt=0
        with open('../../data/' + dataset + '/' + dataset + '.txt', 'r', encoding='utf8') as f:
            # 读入输入文件，记录每个点的度和边
            lines = f.readlines()
            f = self.line_genor(lines)
            n_g = int(next(f).strip())
            #with open('../../data/' + dataset + '/' + dataset + '_aug', 'w', encoding='utf8') as f1:
                #f1.write(str(n_g)+'\n')
            start= time.clock()
            sum_error = 0
            for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
                g = self.gen_graph(f, i, label_dict, feat_dict, args.deg_as_tag)#g有node,edge,node_tags,g.label

                #print(g.label)
                '''
                if g.label==0 and cnt<=20:
                    nx.draw_networkx(g,node_color=g.node_tags)
                    plt.show()
                    cnt+=1
                '''
                g, dif = graph_augment(g, dataset)
                g_list.append(g)

            end = time.clock()
            print("time:", end-start)
            print('ave_dif:', sum_dif/(n_g-sum_error))
args = get_args()
G_data = data_augmenter(args).augment_data()