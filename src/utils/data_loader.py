import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from functools import partial
import sys
sys.path.append("..")
sys.path.append("../..")
from scipy.sparse import csr_matrix
import random
import copy

class G_data(object):
    def __init__(self, num_class, feat_dim, g_list):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.g_list = g_list
        self.sep_data()

    def sep_data(self, seed=0):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        labels = [g.label for g in self.g_list]
        self.idx_list = list(skf.split(np.zeros(len(labels)), labels))

    def use_fold_data(self, fold_idx):
        self.fold_idx = fold_idx + 1
        train_idx, test_idx = self.idx_list[fold_idx]
        #test_idx, train_idx = self.idx_list[fold_idx]
        tem_g_list=copy.deepcopy(self.g_list)
        self.train_gs = [copy.deepcopy(self.g_list[i]) for i in train_idx]
        self.test_gs = [copy.deepcopy(self.g_list[i]) for i in test_idx]
        '''
        random.shuffle(self.train_gs)
        for i,tem_g in enumerate(self.train_gs):
            g=copy.deepcopy(tem_g)
            if i<=0*len(self.train_gs):
                continue
            elif i>=0.9*len(self.train_gs):
                break
            #changed_tags=check_single(g.node_tags)
            #g.ori_A=poison(csr_matrix(g.ori_A), csr_matrix(g.ori_feas), np.array(changed_tags).reshape(len(g.node_tags), ))
            g.ori_A=random_poison_matrix(g)
            g.ori_A=torch.FloatTensor(g.ori_A)
            g.ori_feas = torch.tensor([x[1] for x in list(g.degree())])
            g.ori_feas = F.one_hot(g.ori_feas, len(g.tagset))

            poison_g,_=poison_tree(g)
            g.ori_A = g.ori_A + torch.eye(g.number_of_nodes())
            #g.ori_A =torch.matmul(torch.matmul(g.ori_A, g.ori_A), g.ori_A)
            g.A= torch.FloatTensor(nx.to_numpy_matrix(poison_g))
            g.A=g.A+torch.eye(poison_g.number_of_nodes())
            #g.A=torch.matmul(torch.matmul(g.A,g.A),g.A)
            self.train_gs[i]=copy.deepcopy(g)
        '''


        self.g_list=copy.deepcopy(tem_g_list)


class FileLoader(object):
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
        g.add_nodes_from(list(range(n)))
        node_tags = []
        for j in range(n):
            row = next(f).strip().split()
            tmp = int(row[1]) + 2
            row = [int(w) for w in row[:tmp]]
            if row[0] not in feat_dict:
                feat_dict[row[0]] = len(feat_dict)
            for k in range(2, len(row)):
                if j != row[k]:
                    g.add_edge(j, row[k])
            if len(row) > 2:
                node_tags.append(feat_dict[row[0]])
        g.label = label
        g.remove_nodes_from(list(nx.isolates(g)))
        if deg_as_tag:
            g.node_tags = list(dict(g.degree).values())
        else:
            g.node_tags = node_tags
        return g

    def process_g(self, label_dict, tag2index, tagset, g, ori_g):
        ori_g.label = label_dict[g.label]
        #g.new_feas = torch.tensor([tag2index[tag] for tag in g.node_tags])
        #g.new_feas = F.one_hot(g.new_feas, len(tagset))
        ori_g.ori_feas = torch.tensor([tag2index[tag] for tag in ori_g.node_tags])
        ori_g.ori_feas = F.one_hot(ori_g.ori_feas, len(tagset))

        A = torch.FloatTensor(nx.to_numpy_matrix(g))
        A=A+torch.eye(g.number_of_nodes())
        ori_g.new_feas = torch.zeros((A.shape[0], ori_g.ori_feas.shape[1]))
        #ori_g.A=torch.matmul(torch.matmul(A,A),A)
        ori_g.A=A
        ori_A=torch.FloatTensor(nx.to_numpy_matrix(ori_g))
        ori_A = ori_A + torch.eye(ori_g.number_of_nodes())
        #ori_g.ori_A = ori_A
        #ori_A_2= torch.matmul(ori_A,ori_A)
        ori_g.ori_A =torch.matmul(torch.matmul(ori_A, ori_A), ori_A)
        #ori_g.ori_A = ori_A_2
        ori_g.tag2index=tag2index
        ori_g.tagset=tagset

        return ori_g

    def load_data(self):
        args = self.args
        print('loading data ...')
        g_list = []
        label_dict = {}
        feat_dict = {}

        with open('../data/%s/%s.txt' % (args.data, args.data+'_aug'), 'r') as f:
            lines = f.readlines()
        f = self.line_genor(lines)
        n_g = int(next(f).strip())
        for i in tqdm(range(n_g), desc="Create graph", unit='graphs'):
            g = self.gen_graph(f, i, label_dict, feat_dict, args.deg_as_tag)
            g_list.append(g)

        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))
        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}

        #读取原本的图
        ori_g_list = []
        ori_label_dict = {}
        ori_feat_dict = {}
        with open('../data/%s/%s.txt' % (args.data, args.data), 'r') as f:
            lines = f.readlines()
        f = self.line_genor(lines)
        ori_n_g = int(next(f).strip())
        for i in tqdm(range(ori_n_g), desc="Create original graph", unit='graphs'):
            ori_g = self.gen_graph(f, i, ori_label_dict, ori_feat_dict, args.deg_as_tag)
            ori_g_list.append(ori_g)

        ori_tagset = set([])
        for ori_g in ori_g_list:
            ori_tagset = ori_tagset.union(set(ori_g.node_tags))
        ori_tagset = list(ori_tagset)
        ori_tag2index = {ori_tagset[i]: i for i in range(len(ori_tagset))}

        f_n = partial(self.process_g, ori_label_dict, ori_tag2index, ori_tagset)
        new_g_list = []
        for g,ori_g in tqdm(zip(g_list,ori_g_list), desc="Process graph", unit='graphs'):
            new_g_list.append(f_n(g,ori_g))
        num_class = len(ori_label_dict)
        feat_dim = len(ori_tagset)

        print('# classes: %d' % num_class, '# maximum node tag: %d' % feat_dim)
        return G_data(num_class, feat_dim, new_g_list)
