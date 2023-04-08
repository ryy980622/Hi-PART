import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)
class GraphAttentionLayer(nn.Module):
    def __init__(self,input_size,output_size,dropout,alpha,concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_size=input_size
        self.output_size = output_size
        #self.W=Parameter(torch.empty(size=(input_size,output_size)))
        self.proj = torch.nn.Linear(input_size, output_size)
        #torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.dropout=dropout
        self.alpha=alpha
        self.concat=concat
        #self.a=Parameter(torch.empty(size=(2*output_size,1)))
        self.proj_a = torch.nn.Linear(2*output_size, 1)
        #torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu=torch.nn.LeakyReLU(self.alpha)

    def forward(self,adj,h):
        wh=self.proj(h)
        #wh=torch.matmul(h,self.W) #h:n*n_infeature w:n_infeature*n_outfeature wh:n*n_outfeature
        a_input=self._prepare_attentional_mechanism_input(wh) #n*n*(2out_feature)
        a_output=self.proj_a(a_input).squeeze(2)
        all_conc=self.leakyrelu(a_output)  #n*n
        zero_matrix=-9e15*torch.ones_like(all_conc)
        all_conc=torch.where(adj>0,all_conc,zero_matrix) #n*n
        #att_matrix=all_conc
        all_conc= F.softmax(all_conc)
        all_conc = F.dropout(all_conc, self.dropout, training=self.training)
        output=torch.matmul(all_conc,wh)
        if self.concat:
            output=F.elu(output)


        return output

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.output_size)


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        #print("g:",g.device)
        #print("h:",h.device)
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
