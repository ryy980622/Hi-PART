import math
import torch.nn.functional as F
import torch

from torch.nn.parameter import Parameter
#from torch.nn.modules.module import Module


class GraphAttentionLayer(torch.nn.Module):
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
        self.a=Parameter(torch.empty(size=(2*output_size,1)))
        self.proj_a = torch.nn.Linear(2*output_size, 1)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

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

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.output_size)
    '''
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_size) + ' -> ' + str(self.output_size) + ')'
    '''

