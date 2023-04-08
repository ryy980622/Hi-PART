import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class GINConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)

        return X
class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,p):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear( hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
    def forward(self,x):
        #x = self.drop(x)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        #x = self.drop(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x)
        return x

class GIN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,p):
        super().__init__()

        #self.in_proj = torch.nn.Linear(input_dim, hidden_dim)

        #self.convs = torch.nn.ModuleList()

        #self.proj = GINConv(hidden_dim)
        self.mlp = MLP(input_dim, hidden_dim, output_dim,p)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        #for _ in range(n_layers):
            #self.convs.append(GINConv(hidden_dim))

        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x hiddem_dim*(1+n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x hiddem_dim*(1+n_layers)].
        #self.out_proj = torch.nn.Linear(hidden_dim , output_dim)

    def forward(self, A, X):
        X = self.drop(X)
        X = torch.matmul(A,X)
        X = self.mlp(X)


        return X