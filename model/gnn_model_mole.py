import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv,GATConv, global_max_pool as gmp


num_atom_type = 119 
num_chirality_tag = 4

num_bond_type = 5  
num_bond_stereo = 6
num_bond_direction = 3


# Mole Gra
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=2, emb_dim=78,num_features_xd=78, dropout=0.2):

        super(GCNNet, self).__init__()
        
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # GCN
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*3)
        self.drug1_conv3 = GCNConv(num_features_xd*3, 128)

        
        self.fc_g1 = torch.nn.Linear(128, 1024)
        self.fc_g2 = torch.nn.Linear(1024, 128)
        
        self.line = nn.Linear(312, 128)
        self.dropout = nn.Dropout(0.2)
    def forward(self, data1):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x1 = self.x_embedding1(x1[:, 0].int()) + self.x_embedding2(x1[:, 1].int())
 
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)

        # graph representation
        emb = global_mean_pool(x1, data1.batch)

        return x1, emb