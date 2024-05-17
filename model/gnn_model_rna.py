import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,GATConv


# 1d CNN 
class CNN(torch.nn.Module):
    def __init__(self, hidden_size):
        super(CNN, self).__init__()
        # three size
        kernel_size = [7, 11, 15]

        self.conv_xt_1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2)
        self.conv_xt_2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size[1], padding=(kernel_size[1]-1)//2)
        self.conv_xt_3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=kernel_size[2], padding=(kernel_size[2]-1)//2)

        self.fc1_xt = nn.Linear(128, 128)
        
        self.relu = nn.ReLU()
        
        self.line1 =  nn.Linear(64, 512)
        self.line2 =  nn.Linear(512, 128)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):

        x = x.permute(0, 2, 1)
                
        x1 =  self.conv_xt_1(x)
        x2 =  self.conv_xt_2(x)
        x3 =  self.conv_xt_3(x)

        x = (x1+x2+x3) / 3
        x = x.permute(0, 2, 1)
        x = self.line2(self.dropout(self.relu(self.line1(x))))
        return x



class RNA_feature_extraction(nn.Module):
    def __init__(self, hidden_size,input_dim=128, hidden_dim=256):
        super(RNA_feature_extraction, self).__init__()
        
        # RNA Seq
        self.CNN = CNN(hidden_size)
        
        
        # RNA Gra
        self.conv1 =  GATConv(input_dim, hidden_dim, heads=4, dropout=0.1,concat=False)
        self.conv2 =  GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.1,concat=False)
        self.conv3 =  GATConv(hidden_dim , hidden_size, dropout=0.1,concat=False)
        

        self.x_embedding = nn.Embedding(6, input_dim)
        self.x_embedding2 = nn.Embedding(6, input_dim)

        
        self.line1 = nn.Linear(128, hidden_size)
        self.hidden_size = hidden_size
        
        self.line_emb = nn.Linear(640, 128)
        self.line_g= nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()


    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index
        emb = data.emb

        x_r = self.x_embedding(x[:, 0].int())
        x_g = self.x_embedding2(x[:, 0].int())

        # RNA graph
        x = self.relu(self.conv1(x_g, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        
        emb_graph = global_mean_pool(x, data.batch)

        emb = F.relu(self.line_emb(emb))
        flag = 0
        node_len = data.rna_len
        out_graph = []
        out_seq = []
        out_r = []
        mask = []
        
        for i in node_len:
            count_i = i  
            
            # RNA graph output
            mask.append([] + count_i * [1] + (512 - count_i) * [0])
            x1 = x[flag:flag + count_i]
            temp1 = torch.zeros((512 - x1.size(0), self.hidden_size), device=device)
            x1 = torch.cat((x1, temp1),0)
            out_graph.append(x1)
            
            # RNA pre-trained emb 
            emb1 = emb[flag:flag + count_i]
            temp2 = torch.zeros((512-emb1.size()[0]), 128).to(device)
            emb1 = torch.cat((emb1, temp2),0)
            out_seq.append(emb1)
            
            # RNA CNN initial
            x_r1 = x_r[flag:flag + count_i]
            temp3 = torch.zeros((512-x_r1.size()[0]), 128).to(device)
            x_r1 = torch.cat((x_r1, temp3),0)
            out_r.append(x_r1)
            
            flag += count_i
        out_graph = torch.stack(out_graph).to(device)
        out_seq = torch.stack(out_seq).to(device)
        out_r = torch.stack(out_r).to(device)
        
        
        mask_graph = torch.tensor(mask, dtype=torch.float)
        mask_seq = torch.tensor(mask, dtype=torch.float)
        
        # RNA seq
        out_r = (out_r + out_seq) / 2
        
        out_seq_cnn = self.CNN(out_r)

        emb_seq = (out_seq_cnn*(mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)
        
        return out_seq_cnn, out_graph, mask_seq, mask_graph, emb_seq, emb_graph
