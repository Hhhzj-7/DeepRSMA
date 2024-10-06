import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch
from data import RNA_dataset, Molecule_dataset, RNA_dataset_independent, Molecule_dataset_independent, WordVocab
from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr,spearmanr
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import random
torch.set_printoptions(profile="full")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hidden_dim = 16

EPOCH = 200
RNA_type = 'Viral_RNA_independent'
rna_dataset = RNA_dataset(RNA_type)
molecule_dataset = Molecule_dataset(RNA_type)

rna_dataset_in = RNA_dataset_independent()
molecule_dataset_in = Molecule_dataset_independent()

seed = 1



# set random seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)
set_seed(seed)

# combine two pyg dataset
class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)  



def average_multiple_lists(lists):
    return [sum(item)/len(lists) for item in zip(*lists)]





# DeepRSMA architecture
class DeepRSMA(nn.Module):
    def __init__(self):
        super(DeepRSMA, self).__init__()
        # RNA graph + seq
        self.rna_graph_model = RNA_feature_extraction(hidden_dim)
        
        # Mole graph
        self.mole_graph_model = GNN_molecule(hidden_dim)
        # Mole seq
        self.mole_seq_model = mole_seq_model(hidden_dim)

        # Cross fusion module
        self.cross_attention = cross_attention(hidden_dim)
        
        self.line1 = nn.Linear(hidden_dim*2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        
        self.rna1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim*4)
        
        self.rna2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim*4, hidden_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, rna_batch, mole_batch):
        rna_out_seq,rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, device)
        
        mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
        
        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)
        
        mole_seq_final = (mole_seq_emb[-1]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)


        # mole graph
        flag = 0
        mole_out_graph = []
        mask = []
        for i in mole_batch.graph_len:
            count_i = i
            x = mole_graph_emb[flag:flag+count_i]
            temp = torch.zeros((128-x.size()[0]), hidden_dim).to(device)
            x = torch.cat((x, temp),0)
            mole_out_graph.append(x)
            mask.append([] + count_i * [1] + (128 - count_i) * [0])
            flag += count_i
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)
        
        context_layer, attention_score = self.cross_attention([rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph], [rna_mask_seq.to(device), rna_mask_graph.to(device), mole_mask_seq.to(device), mole_mask_graph.to(device)], device)

        
        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]
        
        # Affinity Prediction Module
        rna_cross_seq = ((out_rna[:, 0:512]*(rna_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_seq_final ) / 2
        rna_cross_stru = ((out_rna[:, 512:]*(rna_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_graph_final) / 2        

        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout((self.relu(self.rna1(rna_cross)))))

        
        mole_cross_seq = ((out_mole[:,0:128]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_seq_final) / 2
        mole_cross_stru = ((out_mole[:,128:]*(mole_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_graph_final) / 2
        
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout((self.relu(self.mole1(mole_cross)))))   
        
        out = torch.cat((rna_cross, mole_cross),1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        

        return out


# use viral RNA to train
train_dataset = CustomDualDataset(rna_dataset, molecule_dataset)
# independent test
test_dataset = CustomDualDataset(rna_dataset_in, molecule_dataset_in)


train_loader = DataLoader(
    train_dataset, batch_size=8, num_workers=1, drop_last=False, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False
)



model = DeepRSMA()
model.to(device)

y_pred_all = []
max_p = -1

optimizer = optim.Adam(model.parameters(), lr=6e-5 , weight_decay=1e-5)
optimal_loss = 1e10
loss_fct = torch.nn.MSELoss()
for epoch in range(0,EPOCH):
    train_loss = 0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        pre = model(batch[0].to(device), batch[1].to(device))

        y = batch[0].y
        
        loss = loss_fct(pre.squeeze(dim=1), y.float())
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss
    with torch.set_grad_enabled(False):
        model.eval()
        y_label = []
        y_pred = []
        for step, (batch_v) in enumerate(test_loader):
            label = Variable(torch.from_numpy(np.array(batch_v[0].y))).float()
            score = model(batch_v[0].to(device), batch_v[1].to(device))

            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()

            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        p = pearsonr(y_label, y_pred)
        s = spearmanr(y_label, y_pred)
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
        print( 'epo:',epoch, 'pcc:',p[0],'scc: ',s[0], 'rmse:',rmse)

        if max_p < p[0]:
            max_p = p[0]
            print(' ')
            print('Best:', 'epo:',epoch, 'pcc:',p[0],'scc: ',s[0],'rmse:',rmse)

            torch.save(model.state_dict(), 'save/' + 'model_independent_'+str(seed)+'.pth')

        
        model.train()
