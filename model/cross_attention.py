import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import collections


class cross_attention(nn.Sequential):
    def __init__(self, hidden_dim):
        super(cross_attention, self).__init__()
        transformer_emb_size_drug = hidden_dim
        # transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 4
        transformer_intermediate_size_drug = hidden_dim
        transformer_num_attention_heads_drug = 4
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1
        
        self.encoder = Encoder_1d(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    
    def forward(self, emb, ex_e_mask,device1):
        global device
        device = device1

        encoded_layers, attention_scores = self.encoder(emb, ex_e_mask)
        return encoded_layers, attention_scores


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class CrossFusion(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossFusion, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_rna = nn.Linear(hidden_size, self.all_head_size)
        self.key_rna = nn.Linear(hidden_size, self.all_head_size)
        self.value_rna = nn.Linear(hidden_size, self.all_head_size)
        
        self.query_mole = nn.Linear(hidden_size, self.all_head_size)
        self.key_mole = nn.Linear(hidden_size, self.all_head_size)
        self.value_mole = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        
        # rna
        rna_hidden = hidden_states[0]
        rna_mask = attention_mask[0]
        
        # mole
        mole_hidden = hidden_states[1]
        mole_mask = attention_mask[1]
        
        rna_mask = rna_mask.unsqueeze(1).unsqueeze(2)
        rna_mask = ((1.0 - rna_mask) * -10000.0).to(device)
        
        mole_mask = mole_mask.unsqueeze(1).unsqueeze(2)
        mole_mask = ((1.0 - mole_mask) * -10000.0).to(device)
      
        mixed_query_layer_rna = self.query_rna(rna_hidden)
        mixed_key_layer_rna = self.key_rna(rna_hidden)
        mixed_value_layer_rna = self.value_rna(rna_hidden)

        query_layer_rna = self.transpose_for_scores(mixed_query_layer_rna)
        key_layer_rna = self.transpose_for_scores(mixed_key_layer_rna)
        value_layer_rna = self.transpose_for_scores(mixed_value_layer_rna)
        
        mixed_query_layer_mole = self.query_mole(mole_hidden)
        mixed_key_layer_mole = self.key_mole(mole_hidden)
        mixed_value_layer_mole = self.value_mole(mole_hidden)
        
        query_layer_mole = self.transpose_for_scores(mixed_query_layer_mole)
        key_layer_mole = self.transpose_for_scores(mixed_key_layer_mole)
        value_layer_mole = self.transpose_for_scores(mixed_value_layer_mole)

        # mole as query, rna as key,value
        attention_scores_mole = torch.matmul(query_layer_mole, key_layer_rna.transpose(-1, -2))
        attention_scores_mole = attention_scores_mole / math.sqrt(self.attention_head_size)
        attention_scores_mole = attention_scores_mole + rna_mask
        attention_probs_mole = nn.Softmax(dim=-1)(attention_scores_mole)
        attention_probs_mole = self.dropout(attention_probs_mole)
        
        context_layer_mole = torch.matmul(attention_probs_mole, value_layer_rna)
        context_layer_mole = context_layer_mole.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_mole = context_layer_mole.size()[:-2] + (self.all_head_size,)
        context_layer_mole = context_layer_mole.view(*new_context_layer_shape_mole)
        
        # rna as query, mole as key,value
        attention_scores_rna = torch.matmul(query_layer_rna, key_layer_mole.transpose(-1, -2))
        attention_scores_rna = attention_scores_rna / math.sqrt(self.attention_head_size)
        attention_scores_rna = attention_scores_rna + mole_mask
        attention_probs_rna = nn.Softmax(dim=-1)(attention_scores_rna)
        attention_probs_rna = self.dropout(attention_probs_rna)
        
        context_layer_rna = torch.matmul(attention_probs_rna, value_layer_mole)
        context_layer_rna = context_layer_rna.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_rna = context_layer_rna.size()[:-2] + (self.all_head_size,)
        context_layer_rna = context_layer_rna.view(*new_context_layer_shape_rna)
        
        # output of cross fusion
        context_layer = [context_layer_rna, context_layer_mole]
        # attention of cross fusion
        attention_probs = [attention_probs_rna, attention_probs_mole]

        return context_layer, attention_probs
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense_rna = nn.Linear(hidden_size, hidden_size)
        self.dense_mole = nn.Linear(hidden_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states_rna = self.dense_rna(hidden_states[0])
        hidden_states_rna = self.dropout(hidden_states_rna)

        hidden_states_rna = self.LayerNorm(hidden_states_rna + input_tensor[0])
        
        hidden_states_mole = self.dense_mole(hidden_states[1])
        hidden_states_mole = self.dropout(hidden_states_mole)
        hidden_states_mole = self.LayerNorm(hidden_states_mole + input_tensor[1])
        return [hidden_states_rna, hidden_states_mole]    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = CrossFusion(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_scores = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_scores    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense_rna = nn.Linear(hidden_size, hidden_size)
        self.dense_mole = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        
        hidden_states_rna = self.dense_rna(hidden_states[0])
        hidden_states_rna = F.relu(hidden_states_rna)
        
        hidden_states_mole = self.dense_mole(hidden_states[1])
        hidden_states_mole = F.relu(hidden_states_mole)
        
        return [hidden_states_rna, hidden_states_mole]    

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense_rna = nn.Linear(hidden_size, hidden_size)
        self.dense_mole = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):

        hidden_states_rna = self.dense_rna(hidden_states[0])
        hidden_states_rna = self.dropout(hidden_states_rna)
        hidden_states_rna = self.LayerNorm(hidden_states_rna + input_tensor[0])
        
        hidden_states_mole = self.dense_mole(hidden_states[1])
        hidden_states_mole = self.dropout(hidden_states_mole)
        hidden_states_mole = self.LayerNorm(hidden_states_mole + input_tensor[1])
        return [hidden_states_rna, hidden_states_mole]    
    
    
class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_scores = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_scores    

    
class Encoder_1d(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_1d, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])
        
        # 1: rna_seq; 2: rna_stru; 3: mole_seq; 4: mole_stru
        # self.cls = nn.Embedding(5, hidden_size,padding_idx=0)
        
        # modality embedding; 0 for seq; 1 for stru;
        self.mod = nn.Embedding(2, hidden_size)
        
        # # mole type embedding; 0 for rna; 1 for small mole
        # self.type_e = nn.Embedding(2, 256)
        
    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # add 1 for mask
        # attention_mask[0] = torch.cat((torch.ones(attention_mask[0].size()[0]).unsqueeze(1).to(device), attention_mask[0]), dim=1).to(device)
        # attention_mask[1] = torch.cat((torch.ones(attention_mask[1].size()[0]).unsqueeze(1).to(device), attention_mask[1]), dim=1).to(device)
        # attention_mask[2] = torch.cat((torch.ones(attention_mask[2].size()[0]).unsqueeze(1).to(device), attention_mask[2]), dim=1).to(device)
        # attention_mask[3] = torch.cat((torch.ones(attention_mask[3].size()[0]).unsqueeze(1).to(device), attention_mask[3]), dim=1).to(device)
        
        # cls_rna_seq = torch.tensor([1]).expand(hidden_states[0].size()[0],1).to(device)
        # cls_rna_seq = self.cls(cls_rna_seq)
        # hidden_states[0] = torch.cat((cls_rna_seq, hidden_states[0]),dim=1)
        
        # cls_rna_stru = torch.tensor([2]).expand(hidden_states[1].size()[0],1).to(device)
        # cls_rna_stru = self.cls(cls_rna_stru)
        # hidden_states[1] = torch.cat((cls_rna_stru, hidden_states[1]),dim=1)
       
        # cls_mole_seq = torch.tensor([3]).expand(hidden_states[2].size()[0],1).to(device)
        # cls_mole_seq = self.cls(cls_mole_seq)
        # hidden_states[2] = torch.cat((cls_mole_seq, hidden_states[2]),dim=1)
        
        # cls_mole_stru = torch.tensor([4]).expand(hidden_states[3].size()[0],1).to(device)
        # cls_mole_stru = self.cls(cls_mole_stru)
        # hidden_states[3] = torch.cat((cls_mole_stru, hidden_states[3]),dim=1)


        # for seq
        seq_rna_emb1 = torch.tensor([0]).expand(hidden_states[0].size()[0],hidden_states[0].size()[1]).to(device)
        seq_rna_emb1 = self.mod(seq_rna_emb1)
        hidden_states[0] = hidden_states[0] + seq_rna_emb1
        
        seq_mole_emb1 = torch.tensor([0]).expand(hidden_states[2].size()[0],hidden_states[2].size()[1]).to(device)
        seq_mole_emb1 = self.mod(seq_mole_emb1)
        hidden_states[2] = hidden_states[2] + seq_mole_emb1
        
        # for stru
        stru_rna_emb1 = torch.tensor([1]).expand(hidden_states[1].size()[0],hidden_states[1].size()[1]).to(device)
        stru_rna_emb1 = self.mod(stru_rna_emb1)
        hidden_states[1] = hidden_states[1] + stru_rna_emb1
        
        stru_mole_emb1 = torch.tensor([1]).expand(hidden_states[3].size()[0],hidden_states[3].size()[1]).to(device)
        stru_mole_emb1 = self.mod(stru_mole_emb1)
        hidden_states[3] = hidden_states[3] + stru_mole_emb1
        
        # # for rna
        # seq_rna_emb2 = torch.tensor([0]).expand(hidden_states[0].size()[0],hidden_states[0].size()[1]).to(device)
        # seq_rna_emb2 = self.type_e(seq_rna_emb2)
        # hidden_states[0] = hidden_states[0] + seq_rna_emb2
        
        # stru_rna_emb2 = torch.tensor([0]).expand(hidden_states[2].size()[0],hidden_states[2].size()[1]).to(device)
        # stru_rna_emb2 = self.type_e(stru_rna_emb2)
        # hidden_states[2] = hidden_states[2] + stru_rna_emb2
        
        # # for small mole
        # seq_mole_emb2 = torch.tensor([1]).expand(hidden_states[1].size()[0],hidden_states[1].size()[1]).to(device)
        # seq_mole_emb2 = self.type_e(seq_mole_emb2)
        # hidden_states[1] = hidden_states[1] + seq_mole_emb2
        
        # stru_mole_emb2 = torch.tensor([1]).expand(hidden_states[3].size()[0],hidden_states[3].size()[1]).to(device)
        # stru_mole_emb2 = self.type_e(stru_mole_emb2)
        # hidden_states[3] = hidden_states[3] + stru_mole_emb2
        rna_hidden = torch.cat((hidden_states[0], hidden_states[1]), dim=1)
        mole_hidden = torch.cat((hidden_states[2], hidden_states[3]), dim=1)
        
        rna_mask = torch.cat((attention_mask[0], attention_mask[1]), dim=1)
        mole_mask = torch.cat((attention_mask[2], attention_mask[3]), dim=1)

        hidden_states = [rna_hidden, mole_hidden]        
        attention_mask = [rna_mask, mole_mask]
        
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
               all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attention_scores
    