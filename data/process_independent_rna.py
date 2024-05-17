import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from rdkit import Chem
import numpy as np
# from features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data
import math


def KD_to_pKD(KD):
    return -math.log10(KD)

class RNA_dataset_independent(InMemoryDataset):
    def __init__(self,
                 root="dataset/rna",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        root = "dataset/rna_independent/"

        csv_file_path_qsar = 'independent_data.csv'

        # 使用pandas读取CSV文件
        self.df_qsar = pd.read_csv(csv_file_path_qsar, delimiter=',')

        
        self.concat_folder_path_qsar = 'data/RNA_contact/qsar' 
        self.emb_folder_path = 'data/representations_independent'
        self.emb_file_path_qsar = os.path.join(self.emb_folder_path, "hiv.npy")

        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return "data_rna.pt"

    def process(self):

        data_list = []
        for index in range(0,48):
            sequence = 'GGCAGAUCUGAGCCUGGGAGCUCUCUGCC'
            file_path = os.path.join(self.concat_folder_path_qsar, f"hiv.prob_single")

            matrix = np.loadtxt(file_path)
            matrix[matrix < 0.5] = 0
            matrix[matrix > 0.5] = 1

            one_hot_sequence = [char_to_one_hot(char) for char in sequence]

            edges = np.argwhere(matrix == 1)

            x = torch.tensor(one_hot_sequence, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            y = KD_to_pKD((self.df_qsar.iloc[index])['KD'])
            

            if os.path.exists(self.emb_file_path_qsar):
                rna_emb = torch.tensor(np.load(self.emb_file_path_qsar))
            else:
                print('bad')
                
            rna_len = x.size()[0]
            data = Data(x=x,y=y ,edge_index=edge_index, emb=rna_emb, rna_len = rna_len )

            data_list.append(data)

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])



def char_to_one_hot(char):
    if char == 'T':
        print("T")
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'T':1, 'X':4, 'Y':5}
    return [mapping[char]]
