import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
from torch_geometric.data import Data

class RNA_dataset(InMemoryDataset):
    def __init__(self,
                 RNA_type,
                 root="dataset/rna",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        root = "dataset/rna/" + RNA_type
            

        # All RNA or 6 RNA subtype: All_sf; Aptamers; miRNA; Repeats; Ribosomal; Riboswitch; Viral_RNA;
        csv_file_path = 'data/RSM_data/' + RNA_type + '_dataset_v1.csv'  
        self.df = pd.read_csv(csv_file_path, delimiter='\t')
      
        # contact map folder
        self.concat_folder_path1 = 'data/RNA_contact/Aptamers_contact' 
        self.concat_folder_path2 = 'data/RNA_contact/miRNA_contact' 
        self.concat_folder_path3 = 'data/RNA_contact/Repeats_contact' 
        self.concat_folder_path4 = 'data/RNA_contact/Ribosomal_contact' 
        self.concat_folder_path5 = 'data/RNA_contact/Riboswitch_contact' 
        self.concat_folder_path6 = 'data/RNA_contact/Viral_RNA_contact' 
        self.concat_folder_path = [self.concat_folder_path1,self.concat_folder_path2,self.concat_folder_path3,self.concat_folder_path4,self.concat_folder_path5,self.concat_folder_path6]

        # language model embedding floder
        self.emb_folder_path = 'data/representations_cv'
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        


    @property
    def processed_file_names(self):
        return "data_rna.pt"


    def process(self):

        data_list = []
        for index, row in self.df.iterrows():
            id_value = row['Entry_ID']
            if len(row['Target_RNA_sequence']) > 512:
                sequence = row['Target_RNA_sequence'][0:511]
            else:  
                sequence = row['Target_RNA_sequence']
            target_id = row["Target_RNA_ID"]
            
            for i in range(0,6):
                file_path = os.path.join(self.concat_folder_path[i], f"{id_value}.prob_single")
                if os.path.exists(file_path):
                    break
            if os.path.exists(file_path):
                matrix = np.loadtxt(file_path)
                # contact map
                matrix[matrix < 0.5] = 0
                matrix[matrix > 0.5] = 1
                one_hot_sequence = [char_to_one_hot(char) for char in sequence]
                edges = np.argwhere(matrix == 1)

                x = torch.tensor(one_hot_sequence, dtype=torch.float32)
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                y = row['pKd']  
                T_id = row['Target_RNA_ID']
                E_id = row['Entry_ID']
                
                # read language model embedding
                emb_file_path = os.path.join(self.emb_folder_path, f"{target_id}.npy")
                if os.path.exists(emb_file_path):
                    rna_emb = torch.tensor(np.load(emb_file_path))
                else:
                    print('bad', target_id)
                    
                rna_len = x.size()[0]
                data = Data(x=x, edge_index=edge_index, y=y, t_id=T_id, e_id=E_id, emb=rna_emb, rna_len = rna_len )

                data_list.append(data)
            else:
                print(f"File not found for id {id_value}")
        
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

# nucleotide to one-hot
def char_to_one_hot(char):
    if char == 'T':
        print("T")
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'T':1, 'X':4, 'Y':5}
    return [mapping[char]]