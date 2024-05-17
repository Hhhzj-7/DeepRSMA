import os
import sys
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from rdkit import Chem
import numpy as np
from .feature import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data
from .vocab import *
max_len = 128
atom_dict = {5: 'C',
             6: 'C',
             9: 'O',
             12: 'N',
             15: 'N',
             21: 'F',
             23: 'S',
             25: 'Cl',
             26: 'S',
             28: 'O',
             34: 'Br',
             36: 'P',
             37: 'I',
             39: 'Na',
             40: 'B',
             41: 'Si',
             42: 'Se',
             44: 'K',
             }
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



class Alphabets():
    def __init__(self, chars, encoding=None, missing=255):
        self.chars = np.frombuffer(chars, dtype='uint8')
        self.size = len(self.chars)
        self.encoding = np.zeros(256, dtype='uint8') + missing
        if encoding == None:
            self.encoding[self.chars] = np.arange(self.size)
        else:
            self.encoding[self.chars] = encoding
            
    def encode(self, s):
        s = np.frombuffer(s, dtype='uint8')
        return self.encoding[s]
class Smiles(Alphabets):
    def __init__(self):
        chars = b'#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty'
        super(Smiles, self).__init__(chars)

smilebet = Smiles()


class Molecule_dataset_independent(InMemoryDataset):
    def __init__(self,
                 root="dataset/small_molecule/",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        root = "dataset/small_molecule_independent/"
       
        csv_file_path_qsar = 'data/independent_data.csv'

        self.df_qsar = pd.read_csv(csv_file_path_qsar, delimiter=',')
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data_sm.pt"



    def process(self):

        drug_vocab = WordVocab.load_vocab('data/smiles_vocab.pkl')
        data_list = []
        for index in range(0,48):
                data = Data()
                mol = Chem.MolFromSmiles((self.df_qsar.iloc[index])['SMILES'])
                if mol == None:
                    mol = Chem.MolFromSmiles((self.df_qsar.iloc[index])['SMILES'], sanitize=False)
                    mol.UpdatePropertyCache(strict=False)
                
                # atoms
                atom_features_list = []
                for atom in mol.GetAtoms():
                    atom_features_list.append(atom_to_feature_vector(atom))
                x = np.array(atom_features_list, dtype=np.int64)
                if len(x) > 128:
                    x = x[:128]
                # bonds
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():

                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    if i >= 128 or j >=128:
                        continue 

                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((i, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, i))
                    edge_features_list.append(edge_feature)

                edge_index = np.array(edges_list, dtype=np.int64).T
                edge_attr = np.array(edge_features_list, dtype=np.int64)
                data.x = torch.from_numpy(x).to(torch.int64)
                data.graph_len = len(data.x)
                data.x2 = torch.from_numpy(x2).to(torch.int64)

                data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
                data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
                data.smiles_ori = (self.df_qsar.iloc[index])['SMILES']
                # data.y = row['pKd']
                # E_id = row['Entry_ID']
                
                content = []
                flag = 0
                sm = (self.df_qsar.iloc[index])['SMILES']
                for i in range(len(sm)):
                    if flag >= len(sm):
                        break
                    if (flag + 1 < len(sm)):
                        if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                            content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                            flag = flag + 2
                            continue
                    content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
                    flag = flag + 1

                if len(content) > max_len:
                    content = content[:max_len]

                # X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
                data.smile_len = len(content)
                out = torch.ones(128)
                out[len(content):128] = 0
                data.mask = out
                X = content
                if max_len > len(X):
                    padding = [drug_vocab.pad_index] * (max_len - len(X))
                    X.extend(padding)
                tem = []
                for i, c in enumerate(X):
                    if atom_dict.__contains__(c):
                        # if c == 28 and X[i-1] == 6 and (X[i+1] == 29 or X[i+1] == 19):
                        #     continue
                        tem.append(i)


                smile_emb = torch.tensor(X)
                print(smile_emb.size())
                # print(smile_emb)
                data.smile_emb = smile_emb
                
                data.atom_len = tem
                
                
                
                smiles_f = (self.df_qsar.iloc[index])['SMILES'].encode('utf-8').upper()
                smiles_f = torch.from_numpy(smilebet.encode(smiles_f)).long()
                data.smiles_f = smiles_f
                
                # data.smiles = row['SMILES']
                if len(tem) != data.x.size()[0]:
                    print("bad")
                    print(len(tem), data.x.size()[0])
                    print(smile_emb)
                
                data_list.append(data)
        
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

