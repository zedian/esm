import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs

vocab_path = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205
max_p = 545
 

def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def drug2emb_encoder(x):
    max_d = 50
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        #d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        
        #d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)
        p_v, input_mask_p = protein2emb_encoder(p)
        
        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y

# tok_to_idx = {
#     "<cls>": 0,
#     "<pad>": 91,
#     "<sep>": 2,
#     "<unk>": 3,
#     "<mask>": 1
# }

tok_to_idx = {
    "<cls>": 32,
    "<pad>": 1,
    "<sep>": 34,
    "<unk>": 3,
    "<mask>": 33
}

MAX_DRUG_LEN = 50
MAX_PROT_LEN = 545

# smile_to_idx = { "#": 4, "%": 5, ")": 6, "(": 7, "+": 8, "-": 9, 
#                 ".": 10, "1": 11, "0": 12, "3": 13, "2": 14, "5": 15, 
#                 "4": 16, "7": 17, "6": 18, "9": 19, "8": 20, "=": 21, 
#                 "A": 22, "C": 23, "B": 24, "E": 25, "D": 26, "G": 27,
#                 "F": 28, "I": 29, "H": 30, "K": 31, "M": 32, "L": 33, 
#                 "O": 34, "N": 35, "P": 36, "S": 37, "R": 38, "U": 39, 
#                 "T": 40, "W": 41, "V": 42, "Y": 43, "[": 44, "Z": 45, 
#                 "]": 46, "_": 47, "a": 48, "c": 49, "b": 50, "e": 51, 
#                 "d": 52, "g": 53, "f": 54, "i": 55, "h": 56, "m": 57, 
#                 "l": 58, "o": 59, "n": 60, "s": 61, "r": 62, "u": 63,
#                 "t": 64, "y": 65}

# fasta_to_idx = { "A": 66, "C": 67, "B": 68, "E": 69, "D": 70, "G": 71, 
#                 "F": 72, "I": 73, "H": 74, "K": 75, "M": 76, "L": 77, 
#                 "O": 78, "N": 79, "Q": 80, "P": 81, "S": 82, "R": 83, 
#                 "U": 84, "T": 85, "W": 86, "V": 87, "Y": 88, "X": 89, 
#                 "Z": 90 }

fasta_to_idx = {'<null_0>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
  'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
  'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N':17, 'F':18, 'Y': 19,
  'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24,'B': 25,'U': 26,
  'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<cls>': 32,
  '<mask>': 33, '<sep>': 34}

smile_to_idx = {'#': 35,'%': 36,'(': 38,')': 37,'+': 39,'-': 40,'.': 41,
 '0': 43,'1': 42,'2': 45,'3': 44,'4': 47,'5': 46,'6': 49,'7': 48,'8': 51,
 '9': 50,'=': 52,'A': 53,'B': 55,'C': 54,'D': 57,'E': 56,'F': 59,'G': 58,
 'H': 61,'I': 60,'K': 62,'L': 64,'M': 63,'N': 66,'O': 65,'P': 67,'R': 69,
 'S': 68,'T': 71,'U': 70,'V': 73,'W': 72,'Y': 74,'Z': 76,'[': 75,']': 77,
 '_': 78,'a': 79,'b': 81,'c': 80,'d': 83,'e': 82,'f': 85,'g': 84,'h': 87,
 'i': 86,'l': 89,'m': 88,'n': 91,'o': 90,'r': 93,'s': 92,'t': 95,'u': 94,'y': 96}

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
#     line = pbpe.process_line(line).split() 
    X = np.ones(MAX_SMI_LEN)*91
    for i, ch in enumerate(line[:MAX_SMI_LEN]): #x, smi_ch_ind, y
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
        else:
            X[i] = 3
    
    l = len(X)
   
    if l < MAX_SMI_LEN:
        input_mask = ([1] * l) + ([0] * (MAX_SMI_LEN - l))
    else:
        input_mask = [1] * MAX_SMI_LEN

    return X, input_mask #X.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
#     line = pbpe.process_line(line).split() 
    X = np.ones(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
        else:
            X[i] = 3
    l = len(X)
     
    if l < MAX_SEQ_LEN:
        input_mask = ([1] * l) + ([0] * (MAX_SMI_LEN - l))
    else:
        input_mask = [1] * MAX_SEQ_LEN
    return X, input_mask #X.tolist()


class BIN_combined_encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti, sep=False):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.sep=sep
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
   
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        #d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        
        #d_v = drug2single_vector(d)
#         d_v, input_mask_d = drug2emb_encoder(d)
#         p_v, input_mask_p = protein2emb_encoder(p)
        
        encoded_drug, input_mask_d = label_smiles(d, MAX_DRUG_LEN, smile_to_idx)
        encoded_target, input_mask_p = label_sequence(p, MAX_PROT_LEN, fasta_to_idx)
        
        y = self.labels[index]
        if self.sep:
            return np.array(encoded_drug), np.array(encoded_target), np.array(input_mask_d), np.array(input_mask_p), y
        
        feature_vec = np.concatenate([np.concatenate([encoded_drug,[91]], axis=0),encoded_target], axis=0)
        feature_mask = np.concatenate([np.concatenate([input_mask_d, [91]], axis=0), input_mask_p], axis=0)
        

        return feature_vec, feature_mask, y