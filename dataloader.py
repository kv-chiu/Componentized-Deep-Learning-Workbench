import numpy as np
import pandas as pd

import torch
from torch.utils import data

class myDataset(data.Dataset):
    def __init__(self, input_path, label_path, mode, has_header=False, has_index=False):
        self.mode = mode
        self.has_header = has_header
        self.has_index = has_index
        
        self.input_data = self.load_file(input_path)
        self.input_data = self.input_data.values
        self.input_data = self.input_data.astype(np.float32)
        self.input_data = torch.from_numpy(self.input_data)

        self.label_data = self.load_file(label_path)
        # 全部减一
        self.label_data = self.label_data - 1
        self.label_data = self.label_data.values
        self.label_data = self.label_data.astype(np.float32)
        self.label_data = torch.from_numpy(self.label_data)
        self.label_data = self.label_data.long()
        
        self.input_data, self.label_data = self.input_data.to('cuda:0'), self.label_data.to('cuda:0')

    def __getitem__(self, index):
        input_data = self.input_data[index]
        label_data = self.label_data[index]
        return input_data, label_data
    
    def __len__(self):
        return len(self.input_data)
    
    def load_file(self, file_path):
        if self.has_header:
            data = pd.read_csv(file_path, header=0)
        else:
            data = pd.read_csv(file_path, header=None)
        if self.has_index:
            data = data.drop([0], axis=1)
        return data
