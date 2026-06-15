import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Load data and store as numpy array
class WineQuality(Dataset):

    def __init__(self, csv_path):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = int(self.data[idx, -1]) -3
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)