import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset import WineQuality
from model import Net

dataset_test = WineQuality("winequality_red_test.csv")
dataloader_test = DataLoader(dataset_test, 
                             batch_size=4, 
                             shuffle=False)

net = Net()

net.load_state_dict(torch.load("wine_net.pth"))

net.eval()
all_preds = []

with torch.no_grad():
    for features, _ in dataloader_test:
        outputs = net(features)
        preds = outputs.argmax(dim=1) + 3  # shift back to original quality scale
        all_preds.extend(preds.tolist())

results = []

net.eval()
with torch.no_grad():
    for features, labels in dataloader_test:
        preds = net(features).argmax(dim=1) + 3
        actuals = labels + 3
        for p, a in zip(preds.tolist(), actuals.tolist()):
            results.append({"predicted": p, "actual": a})

pd.DataFrame(results).to_csv("predictions.csv", index=False)