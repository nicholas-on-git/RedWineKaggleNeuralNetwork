import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset import WineQuality
from model import Net

dataset_train = WineQuality("winequality-red.csv")

dataloader_train = DataLoader(
    dataset_train,
    batch_size=4,
    shuffle=True
)

features, labels = next(iter(dataloader_train))

print(f"features:  {features}, \nLabels: {labels}")

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_history = []

for epoch in range(300):
    epoch_loss = 0
    for data in dataloader_train:
        optimizer.zero_grad()
        features, target = data
        pred = net(features)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader_train)
    loss_history.append({"epoch": epoch + 1, "loss": float(avg_loss)})

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/1000, Loss: {avg_loss:.4f}")

pd.DataFrame(loss_history).to_csv("training_log.csv", index=False)

net.eval()

training_set = []

with torch.no_grad():
    sample_features, actual_label = dataset_train[0]
    
    output = net(sample_features)
    predicted_quality = output.argmax().item() + 3
    actual_quality = actual_label.item() + 3

    print(f"Predicted: {predicted_quality}")
    print(f"Actual:    {actual_quality}")


torch.save(net.state_dict(), "wine_net.pth")