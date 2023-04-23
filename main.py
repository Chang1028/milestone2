# import packages
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn

input_size = 24
batch_size = 64

# data preprocess
data = pd.read_csv("smoking.csv")
data = data.drop(["ID", "oral"], axis=1)
data = pd.get_dummies(data, drop_first=True)

scaler = StandardScaler()
scaled = scaler.fit_transform(data)
data = pd.DataFrame(scaled, columns=data.columns)

training_data, test_data = train_test_split(data, test_size=0.2)
training_data = training_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

class SmokeDataset(Dataset):
    def __init__(self, dataset, label="smoking"):
        self.data = dataset
        self.labels = self.data[label]
        self.data = self.data.drop([label], axis = 1)
        self.data = self.data


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        observation = self.data.iloc[idx, :]
        label = self.labels.iloc[idx]
        observation = torch.tensor(observation, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return observation, label


training_data = SmokeDataset(training_data)
test_data = SmokeDataset(test_data)


train_dataloader = DataLoader(
    dataset = training_data,
    batch_size = batch_size,
    shuffle = True
)
test_dataloader = DataLoader(
    dataset = test_data,
    batch_size = batch_size,
    shuffle = True
)

num_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class SmokeNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SmokeNetwork, self).__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Sigmoid(),
            nn.Linear(16, num_classes),
        )


    def forward(self, x):
        logits = self.linear_relu(x)
        return logits


model = SmokeNetwork().to(device)
model.train()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_list = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            loss_list.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss_list


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


epochs = 40
test_loss_list = []
train_loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss_list.append(
        train(train_dataloader, model, loss_fn, optimizer))
    test_loss_list.append(test(test_dataloader, model))
print("Done!")

# loss function curve
flat_train_loss_list = [item for sublist in train_loss_list for item in sublist]
plt.plot(np.linspace(0, len(flat_train_loss_list), len(flat_train_loss_list)), np.array(flat_train_loss_list))
plt.ylabel('train loss function value')
plt.show()

plt.plot(test_loss_list)
plt.ylabel('test loss function value')
plt.show()
# %%
