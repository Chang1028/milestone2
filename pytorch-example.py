# %%
# load package
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from torch import nn

# %%
# load data
local_folder = '~/Dropbox/Teaching/STAT5630/STAT5630_2021/Lectures/NeuralNetwork/example-code/datasets'
# new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
# datasets.MNIST.resources = [
#    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
#    for url, md5 in datasets.MNIST.resources
# ]
training_data = datasets.MNIST(
    root=local_folder,
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root=local_folder,
    train=False,
    download=True,
    transform=ToTensor()
)

training_one = training_data[100]
training_one = training_one[0]
input_size = training_one.shape[0] * training_one.shape[1] * training_one.shape[2]

# %%
# algorithm optimization
batch_size = 64
train_dataloader = DataLoader(
    dataset=training_data, batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)
# %%
# define model
num_classes = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # change image to a vector
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 50), 
            nn.ReLU(),
            nn.Linear(50, 50), 
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
# model.eval()
# %%
# Optimizing the Model Parameters
model = NeuralNetwork().to(device)
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss
# %%
epochs = 20 
test_loss_list = []
train_loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss_list.append(
        train(train_dataloader, model, loss_fn, optimizer))
    test_loss_list.append(test(test_dataloader, model)) 
print("Done!")
# %%
# loss function curve
flat_train_loss_list = [item for sublist in train_loss_list for item in sublist]
plt.plot(np.linspace(0, len(flat_train_loss_list), len(flat_train_loss_list)), np.array(flat_train_loss_list))
plt.ylabel('train loss function value')
plt.show()

plt.plot(test_loss_list)
plt.ylabel('test loss function value')
plt.show()
# %%
