import PIL
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import torch.nn.functional as F


def testNNModel(file, valid_loader):
    mymodel = torch.load(file)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            #images = images.view(images.size(0), -1)
            print("do smth")
            out = mymodel(images)
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Testing accuracy: {} %'.format(100 * correct / total))
    return "done testing"


def get_train_and_validation_data_loader(data_path="images", validation_split_ratio=0.1, seed=42):
    data = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize((160, 120)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split_ratio * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        data,
        num_workers=4,
        batch_size=1,
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        data,
        num_workers=4,
        batch_size=1,
        sampler=valid_sampler
    )
    return train_loader, valid_loader


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 24, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(24 * 40 * 30, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Define relu activation and LogSoftmax output
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flattening the image
        x = x.view(x.size(0), -1)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.LogSoftmax(self.fc4(x))
        return x


if __name__ == '__main__':
    train_loader, valid_loader = get_train_and_validation_data_loader("images", 0.1, 33)
    print("Created Datasets.")
    #testNNModel("nn_firsttry.pt", valid_loader)
    model = NeuralNet()
    lossFunction = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss_ = 0
        count = 0
        for images, labels in train_loader:
            count += 1
            # images = images.reshape(-1, 19200)
            print(count)
            # Forward Pass
            output = model(images)
            # Loss at each iteration by comparing to target(label)
            loss = lossFunction(output, labels)
            # Backpropogating gradient of loss
            optimizer.zero_grad()
            loss.backward()

            # Updating parameters(weights and bias)
            optimizer.step()

            loss_ += loss.item()

        print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(train_loader)))

    torch.save(model, 'nn_firsttry.pt')


