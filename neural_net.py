import torch.nn as nn
import torch.nn.functional as F
import torch


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # convolutional layers
        # self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        # self.conv2 = nn.Conv2d(8, 24, 3, padding=1)
        # Increasing
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 48, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(48 * 40 * 30, 4096)
        self.fc2 = nn.Linear(4096, 512)
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


def testNNModel(file, valid_loader):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    torch.cuda.set_device(dev)

    model = torch.load(file)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images, labels = images.to(dev), labels.to(dev)
            out = model(images)
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return print('Testing accuracy: {} %'.format(100 * correct / total))
