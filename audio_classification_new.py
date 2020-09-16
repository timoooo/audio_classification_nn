import torch
import torch.nn as nn
from torch import optim

import data_provider
import neural_net

if __name__ == '__main__':
    seed = 33
    test_train_ratio = 0.1
    data_path = "images"
    #checks if cuda is available   if so it is used
    print("Cuda availability: "+str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("Using GPU")
        dev = "cuda:0"
    else:
        print("Using CPU")
        dev = "cpu"

    torch.cuda.set_device(dev)
    print('Current cuda device ', torch.cuda.current_device())

    #get train and valid loaders used to train and validate
    train_loader, valid_loader = data_provider.get_train_and_validation_data_loader(data_path, test_train_ratio, seed)
    print("Created Datasets.")
    #init model
    model = neural_net.NeuralNet()
    model.cuda()
    model.to(dev)
    # init start params
    lossFunction = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 30
    for epoch in range(num_epochs):
        loss_ = 0
        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
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

    torch.save(model, 'nn_v6.pt')
    print("Testing network model.")
    neural_net.testNNModel("nn_v6.pt", valid_loader)
