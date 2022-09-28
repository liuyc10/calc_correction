# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pathlib

import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Model

from matplotlib import pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def training():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device(type='cuda', index=0)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_train = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root="./data/", transform=transform, train=False)

    data_loader_train = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
    data_loader_test = DataLoader(dataset=data_test, batch_size=64, shuffle=True)

    model = Model()
    model.to(device)

    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 5
    if pathlib.Path('model_parameter/model_parameter.pkl').exists():
        model.load_state_dict(torch.load('model_parameter/model_parameter.pkl'))

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)
        for data in data_loader_train:
            X_train, y_train = data
            X_train, y_train = Variable(X_train.cuda()), Variable(y_train.cuda())
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += torch.sum(pred == y_train.data)
        testing_correct = 0
        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test.cuda()), Variable(y_test.cuda())
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)
        print(
            "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                        100 * running_correct / len(
                                                                                            data_train),
                                                                                        100 * testing_correct / len(
                                                                                            data_test)))
    torch.save(model.state_dict(), "model_parameter/model_parameter.pkl")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    data_test = datasets.MNIST(root="./data/", transform=transform, train=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=4,
                                                   shuffle=True)

    model = Model()
    model.to(device)

    if pathlib.Path('model_parameter/model_parameter.pkl').exists():
        model.load_state_dict(torch.load('model_parameter/model_parameter.pkl'))

    X_test, y_test = next(iter(data_loader_test))
    inputs = Variable(X_test.cuda())
    pred = model(inputs)
    _, pred = torch.max(pred, 1)

    print("Predict Label is:", [i for i in pred.data])
    print("Real Label is:", [i for i in y_test])

    img = torchvision.utils.make_grid(X_test)
    img = img.numpy().transpose(1, 2, 0)

    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    plt.imshow(img)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
