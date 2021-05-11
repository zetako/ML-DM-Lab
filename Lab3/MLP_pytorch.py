import torch
import torch.nn.functional as func
import numpy as np

# Hyperparameter
NUM_EPOCHS = 5000
DEVICE = 'cuda:0'
USE_CUDA = False
DISPLAY = 1000

# MLP class
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(32*32*3, 16*16*3)
        self.l2 = torch.nn.Linear(8*8*3, 4*4*3)
        self.l3 = torch.nn.Linear(4*4*3, 10)

    def forward(self, x):
        x = func.relu(l1(x))
        x = func.relu(l2(x))
        x = func.softmax(l3(x), dim = 1)
        return x

# Dataset
train_images = np.load("cifar10_train_images.npy")
train_labels = np.load("cifar10_train_labels.npy")
train_size = train_images.shape[0]
test_images = np.load("cifar10_test_images.npy")
test_labels = np.load("cifar10_test_labels.npy")
test_size = test_images.shape[0]


# Linear Model
MLP_model = MLP()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MLP.parameters())

# Dataset to Pytorch
train_images = torch.from_numpy(train_images.astype(np.float32))
train_labels = torch.from_numpy(train_labels)

# Training
if USE_CUDA:
    MLP_model = MLP_model.to(DEVICE)

for epoch in range(NUM_EPOCHS):
    # data this time
    x = train_images.reshape(-1, 32*32*3)
    label = train_labels

    # transfer if use CUDA
    if USE_CUDA:
        x = x.to(DEVICE)
        label = label.to(DEVICE)

    # cal
    optimizer.zero_grad()
    pred = MLP_model(x)
    loss = loss_func(pred, label)
    loss.backward()
    optimizer.step()

    # print
    if DISPLAY and epoch % DISPLAY == 0:
        print("epoch={}/{},loss={:.2%}".format(epoch,NUM_EPOCHS,loss))