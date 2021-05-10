import torch
import numpy as np

# Hyperparameter
NUM_EPOCHS = 5000
DEVICE = 'cuda:0'
USE_CUDA = False
DISPLAY = 1000

# Dataset
train_images = np.load("cifar10_train_images.npy")
train_labels = np.load("cifar10_train_labels.npy")
train_size = train_images.shape[0]
test_images = np.load("cifar10_test_images.npy")
test_labels = np.load("cifar10_test_labels.npy")
test_size = test_images.shape[0]


# Linear Model
linear_model = torch.nn.Linear(32*32*3, 10)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_model.parameters())

# Dataset to Pytorch
train_images = torch.from_numpy(train_images.astype(np.float32))
train_labels = torch.from_numpy(train_labels)

# Training
if USE_CUDA:
    linear_model = linear_model.to(DEVICE)

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
    pred = linear_model(x)
    loss = loss_func(pred, label)
    loss.backward()
    optimizer.step()

    # print
    if DISPLAY and epoch % DISPLAY == 0:
        print("epoch={}:idx={},loss={:g}".format(epoch,NUM_EPOCHS,loss))
