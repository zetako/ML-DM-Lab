import torch
import torch.nn.functional as func
import torch.utils.data as data
import numpy as np
import time

# Hyperparameter
NUM_EPOCHS = 100
DEVICE = 'cuda:0'
USE_CUDA = True
DISPLAY = 1
BATCH = 100
LEARN_RATE = 0.05
MLP_LAYER = 'normal'

# MLP class
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(32*32*3, 16*16*3)
        self.l2 = torch.nn.Linear(16*16*3, 8*8*3)
        self.l3 = torch.nn.Linear(8*8*3, 4*4*3)
        self.l4 = torch.nn.Linear(4*4*3, 10)

    def forward(self, x):
        x = x.reshape(-1, 32*32*3)
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = func.relu(self.l3(x))
        x = func.softmax(self.l4(x), dim = 1)
        return x

# MLP class with more layer
class MLP_PP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(32*32*3, 32*16*3)
        self.l2 = torch.nn.Linear(32*16*3, 16*16*3)
        self.l3 = torch.nn.Linear(16*16*3, 16*8*3)
        self.l4 = torch.nn.Linear(16*8*3, 8*8*3)
        self.l5 = torch.nn.Linear(8*8*3, 8*4*3)
        self.l6 = torch.nn.Linear(8*4*3, 4*4*3)
        self.l7 = torch.nn.Linear(4*4*3, 10)

    def forward(self, x):
        x = x.reshape(-1, 32*32*3)
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = func.relu(self.l3(x))
        x = func.relu(self.l4(x))
        x = func.relu(self.l5(x))
        x = func.relu(self.l6(x))
        x = func.softmax(self.l7(x), dim = 1)
        return x


# Dataset
train_images = np.load("cifar10_train_images.npy")
train_labels = np.load("cifar10_train_labels.npy")
train_size = train_images.shape[0]
test_images = np.load("cifar10_test_images.npy")
test_labels = np.load("cifar10_test_labels.npy")
test_size = test_images.shape[0]


# Linear Model
if MLP_LAYER == 'plus':
    MLP_model = MLP_PP()
else:
    MLP_model = MLP()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MLP_model.parameters(), lr = LEARN_RATE)

# Dataset to Pytorch
train_images = torch.from_numpy(train_images.astype(np.float32)).permute(0, 3, 1, 2) / 255
train_labels = torch.from_numpy(train_labels)
train_dataSet = data.TensorDataset(train_images, train_labels)
train_dataLoader = data.DataLoader(train_dataSet, batch_size = BATCH, shuffle = True)
test_images = torch.from_numpy(test_images.astype(np.float32)).permute(0, 3, 1, 2) / 255
test_labels = torch.from_numpy(test_labels)

# Training
if USE_CUDA:
    MLP_model = MLP_model.to(DEVICE)
    test_images = test_images.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

correct_history = []
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    for index, (image, label) in enumerate(train_dataLoader):
        if USE_CUDA:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
        # cal
        optimizer.zero_grad()
        pred = MLP_model(image)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        
    test_pred = MLP_model(test_images)
    test_pred = torch.max(test_pred, 1)[1]
    correct = test_pred.eq(test_labels).sum().item()
    correct = correct / test_size
    correct_history.append(correct)

    # print
    if DISPLAY and epoch % DISPLAY == 0:
        print("epoch={}/{}, loss={:g}, correct={:.4%}, time used={:.4f}ms".format(epoch, NUM_EPOCHS, loss, correct, epoch_dur))

correct_history = np.array(correct_history)
np.save(MLP_LAYER+'.npy', correct_history)