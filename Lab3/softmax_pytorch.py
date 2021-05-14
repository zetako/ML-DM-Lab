import torch
import numpy as np
import torch.utils.data as data
import time

# Hyperparameter
NUM_EPOCHS = 1000
BATCH = 100
DEVICE = 'cuda:0'
USE_CUDA = False
DISPLAY = 10
LEARN_RATE = 0.05
MOMENTUM = 0.9
OPTIM_TYPE = 'SGD'

class LC(torch.nn.Module):
    def __init__(self):
        super(LC, self).__init__()
        self.linear = torch.nn.Linear(3*32*32, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 3*32*32)
        x = self.linear(x)
        return x

# Dataset
train_images = np.load("cifar10_train_images.npy")
train_labels = np.load("cifar10_train_labels.npy")
train_size = train_images.shape[0]
test_images = np.load("cifar10_test_images.npy")
test_labels = np.load("cifar10_test_labels.npy")
test_size = test_images.shape[0]

# Linear Model
linear_model = LC()
loss_func = torch.nn.CrossEntropyLoss()
if OPTIM_TYPE == 'SGD':
    optimizer = torch.optim.SGD(linear_model.parameters(), lr = LEARN_RATE)
elif OPTIM_TYPE == 'SGDM':
    optimizer = torch.optim.SGD(linear_model.parameters(), lr = LEARN_RATE, momentum = MOMENTUM)
else:
    optimizer = torch.optim.Adam(linear_model.parameters())
    
# Dataset to Pytorch
train_images = torch.from_numpy(train_images.astype(np.float32)).permute(0, 3, 1, 2) / 255
train_labels = torch.from_numpy(train_labels)
train_dataSet = data.TensorDataset(train_images, train_labels)
train_dataLoader = data.DataLoader(train_dataSet, batch_size = BATCH, shuffle = True)
test_images = torch.from_numpy(test_images.astype(np.float32)).permute(0, 3, 1, 2) / 255
test_labels = torch.from_numpy(test_labels)

# Training
if USE_CUDA:
    linear_model = linear_model.to(DEVICE)
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
        pred = linear_model(image)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    epoch_dur = (end_time - start_time) * 1000
        
    test_pred = linear_model(test_images)
    test_pred = torch.max(test_pred, 1)[1]
    correct = test_pred.eq(test_labels).sum().item()
    correct = correct / test_size
    correct_history.append(correct)

    # print
    if DISPLAY and epoch % DISPLAY == 0:
        print("epoch={}/{}, loss={:g}, correct={:.4%}, time used={:.4f}ms".format(epoch, NUM_EPOCHS, loss, correct, epoch_dur))

correct_history = np.array(correct_history)
np.save(OPTIM_TYPE+'.npy', correct_history)

### Running Code




