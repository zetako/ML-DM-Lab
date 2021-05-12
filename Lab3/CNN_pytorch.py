import torch
import torch.nn.functional as func
import torch.utils.data as data
import numpy as np

### Hyperparameter
NUM_EPOCHS = 100
DEVICE = 'cuda:0'
USE_CUDA = False
DISPLAY = 1
BATCH = 100
LEARN_RATE = 0.05
CNN_TYPE = 'LeNet'

### CNN class
# Share Function: expansion of data
def expansion(x):
    dims = x.size()[1:]
    ret = 1
    for d in dims:
        ret *= d
    return ret

# Original LeNet
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.l1 = torch.nn.Linear(16*5*5, 120)
        self.l2 = torch.nn.Linear(120, 84)
        self.l3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(func.relu(x))
        x = self.conv2(x)
        x = self.pool(func.relu(x))
        x = x.reshape(-1, expansion(x))
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = func.softmax(self.l3(x), dim = 1)
        return x

# LeNet with add conv 
class LeNet_Add_Conv(torch.nn.Module):
    def __init__(self):
        super(LeNet_Add_Conv, self).__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 8, 5)
        self.conv3 = torch.nn.Conv2d(8, 16, 5)
        self.l1 = torch.nn.Linear(16*5*5, 120)
        self.l2 = torch.nn.Linear(120, 84)
        self.l3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(func.relu(x))
        x = self.conv2(x)
        x = self.pool(func.relu(x))
        x = self.conv3(x)
        x = self.pool(func.relu(x))
        x = x.reshape(-1, expansion(x))
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = func.softmax(self.l3(x), dim = 1)
        return x
    
    def expansion(x):
        dims = x.size()[1:]
        ret = 1
        for d in dims:
            ret *= d
        return ret

# LeNet with avg_pool
class LeNet_Avg_Pool(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.l1 = torch.nn.Linear(16*5*5, 120)
        self.l2 = torch.nn.Linear(120, 84)
        self.l3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(func.relu(x))
        x = self.conv2(x)
        x = self.pool(func.relu(x))
        x = x.reshape(-1, expansion(x))
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = func.softmax(self.l3(x), dim = 1)
        return x

# LeNet with larger filter
class LeNet_Large_Filter(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(3, 12, 5)
        self.conv2 = torch.nn.Conv2d(12, 32, 5)
        self.l1 = torch.nn.Linear(32*5*5, 120)
        self.l2 = torch.nn.Linear(120, 84)
        self.l3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(func.relu(x))
        x = self.conv2(x)
        x = self.pool(func.relu(x))
        x = x.reshape(-1, expansion(x))
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = func.softmax(self.l3(x), dim = 1)
        return x

### Running Code
# Dataset
train_images = np.load("cifar10_train_images.npy")
train_labels = np.load("cifar10_train_labels.npy")
train_size = train_images.shape[0]
test_images = np.load("cifar10_test_images.npy")
test_labels = np.load("cifar10_test_labels.npy")
test_size = test_images.shape[0]

# Linear Model
CNN_model = eval(CNN_TYPE+'()')
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(CNN_model.parameters(), lr = LEARN_RATE)

# Dataset to Pytorch
train_images = torch.from_numpy(train_images.astype(np.float32)).permute(0, 3, 1, 2) / 255
train_labels = torch.from_numpy(train_labels)
train_dataSet = data.TensorDataset(train_images, train_labels)
train_dataLoader = data.DataLoader(train_dataSet, batch_size = BATCH, shuffle = True)
test_images = torch.from_numpy(test_images.astype(np.float32)).permute(0, 3, 1, 2) / 255
test_labels = torch.from_numpy(test_labels)

# Training
if USE_CUDA:
    CNN_model = CNN_model.to(DEVICE)
    test_images = test_images.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

for epoch in range(NUM_EPOCHS):
    for index, (image, label) in enumerate(train_dataLoader):
        if USE_CUDA:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
        # cal
        optimizer.zero_grad()
        pred = CNN_model(image)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        
    test_pred = CNN_model(test_images)
    test_pred = torch.max(test_pred, 1)[1]
    correct = test_pred.eq(test_labels).sum().item()
    correct = correct / test_size

    # print
    if DISPLAY and epoch % DISPLAY == 0:
        print("epoch={}/{}, loss={:g}, correct={:.4%}".format(epoch, NUM_EPOCHS, loss, correct))