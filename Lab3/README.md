###### Cifar10数据集格式

- `cifar10_train_images.npy`：训练集包含5万张 32*32的彩色图片。共10个类别，每个类别5000张图片。
- `cifar10_train_labels.npy`：训练集标签，取值范围为$[0,9]$。
- `cifar10_test_images.npy`：测试集包含1万张 32*32的彩色图片。共10个类别，每个类别1000张图片。
- `cifar10_test_labels.npy`：测试集标签，取值范围为$[0,9]$。

###### 读取方法实例

```python
import numpy as np 

train_images = np.load("cifar10_train_images.npy")
train_labels = np.load("cifar10_train_labels.npy")
test_images = np.load("cifar10_test_images.npy")
test_labels = np.load("cifar10_test_labels.npy")

print(train_images.shape) #(50000, 32, 32, 3)
print(train_labels.shape) #(50000, )
print(test_images.shape)  #(10000, 32, 32, 3)
print(test_labels.shape)  #(10000, )

```

