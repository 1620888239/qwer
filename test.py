import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# root是存储数据的文件夹，download=True指定如果数据不存在先下载数据
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)


trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=True)


class LeNet(nn.Module):

    # 在__init__中定义网络需要的操作算子
    def __init__(self):
        super(LeNet, self).__init__()

        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)

        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    # 前向传播
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # 把二维特征图变为一维
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0")
net = LeNet().to(device)

import torch.optim as optim

# CrossEntropyLoss 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Start Training...")
for epoch in range(50):
    loss1 = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss1 += loss.item()
        print('[Epoch %d, Batch %5d] loss: %.3f' %(epoch + 1, i + 1, loss1))
        loss1 = 0.0

print("Finish Training!")

# 构造测试的dataloader
dataiter = iter(testloader)

# 预测正确的数量和总数量
correct = 0
total = 0

# 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 预测
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('\nAccuracy: %d %%' % (
    100 * correct / total))
