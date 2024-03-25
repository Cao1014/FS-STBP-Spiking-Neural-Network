import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda, Compose, Resize
import matplotlib.pyplot as plt
import time
import numpy as np


# Get cpu or gpu device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {device} ", device)

# 网络结构：
cfg_fc = [121, 64, 124, 64, 124, 10]

# 数据集：len(trainset)=60000，len(testset)=10000
data_path = './raw/'  # todo: input your data path
batch_size = 100
compress_size = 11


train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=Compose([Resize(compress_size), ToTensor()])) #裁剪为11*11
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=Compose([Resize(compress_size), ToTensor()])) #裁剪为11*11
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)




# The FS-conversion weights
# x:加权和(激活值), h:spike后膜电位的衰减，d:模拟激活函数, T:阈值, K:窗, z: spike
# 传入当前step的h、T用于更新mem

K = [4, 3, 3, 3, 4]
alpha = [1, 1, 1, 1, 1]  # 根据原先网路中relu层输出最大值来设置

# K = [4, 4, 4, 3, 2]
# alpha = [1, 1, 1, 1, 1]  # 根据原先网路中relu层输出最大值来设置

d1 = H1 = T1 = alpha[0] * 2 ** (-K[0]) * np.array([float(2 ** (K[0] - i)) for i in range(1, K[0] + 1)]).astype(np.float32)
d2 = H2 = T2 = alpha[1] * 2 ** (-K[1]) * np.array([float(2 ** (K[1] - i)) for i in range(1, K[1] + 1)]).astype(np.float32)
d3 = H3 = T3 = alpha[2] * 2 ** (-K[2]) * np.array([float(2 ** (K[2] - i)) for i in range(1, K[2] + 1)]).astype(np.float32)
d4 = H4 = T4 = alpha[3] * 2 ** (-K[3]) * np.array([float(2 ** (K[3] - i)) for i in range(1, K[3] + 1)]).astype(np.float32)
d5 = H5 = T5 = alpha[4] * 2 ** (-K[4]) * np.array([float(2 ** (K[4] - i)) for i in range(1, K[4] + 1)]).astype(np.float32)

lens = 0.5  # hyper-parameters of approximate function

num_epochs = 80
# num_updates = 1

learning_rate = 1e-3

bias = False



class ActFun(torch.autograd.Function):
    '''
    Approaximation function of spike firing rate function
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # ctx：类似于self，这里保存下来input供反传用
        return input.gt(0.).float()  # input：膜电位，gt比较是否超过阈值thresh，return 脉冲0/1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens# 膜电压在门限一定范围内，置1，发射脉冲，temp置1，该时间步的反传梯度为1
        return grad_input * temp.float()


act_fun = ActFun.apply


# fs-neuron 更新膜电位
def fs_mem_update(ops, x, mem, d):  # self.fc1, h0_spike, h1(zeros), d[step]
    mem = mem + ops(x * d)
    return mem                   # h1


# fs-neuron 编码
def fs_coding(mem, decay, th):   # h0, H[step], T[step]
    z = act_fun(mem - th) #* 0.5
    mem = mem - decay * z #* 2
    return mem, z              # h0, h0_spike




# Decay learning_rate

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer




class CNN_to_SNN(nn.Module):
    def __init__(self):
        super(CNN_to_SNN, self).__init__()

        self.fc1 = nn.Linear(cfg_fc[0], cfg_fc[1])          # 121 64
        self.fc2 = nn.Linear(cfg_fc[1], cfg_fc[2])          # 64  124
        self.fc3 = nn.Linear(cfg_fc[2], cfg_fc[3])          # 124 64
        self.fc4 = nn.Linear(cfg_fc[3], cfg_fc[4])          # 64  124
        self.fc5 = nn.Linear(cfg_fc[4], cfg_fc[5])          # 124 10

        self.bn1 = nn.BatchNorm1d(num_features=cfg_fc[1])   # 64
        self.bn2 = nn.BatchNorm1d(num_features=cfg_fc[2])   # 124
        self.bn3 = nn.BatchNorm1d(num_features=cfg_fc[3])   # 64
        self.bn4 = nn.BatchNorm1d(num_features=cfg_fc[4])   # 124

    def forward(self, input: torch.Tensor):
        # 初始化每层的膜电位、累计发放脉冲数

        # input = torch.flatten(input, 1)  # flatten all dimensions except the batch dimension     # 11*11=121
        input = input.view(-1, compress_size * compress_size)
        h0 = input  # > torch.rand(input.size(), device=device)

        h0_spikes = torch.zeros(batch_size, cfg_fc[0], device=device)  # 121
        h1_spikes = h1 = torch.zeros(batch_size, cfg_fc[1], device=device)  # 64
        h2_spikes = h2 = torch.zeros(batch_size, cfg_fc[2], device=device)  # 124
        h3_spikes = h3 = torch.zeros(batch_size, cfg_fc[3], device=device)  # 64
        h4_spikes = h4 = torch.zeros(batch_size, cfg_fc[4], device=device)  # 124
        h5_spikes = h5 = torch.zeros(batch_size, cfg_fc[5], device=device)  # 10

        # 前传
        for step in range(K[0]):
            # 输入值编码
            h0, h0_spike = fs_coding(h0, H1[step], T1[step])
            # 更新c1_mem
            h0_spikes += h0_spike
            h1 = fs_mem_update(self.fc1, h0_spike, h1, d1[step])
            # self.fc1.weight.data = self.fc1.weight.data * 0.5             #在weight上乘0.5会使SNN无法训练，因此改为0.5乘到act_fun后面

        for step in range(K[1]):
            # c1_mem编码
            h1, h1_spike = fs_coding(h1, H2[step], T2[step])
            h1_spikes += h1_spike
            # x = F.avg_pool2d(h1_spike, 2)
            # 更新c2_mem
            h2 = fs_mem_update(self.fc2, h1_spike, h2, d2[step])
            # self.fc2.weight.data = self.fc2.weight.data * 0.5
        for step in range(K[2]):
            # c2_mem编码
            h2, h2_spike = fs_coding(h2, H3[step], T3[step])
            h2_spikes += h2_spike
            # x = F.avg_pool2d(h2_spike, 2)
            # x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
            # 更新h1_mem
            h3 = fs_mem_update(self.fc3, h2_spike, h3, d3[step])
            # self.fc3.weight.data = self.fc3.weight.data * 0.5
        for step in range(K[3]):
            # h1_mem编码
            h3, h3_spike = fs_coding(h3, H4[step], T4[step])
            h3_spikes += h3_spike
            # 更新h2_mem
            h4 = fs_mem_update(self.fc4, h3_spike, h4, d4[step])
            # h22 = h2.cpu().numpy()
            # self.fc4.weight.data = self.fc4.weight.data * 0.5
        for step in range(K[4]):
            # h2_mem编码
            h4, h4_spike = fs_coding(h4, H5[step], T5[step])
            h4_spikes += h4_spike
            # 更新h3_mem
            h5 = fs_mem_update(self.fc5, h4_spike, h5, d5[step])
            # self.fc5.weight.data = self.fc5.weight.data * 0.5

        #统计该轮平均脉冲数
        sh1 = torch.sum(h0_spikes) / (cfg_fc[0] * batch_size)  # 121x100
        sh2 = torch.sum(h1_spikes) / (cfg_fc[1] * batch_size)  # 64x100
        sh3 = torch.sum(h2_spikes) / (cfg_fc[2] * batch_size)  # 124x100
        sh4 = torch.sum(h3_spikes) / (cfg_fc[3] * batch_size)  # 64x100
        sh5 = torch.sum(h4_spikes) / (cfg_fc[4] * batch_size)  # 124x100
        s_ave = (torch.sum(h0_spikes) + torch.sum(h1_spikes) + torch.sum(h2_spikes) + torch.sum(h3_spikes) + torch.sum(
            h4_spikes)) / (cfg_fc[0] * batch_size + cfg_fc[1] * batch_size +
                           cfg_fc[2] * batch_size + cfg_fc[3] * batch_size + cfg_fc[4] * batch_size)

        # 最后一层不发放脉冲，直接使用膜电压分类
        outputs = h5
        return outputs, sh1, sh2, sh3, sh4, sh5, s_ave


snn = CNN_to_SNN()
snn.to(device)



optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

acc_record = list([])
loss_train_record = list([])
epoch_list = []


# spi_record = list([])
# loss_test_record = list([])
total_best_acc = []
total_acc_record = []
total_hid_state = []

total = 0.
correct = 0.



for epoch in range(num_epochs):
    epoch_list.append(epoch)
    running_loss = 0.
    # snn.train()

    start_time = time.time()
    print('\n\n\n', 'Iters:', epoch)

    # 训练
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        images = images.float().to(device)

        # optim_base.zero_grad()
        optimizer.zero_grad()


        images = images.to(device)
        #images = transforms.CenterCrop(11)(images)  # 裁切为11*11

        outputs, sh1, sh2, sh3, sh4, sh5, s_average = snn(images)  # 这里改成求平均脉冲数
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)

        # loss_reg = torch.norm(eta1, p=2) + torch.norm(eta2, p=2)
        loss = criterion(outputs.cpu(), labels_)  # + lambdas * loss_reg.cpu()
        running_loss += loss.item()
        loss.backward()
        # optim_base.step()
        optimizer.step()  # 通过梯度下降执行一步参数更新
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f,Time elasped：%.5f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss,
                     time.time() - start_time))
            if (i + 1) == 600:
                loss_train_record.append(running_loss)
            running_loss = 0

    correct = 0.
    total = 0.

    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    # 测试
    with torch.no_grad():
        sh1_sum, sh2_sum, sh3_sum, sh4_sum, sh5_sum = 0, 0, 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs = inputs.to(device)
            #inputs = transforms.CenterCrop(11)(inputs)      # 裁切为11*11

            optimizer.zero_grad()
            outputs, sh1, sh2, sh3, sh4, sh5, s_average = snn(inputs)  # 这里改成求平均脉冲数
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            sh1_sum += sh1
            sh2_sum += sh2
            sh3_sum += sh3
            sh4_sum += sh4
            sh5_sum += sh5

            if (batch_idx + 1) % 50 == 0:
                print(batch_idx + 1, "sh1:%.5f, sh2:%.5f, sh3:%.5f, sh4:%.5f, sh5:%.5f" % (sh1, sh2, sh3, sh4, sh5))


    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)

    if acc > best_acc:  # 最佳准确率
        print('\n', 'Best test Accuracy of the snn on the 10000 test images: %.3f' % (acc))
        print("Average spikes:sc1_ave:%.5f, sc2_ave:%.5f, s1_ave:%.5f, s2_ave:%.5f, s3_ave:%.5f"
         % (sh1_sum / (batch_idx + 1), sh2_sum / (batch_idx + 1), sh3_sum / (batch_idx + 1),
            sh4_sum / (batch_idx + 1),
            sh5_sum / (batch_idx + 1)), '\n')
        best_acc = acc

# 绘图
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
plt.plot(epoch_list, acc_record, color='blue')
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlabel('epochs', size=18)
plt.ylabel('accuracy', size=18)
plt.title("SNN" + " (" + str(learning_rate) + ")", fontsize=18)

ax2 = fig.add_subplot(3, 1, 3)
plt.plot(epoch_list, loss_train_record, color='red')
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlabel('epochs', size=18)
plt.ylabel('loss during training', size=18)
plt.title("SNN" + " (" + str(learning_rate) + ")", fontsize=18)

plt.show()






