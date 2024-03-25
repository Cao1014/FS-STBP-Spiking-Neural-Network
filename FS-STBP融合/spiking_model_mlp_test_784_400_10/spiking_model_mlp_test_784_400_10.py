import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.2  # decay constants LIF中的leaky项

num_classes = 10
batch_size = 100
learning_rate = 1e-3
num_epochs = 80  # max epoch


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # ctx：类似于self，这里保存下来input供反传用
        ctx.save_for_backward(input)
        return input.gt(thresh).float()  # input：膜电位，gt比较是否超过阈值thresh，return 脉冲0/1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens  # 膜电压在门限一定范围(lens)内，temp置1，该时间步的反传梯度为1
        return grad_input * temp.float()


act_fun = ActFun.apply  # .apply调用自定义的t.autograd.Function


# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike     #spike为0/1

'''
# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3), ]
# kernel size
cfg_kernel = [28, 14, 7]
'''
# fc layer
# cfg_fc = [400,300,10]
cfg_fc = [400, 10]
# Test Accuracy of the model on the 10000 test images: 97.060
# Epoch [76/80], Step [100/600], Loss: 0.13000
# Time elasped: 1.2581334114074707
# Epoch [76/80], Step [200/600], Loss: 0.13800
# Time elasped: 2.273649215698242
# Epoch [76/80], Step [300/600], Loss: 0.14200
# Time elasped: 3.2735540866851807
# Epoch [76/80], Step [400/600], Loss: 0.12900
# Time elasped: 4.257814407348633
# Epoch [76/80], Step [500/600], Loss: 0.15500
# Time elasped: 5.271173715591431
# Epoch [76/80], Step [600/600], Loss: 0.13800
# Time elasped: 6.271069526672363
# 0 100  Acc: 99.00000
# s1:0.07240 s2:0.09900
# Iters: 75


# Decay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        self.fc1 = nn.Linear(784, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        # self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

    def forward(self, input, time_window=4):
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)  #[100,400]
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        # h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)

        for step in range(time_window):  # simulation time steps
            x = input > torch.rand(input.size(), device=device)  # prob. firing  [100,1,28,28]
            x = x.view(batch_size, -1) #[100,784]
            h1_mem, h1_spike = mem_update(self.fc1, x.float(), h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            # h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            # h3_sumspike += h3_spike

        s1=torch.sum(h1_sumspike)/(400*100)
        s2=torch.sum(h2_sumspike)/(10*100)
        # s3=torch.sum(h3_sumspike)/(10*100)

        outputs = h2_sumspike / time_window
        return outputs,s1,s2#,s3
