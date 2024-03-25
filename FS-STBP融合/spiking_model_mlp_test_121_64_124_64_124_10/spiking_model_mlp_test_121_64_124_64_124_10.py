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
#cfg_fc = [400,300,10]
cfg_fc = [64, 124, 64, 124, 10]
compress_size = 11

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

        self.fc1 = nn.Linear(121, cfg_fc[0])    # 121 64
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])  # 64 124
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])  # 124 64
        self.fc4 = nn.Linear(cfg_fc[2], cfg_fc[3])  # 64 124
        self.fc5 = nn.Linear(cfg_fc[3], cfg_fc[4])  # 124 10

    def forward(self, input, time_window=4):
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)  #[100,400]
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)
        h4_mem = h4_spike = h4_sumspike = torch.zeros(batch_size, cfg_fc[3], device=device)
        h5_mem = h5_spike = h5_sumspike = torch.zeros(batch_size, cfg_fc[4], device=device)

        for step in range(time_window):  # simulation time steps
            #x = input > torch.rand(input.size(), device=device)  # prob. firing  [100,1,28,28]
            #x = x.view(batch_size, -1) #[100,784]
            x = input
            x = x.view(-1, compress_size * compress_size)
            h1_mem, h1_spike = mem_update(self.fc1, x.float(), h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike
            h4_mem, h4_spike = mem_update(self.fc4, h3_spike, h4_mem, h4_spike)
            h4_sumspike += h4_spike
            h5_mem, h5_spike = mem_update(self.fc5, h4_spike, h5_mem, h5_spike)
            h5_sumspike += h5_spike


        s1 = torch.sum(h1_sumspike)/(64*100)
        s2 = torch.sum(h2_sumspike)/(124*100)
        s3 = torch.sum(h3_sumspike)/(64*100)
        s4 = torch.sum(h4_sumspike) / (124 * 100)
        s5 = torch.sum(h5_sumspike) / (10 * 100)

        outputs = h5_sumspike / time_window
        return outputs, s1, s2, s3, s4, s5
