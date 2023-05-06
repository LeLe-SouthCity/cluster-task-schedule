import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import h5py
from timeit import default_timer
import scipy.io
# from torchsummary import summary
from model_set import *
import argparse

parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
parser.add_argument("--gpus",  type=int,default=2, help="the GPUs need.")
parser.add_argument("--batchsize",  type=int,default=256, help="the batchsize.")
parser.add_argument("--device_num",  type=int,default=0, help="the batchsize.")
parser.add_argument("--type",  type=int,default=1, help="the task type.")
parser.add_argument("--usingtype",  type=int,default=1, help="usingtype.")
parser.add_argument("--epochs",  type=int,default=200, help="epochs nums.")

args = parser.parse_args()

TRAIN_PATH = 'TrainData.mat'

# 通道数 = 卷积核  = N
channel = 3
# 移动步长
stride = 2
#填充数目
padding = 1
# 卷积核大小 4*4
kernel_size = 4
#卷积核数量nfeats = 48
nfeats = 48
#输入尺寸--固定
I = 96

start = default_timer()
runtime = np.zeros(2, )
t1 = default_timer()

################################################################
# load data
################################################################
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('Tlist').permute(3, 2, 0, 1)
train_u = reader.read_field('Qlist').permute(3, 2, 0, 1)

train_dataset = torch.utils.data.TensorDataset(train_a, train_u)
train_size = int(1 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset,test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

t2 = default_timer()
print('preprocessing finished, time used:', t2 - t1)
################################################################
# training and evaluation
###############################################################
from fifo_scheclue import *
# 修改gpus的值即可
print("args.gpus",args.gpus)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize)
model,Memory_need = Set_Model(I=I,kernel_size=kernel_size,padding=padding,stride=stride,batch_size=args.batchsize,channel=channel)
print("Memory_need",Memory_need)
# gpus = [1]
# if args.gpus == 2:
#     gpus = [0,1]
# else:
#     gpus = [0]
# batchsize = args.batchsize
# 训练---------------------------------------------------------------------
# 第一步优化
batchsize,gpus = fifo_schedule_Type1(Memory_need = Memory_need,GPU_need=args.gpus)

# 第二步优化
# batchsize,gpus = fifo_schedule_Type2(Memory_need = Memory_need,GPU_need=args.gpus,Task_type=args.type)

#测试优化greedy-多gpu Altruistic-少gpu
# batchsize,gpus = greedyOrAltruistic(Memory_need= Memory_need,usingtype =args.usingtype )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize,num_workers=1)
model,Memory_need = Set_Model(I=I,kernel_size=kernel_size,padding=padding,stride=stride,batch_size=batchsize,channel=channel)
device = torch.cuda.set_device('cuda:{}'.format(gpus[args.device_num]))
model = nn.DataParallel(model.to(device), device_ids=gpus)
#-------------设置模型参数-传出模型
optimizer,scheduler= setmodel(model) 

for ep in range(args.epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)
        # gpu_tracker.track()                     # run function between the code line where uses GPU 追踪显存使用
        mse = F.mse_loss(out, y, reduction='mean')
        mse.backward()

        optimizer.step()
        train_mse += mse.item()

    scheduler.step()
    model.eval()

    with torch.no_grad():
        t2 = default_timer()
        print("222222222222222222-Gpu-zhongda-------Epoch:", ep, "; Elapsed time:", t2 - t1, "; Traning Loss: ", train_mse)

end = default_timer()        
print("Time1:",end-start)
################################################################
# Save model
################################################################
torch.save(model.state_dict(), "model.pt")
