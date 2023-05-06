from sqlalchemy import false
from torch import nn
import torch
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import scipy.io
from model_set import *

# 权值参数所占空间
Pcs =0
#输出尺寸
Oc =0 
# 中间结果所占空间
Ocs = 0
# 总计算量
Cms = 0
batchsize_Ocs = 0
# 权值参数占用空间
def init_nums():
    global Oc
    global Pcs
    global Cms
    global Ocs
    global batchsize_Ocs
    Oc =0
    Pcs =0
    Ocs = 0
    Cms = 0
    batchsize_Ocs = 0
# 权值参数占用空间
def Pc_compute(channelin_pc,channelout_pc,kernel_size_pc,V1 = 4):
    Pc = V1*(channelin_pc*kernel_size_pc*kernel_size_pc*channelout_pc+channelout_pc)*4
    return Pc    

# 网络设置
class Net(nn.Module):
    
    def __init__(self, channel, nfeats,kernel_size,stride,padding):
        """
        Args:
            channel (int): Number of channels in the input image
            nfeats (int): Number of channels produced by the convolution
            kernel_size (int):  Size of the convolving kernel
            stride (int): Stride of the convolution. 
            padding (int): Padding added to all four sides of
            the input.
        Example:
            model =  Net(3, 48,4,2,1)
        """
        global Pcs
        global Cms
        global Ocs
        global batchsize_Ocs
        nums = 1
        super(Net, self).__init__()
        #------------------1
        channelin = channel
        channelout = nfeats
        self.conv1 = nn.Conv2d(channel          , nfeats       , kernel_size   , stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(nfeats * 2)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------2
        channelin = nfeats
        channelout = nfeats * 2
        self.conv2 = nn.Conv2d(nfeats             , nfeats * 2  , kernel_size    , stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------3
        channelin = nfeats * 2    
        channelout = nfeats * 4
        self.conv3 = nn.Conv2d(nfeats * 2         , nfeats * 4   , kernel_size   , stride, padding, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------4
        channelin = nfeats * 4
        channelout =  nfeats * 8 
        self.conv4 = nn.Conv2d(nfeats * 4         , nfeats * 8   , kernel_size    , stride, padding, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 8)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------5
        channelin =  nfeats * 8 
        channelout = nfeats * 4
        self.conv5 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4 , kernel_size   , stride, padding, bias=False)
        self.bn5 = nn.BatchNorm2d(nfeats * 4)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------6
        channelin = nfeats * 4
        channelout = nfeats * 2
        self.conv6 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2 , kernel_size   , stride, padding, bias=False)
        self.bn6 = nn.BatchNorm2d(nfeats * 2)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------7
        channelin = nfeats * 2
        channelout = nfeats
        self.conv7 = nn.ConvTranspose2d(nfeats * 2, nfeats     , kernel_size   , stride, padding, bias=False)
        self.bn7 = nn.BatchNorm2d(nfeats)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------8
        channelin = nfeats
        channelout = 1
        self.conv8 = nn.ConvTranspose2d(nfeats, 1              , kernel_size    , stride, padding, bias=False)
        self.bn8 = nn.BatchNorm2d(1)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        # print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.bn8(self.conv8(x)), 0.2)
        x = torch.sigmoid(x)
        return x

#输出尺寸
def Oc_compute(I_Oc,kernel_size_Oc,pad_Oc,stride_Oc):
    global Oc
    Oc = (I_Oc-kernel_size_Oc+2*pad_Oc)/stride_Oc + 1
    # print("Oc:",Oc)

# 每层样本中间结果占用空间
def Ocs_compute(batchsize_Ocs2,channelout_Ocs):
    global Oc
    global batchsize_Ocs
    # print("Oc:",Oc)
    ans = 2*Oc*Oc*channelout_Ocs*4
    # 给batchsize赋值
    batchsize_Ocs = batchsize_Ocs2
    # print("Ocs:",ans)
    return ans

# 总计算占用显存
def compute_memory(channel_in,channel_out,outM_Cm,kernel_size,V2=2):
    Cm = V2*2*outM_Cm*outM_Cm*kernel_size*kernel_size*channel_in*channel_out
    # print("Cm: ",Cm)
    return Cm

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):

        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# 网络设置  

# 模型设置
def setmodel(model):
    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.5
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    return optimizer,scheduler

def Get_Memory_Efficiency():
    global Pcs
    global Ocs
    global Cms
    memory_efficiency= Cms/(Ocs+Pcs)
    return Ocs,Pcs
