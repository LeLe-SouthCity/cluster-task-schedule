import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import h5py
from timeit import default_timer
import scipy.io
from getGPuInf import getNvidiaGPU
from model_set import *
import time


# 任务内存所需
Memory_need = 0
# 权值参数所占空间
Pcs =0
# 中间结果所占空间
Ocs = 0
batchsize = 0
# 获取可分配GPU 不在函数内运行 返回device元组
def fifo_schedule_Type1(Memory_need,GPU_need):
	global batchsize
	# 服务器GPU数目gups_nums
	gups_nums = torch.cuda.device_count()
	usedMemory,totalMemory,Memoryuse,gpu_use = 0,0,0,0
	# 可用GPu序号
	Gpu_can = []
	# 一直等待到可用GPU数大于1
	while (True):
		for i in range(gups_nums):
			usedMemory,totalMemory,Memoryuse,gpu_use = getNvidiaGPU(i)
			print("Memoryuse,gpu_use",Memoryuse,gpu_use)
			if (int(Memoryuse) <= 70 and Memory_need < totalMemory-usedMemory ):
				Gpu_can.append(i)
		print("len(Gpu_can)--------------------------------------------------",len(Gpu_can))
		if len(Gpu_can) > 0 :
			break
		else :
			print("显存不足")
			time.sleep(10)
			
	return	 batchsize,Gpu_can	

def fifo_schedule_Type2(Memory_need,GPU_need,Task_type): 
	global batchsize
	# 服务器GPU数目gups_nums
	gups_nums = torch.cuda.device_count()
	usedMemory,totalMemory,Memoryuse,gpu_use = 0,0,0,0
	#GPu中剩余可用显存
	Memory_device=[]
	# 可用GPu序号
	Gpu_can0 = []
	Gpu_can1 = []
	# GPU中最小可用显存
	Min_Memory = 1000000
	# 一直等待到可用GPU数大于1
	# A------------------------------------------------------------遍历服务器中可用的GPU
	
	while (True):
    	# 遍历可用的GPU
		for i in range(gups_nums):
			usedMemory,totalMemory,Memoryuse,gpu_use = getNvidiaGPU(i)
			# 内存添加
			Memory_device.append(totalMemory-usedMemory)
			print("Memoryuse,gpu_use",Memoryuse,gpu_use)
			if (int(Memoryuse) <= 80 and Memory_need <(totalMemory-usedMemory) ):
				Min_Memory = min(Min_Memory,totalMemory-usedMemory)
				Gpu_can1.append(i)
				if int( gpu_use) <= 80:
    					Gpu_can0.append(i)
		print("Min_Memory",Min_Memory)
		print("len(Gpu_can0)--------------------------------------------------",len(Gpu_can0))
		if len(Gpu_can1) > 0 :
			break
		else :
			print("显存不足")
			time.sleep(10)
	print("输出对应GPU剩余的内存Memory_device",Memory_device)

	# B---------------------------------------------------------判断任务类型区别任务类型 0:计算密集型 1：访存密集型
	device_return = []
	if Task_type==0:
			if len(Gpu_can0)>GPU_need:
				for Mem in Gpu_can0: 
					if Memory_need*1.2 < Memory_device[Mem]:
						device_return.append(Mem)
				return batchsize,device_return
	else:
		# if 任务占用显存 < GPu中可用显存:
		# for i in range(Memory_device):
		if Memory_need < Min_Memory:
			# 找到最合适的batchsize,看能否分配较少的GPU
			test_b = batchsize
			while Memory_need*1 < Min_Memory:
				test_b+=1
				Memory_need = Get_Memoryneed(test_b)
			batchsize = test_b
	
	for i in range(gups_nums): 
		if Memory_need < Memory_device[i]:
			device_return.append(i)
		if len(device_return) > 0:
			break
	
	print("batchsize,device_return",batchsize,device_return)
	return	 batchsize,device_return	

# 尽可能多的给任务分配gpu
def greedyOrAltruistic(Memory_need,usingtype):
	global batchsize
    	# 服务器GPU数目gups_nums
	gups_nums = torch.cuda.device_count()
	usedMemory,totalMemory,Memoryuse,gpu_use = 0,0,0,0
	#GPu中剩余可用显存
	Memory_device=[]
	# 可用GPu序号
	Gpu_can = []
	# 一直等待到可用GPU数大于1
	while (True):
		for i in range(gups_nums):
			usedMemory,totalMemory,Memoryuse,gpu_use = getNvidiaGPU(i)
			Memory_device.append(totalMemory-usedMemory)
			print("Memoryuse,gpu_use",Memoryuse,gpu_use)
			# and int(gpu_use) <= 70
			if (int(Memoryuse) <= 80 and Memory_need*1.1 < Memory_device[i]):
				Gpu_can.append(i)
		print("len(Gpu_can)--------------------------------------------------",len(Gpu_can))
		if len(Gpu_can) > 0 :
			break
		else :
			print("显存不足")
			time.sleep(10)
	print("Memory_device",Memory_device)
	device_return = []
	for i in range(gups_nums):
		if Memory_need < Memory_device[i]:
			device_return.append(i)
		# Altruistic尽可能少的给任务分配GPU
		if usingtype==1 and len(device_return)>0:
    			break
	return	 batchsize,device_return	


def Set_Model(I,kernel_size,padding,stride,batch_size,channel):
	global Pcs
	global Ocs
	global batchsize
	batchsize = batch_size
	init_nums()
	Oc_compute(I,kernel_size,padding,stride) #------------------------------ 计算输出尺寸
	Ocs_compute(batch_size,channel)             #------------------------------ 计算总中间结果占用空间
	# print(train_u.shape,batch_size)
	# 判断该batchsize下模型是否满足条件
	model = Net(channel, 48,kernel_size,stride,padding).cuda()
	Ocs,Pcs = Get_Memory_Efficiency()
	Memory_need = (Pcs+Ocs*batch_size)/1024/1024
	# print("memory_efficiency0:权值参数占用空间Pcs:  ",Pcs/1024)54r
	return model,Memory_need

def Get_Memoryneed(batch_size):
	global Pcs
	global Ocs
	Memory_need = (Pcs+Ocs*batch_size)/1024/1024
	# print("memory_efficiency0:权值参数占用空间Pcs:  ",Pcs/1024)
	# print("memory_efficiency0:Ocs_memory:",Ocs*batch_size/1024)
	# print("memory_efficiency0:sum memory",Memory_need)
	return Memory_need
