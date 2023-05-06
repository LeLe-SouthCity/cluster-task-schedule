import pynvml
import time

pynvml.nvmlInit()

# 获取显存的使用率和内存占用
def getNvidiaGPU(gpu_id):
    # get GPU temperature
    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    # get GPU memory total
    totalMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
    # get GPU memory used
    usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
    UtilizationRates = pynvml.nvmlDeviceGetUtilizationRates(gpu_device)

    # 获取显存占用
    # print("gpu_id:",gpu_id)
    # print("MemoryInfo：{0}M/{1}M，使用率：{2}%".format("%.1f" % (usedMemory / 1024 / 1024), "%.1f" % (totalMemory / 1024 / 1024), "%.1f" % (usedMemory/totalMemory*100)))
    Memoryuse = "%.0f" % (usedMemory/totalMemory*100)
    # print("gpu_id:",gpu_id)
    # print("UtilizationRates: {0}".format(UtilizationRates.gpu))#获取利用率
    gpu_use = UtilizationRates.gpu
    # 返回显存利用率和GPU利用率
    return usedMemory/ 1024 / 1024,totalMemory/ 1024 / 1024,Memoryuse,gpu_use
    
