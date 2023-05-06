#!/bin/bash

#SBATCH -n 8 # 核心数
#SBATCH -N 1 # 节点数
#SBATCH --mem=10G # 内存
#SBATCH --partition=debug
#SBATCH -e err.log # 错误输出
#SBATCH --output=array_%A-%a.out    # Standard output and error log
#SBATCH --array=1-16%2                # Array range
#SBATCH --job-name=test # 任务名


echo "START: $SLURM_JOBID"
startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`
nums=$[$RANDOM%2+1]
types=$[$RANDOM%2]
batchsizeRan=$[$RANDOM%512+128]
epochRan=$[$RANDOM%100+100]
# 提交任务也可以是任何别的命令 &可以使得两个任务并行运行

# python /home/roota/4TB/lele/MYstudy/schedule/slurm/fifo_End/EndToEnd_test.py --epochs $epochRan --gpus $nums --batchsize $batchsizeRan --type $types
python /home/roota/4TB/lele/MYstudy/schedule/slurm/fifo_End/EndToEnd_test.py --epochs $epochRan --gpus $nums --batchsize $batchsizeRan
# python /home/roota/4TB/lele/MYstudy/schedule/slurm/fifo_End/EndToEnd_test.py --gpus 2 --batchsize 256 
# sleep(10)
# echo " job 1 finish -------------------------------------------------------------------------------------------- "
# python /home/roota/4TB/lele/MYstudy/schedule/slurm/fifo_End/EndToEnd_test.py --gpus 1 --batchsize 256 &
# python /home/roota/4TB/lele/MYstudy/schedule/slurm/fifo_End/EndToEnd_test.py --gpus 1 --batchsize 256 &

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[$endTime_s - $startTime_s ]
Cankao=$[295*$epochRan/$batchsizeRan] #参考时间
lww=$[$sumTime/$Cankao] 
echo "using gpu ------------------ $nums using batchsize---------------------$batchsizeRan"
echo "参考时间----------------  $Cankao  ---------------标准化延迟-----------$lww "
echo "$startTime ---> $endTime" "Totl:$sumTime seconds"

wait
