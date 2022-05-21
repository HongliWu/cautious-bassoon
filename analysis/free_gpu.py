import os

command = os.popen("nvidia-smi -q -d PIDS | grep Processes")
lines = command.read().split("\n")  # 如果显卡上有进程那么这一行只会有一个Processes
free_gpu = []
for i in range(len(lines)):
    if "None" in lines[i]:
        free_gpu.append(i)

print(free_gpu)