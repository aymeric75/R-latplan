from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetComputeRunningProcesses

# Initialize NVML
nvmlInit()

device_count = 1  # Adjust based on the number of GPUs
for i in range(device_count):
    handle = nvmlDeviceGetHandleByIndex(i)
    processes = nvmlDeviceGetComputeRunningProcesses(handle)
    print(f"Processes on GPU {i}:")
    for p in processes:
        print(f"  PID: {p.pid}, Used Memory: {p.usedGpuMemory / 1024**2} MB")
