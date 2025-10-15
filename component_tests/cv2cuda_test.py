import cv2
import numpy as np
import time

print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())

if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("ERROR: No CUDA support detected!")
    exit(1)

# Test GPU operations
print("\nTesting GPU operations...")

# Create test image
img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

# CPU timing
start = time.perf_counter()
for _ in range(100):
    cpu_result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cpu_time = time.perf_counter() - start

# GPU timing
cv2.cuda.setDevice(0)  # Your remapped GPU
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)

start = time.perf_counter()
for _ in range(100):
    gpu_result = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB)
    result = gpu_result.download()  # Include download time for fair comparison
gpu_time = time.perf_counter() - start

print(f"\nCPU time: {cpu_time*1000:.2f}ms")
print(f"GPU time: {gpu_time*1000:.2f}ms")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")