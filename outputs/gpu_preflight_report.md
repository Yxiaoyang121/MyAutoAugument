# GPU Preflight Report

- Started: 2026-05-17T13:08:54
- Finished: 2026-05-17T13:09:06
- Project root: `E:\TJGY\MinPaper\MyAutoAugument`

## Summary

- `torch.cuda.is_available()` = False
- PyTorch: 2.4.1+cpu
- CUDA version reported by PyTorch: None
- CUDA device count: 0
- GPU device name: NO CUDA
- `nvidia-smi` GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- `nvidia-smi` driver: 560.81
- `nvidia-smi` memory: 6144 MiB
- Ultralytics: 8.4.48
- `yolo checks` success: True
- YOLO GPU smoke status: not_run_cuda_unavailable
- Formal training allowed: False

## Conclusion

torch.cuda.is_available() = False. Current environment may be CPU-only PyTorch, missing CUDA runtime, or unavailable GPU driver. Only CPU smoke tests can run here, and they cannot be used as formal experiment results.

Inference: nvidia-smi detects a GPU, but the active PyTorch build reports 2.4.1+cpu and cuda_version=None; the primary blocker is likely CPU-only PyTorch in the active Python environment.

Required no-GPU note:

- `torch.cuda.is_available() = False`
- Current environment may be CPU-only PyTorch, CUDA not installed, or driver unavailable.
- Only CPU smoke tests can run here; CPU smoke results cannot be used as formal experiment results.

## Commands

### python_version

- Command: `python --version`
- Return code: `0`
- Duration seconds: `0.031`

stdout:

```text
Python 3.12.4
```

stderr:

```text

```

### torch_cuda_info

- Command: `python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"`
- Return code: `0`
- Duration seconds: `3.406`

stdout:

```text
torch: 2.4.1+cpu
cuda_available: False
cuda_version: None
device_count: 0
device_name: NO CUDA
```

stderr:

```text

```

### ultralytics_version

- Command: `python -c "import ultralytics; print('ultralytics:', ultralytics.__version__)"`
- Return code: `0`
- Duration seconds: `3.641`

stdout:

```text
ultralytics: 8.4.48
```

stderr:

```text

```

### yolo_checks

- Command: `yolo checks`
- Return code: `0`
- Duration seconds: `4.703`

stdout:

```text
[2K
[2K
Ultralytics 8.4.48  Python-3.12.4 torch-2.4.1+cpu CPU (AMD Ryzen 7 5800H with Radeon Graphics)
Setup complete  (16 CPUs, 15.9 GB RAM, 312.5/476.9 GB disk)

OS                     Windows-11-10.0.26200-SP0
Environment            Windows
Python                 3.12.4
Install                pip
Path                   D:\Anaconda\Lib\site-packages\ultralytics
RAM                    15.86 GB
Disk                   312.5/476.9 GB
CPU                    AMD Ryzen 7 5800H with Radeon Graphics
CPU count              16
GPU                    None
GPU count              None
CUDA                   None

numpy                   1.26.4>=1.23.0
matplotlib              3.8.4>=3.3.0
opencv-python           4.10.0.84>=4.6.0
pillow                  10.3.0>=7.1.2
pyyaml                  6.0.1>=5.3.1
requests                2.32.2>=2.23.0
scipy                   1.13.1>=1.4.1
torch                   2.4.1>=1.8.0
torch                   2.4.1!=2.4.0,>=1.8.0; sys_platform == "win32"
torchvision             0.19.1>=0.9.0
psutil                  5.9.0>=5.8.0
polars                  1.40.1>=0.20.0
ultralytics-thop        2.0.19>=2.0.18
```

stderr:

```text

```

### nvidia_smi

- Command: `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`
- Return code: `0`
- Duration seconds: `0.047`

stdout:

```text
NVIDIA GeForce RTX 3060 Laptop GPU, 560.81, 6144 MiB
```

stderr:

```text

```

## YOLO GPU Smoke

- status: `not_run_cuda_unavailable`
- success: `None`
- reason: `torch.cuda.is_available() = False`
