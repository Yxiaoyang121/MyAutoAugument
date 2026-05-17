# GPU Preflight Report

- Started: 2026-05-17T13:36:20
- Finished: 2026-05-17T13:37:31
- Project root: `E:\TJGY\MinPaper\MyAutoAugument`

## Summary

- Conda env name: pytorch
- `sys.executable`: `D:\Anaconda\envs\pytorch\python.exe`
- YOLO executable: `D:\Anaconda\envs\pytorch\Scripts\yolo.exe`
- YOLO config dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\ultralytics_config`
- `torch.cuda.is_available()` = True
- PyTorch: 2.4.1
- CUDA version reported by PyTorch: 12.4
- CUDA device count: 1
- GPU device name: NVIDIA GeForce RTX 3060 Laptop GPU
- `nvidia-smi` GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- `nvidia-smi` driver: 560.81
- `nvidia-smi` memory: 6144 MiB
- Ultralytics: 8.3.221
- `yolo checks` success: True
- YOLO GPU smoke status: success
- Formal training allowed: True

## Conclusion

CUDA is available and the minimal YOLO GPU smoke test completed successfully.

Inference: n/a

## Commands

### python_version

- Command: `D:\Anaconda\envs\pytorch\python.exe --version`
- Return code: `0`
- Duration seconds: `0.047`

stdout:

```text
Python 3.9.19
```

stderr:

```text

```

### torch_cuda_info

- Command: `D:\Anaconda\envs\pytorch\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"`
- Return code: `0`
- Duration seconds: `3.516`

stdout:

```text
torch: 2.4.1
cuda_available: True
cuda_version: 12.4
device_count: 1
device_name: NVIDIA GeForce RTX 3060 Laptop GPU
```

stderr:

```text

```

### ultralytics_version

- Command: `D:\Anaconda\envs\pytorch\python.exe -c "import ultralytics; print('ultralytics:', ultralytics.__version__)"`
- Return code: `0`
- Duration seconds: `3.718`

stdout:

```text
Creating new Ultralytics Settings v0.0.6 file
View Ultralytics Settings with 'yolo settings' or at 'E:\TJGY\MinPaper\MyAutoAugument\outputs\ultralytics_config\Ultralytics\settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
ultralytics: 8.3.221
```

stderr:

```text

```

### yolo_checks

- Command: `D:\Anaconda\envs\pytorch\Scripts\yolo.exe checks`
- Return code: `0`
- Duration seconds: `4.735`

stdout:

```text
[2K
[2K
Ultralytics 8.3.221  Python-3.9.19 torch-2.4.1 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
Setup complete  (16 CPUs, 15.9 GB RAM, 312.5/476.9 GB disk)

OS                     Windows-10-10.0.26200-SP0
Environment            Windows
Python                 3.9.19
Install                pip
Path                   D:\Anaconda\envs\pytorch\Lib\site-packages\ultralytics
RAM                    15.86 GB
Disk                   312.5/476.9 GB
CPU                    AMD Ryzen 7 5800H with Radeon Graphics
CPU count              16
GPU                    NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB
GPU count              1
CUDA                   12.4

numpy                   1.26.4>=1.23.0
matplotlib              3.9.2>=3.3.0
opencv-python           4.7.0>=4.6.0
pillow                  10.4.0>=7.1.2
pyyaml                  6.0.1>=5.3.1
requests                2.32.3>=2.23.0
scipy                   1.13.1>=1.4.1
torch                   2.4.1>=1.8.0
torch                   2.4.1!=2.4.0,>=1.8.0; sys_platform == "win32"
torchvision             0.19.1>=0.9.0
psutil                  5.9.0
polars                  1.34.0
ultralytics-thop        2.0.6>=2.0.0
```

stderr:

```text

```

### nvidia_smi

- Command: `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`
- Return code: `0`
- Duration seconds: `0.078`

stdout:

```text
NVIDIA GeForce RTX 3060 Laptop GPU, 560.81, 6144 MiB
```

stderr:

```text

```

## YOLO GPU Smoke

- status: `success`
- success: `True`
- returncode: `0`
- command: `D:\Anaconda\envs\pytorch\Scripts\yolo.exe detect train model=E:\TJGY\MinPaper\MyAutoAugument\yolo11n.pt data=E:\TJGY\MinPaper\MyAutoAugument\outputs\tiled_dataset_smoke\data.yaml imgsz=640 batch=1 epochs=1 workers=0 device=0 project=E:\TJGY\MinPaper\MyAutoAugument\outputs\gpu_preflight_smoke name=train exist_ok=True mosaic=0 mixup=0 copy_paste=0 hsv_h=0 hsv_s=0 hsv_v=0 degrees=0 translate=0 scale=0 shear=0 perspective=0 fliplr=0 flipud=0`
- data_yaml: `E:\TJGY\MinPaper\MyAutoAugument\outputs\tiled_dataset_smoke\data.yaml`
- model: `E:\TJGY\MinPaper\MyAutoAugument\yolo11n.pt`
- output_dir: `E:\TJGY\MinPaper\MyAutoAugument\outputs\gpu_preflight_smoke`
- train_command_txt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\gpu_preflight_smoke\train_command.txt`
- train_stdout_log: `E:\TJGY\MinPaper\MyAutoAugument\outputs\gpu_preflight_smoke\train_stdout.log`
- train_stderr_log: `E:\TJGY\MinPaper\MyAutoAugument\outputs\gpu_preflight_smoke\train_stderr.log`
