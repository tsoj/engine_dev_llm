# Setup

 TODO file bug cmake -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr -DCOMPUTE_BACKEND=hip -S .

```bash
conda create -n chessgpt python=3.10
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
pip install transformers datasets peft
```
Also install bitsandbytes:
Nvidia:
```bash
pip install bitsandbytes
```
AMD:
```bash
dnf install rocm-hip-devel hipblas-devel hiprand-devel hipsparse-devel
./install_amd_bitsandbytes.sh
```
