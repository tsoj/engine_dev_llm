# Setup

 TODO file bug cmake -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr -DCOMPUTE_BACKEND=hip -S .

```bash
conda create -n finetuning python=3.12
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
pip install transformers datasets peft
```
