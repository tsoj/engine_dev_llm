# Setup

```bash
conda create -n finetuning python=3.12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft bitsandbytes dataclasses-json
```

# Train

```bash
# replace 3 with the number of GPUs you want to train on
torchrun --nproc_per_node 3 main.py
```

If one GPU is not enough to hold an entire model, then enable gradient_checkpoint again, adjust the batchsize and gradient accumulation steps.
Then run the main.py script normally with `python main.py`.
