#!/bin/bash

set -e

mkdir -p ext
cd ext
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes/
git switch multi-backend-refactor
pip install -r requirements-dev.txt
cmake -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr -DCOMPUTE_BACKEND=hip -S .
make
pip install -e .
