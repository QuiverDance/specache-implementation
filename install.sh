#!/bin/bash
set -e

# intall pytorch and required library
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

echo ""
echo "install complete."
