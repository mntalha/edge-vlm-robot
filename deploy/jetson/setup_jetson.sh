#!/usr/bin/env bash
# One-time Jetson preparation for running this project (JetPack 6.x / 7.x).
# Read before running - every step is optional and explained.
set -euo pipefail

echo "== JetPack / L4T version =="
cat /etc/nv_tegra_release 2>/dev/null || true
dpkg -l | grep -E "nvidia-jetpack|tensorrt " | head -5 || true

echo "== Power mode: list and select max performance (MAXN / MAXN SUPER) =="
sudo nvpmodel -q || true
# The id of the max mode differs per module - check /etc/nvpmodel.conf. Often 0 or 2.
# sudo nvpmodel -m 0
# sudo jetson_clocks            # pin clocks to max for benchmarking

echo "== Free memory for the model (8GB is shared by CPU+GPU!) =="
# Boot to console instead of the desktop GUI (~0.8-1 GB saved):
#   sudo systemctl set-default multi-user.target     (undo: graphical.target)
# Add swap on NVMe so builds don't OOM (swap is NOT used by the GPU, only helps the build):
#   sudo fallocate -l 8G /ssd/8GB.swap && sudo mkswap /ssd/8GB.swap && sudo swapon /ssd/8GB.swap

echo "== Monitoring =="
# tegrastats             -> RAM, GPU %, power rails, temps (built in)
# sudo pip3 install -U jetson-stats && jtop   -> nicer TUI

echo "== Containers (recommended: avoids building PyTorch for aarch64 yourself) =="
# git clone https://github.com/dusty-nv/jetson-containers && bash jetson-containers/install.sh
# jetson-containers run -v $PWD:/workspace $(autotag l4t-pytorch)
#   then inside: cd /workspace && pip install -e . && python scripts/benchmark.py
echo "done - see notebooks/06_tensorrt_jetson_deployment.ipynb"
