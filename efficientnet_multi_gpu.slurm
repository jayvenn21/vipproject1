#!/bin/bash
#SBATCH --job-name=efficientnet_multi_gpu
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx_6000:4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Load necessary modules
module load python/3.10.10
module load cuda/12.6.1

# Print Python and pip versions for debugging
python --version
python -m pip --version

# Install packages (if not already installed)
python -m pip install --user --no-cache-dir datasets torch torchvision transfo>

# Upgrade transformers to the latest version
python -m pip install --user --upgrade transformers

# Set PYTHONPATH to include user-installed packages
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

# Print PYTHONPATH for debugging
echo "PYTHONPATH: $PYTHONPATH"

# Start GPU memory usage logging
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > gpu_memory_us>
NVIDIA_SMI_PID=$!

# Run your Python script using torch.distributed.launch
echo "Starting EfficientNet Multi-GPU Benchmark..."
python -m torch.distributed.launch --nproc_per_node=4 /home/hice1/jvennamreddy>

# Stop GPU memory usage logging
kill $NVIDIA_SMI_PID

echo "Execution Complete."

# Print peak GPU memory usage
echo "Peak GPU Memory Usage:"
awk -F',' '{print $2}' gpu_memory_usage_multi.log | sort -n | tail -n 1