#!/bin/bash
#SBATCH --job-name=efficientnet_test_cpu
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=187G
#SBATCH --time=00:30:00

# Load necessary modules
module load python/3.10.10

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

# Function to get current memory usage
get_memory_usage() {
    free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*1>
}

# Run your Python script
echo "Starting EfficientNet Benchmark on CPU..."
python -c "import sys; print(sys.path)"

# Record start time
start_time=$(date +%s.%N)

# Run the script and periodically check memory usage
python /home/hice1/jvennamreddy3/efficientnet_test.py &
PID=$!

while kill -0 $PID 2>/dev/null; do
    get_memory_usage
    sleep 1
done

# Record end time
end_time=$(date +%s.%N)

# Calculate execution time
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Execution Time: $execution_time seconds"

echo "Execution Complete."

# Print final memory usage
get_memory_usage
