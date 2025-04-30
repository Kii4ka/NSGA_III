# NSGA-III: CPU and GPU-Accelerated Implementation

This repository contains an implementation of the NSGA-III (Non-dominated Sorting Genetic Algorithm III) algorithm with CPU parallel processing and GPU acceleration using Apple's Metal Performance Shaders.

## Features

- CPU implementation with parallel processing
- GPU-accelerated implementation using Metal Performance Shaders
- Comprehensive benchmarking tools
- Performance visualization
- Support for various dataset dimensions

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NumPy, Pandas, Matplotlib, Seaborn
- PyTorch (for GPU acceleration)
- macOS with Metal support or compatible GPU system

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Kii4ka/NSGA_III.git
cd NSGA_III
```

2. Switch to the implementation branch for the final version:

```bash
git checkout framework_impl
```

### Setting Up Data

1. Unzip the dataset files:

```bash
# Navigate to the data directory
cd data/add_dim
# Unzip additional dimension files
unzip *.zip
# Return to main directory
```

## Running Experiments

### Extended Experiments

To run the full suite of experiments with both CPU and GPU implementations:

```bash
python experiments/experiment_update.py
```

This will:
- Process datasets across multiple categories
- Run the NSGA-III algorithm with both CPU and GPU implementations
- Generate performance metrics and comparisons
- Output results to CSV files

### Checking Results

After running the experiments, you'll find:

- nsga_performance.csv - Contains performance metrics for CPU vs GPU implementations
- results directory - Individual experiment results by dataset category

### Visualizing Results

Generate performance comparison graphs:

```bash
python graph.py
```

This creates:
- nsga_performance_bars.png - Bar charts showing time, memory usage, throughput, and speedup
- gpu_time_breakdown.png - Pie chart showing GPU time allocation

## Performance Metrics

When examining results, look for:

- **Execution Time**: Comparison between CPU and GPU implementations (ms)
- **Memory Usage**: Memory footprint difference (MB)
- **Throughput**: Operations per second (higher is better)
- **Speedup**: Relative performance gain of GPU vs CPU (×)
- **Memory Transfer**: Time spent transferring data to GPU (ms)

## Example Output

```
==== NSGA-III Performance Comparison ====
Implementation       | Time (ms)  | Speedup    | Memory (MB)  | Throughput     
---------------------------------------------------------------------------
CPU Parallel         | 249183.79  | 1.00      x | 108.45       | 30.10          
GPU MPS              | 3942.46    | 63.21     x | 28.83        | 1902.37
```

This indicates the GPU implementation is approximately 63× faster while using only about 26% of the memory compared to the CPU implementation.

## Troubleshooting

If you encounter issues with duplicate output messages, ensure `QUIET_MODE` is properly set.

For GPU-related errors, verify your system has proper Metal/GPU support by checking:

```python
import torch
print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
print(f"PyTorch MPS built: {torch.backends.mps.is_built()}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The NSGA-III algorithm was proposed by Deb and Jain (2014)
- Implementation inspired by PyNSGA3 and other open source MOEA frameworks
