# Open-source AI & GPU Computing

I'm a university student exploring open-source generative AI and GPU computing, with an interest in image and video generation, memory optimization, and inference performance.

## GPU benchmark starter

This repository includes a small Python project for learning how to measure compute time and GPU memory usage with PyTorch.

- Runs square matrix multiplication on CPU or an available AMD ROCm / NVIDIA CUDA GPU.
- Includes warmup iterations and synchronizes GPU work before recording elapsed time.
- Records individual timings, mean, median, minimum, tensor memory, and peak GPU memory allocation.
- Saves settings and environment details with the measurements in a JSON file.
- Supports float32 and float16 for exploring precision and memory tradeoffs.

### Get started

Use Python 3.10+ and install the PyTorch build appropriate for your hardware using the [official installation instructions](https://pytorch.org/get-started/locally/). On AMD Developer Cloud, use a compatible ROCm environment and PyTorch build.

Clone this repository:

```bash
git clone https://github.com/NguyenVietThanh14536/NguyenVietThanh14536.git
cd NguyenVietThanh14536
```

Run a small CPU check:

```bash
python benchmark.py --device cpu --size 256 --runs 5 --output results/cpu.json
```

Run on an available GPU:

```bash
python benchmark.py --device cuda --size 2048 --runs 20 --warmup 5 --output results/gpu-fp32.json
python benchmark.py --device cuda --size 2048 --dtype float16 --runs 20 --warmup 5 --output results/gpu-fp16.json
```

PyTorch uses the `cuda` device name for AMD ROCm as well as NVIDIA CUDA. The report identifies the backend as `rocm`, `cuda`, or `cpu`. An explicit `--device cuda` request fails with a clear message when no compatible GPU is available. See [PyTorch's HIP/ROCm documentation](https://docs.pytorch.org/docs/stable/notes/hip.html).

### Interpreting measurements

The current workload is a matrix multiplication microbenchmark. Timings include Python dispatch and, on GPU, synchronization overhead. Inputs and output are allocated before timing. `peak_allocated_mib` measures PyTorch tensor allocation, not total device memory use; CPU runs report this field as `null`. `tensor_memory_mib` describes the three benchmark tensors.

Keep matrix size, precision, software versions, and hardware settings consistent when comparing runs. Float16 changes numerical precision. Increase matrix sizes gradually if memory is limited.

### Tests

```bash
python -m unittest discover -s tests -v
```

The tests check real CPU execution, JSON output, invalid settings, automatic CPU selection, and clear failure when a requested GPU is unavailable.

## Next experiments

I plan to use AMD Developer Cloud to test larger workloads than my local hardware can support. Next steps include benchmarking open-source image and video generation models, comparing memory-saving techniques, and documenting ROCm compatibility findings.

AMD GPU and generative-model measurements will be added after those experiments are run.
