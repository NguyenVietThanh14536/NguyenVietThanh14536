"""A small matrix multiplication benchmark for CPU, AMD ROCm, and NVIDIA CUDA."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time


def run_benchmark(size=1024, runs=10, warmup=3, device="auto", dtype="float32"):
    """Measure repeated matrix multiplications and return a JSON-ready report."""
    if size <= 0 or runs <= 0 or warmup < 0:
        raise ValueError("size and runs must be positive; warmup must be nonnegative")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if dtype not in {"float32", "float16"}:
        raise ValueError("dtype must be float32 or float16")

    import torch

    gpu_available = torch.cuda.is_available()
    if device == "cuda" and not gpu_available:
        raise RuntimeError("GPU unavailable. Install a compatible PyTorch build or use --device cpu")
    chosen = "cuda" if device == "auto" and gpu_available else device
    if chosen == "auto":
        chosen = "cpu"
    using_gpu = chosen == "cuda"
    backend = ("rocm" if torch.version.hip else "cuda") if using_gpu else "cpu"
    if using_gpu:
        torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device=chosen).manual_seed(0)
    options = {"device": chosen, "dtype": getattr(torch, dtype)}
    left = torch.randn(size, size, generator=generator, **options)
    right = torch.randn(size, size, generator=generator, **options)
    result = torch.empty(size, size, **options)

    def synchronize():
        if using_gpu:
            torch.cuda.synchronize()

    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            torch.mm(left, right, out=result)
        synchronize()
        for _ in range(runs):
            start = time.perf_counter_ns()
            torch.mm(left, right, out=result)
            synchronize()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "matrix_multiply",
        "size": size,
        "runs": runs,
        "warmup": warmup,
        "seed": 0,
        "dtype": dtype,
        "device": chosen,
        "backend": backend,
        "hardware": torch.cuda.get_device_name() if using_gpu else platform.machine(),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "cpu_threads": torch.get_num_threads() if not using_gpu else None,
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "tensor_memory_mib": 3 * left.numel() * left.element_size() / (1024 ** 2),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2) if using_gpu else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--size", type=int, default=1024, help="Square matrix dimension")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/latest.json"))
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.size, args.runs, args.warmup, args.device, args.dtype)
    except ImportError:
        parser.error("Install a PyTorch build for your hardware: https://pytorch.org/get-started/locally/")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
