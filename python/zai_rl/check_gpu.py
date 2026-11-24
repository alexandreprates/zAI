"""Report the PyTorch build and verify it can run on the target GPU (Tesla P100, sm_60).

Usage: python -m zai_rl.check_gpu [--require-gpu]
Exits non-zero if the CUDA build lacks sm_60 kernels, or if --require-gpu is set
and no CUDA device is visible.
"""

import argparse
import sys

import torch

REQUIRED_ARCH = "sm_60"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="fail if no CUDA device")
    args = parser.parse_args()

    print(f"torch {torch.__version__}, built with CUDA {torch.version.cuda}")
    if torch.version.cuda is None:
        print("CPU-only build")
        return 1 if args.require_gpu else 0

    arch_flags = torch._C._cuda_getArchFlags()
    print(f"compiled CUDA archs: {arch_flags}")
    if REQUIRED_ARCH not in arch_flags.split():
        print(f"ERROR: this build has no {REQUIRED_ARCH} kernels (needed for Tesla P100)")
        return 1

    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return 1 if args.require_gpu else 0

    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        print(f"device {i}: {torch.cuda.get_device_name(i)} (sm_{major}{minor})")
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"matmul check ok: {float((x @ x).sum()):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
