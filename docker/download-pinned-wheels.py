#!/usr/bin/env python3
"""Resolve exact PyTorch CUDA dependency wheels and emit an aria2 input file."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


PINNED_WHEELS = {
    "nvidia-cuda-nvrtc-cu11": "11.8.89",
    "nvidia-cuda-runtime-cu11": "11.8.89",
    "nvidia-cuda-cupti-cu11": "11.8.87",
    "nvidia-cudnn-cu11": "9.1.0.70",
    "nvidia-cublas-cu11": "11.11.3.6",
    "nvidia-cufft-cu11": "10.9.0.58",
    "nvidia-curand-cu11": "10.3.0.86",
    "nvidia-cusolver-cu11": "11.4.1.48",
    "nvidia-cusparse-cu11": "11.7.5.86",
    "nvidia-nccl-cu11": "2.20.5",
    "nvidia-nvtx-cu11": "11.8.86",
    "triton": "3.0.0",
}


def compatible(package, filename):
    if package == "triton":
        return "-cp38-cp38-" in filename and filename.endswith("x86_64.whl")
    return filename.endswith("-py3-none-manylinux2014_x86_64.whl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aria-input", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for package, version in PINNED_WHEELS.items():
        metadata_url = f"https://pypi.org/pypi/{package}/{version}/json"
        with urlopen(metadata_url, timeout=60) as response:
            metadata = json.load(response)
        matches = [item for item in metadata["urls"] if compatible(package, item["filename"])]
        if len(matches) != 1:
            names = [item["filename"] for item in matches]
            raise RuntimeError(f"Expected one compatible wheel for {package}=={version}, got {names}")
        item = matches[0]
        lines.extend(
            [
                item["url"],
                f"  dir={output_dir}",
                f"  out={item['filename']}",
                f"  checksum=sha-256={item['digests']['sha256']}",
            ]
        )

    Path(args.aria_input).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Resolved {len(PINNED_WHEELS)} pinned CUDA wheels", flush=True)


if __name__ == "__main__":
    main()
