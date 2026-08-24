#!/usr/bin/env python3
"""Measure network-only CUDA latency and parameter count for both checkpoints."""

import argparse
import json
import os
import subprocess
import sys


def child(model, repo, repeats, warmup):
    import numpy as np
    import torch

    if model == "yopo-simple":
        runtime = os.path.join(repo, "YOPO", "simple_runtime")
        checkpoint = os.path.join(repo, "YOPO", "saved", "yopo-simple", "epoch50.pth")
    else:
        runtime = os.path.join(repo, "YOPO")
        checkpoint = os.path.join(repo, "YOPO", "saved", "yopo-minco", "epoch50.pth")
    os.chdir(runtime)
    sys.path.insert(0, runtime)
    from policy.yopo_network import YopoNetwork

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the report latency measurement")
    network = YopoNetwork().cuda().eval()
    network.load_state_dict(torch.load(checkpoint, weights_only=True))
    depth = torch.zeros((1, 1, 96, 160), device="cuda")
    obs = torch.zeros((1, 9, 3, 5), device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            network(depth, obs)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends):
            start.record()
            network(depth, obs)
            end.record()
        torch.cuda.synchronize()
    timings = np.asarray([start.elapsed_time(end) for start, end in zip(starts, ends)])
    state = network.state_dict()
    return {
        "model": model,
        "checkpoint": checkpoint,
        "parameters": int(sum(value.numel() for value in state.values())),
        "head_output_channels": int(state["yopo_head.model.4.weight"].shape[0]),
        "repeats": repeats,
        "warmup": warmup,
        "network_latency_ms": {
            "mean": float(timings.mean()),
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "std": float(timings.std()),
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


def main(args):
    if args.model:
        print(json.dumps(child(args.model, args.repo, args.repeats, args.warmup), sort_keys=True))
        return
    results = {}
    for model in ("yopo-simple", "yopo-minco"):
        command = [sys.executable, os.path.abspath(__file__), "--model", model,
                   "--repo", args.repo, "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
        results[model] = json.loads(subprocess.check_output(command, text=True).strip().splitlines()[-1])
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("yopo-simple", "yopo-minco"))
    parser.add_argument("--repo", default="/workspace/YOPO")
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", default="/workspace/YOPO/docs/report_assets/inference_benchmark.json")
    main(parser.parse_args())
