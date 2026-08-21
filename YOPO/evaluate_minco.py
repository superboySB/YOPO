#!/usr/bin/env python3
"""Evaluate one or more MINCO checkpoints on the same deterministic validation stream."""

import argparse
import json
import os
import random
import tempfile

import numpy as np
import torch

from config.config import cfg
from policy.yopo_trainer import YopoTrainer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="label=path or plain checkpoint path")
    parser.add_argument("--dataset-path", default="../dataset_minco")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="../results/minco_checkpoint_eval.json")
    args = parser.parse_args()

    cfg["dataset_path"] = args.dataset_path
    results = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "checkpoints": {},
    }
    with tempfile.TemporaryDirectory(prefix="yopo-minco-eval-") as log_root:
        for item in args.checkpoints:
            label, path = item.split("=", 1) if "=" in item else (os.path.basename(item), item)
            path = os.path.abspath(path)
            seed_everything(args.seed)
            trainer = YopoTrainer(
                learning_rate=1.5e-4,
                batch_size=args.batch_size,
                tensorboard_path=log_root,
                checkpoint_path=path,
                save_on_exit=False,
                num_workers=0,
            )
            trainer.policy.eval()
            metrics = trainer.eval_one_epoch(0)
            trainer.tensorboard_log.close()
            results["checkpoints"][label] = {
                "path": path,
                "metrics": {key: float(value) for key, value in sorted(metrics.items())},
            }

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
