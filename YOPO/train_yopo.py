import os
import torch
import random
import argparse
import numpy as np
from policy.yopo_trainer import YopoTrainer
from config.config import cfg


def configure_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained model?")
    parser.add_argument("--trial", type=int, default=1, help="trial of pre-trained model")
    parser.add_argument("--epoch", type=int, default=50, help="epoch of pre-trained model")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--dataset-path", default=None, help="dataset path relative to YOPO/ or absolute")
    parser.add_argument("--log-dir", default=None, help="parent directory for YOPO_<n> run folders")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    configure_random_seed(args.seed)
    if args.dataset_path:
        cfg["dataset_path"] = args.dataset_path

    # save the configuration and other files
    log_dir = args.log_dir or os.path.dirname(os.path.abspath(__file__)) + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = log_dir + "/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch) if args.pretrained else ""

    trainer = YopoTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        save_on_exit=True,
        num_workers=args.num_workers,
    )

    trainer.train(epoch=args.epochs, save_interval=args.save_interval)

    print("Run YOPO Finish!")
