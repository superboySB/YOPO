import os
import torch
import random
import argparse
import numpy as np


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
    parser.add_argument("--config", type=str, default="", help="optional YOPO config yaml path")
    parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained model?")
    parser.add_argument("--trial", type=int, default=1, help="trial of pre-trained model")
    parser.add_argument("--epoch", type=int, default=50, help="epoch of pre-trained model")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4, help="optimizer learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--save-root", type=str, default="saved", help="checkpoint/tensorboard root under YOPO/")
    parser.add_argument("--save-interval", type=int, default=10, help="save checkpoint every N epochs; <=0 disables interval saving")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    if args.config:
        os.environ["YOPO_CONFIG_PATH"] = args.config

    from policy.yopo_trainer import YopoTrainer

    configure_random_seed(0)    # set random seed

    # save the configuration and other files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, args.save_root)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = log_dir + "/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch) if args.pretrained else ""

    trainer = YopoTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        num_workers=args.num_workers,
        save_on_exit=True,
    )

    save_interval = args.save_interval if args.save_interval > 0 else None
    trainer.train(epoch=args.epochs, save_interval=save_interval)

    print("Run YOPOv2-Tracker training Finish!")
