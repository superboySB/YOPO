import os
import torch
import random
import argparse
import numpy as np
from policy.yopo_trainer import YopoTrainer


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
    parser.add_argument("--train_epochs", type=int, default=120, help="total training epochs")
    parser.add_argument("--save_interval", type=int, default=50, help="save checkpoint every N epochs (<=0 to disable)")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="learning rate")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    configure_random_seed(0)    # set random seed

    # save the configuration and other files
    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = log_dir + "/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch) if args.pretrained else ""

    trainer = YopoTrainer(
        learning_rate=args.lr,
        batch_size=16,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        save_on_exit=True,
    )

    save_interval = args.save_interval if args.save_interval > 0 else None
    trainer.train(epoch=args.train_epochs, save_interval=save_interval)

    print("Run YOPO Finish!")
