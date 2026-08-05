import argparse
import os
import random

import numpy as np
import torch

from config.config import cfg
from policy.yopo_omni_trainer import YOPOOmniTrainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def configure_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained YOPO-Omni model?")
    parser.add_argument("--trial", type=int, default=0, help="trial of pre-trained model")
    parser.add_argument("--epoch", type=int, default=50, help="epoch of pre-trained model")
    parser.add_argument("--train-epoch", type=int, default=50, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="pose-level batch size")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4, help="AdamW learning rate.")
    parser.add_argument("--guidance-loss", type=str2bool, default=None,
                        help="Override use_guidance_loss.")
    parser.add_argument("--rank-loss", type=str2bool, default=None,
                        help="Enable or disable ranking supervision by overriding w_rank.")
    parser.add_argument("--rank-weight", type=float, default=None,
                        help="Override ranking loss weight. If --rank-loss false is set, w_rank becomes 0.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override omni_num_workers.")
    parser.add_argument("--vdes-speed-min", type=float, default=None,
                        help="Override omni_vdes_speed_min.")
    parser.add_argument("--vdes-speed-max", type=float, default=None,
                        help="Override omni_vdes_speed_max.")
    parser.add_argument("--vel-noise-std", type=float, default=None,
                        help="Override omni_vel_noise_std.")
    parser.add_argument("--acc-noise-std", type=float, default=None,
                        help="Override omni_acc_noise_std.")
    parser.add_argument("--radius-min", type=float, default=None,
                        help="Override omni_radius_min.")
    parser.add_argument("--radius-max", type=float, default=None,
                        help="Override omni_radius_max.")
    parser.add_argument("--radio-range", type=float, default=None,
                        help="Override radio_range and recompute sgm_time unless --sgm-time is also set.")
    parser.add_argument("--sgm-time", type=float, default=None,
                        help="Override trajectory segment time used by loss and polynomial generation.")
    parser.add_argument("--smooth-weight", type=float, default=None,
                        help="Override ws.")
    parser.add_argument("--acc-weight", type=float, default=None,
                        help="Override wa.")
    parser.add_argument("--safety-weight", type=float, default=None,
                        help="Override wc.")
    parser.add_argument("--intent-weight", type=float, default=None,
                        help="Override wi.")
    parser.add_argument("--intent-min-progress", type=float, default=None,
                        help="Override omni_intent_min_progress.")
    parser.add_argument("--explore-weight", type=float, default=None,
                        help="Override w_explore for the endpoint displacement loss.")
    parser.add_argument("--explore-beta", type=float, default=None,
                        help="Override omni_explore_beta for the endpoint displacement loss.")
    parser.add_argument("--run-name", default=None,
                        help="Directory name under YOPO/saved for logs and checkpoints.")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    configure_random_seed(0)

    if args.guidance_loss is not None:
        cfg["use_guidance_loss"] = args.guidance_loss
    if args.rank_loss is not None:
        cfg["w_rank"] = float(cfg["w_rank"] if args.rank_loss else 0.0)
    if args.rank_weight is not None:
        cfg["w_rank"] = float(args.rank_weight)
    if args.num_workers is not None:
        cfg["omni_num_workers"] = args.num_workers
    if args.vdes_speed_min is not None:
        cfg["omni_vdes_speed_min"] = args.vdes_speed_min
    if args.vdes_speed_max is not None:
        cfg["omni_vdes_speed_max"] = args.vdes_speed_max
    if args.vel_noise_std is not None:
        cfg["omni_vel_noise_std"] = args.vel_noise_std
    if args.acc_noise_std is not None:
        cfg["omni_acc_noise_std"] = args.acc_noise_std
    if args.radius_min is not None:
        cfg["omni_radius_min"] = args.radius_min
    if args.radius_max is not None:
        cfg["omni_radius_max"] = args.radius_max
    if args.radio_range is not None:
        cfg["radio_range"] = args.radio_range
        cfg["goal_length"] = 2.0 * args.radio_range
        cfg["sgm_time"] = 2.0 * args.radio_range / float(cfg["vel_max_train"])
    if args.sgm_time is not None:
        cfg["sgm_time"] = args.sgm_time
    if args.smooth_weight is not None:
        cfg["ws"] = args.smooth_weight
    if args.acc_weight is not None:
        cfg["wa"] = args.acc_weight
    if args.safety_weight is not None:
        cfg["wc"] = args.safety_weight
    if args.intent_weight is not None:
        cfg["wi"] = args.intent_weight
    if args.intent_min_progress is not None:
        cfg["omni_intent_min_progress"] = args.intent_min_progress
    if args.explore_weight is not None:
        cfg["w_explore"] = args.explore_weight
    if args.explore_beta is not None:
        cfg["omni_explore_beta"] = args.explore_beta

    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = (
        log_dir + "/YOPO_Omni_{}/epoch{}.pth".format(args.trial, args.epoch)
        if args.pretrained
        else ""
    )

    trainer = YOPOOmniTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        run_name=args.run_name,
        save_on_exit=True,
    )
    trainer.train(epoch=args.train_epoch, save_interval=10)
    print("Run YOPO-Omni Finish!")
