#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = ROOT / "Simulator/src/config/config.yaml"
TRAJ_CONFIG = ROOT / "YOPO/config/traj_opt.yaml"
SIM_DIR = ROOT / "Simulator"


def log(message):
    print(message, flush=True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run(cmd, cwd=ROOT, env=None, dry_run=False):
    printable = " ".join(shlex.quote(str(part)) for part in cmd)
    log(f"[YOPO] $ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def shell_run(command, cwd=ROOT, env=None, dry_run=False):
    log(f"[YOPO] $ {command}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), env=env, shell=True, executable="/bin/bash", check=True)


def replace_top_level_scalar(text, key, value):
    pattern = re.compile(rf"^({re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(?:#.*)?)$", re.MULTILINE)

    def repl(match):
        return f"{match.group(1)}{value}{match.group(3)}"

    text, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise KeyError(f"Cannot find top-level key '{key}'")
    return text


def replace_nested_scalar(text, section, key, value):
    section_match = re.search(rf"^{re.escape(section)}\s*:\s*$", text, flags=re.MULTILINE)
    if not section_match:
        raise KeyError(f"Cannot find section '{section}'")

    start = section_match.end()
    next_section = re.search(r"^[^\s#][^:\n]*\s*:\s*", text[start:], flags=re.MULTILINE)
    end = start + next_section.start() if next_section else len(text)
    block = text[start:end]
    pattern = re.compile(rf"^(\s+{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(?:#.*)?)$", re.MULTILINE)

    def repl(match):
        return f"{match.group(1)}{value}{match.group(3)}"

    block, count = pattern.subn(repl, block, count=1)
    if count != 1:
        raise KeyError(f"Cannot find key '{section}.{key}'")
    return text[:start] + block + text[end:]


def apply_config_overrides(args):
    sim_text = SIM_CONFIG.read_text()
    traj_text = TRAJ_CONFIG.read_text()
    new_sim = sim_text
    new_traj = traj_text

    if args.env_num is not None:
        new_sim = replace_top_level_scalar(new_sim, "env_num", str(args.env_num))
    if args.image_num is not None:
        new_sim = replace_top_level_scalar(new_sim, "image_num", str(args.image_num))
    if args.save_path is not None:
        save_path = args.save_path if args.save_path.endswith("/") else args.save_path + "/"
        new_sim = replace_top_level_scalar(new_sim, "save_path", f'"{save_path}"')
    if args.direction_num is not None:
        new_sim = replace_nested_scalar(new_sim, "omni", "direction_num", str(args.direction_num))
    if args.goal_length is not None:
        new_sim = replace_nested_scalar(new_sim, "omni", "goal_length", str(args.goal_length))
    if args.goal_search_radius is not None:
        new_sim = replace_nested_scalar(new_sim, "omni", "goal_search_radius", str(args.goal_search_radius))
    if args.astar_local_radius is not None:
        new_sim = replace_nested_scalar(new_sim, "omni", "astar_local_radius", str(args.astar_local_radius))

    if args.dataset_path is not None:
        new_traj = replace_top_level_scalar(new_traj, "dataset_path", f'"{args.dataset_path}"')
    if new_sim != sim_text:
        if args.dry_run:
            log(f"[YOPO] would update {SIM_CONFIG.relative_to(ROOT)}")
        else:
            SIM_CONFIG.write_text(new_sim)
            log(f"[YOPO] updated {SIM_CONFIG.relative_to(ROOT)}")
    if new_traj != traj_text:
        if args.dry_run:
            log(f"[YOPO] would update {TRAJ_CONFIG.relative_to(ROOT)}")
        else:
            TRAJ_CONFIG.write_text(new_traj)
            log(f"[YOPO] updated {TRAJ_CONFIG.relative_to(ROOT)}")
    return sim_text, traj_text


def restore_config(original_sim, original_traj, keep_config):
    if keep_config:
        return
    if SIM_CONFIG.read_text() != original_sim:
        SIM_CONFIG.write_text(original_sim)
        log(f"[YOPO] restored {SIM_CONFIG.relative_to(ROOT)}")
    if TRAJ_CONFIG.read_text() != original_traj:
        TRAJ_CONFIG.write_text(original_traj)
        log(f"[YOPO] restored {TRAJ_CONFIG.relative_to(ROOT)}")


def build_env(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "YOPO") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def compile_generator(args, env):
    if args.skip_compile:
        return
    if args.ros_setup:
        shell_run(
            f"source {shlex.quote(args.ros_setup)} && "
            "cmake --build Simulator/build --target dataset_generator "
            f"-j{args.jobs}",
            env=env,
            dry_run=args.dry_run,
        )
    else:
        run(["cmake", "--build", "Simulator/build", "--target", "dataset_generator", f"-j{args.jobs}"],
            env=env, dry_run=args.dry_run)


def generate_dataset(args, env):
    compile_generator(args, env)
    command = "./devel/lib/sensor_simulator/dataset_generator"
    if args.ros_setup:
        command = f"source {shlex.quote(args.ros_setup)} && {command}"
    shell_run(command, cwd=SIM_DIR, env=env, dry_run=args.dry_run)


def train(args, env):
    python = args.python
    cmd = [
        python,
        "YOPO/train_yopo.py",
        "--train-epoch",
        str(args.train_epoch),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.learning_rate is not None:
        cmd += ["--learning-rate", str(args.learning_rate)]
    if args.pretrained:
        cmd += ["--pretrained", "1", "--trial", str(args.trial), "--epoch", str(args.epoch)]
    if args.guidance_loss is not None:
        cmd += ["--guidance-loss", str(args.guidance_loss).lower()]
    if args.rank_loss is not None:
        cmd += ["--rank-loss", str(args.rank_loss).lower()]
    if args.rank_weight is not None:
        cmd += ["--rank-weight", str(args.rank_weight)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    if args.vdes_speed_min is not None:
        cmd += ["--vdes-speed-min", str(args.vdes_speed_min)]
    if args.vdes_speed_max is not None:
        cmd += ["--vdes-speed-max", str(args.vdes_speed_max)]
    if args.vel_noise_std is not None:
        cmd += ["--vel-noise-std", str(args.vel_noise_std)]
    if args.acc_noise_std is not None:
        cmd += ["--acc-noise-std", str(args.acc_noise_std)]
    if args.radius_min is not None:
        cmd += ["--radius-min", str(args.radius_min)]
    if args.radius_max is not None:
        cmd += ["--radius-max", str(args.radius_max)]
    if args.radio_range is not None:
        cmd += ["--radio-range", str(args.radio_range)]
    if args.sgm_time is not None:
        cmd += ["--sgm-time", str(args.sgm_time)]
    if args.smooth_weight is not None:
        cmd += ["--smooth-weight", str(args.smooth_weight)]
    if args.acc_weight is not None:
        cmd += ["--acc-weight", str(args.acc_weight)]
    if args.safety_weight is not None:
        cmd += ["--safety-weight", str(args.safety_weight)]
    if args.intent_weight is not None:
        cmd += ["--intent-weight", str(args.intent_weight)]
    if args.intent_min_progress is not None:
        cmd += ["--intent-min-progress", str(args.intent_min_progress)]
    if args.explore_weight is not None:
        cmd += ["--explore-weight", str(args.explore_weight)]
    if args.explore_beta is not None:
        cmd += ["--explore-beta", str(args.explore_beta)]
    if args.run_name:
        cmd += ["--run-name", args.run_name]
    run(cmd, env=env, dry_run=args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description="One-command YOPO-Omni dataset generation and training pipeline.")
    parser.add_argument("--mode", choices=["all", "generate", "train"], default="all",
                        help="Run dataset generation, training, or both.")
    parser.add_argument("--python", default="python3",
                        help="Python executable for training.")
    parser.add_argument("--ros-setup", default="/opt/ros/noetic/setup.bash",
                        help="ROS setup file to source before building/running the simulator. Use '' to disable.")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel build jobs for dataset_generator.")
    parser.add_argument("--skip-compile", action="store_true", help="Skip rebuilding dataset_generator.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and config changes without running them.")
    parser.add_argument("--keep-config", action="store_true",
                        help="Keep config overrides after the script exits. Default restores original files.")

    parser.add_argument("--env-num", type=int, default=None, help="Override Simulator env_num.")
    parser.add_argument("--image-num", type=int, default=None, help="Override Simulator image_num per map.")
    parser.add_argument("--save-path", default=None, help='Override Simulator save_path, e.g. "../dataset_omni".')
    parser.add_argument("--direction-num", type=int, default=None, help="Override omni.direction_num.")
    parser.add_argument("--goal-length", type=float, default=None, help="Override omni.goal_length.")
    parser.add_argument("--goal-search-radius", type=float, default=None, help="Override omni.goal_search_radius.")
    parser.add_argument("--astar-local-radius", type=float, default=None, help="Override omni.astar_local_radius.")

    parser.add_argument("--dataset-path", default=None,
                        help='Override YOPO dataset_path, e.g. "../dataset_omni".')
    parser.add_argument("--guidance-loss", type=str2bool, default=None,
                        help="Override use_guidance_loss for training.")
    parser.add_argument("--rank-loss", type=str2bool, default=None,
                        help="Enable or disable ranking supervision for training.")
    parser.add_argument("--rank-weight", type=float, default=None,
                        help="Override w_rank. Defaults to current config value when --rank-loss true is used.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override omni_num_workers.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override AdamW learning rate.")
    parser.add_argument("--vdes-speed-min", type=float, default=None, help="Override omni_vdes_speed_min.")
    parser.add_argument("--vdes-speed-max", type=float, default=None, help="Override omni_vdes_speed_max.")
    parser.add_argument("--vel-noise-std", type=float, default=None, help="Override omni_vel_noise_std.")
    parser.add_argument("--acc-noise-std", type=float, default=None, help="Override omni_acc_noise_std.")
    parser.add_argument("--radius-min", type=float, default=None, help="Override omni_radius_min.")
    parser.add_argument("--radius-max", type=float, default=None, help="Override omni_radius_max.")
    parser.add_argument("--radio-range", type=float, default=None,
                        help="Override radio_range for training and recompute sgm_time unless --sgm-time is set.")
    parser.add_argument("--sgm-time", type=float, default=None,
                        help="Override trajectory segment time for training.")
    parser.add_argument("--smooth-weight", type=float, default=None, help="Override ws.")
    parser.add_argument("--acc-weight", type=float, default=None, help="Override wa.")
    parser.add_argument("--safety-weight", type=float, default=None, help="Override wc.")
    parser.add_argument("--intent-weight", type=float, default=None, help="Override wi.")
    parser.add_argument("--intent-min-progress", type=float, default=None,
                        help="Override omni_intent_min_progress.")
    parser.add_argument("--explore-weight", type=float, default=None,
                        help="Override w_explore for the endpoint displacement loss.")
    parser.add_argument("--explore-beta", type=float, default=None,
                        help="Override omni_explore_beta for the endpoint displacement loss.")

    parser.add_argument("--train-epoch", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Pose-level training batch size.")
    parser.add_argument("--run-name", default=None,
                        help="Directory name under YOPO/saved for training logs and checkpoints.")
    parser.add_argument("--pretrained", action="store_true", help="Resume from a saved YOPO-Omni checkpoint.")
    parser.add_argument("--trial", type=int, default=0, help="Checkpoint trial id when --pretrained is set.")
    parser.add_argument("--epoch", type=int, default=50, help="Checkpoint epoch when --pretrained is set.")
    args = parser.parse_args()
    if args.ros_setup == "":
        args.ros_setup = None
    if args.save_path is not None and args.dataset_path is None:
        args.dataset_path = args.save_path
    if args.dataset_path is not None and args.save_path is None and args.mode in ("all", "generate"):
        args.save_path = args.dataset_path
    return args


def main():
    args = parse_args()
    original_sim = SIM_CONFIG.read_text()
    original_traj = TRAJ_CONFIG.read_text()
    env = build_env(args)
    try:
        apply_config_overrides(args)
        if args.mode in ("all", "generate"):
            generate_dataset(args, env)
        if args.mode in ("all", "train"):
            train(args, env)
    finally:
        restore_config(original_sim, original_traj, args.keep_config)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[YOPO] command failed with exit code {exc.returncode}", file=sys.stderr, flush=True)
        sys.exit(exc.returncode)
