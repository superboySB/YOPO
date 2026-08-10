#!/usr/bin/env python3
"""Safely inspect the joystick-to-vdes mapping without starting ROS or motors."""

import argparse
import os
import select
import struct
import sys
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "YOPO"))

from joystick_control import map_dual_sticks


JS_EVENT = struct.Struct("IhBB")
JS_AXIS = 0x02


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/js0")
    parser.add_argument("--axis-x", type=int, default=0, help="Right-stick horizontal axis.")
    parser.add_argument("--axis-y", type=int, default=1, help="Right-stick vertical axis.")
    parser.add_argument("--axis-z", type=int, default=2, help="Left-stick vertical climb/descent axis.")
    parser.add_argument("--axis-yaw", type=int, default=3, help="Left-stick horizontal yaw axis.")
    parser.add_argument("--axis-max", type=float, default=32767.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--invert-x", type=int, choices=(0, 1), default=1)
    parser.add_argument("--invert-y", type=int, choices=(0, 1), default=0)
    parser.add_argument("--invert-z", type=int, choices=(0, 1), default=0)
    parser.add_argument("--invert-yaw", type=int, choices=(0, 1), default=1)
    parser.add_argument("--swap-xy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--speed", type=float, default=6.0, help="Maximum vdes magnitude in m/s.")
    parser.add_argument("--vertical-speed", type=float, default=2.0, help="Maximum climb/descent speed in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=1.0, help="Maximum yaw rate in rad/s.")
    return parser.parse_args()


def main():
    args = parse_args()
    axes = {}
    buffer = b""
    last_print = 0.0
    pending_print = False

    print(
        f"Reading {args.device}: right axes={args.axis_x}/{args.axis_y}, "
        f"left axes={args.axis_z}/{args.axis_yaw}, deadzone={args.deadzone:g}, "
        f"horizontal={args.speed:g}m/s vertical={args.vertical_speed:g}m/s yaw={args.yaw_rate:g}rad/s"
    )
    print("Move both sticks; press Ctrl-C to stop.")

    fd = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
    try:
        while True:
            readable, _, _ = select.select([fd], [], [], 0.1)
            if readable:
                data = os.read(fd, 256)
                if not data:
                    raise RuntimeError(f"joystick disconnected: {args.device}")
                buffer += data
                while len(buffer) >= JS_EVENT.size:
                    _event_time, value, event_type, number = JS_EVENT.unpack(buffer[:JS_EVENT.size])
                    buffer = buffer[JS_EVENT.size:]
                    if event_type & JS_AXIS:
                        axes[number] = value
                        pending_print = True

            now = time.monotonic()
            if pending_print and now - last_print >= 0.05:
                raw_x = axes.get(args.axis_x, 0)
                raw_y = axes.get(args.axis_y, 0)
                raw_z = axes.get(args.axis_z, 0)
                raw_yaw = axes.get(args.axis_yaw, 0)
                translation, yaw_fraction, _planar_magnitude, magnitude = map_dual_sticks(
                    raw_x,
                    raw_y,
                    raw_z,
                    raw_yaw,
                    axis_max=args.axis_max,
                    deadzone=args.deadzone,
                    invert_right_horizontal=bool(args.invert_x),
                    invert_right_vertical=bool(args.invert_y),
                    swap_right_xy=bool(args.swap_xy),
                    invert_left_vertical=bool(args.invert_z),
                    invert_left_horizontal=bool(args.invert_yaw),
                )
                forward, left, up_fraction = translation
                print(
                    f"raw R=({raw_x:+6d},{raw_y:+6d}) L=({raw_z:+6d},{raw_yaw:+6d})  "
                    f"vdes_heading=({args.speed * forward:+.2f}, {args.speed * left:+.2f}, "
                    f"{args.vertical_speed * up_fraction:+.2f})m/s  "
                    f"yaw_rate={args.yaw_rate * yaw_fraction:+.2f}rad/s  translation_fraction={magnitude:.3f}",
                    flush=True,
                )
                last_print = now
                pending_print = False
    except KeyboardInterrupt:
        print()
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
