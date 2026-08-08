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

from joystick_control import map_planar_stick


JS_EVENT = struct.Struct("IhBB")
JS_AXIS = 0x02


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/js0")
    parser.add_argument("--axis-x", type=int, default=0, help="Right-stick horizontal axis.")
    parser.add_argument("--axis-y", type=int, default=1, help="Right-stick vertical axis.")
    parser.add_argument("--axis-max", type=float, default=32767.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--invert-x", type=int, choices=(0, 1), default=1)
    parser.add_argument("--invert-y", type=int, choices=(0, 1), default=0)
    parser.add_argument("--swap-xy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--speed", type=float, default=6.0, help="Maximum vdes magnitude in m/s.")
    return parser.parse_args()


def main():
    args = parse_args()
    axes = {}
    buffer = b""
    last_print = 0.0
    pending_print = False

    print(
        f"Reading {args.device}: horizontal axis={args.axis_x}, vertical axis={args.axis_y}, "
        f"deadzone={args.deadzone:g}, max vdes={args.speed:g}m/s"
    )
    print("Move the right stick; press Ctrl-C to stop.")

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
                body_x, body_y, magnitude = map_planar_stick(
                    raw_x,
                    raw_y,
                    axis_max=args.axis_max,
                    deadzone=args.deadzone,
                    invert_horizontal=bool(args.invert_x),
                    invert_vertical=bool(args.invert_y),
                    swap_xy=bool(args.swap_xy),
                )
                print(
                    f"raw[{args.axis_x}]={raw_x:+6d} raw[{args.axis_y}]={raw_y:+6d}  "
                    f"heading_fraction=({body_x:+.3f}, {body_y:+.3f}, 0.000)  "
                    f"vdes_heading=({args.speed * body_x:+.2f}, {args.speed * body_y:+.2f}, 0.00)m/s  "
                    f"magnitude={magnitude:.3f}",
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
