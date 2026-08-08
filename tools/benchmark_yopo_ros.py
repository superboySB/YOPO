#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "python3"


EXPERIMENT_SETS = {
    "l20x80608_epoch50": [
        {
            "name": "t14_timeonly",
            "weight": "trains/L20x80608/omni_ep100_bs64_focus_t14_timeonly_g0_r0/epoch50.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "t13_timeonly",
            "weight": "trains/L20x80608/omni_ep100_bs64_focus_t13_timeonly_g0_r0/epoch50.pth",
            "sgm_time": 1.3,
        },
        {
            "name": "t14_balanced",
            "weight": "trains/L20x80608/omni_ep100_bs64_t14_balanced_smooth_fast_g0_r0/epoch50.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "t14_smooth_plus",
            "weight": "trains/L20x80608/omni_ep100_bs64_t14_smooth_plus_g0_r0/epoch50.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "t14_progress_plus",
            "weight": "trains/L20x80608/omni_ep100_bs64_t14_progress_plus_g0_r0/epoch50.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "t14_safe_smooth",
            "weight": "trains/L20x80608/omni_ep100_bs64_t14_safe_smooth_g0_r0/epoch50.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "t14_clean_state",
            "weight": "trains/L20x80608/omni_ep100_bs64_t14_clean_state_g0_r0/epoch50.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "t14_mixed_speed",
            "weight": "trains/L20x80608/omni_ep100_bs64_t14_mixed_speed_g0_r0/epoch50.pth",
            "sgm_time": 1.4,
        },
    ],
    "l20x80609_epoch100": [
        {
            "name": "glook_base",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_base_t14_progress/epoch100.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "glook_i18_p10",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_i18_p10/epoch100.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "glook_i16_safe",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_i16_p10_wc14/epoch100.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "glook_safe_strong",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_i20_p12_wc18/epoch100.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "glook_r11",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_rmax11_i20/epoch100.pth",
            "sgm_time": 1.4,
            "radius_max": 11.0,
        },
        {
            "name": "glook_r12_safe",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_rmax12_safe/epoch100.pth",
            "sgm_time": 1.4,
            "radius_max": 12.0,
        },
        {
            "name": "glook_exp_mid",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_exp_mid_beta16_w03/epoch100.pth",
            "sgm_time": 1.4,
        },
        {
            "name": "glook_exp_strong_safe",
            "weight": "trains/L20x80609/omni_ep100_bs64_glook_exp_strong_beta25_w05_safe/epoch100.pth",
            "sgm_time": 1.4,
        },
    ],
}


def parse_vec3(text):
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return parts


def shell_quote(path):
    return "'" + str(path).replace("'", "'\"'\"'") + "'"


def controller_env_prefix():
    return (
        "source /opt/ros/noetic/setup.bash && "
        f"source {shell_quote(ROOT / 'Controller/devel/setup.bash')}"
    )


def simulator_env_prefix():
    return (
        "source /opt/ros/noetic/setup.bash && "
        f"source {shell_quote(ROOT / 'Simulator/devel/setup.bash')}"
    )


def planner_env_prefix():
    return (
        "source /opt/ros/noetic/setup.bash && "
        f"source {shell_quote(ROOT / 'Controller/devel/setup.bash')} && "
        f"source {shell_quote(ROOT / 'Simulator/devel/setup.bash')}"
    )


def yopo_pythonpath():
    entries = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        str(ROOT / "Controller/devel/lib/python3/dist-packages"),
        str(ROOT / "Simulator/devel/lib/python3/dist-packages"),
        str(ROOT / "YOPO"),
    ]
    return ":".join(entries)


def launch_process(command, log_path):
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    process._benchmark_log_file = log_file
    return process


def stop_process(process, grace=4.0):
    if process is None:
        return
    if process.poll() is not None:
        if hasattr(process, "_benchmark_log_file"):
            process._benchmark_log_file.close()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    except ProcessLookupError:
        pass
    deadline = time.time() + grace
    while time.time() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    if hasattr(process, "_benchmark_log_file"):
        process._benchmark_log_file.close()


def wait_monitor(process, timeout):
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def write_controller_launch(output_dir, start):
    launch_path = Path(output_dir) / "benchmark_simulator_attitude_control.launch"
    launch_path.write_text(
        f"""<launch>
    <node pkg="so3_quadrotor_simulator" type="quadrotor_simulator_so3" name="quadrotor_simulator_so3" output="screen">
        <param name="rate/odom" value="100.0"/>
        <param name="simulator/init_state_x" value="{start[0]}"/>
        <param name="simulator/init_state_y" value="{start[1]}"/>
        <param name="simulator/init_state_z" value="{start[2]}"/>
        <remap from="~odom" to="/sim/odom"/>
        <remap from="~imu" to="/sim/imu"/>
        <remap from="~cmd" to="so3_cmd"/>
        <remap from="~force_disturbance" to="force_disturbance"/>
        <remap from="~moment_disturbance" to="moment_disturbance"/>
    </node>

    <node pkg="so3_control" type="network_control_node" name="network_controller_node" output="screen">
        <param name="is_simulation" value="true"/>
        <param name="use_disturbance_observer" value="true"/>
        <param name="hover_thrust" value="0.375"/>
        <remap from="~odom" to="/sim/odom"/>
        <remap from="~imu" to="/sim/imu"/>
        <remap from="~position_cmd" to="/so3_control/pos_cmd"/>
        <remap from="~so3_cmd" to="so3_cmd"/>
        <param name="record_log" value="false"/>
        <param name="logger_file_name" value="$(find so3_control)/logger/"/>
    </node>
</launch>
"""
    )
    return launch_path


def make_controller_cmd(controller_launch):
    return (
        f"{controller_env_prefix()} && "
        f"roslaunch {shell_quote(controller_launch)}"
    )


def make_sensor_cmd():
    return (
        f"{simulator_env_prefix()} && "
        f"cd {shell_quote(ROOT / 'Simulator')} && "
        "rosrun sensor_simulator sensor_simulator_cuda"
    )


def make_planner_cmd(args, exp):
    radius_args = ""
    if exp.get("radius_min") is not None:
        radius_args += f" --radius-min {exp['radius_min']}"
    if exp.get("radius_max") is not None:
        radius_args += f" --radius-max {exp['radius_max']}"
    return (
        f"{planner_env_prefix()} && "
        f"cd {shell_quote(ROOT)} && "
        f"PYTHONPATH={shell_quote(yopo_pythonpath())} "
        f"CUDA_VISIBLE_DEVICES={args.gpu} "
        f"{shell_quote(args.python)} YOPO/test_yopo_ros.py "
        f"--weight {shell_quote(exp['weight'])} "
        f"--velocity {args.velocity} "
        f"--max-depth {args.max_depth} "
        f"--arrive-dist {args.arrive_dist} "
        f"--visualize 0 "
        f"--fixed-yaw "
        f"--sgm-time {exp['sgm_time']}"
        f"{radius_args}"
    )


def make_monitor_cmd(args, name, output_json):
    return (
        f"{planner_env_prefix()} && "
        f"cd {shell_quote(ROOT)} && "
        f"PYTHONPATH={shell_quote(yopo_pythonpath())} "
        f"{shell_quote(args.python)} tools/benchmark_yopo_ros.py --monitor "
        f"--name {name} "
        f"--start={','.join(map(str, args.start))} "
        f"--end={','.join(map(str, args.end))} "
        f"--repeats {args.repeats} "
        f"--arrive-dist {args.arrive_dist} "
        f"--segment-timeout {args.segment_timeout} "
        f"--collision-radius {args.collision_radius} "
        f"--output-json {shell_quote(output_json)}"
    )


def default_output_dir(experiment_set):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if experiment_set == "l20x80609_epoch100":
        return ROOT / "trains/L20x80609" / f"benchmark_fixed_yaw_epoch100_{stamp}"
    return ROOT / "trains/L20x80608" / f"benchmark_fixed_yaw_epoch50_{stamp}"


def run_benchmark(args):
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.experiment_set)
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    controller_launch = write_controller_launch(output_dir, args.start)

    experiments = []
    for item in EXPERIMENT_SETS[args.experiment_set]:
        weight = ROOT / item["weight"]
        if not weight.exists():
            raise FileNotFoundError(weight)
        experiments.append({**item, "weight": weight})

    results = []
    for idx, exp in enumerate(experiments, start=1):
        print(f"[{idx}/{len(experiments)}] {exp['name']} starting", flush=True)
        controller = sensor = planner = monitor = None
        result_json = output_dir / f"{exp['name']}.json"
        try:
            controller = launch_process(make_controller_cmd(controller_launch), logs_dir / f"{exp['name']}_controller.log")
            time.sleep(args.sensor_delay)
            sensor = launch_process(make_sensor_cmd(), logs_dir / f"{exp['name']}_sensor.log")
            time.sleep(args.planner_delay)
            planner = launch_process(
                make_planner_cmd(args, exp),
                logs_dir / f"{exp['name']}_planner.log",
            )
            time.sleep(args.monitor_delay)
            monitor = launch_process(make_monitor_cmd(args, exp["name"], result_json), logs_dir / f"{exp['name']}_monitor.log")
            rc = wait_monitor(monitor, args.model_timeout)
            if rc is None:
                print(f"[{exp['name']}] monitor timeout", flush=True)
                result = {
                    "name": exp["name"],
                    "status": "timeout",
                    "weight": str(exp["weight"]),
                    "sgm_time": exp["sgm_time"],
                }
            elif result_json.exists():
                result = json.loads(result_json.read_text())
                result["status"] = result.get("status", "ok" if rc == 0 else f"monitor_exit_{rc}")
            else:
                result = {
                    "name": exp["name"],
                    "status": f"monitor_exit_{rc}",
                    "weight": str(exp["weight"]),
                    "sgm_time": exp["sgm_time"],
                }
            result["weight"] = str(exp["weight"])
            result["sgm_time"] = exp["sgm_time"]
            if exp.get("radius_min") is not None:
                result["radius_min"] = exp["radius_min"]
            if exp.get("radius_max") is not None:
                result["radius_max"] = exp["radius_max"]
            results.append(result)
            print(
                f"[{exp['name']}] {result.get('status')} "
                f"segments={result.get('completed_segments', 0)} "
                f"avg_time={result.get('avg_time_s')} "
                f"avg_speed={result.get('avg_speed_mps')} "
                f"collisions={result.get('collision_count')}",
                flush=True,
            )
        finally:
            stop_process(monitor)
            stop_process(planner)
            stop_process(sensor)
            stop_process(controller)
            time.sleep(args.cooldown)

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "status",
                "sgm_time",
                "completed_segments",
                "avg_time_s",
                "avg_speed_mps",
                "straight_avg_speed_mps",
                "collision_count",
                "min_clearance_m",
                "total_time_s",
                "total_distance_m",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in writer.fieldnames})

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"summary_csv={summary_path}", flush=True)
    print(f"summary_json={output_dir / 'summary.json'}", flush=True)
    return results


def run_monitor(args):
    import rospy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import PointCloud2
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None
        import numpy as np
    else:
        import numpy as np

    state = {
        "pos": None,
        "last_pos": None,
        "last_time": None,
        "path_length": 0.0,
        "collision_count": 0,
        "in_collision": False,
        "min_clearance": float("inf"),
        "map_tree": None,
        "map_points": None,
        "segments": [],
    }

    start = np.array(args.start, dtype=np.float64)
    end = np.array(args.end, dtype=np.float64)
    goals = [end, start] * int(args.repeats)
    current_goal_idx = 0
    current_segment = None
    last_goal_pub = rospy.Time(0)

    def odom_cb(msg):
        now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        pos = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        if state["last_pos"] is not None and state["last_time"] is not None:
            dt = max(0.0, now - state["last_time"])
            if dt < 1.0:
                state["path_length"] += float(np.linalg.norm(pos - state["last_pos"]))
        state["pos"] = pos
        state["last_pos"] = pos
        state["last_time"] = now

        if state["map_tree"] is not None:
            clearance = float(state["map_tree"].query(pos, k=1)[0])
            state["min_clearance"] = min(state["min_clearance"], clearance)
            colliding = clearance < args.collision_radius
            if colliding and not state["in_collision"]:
                state["collision_count"] += 1
            state["in_collision"] = colliding

    def map_cb(msg):
        if state["map_tree"] is not None:
            return
        points = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append((float(p[0]), float(p[1]), float(p[2])))
        if not points:
            return
        points = np.asarray(points, dtype=np.float64)
        state["map_points"] = points
        if cKDTree is not None:
            state["map_tree"] = cKDTree(points)
        else:
            state["map_tree"] = _NumpyTree(points)
        rospy.loginfo("Benchmark map loaded: %d points", points.shape[0])

    class _NumpyTree:
        def __init__(self, points):
            self.points = points

        def query(self, point, k=1):
            del k
            dist = np.linalg.norm(self.points - point[None, :], axis=1)
            return float(dist.min()), int(dist.argmin())

    rospy.init_node(f"yopo_benchmark_{args.name}", anonymous=True)
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    rospy.Subscriber("/sim/odom", Odometry, odom_cb, queue_size=50)
    rospy.Subscriber("/mock_map", PointCloud2, map_cb, queue_size=1)

    rospy.loginfo("Waiting for odom...")
    start_wait = time.time()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and state["pos"] is None and time.time() - start_wait < 30:
        rate.sleep()
    if state["pos"] is None:
        raise RuntimeError("No odom received")

    # Give /mock_map a chance to arrive; collision stats can still run without it.
    map_wait = time.time()
    while not rospy.is_shutdown() and state["map_tree"] is None and time.time() - map_wait < 8:
        rate.sleep()

    def publish_goal(goal):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(goal[0])
        msg.pose.position.y = float(goal[1])
        msg.pose.position.z = float(goal[2])
        msg.pose.orientation.w = 1.0
        goal_pub.publish(msg)

    result = {
        "name": args.name,
        "start": args.start,
        "end": args.end,
        "repeats": args.repeats,
        "arrive_dist": args.arrive_dist,
        "collision_radius": args.collision_radius,
        "segments": [],
    }

    current_segment = {
        "index": 0,
        "goal": goals[0].tolist(),
        "start_wall_time": time.time(),
        "start_ros_time": rospy.Time.now().to_sec(),
        "start_path_length": state["path_length"],
        "straight_distance": float(np.linalg.norm(goals[0] - state["pos"])),
    }
    publish_goal(goals[0])
    last_goal_pub = rospy.Time.now()
    rospy.loginfo("Benchmark first goal: %s", goals[0])

    while not rospy.is_shutdown() and current_goal_idx < len(goals):
        now = rospy.Time.now()
        goal = goals[current_goal_idx]
        if (now - last_goal_pub).to_sec() > 0.5:
            publish_goal(goal)
            last_goal_pub = now
        if state["pos"] is not None:
            dist = float(np.linalg.norm(state["pos"] - goal))
            elapsed = time.time() - current_segment["start_wall_time"]
            if dist <= args.arrive_dist:
                path_delta = state["path_length"] - current_segment["start_path_length"]
                segment = {
                    **current_segment,
                    "end_wall_time": time.time(),
                    "duration_s": elapsed,
                    "path_distance_m": path_delta,
                    "avg_speed_mps": path_delta / max(elapsed, 1e-6),
                    "straight_avg_speed_mps": current_segment["straight_distance"] / max(elapsed, 1e-6),
                    "final_dist_m": dist,
                }
                result["segments"].append(segment)
                rospy.loginfo(
                    "Segment %d reached: time=%.2fs speed=%.2fm/s final_dist=%.2fm",
                    current_goal_idx,
                    segment["duration_s"],
                    segment["avg_speed_mps"],
                    dist,
                )
                current_goal_idx += 1
                if current_goal_idx >= len(goals):
                    break
                time.sleep(0.5)
                next_goal = goals[current_goal_idx]
                current_segment = {
                    "index": current_goal_idx,
                    "goal": next_goal.tolist(),
                    "start_wall_time": time.time(),
                    "start_ros_time": rospy.Time.now().to_sec(),
                    "start_path_length": state["path_length"],
                    "straight_distance": float(np.linalg.norm(next_goal - state["pos"])),
                }
                publish_goal(next_goal)
                last_goal_pub = rospy.Time.now()
            elif elapsed > args.segment_timeout:
                rospy.logwarn("Segment %d timeout after %.2fs, dist=%.2fm", current_goal_idx, elapsed, dist)
                break
        rate.sleep()

    segments = result["segments"]
    total_time = sum(s["duration_s"] for s in segments)
    total_distance = sum(s["path_distance_m"] for s in segments)
    total_straight = sum(s["straight_distance"] for s in segments)
    result.update(
        {
            "status": "ok" if len(segments) == len(goals) else "incomplete",
            "completed_segments": len(segments),
            "total_time_s": total_time,
            "total_distance_m": total_distance,
            "avg_time_s": total_time / len(segments) if segments else None,
            "avg_speed_mps": total_distance / total_time if total_time > 0 else None,
            "straight_avg_speed_mps": total_straight / total_time if total_time > 0 else None,
            "collision_count": state["collision_count"],
            "min_clearance_m": None if math.isinf(state["min_clearance"]) else state["min_clearance"],
            "map_points": None if state["map_points"] is None else int(state["map_points"].shape[0]),
        }
    )
    Path(args.output_json).write_text(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true", help="Run as the ROS monitor node.")
    parser.add_argument("--name", default="benchmark")
    parser.add_argument(
        "--experiment-set",
        choices=sorted(EXPERIMENT_SETS.keys()),
        default="l20x80608_epoch50",
        help="Named benchmark experiment set to run.",
    )
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--velocity", type=float, default=6.0)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--start", type=parse_vec3, default=parse_vec3("-25,0,2"))
    parser.add_argument("--end", type=parse_vec3, default=parse_vec3("25,0,2"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--arrive-dist", type=float, default=1.0)
    parser.add_argument("--collision-radius", type=float, default=0.45)
    parser.add_argument("--segment-timeout", type=float, default=70.0)
    parser.add_argument("--model-timeout", type=float, default=340.0)
    parser.add_argument("--sensor-delay", type=float, default=2.0)
    parser.add_argument("--planner-delay", type=float, default=5.0)
    parser.add_argument("--monitor-delay", type=float, default=5.0)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-json", default="/tmp/yopo_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.monitor:
        run_monitor(cli_args)
    else:
        try:
            run_benchmark(cli_args)
        except KeyboardInterrupt:
            print("Interrupted", file=sys.stderr)
            raise
