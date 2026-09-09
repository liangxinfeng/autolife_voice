#!/usr/bin/env python3
"""PKL 动作文件验证与测试脚本

功能:
    1. 扫描所有 pkl 文件，验证格式
    2. 显示每个文件的帧数、关节数据结构
    3. (可选) 在连接机器人时回放 PKL 动作

用法:
    conda activate robot_env
    source /opt/ros/jazzy/setup.bash

    python3 test_pkl_actions.py                    # 仅验证格式（不连接机器人）
    python3 test_pkl_actions.py --play wave        # 回放指定动作
    python3 test_pkl_actions.py --play-all         # 回放所有动作
    python3 test_pkl_actions.py --play-all --speed 0.5  # 慢速回放
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

_ASSET_DIR = Path(__file__).resolve().parent
_PKL_DIR = _ASSET_DIR.parent / "example" / "pkl"

JOINT_GROUPS = {
    "neck": ["Joint_Neck_Yaw", "Joint_Neck_Pitch", "Joint_Neck_Roll"],
    "left_arm": [
        "Joint_Left_Shoulder_Inner", "Joint_Left_Shoulder_Outer",
        "Joint_Left_UpperArm", "Joint_Left_Elbow",
        "Joint_Left_Forearm", "Joint_Left_Wrist_Upper", "Joint_Left_Wrist_Lower",
    ],
    "right_arm": [
        "Joint_Right_Shoulder_Inner", "Joint_Right_Shoulder_Outer",
        "Joint_Right_UpperArm", "Joint_Right_Elbow",
        "Joint_Right_Forearm", "Joint_Right_Wrist_Upper", "Joint_Right_Wrist_Lower",
    ],
    "waist_leg": ["Joint_Waist_Yaw", "Joint_Waist_Pitch", "Joint_Knee", "Joint_Ankle"],
}

ALL_JOINT_NAMES = [n for g in JOINT_GROUPS.values() for n in g]


def load_pkl(path: str) -> list[tuple[float, dict]]:
    """加载 PKL 文件并返回帧列表"""
    with open(path, "rb") as f:
        raw = pickle.load(f)

    frames = []
    if isinstance(raw, list) and all(isinstance(r, (list, tuple)) and len(r) >= 2 for r in raw):
        for r in raw:
            dt = float(r[0])
            data = json.loads(r[1]) if isinstance(r[1], str) else r[1]
            frames.append((dt, data))
    elif isinstance(raw, tuple) and len(raw) >= 2:
        dt = float(raw[0])
        data = json.loads(raw[1]) if isinstance(raw[1], str) else raw[1]
        frames.append((dt, data))
    else:
        raise ValueError(f"未知 PKL 格式: type={type(raw)}")

    return frames


def find_pkl(name: str) -> Path | None:
    """优先使用用户录制的 asset，兼容旧 example/pkl 示例文件。"""
    for directory in (_ASSET_DIR, _PKL_DIR):
        path = directory / f"{name}.pkl"
        if path.exists():
            return path
    return None


def analyze_pkl(name: str, path: str) -> dict:
    """分析单个 PKL 文件的内容"""
    result = {
        "name": name,
        "path": path,
        "file_size": os.path.getsize(path),
        "valid": False,
        "frame_count": 0,
        "duration": 0.0,
        "format_type": "unknown",
        "groups_found": [],
        "error": None,
    }

    try:
        frames = load_pkl(path)
        result["valid"] = True
        result["frame_count"] = len(frames)

        # 计算时长
        if len(frames) > 0:
            result["duration"] = frames[-1][0] - frames[0][0]

        # 判断格式
        if len(frames) > 1:
            result["format_type"] = "multi_frame_sequence"
        elif len(frames) == 1:
            result["format_type"] = "single_frame_pose"

        # 分析关节组
        if len(frames) > 0:
            first_frame = frames[0][1]
            groups_found = []
            for group in JOINT_GROUPS:
                key_candidates = [f"{group}_joint_state"]
                if group == "waist_leg":
                    key_candidates.append("leg_waist_joint_state")
                for key in key_candidates:
                    if key in first_frame:
                        state = first_frame[key]
                        if isinstance(state, dict) and "position" in state:
                            groups_found.append(f"{group}({len(state['position'])} joints)")
                        break
            result["groups_found"] = groups_found

    except Exception as e:
        result["error"] = str(e)

    return result


def extract_joints_for_play(data: dict) -> dict:
    """从录制数据中提取关节位置转为 ROS2 目标格式"""
    target: dict[str, list[float]] = {}

    for group, names in JOINT_GROUPS.items():
        state = None
        key_candidates = [f"{group}_joint_state"]
        if group == "waist_leg":
            key_candidates.append("leg_waist_joint_state")

        for key in key_candidates:
            if key in data:
                state = data[key]
                break

        if state is None:
            target_key_candidates = [
                f"{group}_target_joint_state",
                f"{group}_target_joints_position",
            ]
            if group == "waist_leg":
                target_key_candidates.append("leg_waist_target_joint_state")

            for tk in target_key_candidates:
                if tk in data:
                    positions = data[tk]
                    if isinstance(positions, list):
                        target[f"{group}_target_joints_position"] = list(positions[:len(names)])
                        break
            if f"{group}_target_joints_position" not in target:
                target[f"{group}_target_joints_position"] = [0.0] * len(names)
            continue

        positions = state.get("position", [])
        target[f"{group}_target_joints_position"] = list(positions[:len(names)])
        while len(target[f"{group}_target_joints_position"]) < len(names):
            target[f"{group}_target_joints_position"].append(0.0)

    return target


def create_ros2_publisher() -> tuple[Any, Any, Any, Any, bool] | None:
    """为一次完整回放创建并返回一个可复用的 ROS2 发布器。"""
    ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
    robot_id = os.environ.get("ROBOT_ID", "0")
    node_id = f"{ros_domain_id}_{robot_id}"
    topic = f"/topic_arm_whole_body_target_joints_position_{node_id}"

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        # 清除可能影响的环境变量
        for k in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
            os.environ.pop(k, None)

        owns_rclpy = not rclpy.ok()
        if owns_rclpy:
            rclpy.init()

        node = Node(f"pkl_test_{node_id}")
        pub = node.create_publisher(String, topic, 10)
        return rclpy, node, pub, String, owns_rclpy

    except ImportError:
        print("  [WARN] rclpy 不可用，跳过发布")
        return None


def play_pkl(name: str, speed: float = 1.0, loop: int = 1) -> bool:
    """回放 PKL 动作"""
    path = find_pkl(name)
    if path is None:
        print(f"[ERROR] 文件不存在: {_ASSET_DIR / f'{name}.pkl'}")
        return False

    try:
        frames = load_pkl(str(path))
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        return False

    print(f"\n{'='*60}")
    print(f"  回放动作: {name}")
    print(f"  帧数: {len(frames)}, 速度: {speed}x, 循环: {loop}")
    print(f"{'='*60}")

    ros = create_ros2_publisher()
    if ros is None:
        return False
    rclpy, node, pub, String, owns_rclpy = ros

    try:
        for lap in range(loop):
            if loop > 1:
                print(f"  --- 第 {lap + 1}/{loop} 遍 ---")

            prev_time = 0.0
            start_time = time.time()

            for idx, (dt, data) in enumerate(frames):
                if idx == 0:
                    wait_time = dt / speed
                else:
                    wait_time = max(0, (dt - prev_time) / speed)

                if wait_time > 0:
                    time.sleep(wait_time)

                msg = String()
                msg.data = json.dumps(extract_joints_for_play(data))
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                prev_time = dt

            elapsed = time.time() - start_time
            print(f"  完成 (耗时 {elapsed:.1f}s)")
    finally:
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()

    return True


def validate_all_pkls() -> list[dict]:
    """验证所有 PKL 文件"""
    results = []

    pkl_files_by_name: dict[str, Path] = {}
    for directory in (_PKL_DIR, _ASSET_DIR):
        if not directory.is_dir():
            continue
        for pkl_file in directory.glob("*.pkl"):
            # asset 后遍历，故同名文件优先使用新录制的版本。
            pkl_files_by_name[pkl_file.name] = pkl_file

    if not pkl_files_by_name:
        print(f"[ERROR] PKL 目录不存在或为空: {_ASSET_DIR}, {_PKL_DIR}")
        return results

    pkl_files = [pkl_files_by_name[name] for name in sorted(pkl_files_by_name)]
    print(f"发现 {len(pkl_files)} 个 PKL 文件:\n")

    for pkl_file in pkl_files:
        result = analyze_pkl(pkl_file.stem, str(pkl_file))
        results.append(result)

        status = "✓" if result["valid"] else "✗"
        print(f"  [{status}] {result['name']}.pkl")
        print(f"        大小: {result['file_size']:,} bytes")
        if result["valid"]:
            print(f"        格式: {result['format_type']}")
            print(f"        帧数: {result['frame_count']}")
            print(f"        时长: {result['duration']:.2f}s")
            print(f"        关节组: {', '.join(result['groups_found']) or '(无)'}")
        else:
            print(f"        错误: {result['error']}")
        print()

    # 汇总
    valid_count = sum(1 for r in results if r["valid"])
    total_frames = sum(r["frame_count"] for r in results if r["valid"])
    total_duration = sum(r["duration"] for r in results if r["valid"])

    print(f"{'='*60}")
    print(f"  汇总: {valid_count}/{len(results)} 有效, "
          f"共 {total_frames} 帧, 总时长 {total_duration:.2f}s")
    print(f"{'='*60}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PKL 动作文件验证与测试")
    parser.add_argument("--play", type=str, default=None,
                        help="回放指定的 PKL 动作 (e.g. wave)")
    parser.add_argument("--play-all", action="store_true",
                        help="回放所有有效的 PKL 动作")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="播放速度倍率 (default: 1.0)")
    parser.add_argument("--loop", type=int, default=1,
                        help="循环次数 (default: 1)")
    parser.add_argument("--validate-only", action="store_true",
                        help="仅验证格式，不回放")
    args = parser.parse_args()

    print("=" * 60)
    print("  PKL 动作文件验证与测试")
    print("=" * 60)

    # 第一步: 验证所有 PKL 文件
    results = validate_all_pkls()

    if args.validate_only or (args.play is None and not args.play_all):
        print("\n[验证完成] 使用 --play <name> 或 --play-all 来回放动作")
        return

    # 第二步: 回放动作
    valid_actions = [r["name"] for r in results if r["valid"]]

    if args.play:
        if args.play not in valid_actions:
            print(f"\n[ERROR] 无效的动作名: {args.play}")
            print(f"  可用动作: {', '.join(valid_actions)}")
            sys.exit(1)
        play_pkl(args.play, speed=args.speed, loop=args.loop)

    elif args.play_all:
        print(f"\n开始回放所有有效动作 (共 {len(valid_actions)} 个)...")
        for i, name in enumerate(valid_actions):
            print(f"\n--- [{i+1}/{len(valid_actions)}] ---")
            play_pkl(name, speed=args.speed, loop=args.loop)
            if i < len(valid_actions) - 1:
                time.sleep(1)

    print("\n[测试完成]")


if __name__ == "__main__":
    main()
