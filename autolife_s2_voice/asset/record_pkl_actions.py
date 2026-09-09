#!/usr/bin/env python3
"""只读录制机器人关节状态为与现有动作兼容的 PKL 文件。

示例：
    python3 record_pkl_actions.py hello --duration 8
    python3 record_pkl_actions.py greet

不传 --duration 时按 Ctrl-C 结束并保存。脚本只订阅状态话题，绝不向机器人
发布目标、使能或模式切换指令。
"""

import argparse
import json
import os
import pickle
import signal
import sys
import time
from pathlib import Path


NODE_ID = "0_307"
TOPIC_STATUS = f"/topic_arm_whole_body_and_gripper_current_joints_status_{NODE_ID}"
# 脚本与新录制动作均位于 autolife_s2/asset。
PKL_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="录制全身关节状态为 PKL 动作文件")
    parser.add_argument("name", help="动作名，例如 hello、greet、handshake")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="录制时长（秒）；省略时按 Ctrl-C 结束",
    )
    parser.add_argument("--force", action="store_true", help="允许覆盖同名 PKL 文件")
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        parser.error("动作名只能包含字母、数字、下划线和连字符")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须大于 0")
    return args


def main() -> int:
    args = parse_args()
    output_path = PKL_DIR / f"{args.name}.pkl"
    if output_path.exists() and not args.force:
        print(f"文件已存在: {output_path}", file=sys.stderr)
        print("如确认需要覆盖，请显式传入 --force。", file=sys.stderr)
        return 2

    # 与现有控制脚本保持一致，避免 Conda 环境中预加载库干扰 ROS 运行时。
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        os.environ.pop(key, None)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    frames: list[tuple[float, str]] = []
    started_at: float | None = None

    class StateRecorder(Node):
        def __init__(self) -> None:
            super().__init__("pkl_state_recorder_0_307")
            self.create_subscription(String, TOPIC_STATUS, self.on_status, 100)

        def on_status(self, msg: String) -> None:
            nonlocal started_at
            try:
                json.loads(msg.data)
            except json.JSONDecodeError:
                self.get_logger().warning("跳过一条非 JSON 状态消息")
                return
            now = time.monotonic()
            if started_at is None:
                started_at = now
            frames.append((now - started_at, msg.data))

    rclpy.init()
    node = StateRecorder()
    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    deadline = time.monotonic() + args.duration if args.duration else None

    print(f"订阅: {TOPIC_STATUS}")
    print("开始录制。请仅在机器人已按安全流程进入可示教状态后带动关节。")
    print("结束方式: 到达 --duration，或按 Ctrl-C。")
    try:
        while not stopped and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not frames:
        print("未收到任何关节状态，未写入文件。", file=sys.stderr)
        return 1

    PKL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as file:
        pickle.dump(frames, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, output_path)

    duration = frames[-1][0]
    rate = (len(frames) - 1) / duration if duration > 0 else 0.0
    print(f"已保存: {output_path}")
    print(f"帧数: {len(frames)}, 时长: {duration:.2f}s, 实际采样率: {rate:.1f}Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
