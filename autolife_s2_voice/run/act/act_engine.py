"""S2 动作模块 — PKL 动作回放 + 预置关节位姿 + 技能注册

PKL 预设动作（8 个，录制自机器人关节状态）:
  idle1, idle2, idle3       — 待机位姿（呼吸式微动）
  wave                      — 双手挥手
  left_wave                 — 左手挥手
  right_wave                — 右手挥手
  package_pos               — 包裹位姿
  standup_pos               — 站立位姿

预置关节位姿（4 个，来自 joint_controller.py）:
  home                      — 归零
  left_open                 — 左手张开
  right_open                — 右手张开
  both_open                 — 双手张开

技能注册:
  所有动作通过 get_skills() 导出为 Skill 列表，可合并到 SkillRegistry。
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── 路径 ─────────────────────────────────────────────────
_ACT_DIR = Path(__file__).resolve().parent
# 优先扫描 asset 目录（用户自定义动作：shake_hand, welcome, heart 等）
_ASSET_DIR = (_ACT_DIR / ".." / ".." / "asset").resolve()
# 再扫描 example/pkl 目录（基础动作：wave, idle 等）
_EXAMPLE_PKL_DIR = (_ACT_DIR / ".." / ".." / "example" / "pkl").resolve()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("act")

# ── ROS2 常量 ────────────────────────────────────────────
_ROS_DOMAIN_ID = os.environ.get("ROS_DOMAIN_ID", "0")
_ROBOT_ID = os.environ.get("ROBOT_ID", "0")
_NODE_ID = f"{_ROS_DOMAIN_ID}_{_ROBOT_ID}"
_TOPIC_TARGET = f"/topic_arm_whole_body_target_joints_position_{_NODE_ID}"
_TOPIC_STATUS = f"/topic_arm_whole_body_and_gripper_current_joints_status_{_NODE_ID}"

STATUS_FIELDS = {
    "neck": "neck_joint_state",
    "left_arm": "left_arm_joint_state",
    "right_arm": "right_arm_joint_state",
    "waist_leg": "leg_waist_joint_state",
}

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

TARGET_FIELDS = {
    "neck": "neck_target_joints_position",
    "left_arm": "left_arm_target_joints_position",
    "right_arm": "right_arm_target_joints_position",
    "waist_leg": "leg_waist_target_joints_position",
}

JOINT_LIMITS_HARDWARE: dict[str, tuple[float, float]] = {
    "Joint_Neck_Yaw": (-60, 60), "Joint_Neck_Pitch": (-45, 30), "Joint_Neck_Roll": (-20, 20),
    "Joint_Left_Shoulder_Inner": (-168, 168), "Joint_Left_Shoulder_Outer": (-180, 18),
    "Joint_Left_UpperArm": (-167, 167), "Joint_Left_Elbow": (0, 150),
    "Joint_Left_Forearm": (-169, 169), "Joint_Left_Wrist_Upper": (-80, 80),
    "Joint_Left_Wrist_Lower": (-83, 83),
    "Joint_Right_Shoulder_Inner": (-168, 168), "Joint_Right_Shoulder_Outer": (-18, 180),
    "Joint_Right_UpperArm": (-167, 167), "Joint_Right_Elbow": (-150, 0),
    "Joint_Right_Forearm": (-169, 169), "Joint_Right_Wrist_Upper": (-80, 80),
    "Joint_Right_Wrist_Lower": (-83, 83),
    "Joint_Waist_Yaw": (-180, 180), "Joint_Waist_Pitch": (-80, 149),
    "Joint_Knee": (-155, 155), "Joint_Ankle": (-85, 85),
}
"""硬件物理限位（厂商规格），仅作参考"""


def load_joint_limits(config_path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    """从 configs/joint_limits.yaml 加载运行时关节限位

    若配置文件存在且解析成功，返回配置值；否则回退到硬件默认值。
    YAML 格式: Joint_Name: [min, max]

    Parameters
    ----------
    config_path:
        YAML 配置文件路径，默认 robots/autolife_s2/configs/joint_limits.yaml
    """
    if config_path is None:
        config_path = (Path(__file__).resolve().parents[2]
                       / "configs" / "joint_limits.yaml")
    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        limits: dict[str, tuple[float, float]] = {}
        for name, vals in data.items():
            if isinstance(vals, (list, tuple)) and len(vals) == 2:
                limits[name] = (float(vals[0]), float(vals[1]))
        if limits:
            _logger.info("关节限位加载自 %s (%d 个)", config_path, len(limits))
            return limits
    except Exception as e:
        _logger.warning("加载关节限位配置失败 (%s)，使用硬件默认值", e)
    return dict(JOINT_LIMITS_HARDWARE)


# 运行时关节限位（加载一次）
JOINT_LIMITS = load_joint_limits()

# ── 预置关节位姿 ─────────────────────────────────────────
POSES: dict[str, dict[str, float]] = {
    "home": {n: 0.0 for n in ALL_JOINT_NAMES},
    "left_open": {
        "Joint_Neck_Yaw": 0.0, "Joint_Neck_Pitch": 0.0, "Joint_Neck_Roll": 0.0,
        "Joint_Left_Shoulder_Inner": 68.0, "Joint_Left_Shoulder_Outer": -70.0,
        "Joint_Left_UpperArm": -25.0, "Joint_Left_Elbow": 93.0,
        "Joint_Left_Forearm": -76.0, "Joint_Left_Wrist_Upper": 0.0,
        "Joint_Left_Wrist_Lower": -44.0,
        "Joint_Right_Shoulder_Inner": 0.0, "Joint_Right_Shoulder_Outer": 0.0,
        "Joint_Right_UpperArm": 0.0, "Joint_Right_Elbow": 0.0,
        "Joint_Right_Forearm": 0.0, "Joint_Right_Wrist_Upper": 0.0,
        "Joint_Right_Wrist_Lower": 0.0,
        "Joint_Waist_Yaw": 0.0, "Joint_Waist_Pitch": 0.0,
        "Joint_Knee": 0.0, "Joint_Ankle": 0.0,
    },
    "right_open": {
        "Joint_Neck_Yaw": 0.0, "Joint_Neck_Pitch": 0.0, "Joint_Neck_Roll": 0.0,
        "Joint_Left_Shoulder_Inner": 0.0, "Joint_Left_Shoulder_Outer": 0.0,
        "Joint_Left_UpperArm": 0.0, "Joint_Left_Elbow": 0.0,
        "Joint_Left_Forearm": 0.0, "Joint_Left_Wrist_Upper": 0.0,
        "Joint_Left_Wrist_Lower": 0.0,
        "Joint_Right_Shoulder_Inner": -68.0, "Joint_Right_Shoulder_Outer": 70.0,
        "Joint_Right_UpperArm": 25.0, "Joint_Right_Elbow": -93.0,
        "Joint_Right_Forearm": 76.0, "Joint_Right_Wrist_Upper": 0.0,
        "Joint_Right_Wrist_Lower": 44.0,
        "Joint_Waist_Yaw": 0.0, "Joint_Waist_Pitch": 0.0,
        "Joint_Knee": 0.0, "Joint_Ankle": 0.0,
    },
    "both_open": {
        "Joint_Neck_Yaw": 0.0, "Joint_Neck_Pitch": 0.0, "Joint_Neck_Roll": 0.0,
        "Joint_Left_Shoulder_Inner": 68.0, "Joint_Left_Shoulder_Outer": -70.0,
        "Joint_Left_UpperArm": -25.0, "Joint_Left_Elbow": 93.0,
        "Joint_Left_Forearm": -76.0, "Joint_Left_Wrist_Upper": 0.0,
        "Joint_Left_Wrist_Lower": -44.0,
        "Joint_Right_Shoulder_Inner": -68.0, "Joint_Right_Shoulder_Outer": 70.0,
        "Joint_Right_UpperArm": 25.0, "Joint_Right_Elbow": -93.0,
        "Joint_Right_Forearm": 76.0, "Joint_Right_Wrist_Upper": 0.0,
        "Joint_Right_Wrist_Lower": 44.0,
        "Joint_Waist_Yaw": 0.0, "Joint_Waist_Pitch": 0.0,
        "Joint_Knee": 0.0, "Joint_Ankle": 0.0,
    },
}


# ============================================================================
# 工具函数
# ============================================================================
def clamp_joints(joints: dict) -> dict:
    return {n: max(lo, min(hi, joints[n])) for n, (lo, hi) in JOINT_LIMITS.items()
            if n in joints and n in JOINT_LIMITS}


def joints_to_target(joints: dict) -> dict:
    """将 {Joint_name: deg} 转为 ROS2 目标格式"""
    result = {}
    for group, names in JOINT_GROUPS.items():
        result[TARGET_FIELDS[group]] = [joints.get(n, 0.0) for n in names]
    return result


# ============================================================================
# ROS2 主题发布器（线程安全）
# ============================================================================
class _RosPublisher:
    """ROS2 单例发布器 — 初始化一次，线程安全发布"""

    _instance: _RosPublisher | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _init(self):
        if self._initialized:
            return
        for k in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
            os.environ.pop(k, None)

        # 确保 ROS2 环境已 source
        # 方法：直接用 bash source 后再 import，确保 LD_LIBRARY_PATH 等在动态链接器可见
        if not os.environ.get("ROS_DISTRO") or not os.environ.get("AMENT_PREFIX_PATH"):
            ros_setup = "/opt/ros/jazzy/setup.bash"
            if os.path.exists(ros_setup):
                import subprocess
                try:
                    result = subprocess.run(
                        ["/bin/bash", "-c", f"source {ros_setup} && env -0"],
                        capture_output=True, timeout=5
                    )
                    # env -0 用 null 分隔
                    env_data = result.stdout.decode('utf-8', errors='replace')
                    for entry in env_data.split('\x00'):
                        if '=' in entry:
                            k, v = entry.split('=', 1)
                            if k and k not in ("PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "_", "BASH_FUNC_source%%"):
                                os.environ[k] = v
                    _logger.info("ROS2 环境变量已加载 (ROS_DISTRO=%s)", os.environ.get("ROS_DISTRO", "?"))
                except Exception as e:
                    _logger.error("加载 ROS2 环境变量失败: %s", e)

        # 关键：在 import rclpy 之前预加载 librcl 动态库
        # 确保动态链接器能找到 ROS2 的 .so 文件
        import ctypes
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        for path in ld_path.split(":"):
            path = path.strip()
            if path and os.path.isdir(path):
                try:
                    os.add_dll_directory(path)
                except (OSError, AttributeError):
                    pass  # Linux 上可能不支持，忽略

        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        # 防止 rclpy.init() 重复调用
        try:
            rclpy.init()
            _logger.info("rclpy.init() 成功")
        except (RuntimeError, Exception) as e:
            _logger.warning("rclpy.init() 失败: %s，尝试继续", e)
            if rclpy.ok():
                _logger.info("rclpy 已初始化且 ok，继续")
            else:
                raise RuntimeError(f"rclpy 不可用: {e}") from e

        self._rclpy = rclpy
        self._String = String
        self._node = Node(f"action_player_{_NODE_ID}")
        self._pub = self._node.create_publisher(String, _TOPIC_TARGET, 10)

        # 复位主题
        _TOPIC_RESET = f"/control_reset_{_NODE_ID}"
        self._pub_reset = self._node.create_publisher(String, _TOPIC_RESET, 10)

        # 订阅当前关节状态（用于平滑插值）
        self._status_data: dict | None = None
        self._node.create_subscription(String, _TOPIC_STATUS, self._status_cb, 10)

        # 清除错误 + 使能
        self._publish_enable()
        self._initialized = True
        _logger.info("ROS2 发布器已初始化 (topic=%s, status=%s, reset=%s)",
                     _TOPIC_TARGET, _TOPIC_STATUS, _TOPIC_RESET)

    def _publish_enable(self):
        from std_msgs.msg import String
        enable_topic = f"/topic_arm_joints_set_enable_state_{_NODE_ID}"
        pub = self._node.create_publisher(String, enable_topic, 10)
        msg = String()
        msg.data = json.dumps({j: True for j in ALL_JOINT_NAMES})
        pub.publish(msg)
        self._rclpy.spin_once(self._node, timeout_sec=0.2)

    def _status_cb(self, msg):
        """接收关节状态更新"""
        self._status_data = json.loads(msg.data)

    def read_current(self) -> dict[str, float]:
        """读取当前关节位置，spin 等待获取最新数据"""
        self._init()
        # 如果还没有收到状态数据，等一会儿
        if not self._status_data:
            _logger.info("等待关节状态数据...")
            start = time.time()
            while not self._status_data and time.time() - start < 3.0:
                self._rclpy.spin_once(self._node, timeout_sec=0.5)
        if not self._status_data:
            _logger.warning("未收到关节状态数据，使用零位姿作为当前值")
            return {n: 0.0 for n in ALL_JOINT_NAMES}
        result: dict[str, float] = {}
        for group, names in JOINT_GROUPS.items():
            positions = self._status_data.get(STATUS_FIELDS[group], {}).get("position", [])
            for i, name in enumerate(names):
                if i < len(positions):
                    result[name] = round(positions[i], 2)
        return result

    def publish_smooth(self, target_joints: dict[str, float],
                       move_time: float = 5.0, hold_time: float = 1.0,
                       freq: float = 50.0) -> None:
        """平滑移动到目标关节位置（线性插值）然后保持

        读取当前关节位置 → 在 move_time 秒内线性插值到目标 → 保持 hold_time 秒。
        模仿 joint_controller.move_to_smooth() 的行为。

        Args:
            target_joints: 目标关节位置 {name: deg}
            move_time: 插值移动时长（秒）
            hold_time: 到达后保持时长（秒）
            freq: 发布频率（Hz）
        """
        target_joints = clamp_joints(target_joints)
        current = self.read_current()

        # 如果没读到当前值，直接跳到目标
        if not current:
            _logger.warning("无法读取当前关节状态，直接发布目标位姿")
            target = joints_to_target(target_joints)
            for _ in range(int(hold_time * freq)):
                self.publish(target)
                time.sleep(1 / freq)
            return

        target_fmt = joints_to_target(target_joints)

        interval = 1.0 / freq
        n_move = int(move_time * freq)
        n_hold = int(hold_time * freq)

        _logger.info("平滑移动: %.1fs → 保持 %.1fs (%d 步)", move_time, hold_time, n_move)

        # 移动阶段：线性插值
        for i in range(n_move):
            t = (i + 1) / n_move
            interp: dict[str, float] = {}
            for name in ALL_JOINT_NAMES:
                va = current.get(name, 0.0)
                vb = target_joints.get(name, 0.0)
                interp[name] = va + (vb - va) * t
            interp = clamp_joints(interp)
            self.publish(joints_to_target(interp), spin=False)
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            time.sleep(interval)

        # 保持阶段
        for i in range(n_hold):
            self.publish(target_fmt, spin=False)
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            time.sleep(interval)

        _logger.info("平滑移动完成")

    def publish_reset(self, wait_subscriber: float = 10.0, wait_complete: float = 15.0) -> None:
        """发布复位指令到 /control_reset_<NODE_ID>（参考 joint_controller.reset）

        Args:
            wait_subscriber: 等待订阅者连接的最长时间（秒）
            wait_complete: 发送指令后等待机器人完成复位的时长（秒）
        """
        self._init()
        _logger.info("等待复位 subscriber 连接...")
        start = time.time()
        dots = []
        while time.time() - start < wait_subscriber:
            self._rclpy.spin_once(self._node, timeout_sec=0.5)
            if self._pub_reset.get_subscription_count() > 0:
                break
            dots.append(".")
        if dots:
            _logger.info("".join(dots))

        if self._pub_reset.get_subscription_count() == 0:
            _logger.warning("未检测到复位 subscriber，尝试发送...")

        msg = self._String()
        msg.data = "reset"
        self._pub_reset.publish(msg)
        self._rclpy.spin_once(self._node, timeout_sec=0.5)
        _logger.info("已发送复位指令，等待 %.0fs...", wait_complete)
        time.sleep(wait_complete)

    def publish(self, data: dict, spin: bool = True):
        self._init()
        msg = self._String()
        msg.data = json.dumps(data)
        self._pub.publish(msg)
        if spin:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def shutdown(self):
        if not self._initialized:
            return
        self._node.destroy_node()
        self._rclpy.shutdown()
        _RosPublisher._instance = None
        self._initialized = False
        _logger.info("ROS2 发布器已关闭")


# ============================================================================
# PKL 动作回放器
# ============================================================================
class PklPlayer:
    """PKL 动作回放器 — 按录制帧序列回放到 ROS2 关节目标主题

    用法::

        player = PklPlayer("wave")          # 按名字加载
        player = PklPlayer.from_path("/path/to/wave.pkl")
        player.play()                       # 同步播放（耗时）
        player.play_async()                 # 异步播放（返回 threading.Thread）
    """

    # 已注册的 PKL 动作
    PKL_MAP: dict[str, str] = {}

    def __init__(self, name_or_path: str):
        path = self._resolve(name_or_path)
        self._name = os.path.splitext(os.path.basename(path))[0]
        self._frames = self._load(path)
        _logger.info("PKL 动作 '%s': %d 帧", self._name, len(self._frames))

    @classmethod
    def _resolve(cls, name_or_path: str) -> str:
        # 先查已注册映射
        if name_or_path in cls.PKL_MAP:
            return cls.PKL_MAP[name_or_path]
        # 直接路径
        if os.path.exists(name_or_path):
            return name_or_path
        # 在 asset 目录下查找（用户自定义动作）
        p = _ASSET_DIR / f"{name_or_path}.pkl"
        if p.exists():
            return str(p)
        # 在 example/pkl 目录下查找（基础动作）
        p = _EXAMPLE_PKL_DIR / f"{name_or_path}.pkl"
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"PKL 动作 '{name_or_path}' 未找到，已知: {list(cls.PKL_MAP)}")

    @staticmethod
    def _load(path: str) -> list[tuple[float, dict]]:
        with open(path, "rb") as f:
            raw = pickle.load(f)

        frames = []
        # 格式1: list of [dt, json_str]
        if isinstance(raw, list) and all(isinstance(r, (list, tuple)) and len(r) >= 2 for r in raw):
            for r in raw:
                dt = float(r[0])
                data = json.loads(r[1]) if isinstance(r[1], str) else r[1]
                frames.append((dt, data))
        # 格式2: single tuple (dt, json_str) — 单帧位姿
        elif isinstance(raw, tuple) and len(raw) >= 2:
            dt = float(raw[0])
            data = json.loads(raw[1]) if isinstance(raw[1], str) else raw[1]
            frames.append((dt, data))
            _logger.info("  单帧位姿 → 回放 3 秒")
        else:
            raise ValueError(f"未知 PKL 格式: type={type(raw)}")

        return frames

    @staticmethod
    def _extract_joints(data: dict) -> dict:
        """从录制数据中提取关节位置并转为 ROS2 目标格式"""
        target: dict[str, list[float]] = {}

        for group, names in JOINT_GROUPS.items():
            # 尝试多种 key 名（录制格式差异）
            state = None
            for key_candidate in [
                f"{group}_joint_state",
                # waist_leg 在录制中可能叫 leg_waist_joint_state
                *([f"leg_waist_joint_state"] if group == "waist_leg" else []),
            ]:
                if key_candidate in data:
                    state = data[key_candidate]
                    break

            if state is None:
                # 尝试从 target 字段直接读取
                for target_key in [f"{group}_target_joint_state",
                                   f"{group}_target_joints_position",
                                   *([f"leg_waist_target_joint_state"] if group == "waist_leg" else [])]:
                    if target_key in data:
                        positions = data[target_key]
                        if isinstance(positions, list):
                            target[TARGET_FIELDS[group]] = list(positions[:len(names)])
                            break
                if TARGET_FIELDS[group] not in target:
                    target[TARGET_FIELDS[group]] = [0.0] * len(names)
                continue

            positions = state.get("position", [])
            target[TARGET_FIELDS[group]] = list(positions[:len(names)])
            # 确保长度匹配
            while len(target[TARGET_FIELDS[group]]) < len(names):
                target[TARGET_FIELDS[group]].append(0.0)

        return target

    def _extract_joints_flat(self, data: dict) -> dict[str, float]:
        """从录制数据中提取关节位置并转为展平 {name: deg} 字典"""
        result: dict[str, float] = {}
        for group, names in JOINT_GROUPS.items():
            state: dict | None = None
            for key_candidate in [
                f"{group}_joint_state",
                *([f"leg_waist_joint_state"] if group == "waist_leg" else []),
            ]:
                if key_candidate in data:
                    state = data[key_candidate]
                    break
            if state is None:
                continue
            positions = state.get("position", [])
            for i, name in enumerate(names):
                if i < len(positions):
                    result[name] = float(positions[i])
        return result

    def play(self, loop: int = 1, speed: float = 1.0) -> None:
        """同步回放 PKL 动作

        Args:
            loop: 循环次数
            speed: 播放速度倍率（>1 加速，<1 减速）
        """
        ros = _RosPublisher()
        try:
            # ── 预插值：从当前位置平滑过渡到 PKL 首帧 ──
            if self._frames:
                first_data = self._frames[0][1]
                if isinstance(first_data, str):
                    first_data = json.loads(first_data)
                first_joints = self._extract_joints_flat(first_data)

                # 读取当前位置
                current = ros.read_current()
                if current and any(abs(current.get(n, 0.0) - first_joints.get(n, 0.0)) > 5.0
                                   for n in ALL_JOINT_NAMES):
                    _logger.info("预插值: 当前位置 → PKL 首帧 (3.0s)")
                    ros.publish_smooth(first_joints, move_time=3.0, hold_time=0.0)

            for lap in range(loop):
                if lap > 0:
                    _logger.info("第 %d/%d 遍...", lap + 1, loop)
                prev_time = 0.0
                start_time = time.time()

                for idx, (dt, data) in enumerate(self._frames):
                    if idx == 0:
                        wait_time = dt / speed
                    else:
                        wait_time = max(0, (dt - prev_time) / speed)

                    if wait_time > 0:
                        time.sleep(wait_time)

                    target = self._extract_joints(data)
                    ros.publish(target)

                    prev_time = dt

                _logger.info("第 %d 遍完成 (%.1fs)", lap + 1, time.time() - start_time)

        except Exception as e:
            _logger.error("PKL 回放异常: %s", e)

    # 全局默认播放速度（<1 减速，>1 加速，1.0 原始速度）
    DEFAULT_SPEED: float = 0.4

    def play_async(self, loop: int = 1, speed: float | None = None) -> threading.Thread:
        """异步回放 — 返回后台线程

        Args:
            loop: 循环次数
            speed: 播放速度倍率（None=使用 DEFAULT_SPEED，0.5=半速，1.0=原速）
        """
        _speed = speed if speed is not None else self.DEFAULT_SPEED
        t = threading.Thread(target=self.play, args=(loop, _speed), daemon=True)
        t.start()
        return t

    @property
    def name(self) -> str:
        return self._name

    @property
    def frame_count(self) -> int:
        return len(self._frames)


# ── 自动扫描注册所有 PKL 文件 ────────────────────────────
def _scan_pkl_actions() -> None:
    search_dirs = [_ASSET_DIR, _EXAMPLE_PKL_DIR]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.pkl")):
            name = f.stem  # e.g. "wave", "idle1", "shake_hand"
            # 避免重复注册（asset 优先级高于 example）
            if name not in PklPlayer.PKL_MAP:
                PklPlayer.PKL_MAP[name] = str(f)
    _logger.info("扫描到 %d 个 PKL 预设动作: %s", len(PklPlayer.PKL_MAP),
                 sorted(PklPlayer.PKL_MAP.keys()))


_scan_pkl_actions()


# ============================================================================
# 动作注册表
# ============================================================================
# 类型: (handler_function, description, is_async)
# handler 签名: def handler() -> None  (同步)  或  def handler() -> threading.Thread  (返回线程)
ACTION_REGISTRY: dict[str, tuple[Any, str]] = {}


def register_action(name: str, description: str = ""):
    def decorator(func):
        ACTION_REGISTRY[name] = (func, description)
        return func
    return decorator


# ════════════════════════════════════════════════════════════
# PKL 动作（8 个）
# ════════════════════════════════════════════════════════════

@register_action("idle1", "待机位姿1（呼吸式微动）")
def _action_idle1() -> threading.Thread:
    return PklPlayer("idle1").play_async()


@register_action("idle2", "待机位姿2（呼吸式微动）")
def _action_idle2() -> threading.Thread:
    return PklPlayer("idle2").play_async()


@register_action("idle3", "待机位姿3（呼吸式微动）")
def _action_idle3() -> threading.Thread:
    return PklPlayer("idle3").play_async()


@register_action("wave", "双手挥手打招呼")
def _action_wave() -> threading.Thread:
    return PklPlayer("wave").play_async()


@register_action("left_wave", "左手挥手")
def _action_left_wave() -> threading.Thread:
    return PklPlayer("left_wave").play_async()


@register_action("right_wave", "右手挥手")
def _action_right_wave() -> threading.Thread:
    return PklPlayer("right_wave").play_async()


@register_action("shake_hand", "握手")
def _action_shake_hand() -> threading.Thread:
    return PklPlayer("shake_hand").play_async()


@register_action("welcome", "欢迎")
def _action_welcome() -> threading.Thread:
    return PklPlayer("welcome").play_async()


@register_action("heart", "比心")
def _action_heart() -> threading.Thread:
    return PklPlayer("heart").play_async()


@register_action("package_pos", "包裹位姿")
def _action_package_pos() -> threading.Thread:
    return PklPlayer("package_pos").play_async()


@register_action("standup_pos", "站立位姿")
def _action_standup_pos() -> threading.Thread:
    return PklPlayer("standup_pos").play_async()


# ════════════════════════════════════════════════════════════
# 预置关节位姿（4 个）
# ════════════════════════════════════════════════════════════

@register_action("home", "归零位姿")
def _action_home() -> None:
    """归零位姿 — 平滑插值移动到 home"""
    ros = _RosPublisher()
    try:
        ros.publish_smooth(POSES["home"], move_time=5.0, hold_time=1.0)
        _logger.info("动作 home 完成")
    except Exception as e:
        _logger.error("动作 home 失败: %s", e)


@register_action("left_open", "左手张开")
def _action_left_open() -> None:
    ros = _RosPublisher()
    try:
        ros.publish_smooth(POSES["left_open"], move_time=5.0, hold_time=1.0)
        _logger.info("动作 left_open 完成")
    except Exception as e:
        _logger.error("动作 left_open 失败: %s", e)


@register_action("right_open", "右手张开")
def _action_right_open() -> None:
    ros = _RosPublisher()
    try:
        ros.publish_smooth(POSES["right_open"], move_time=5.0, hold_time=1.0)
        _logger.info("动作 right_open 完成")
    except Exception as e:
        _logger.error("动作 right_open 失败: %s", e)


@register_action("both_open", "双手张开")
def _action_both_open() -> None:
    ros = _RosPublisher()
    try:
        ros.publish_smooth(POSES["both_open"], move_time=5.0, hold_time=1.0)
        _logger.info("动作 both_open 完成")
    except Exception as e:
        _logger.error("动作 both_open 失败: %s", e)


@register_action("reset", "重新复位机器人（复位+解锁安全锁+归零）")
def _action_reset() -> None:
    """语音指令/LLM 触发的复位动作"""
    reset_robot()


# ============================================================================
# 公开 API
# ============================================================================
def execute_action(name: str, **kwargs) -> Any:
    """执行已注册的动作

    Returns:
        动作结果。PKL 动作返回 threading.Thread，预置位姿返回 None。
    """
    if name not in ACTION_REGISTRY:
        raise KeyError(f"未知动作: '{name}'，可用: {list(ACTION_REGISTRY.keys())}")
    handler, _desc = ACTION_REGISTRY[name]
    return handler(**kwargs)


def reset_robot() -> None:
    """复位机器人（发送 /control_reset 指令 + 解锁安全锁 + 归零）

    参考 joint_controller.py:reset()，完整流程：
      1. 发布复位指令到 /control_reset_<NODE_ID>，等待 15s
      2. 清除错误 + 使能所有关节（解锁安全锁）
      3. 发布归零位姿（home），稳定 2s

    Raises:
        RuntimeError: 复位失败时抛出，不再静默吞掉异常
    """
    ros = _RosPublisher()
    try:
        # 1. 发送复位指令（等待 subscriber + 等待完成）
        _logger.info("开始复位流程: 发送 /control_reset 指令...")
        ros.publish_reset(wait_subscriber=10.0, wait_complete=15.0)

        # 2. 清除错误 + 使能（解锁安全锁）
        _logger.info("清除错误 + 使能关节...")
        ros._publish_enable()
        time.sleep(0.5)

        # 3. 执行归零位姿（平滑移动）
        _logger.info("平滑归零...")
        ros.publish_smooth(POSES["home"], move_time=3.0, hold_time=1.0)
        _logger.info("机器人复位完成 ✓ (复位指令 + 安全锁已解锁 + 姿态已归零)")
    except Exception as e:
        _logger.error("机器人复位失败: %s", e, exc_info=True)
        raise RuntimeError(f"reset_robot() 失败: {e}") from e


def get_skills() -> list:
    """导出为 SkillGraph 兼容的 Skill 列表

    可在 Robot 的 register_skills_to_registry() 中调用:
        from autolife_s2_voice.run.act import get_skills
        self.registry.register_many(get_skills())
    """
    from void_cog.agent.skill_graph.decorators import skill
    from void_cog.models.skill_models import SkillSpec

    result = []
    for name, (handler, desc) in ACTION_REGISTRY.items():
        # 包装为 skill
        @skill(name=name, description=desc)
        async def _wrapped(_ctx=None, *, __name=name, **kw):
            import asyncio
            fn, _ = ACTION_REGISTRY[__name]
            return await asyncio.to_thread(fn)
        result.append(_wrapped)
    return result


__all__ = [
    "PklPlayer", "ACTION_REGISTRY", "POSES", "JOINT_GROUPS", "ALL_JOINT_NAMES",
    "execute_action", "register_action", "get_skills",
    "clamp_joints", "joints_to_target", "_RosPublisher",
]
