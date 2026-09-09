"""颈部跟踪控制器 — 像素偏差 → 颈部关节指令（ROS2，限位可配）"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger("track.neck")

# ── ROS2 常量（与 act_engine.py 一致） ─────────────────────
_ROS_DOMAIN_ID = os.environ.get("ROS_DOMAIN_ID", "0")
_ROBOT_ID = os.environ.get("ROBOT_ID", "307")
_NODE_ID = f"{_ROS_DOMAIN_ID}_{_ROBOT_ID}"
_TOPIC_TARGET = f"/topic_arm_whole_body_target_joints_position_{_NODE_ID}"
_TOPIC_STATUS = f"/topic_arm_whole_body_and_gripper_current_joints_status_{_NODE_ID}"

# ── 头部跟踪控制参数 ──────────────────────────────────────
_DEAD_ZONE = 0.04        # 归一化死区（画面半宽的 4%）
_YAW_GAIN = 20.0         # 满半幅偏移 → 20° Yaw 增量（保守）
_PITCH_GAIN = 15.0       # 满半幅偏移 → 15° Pitch 增量
_MAX_STEP = 1.5          # 每帧最大角度变化（度）— 防突变，慢速安全

# 默认颈部限位（joint_limits.yaml 不存在时的回退值）
_DEFAULT_NECK_LIMITS: dict[str, tuple[float, float]] = {
    "Joint_Neck_Yaw": (-45, 45),
    "Joint_Neck_Pitch": (-20, 20),
    "Joint_Neck_Roll": (-5, 5),
}
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_JOINT_LIMITS_PATH = _CONFIG_DIR / "joint_limits.yaml"


def _load_neck_limits() -> dict[str, tuple[float, float]]:
    """优先从 configs/joint_limits.yaml 加载颈部限位，否则用默认值"""
    if not _JOINT_LIMITS_PATH.exists():
        return dict(_DEFAULT_NECK_LIMITS)
    try:
        import yaml
        with open(_JOINT_LIMITS_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        limits: dict[str, tuple[float, float]] = {}
        for key in ("Joint_Neck_Yaw", "Joint_Neck_Pitch", "Joint_Neck_Roll"):
            val = data.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                limits[key] = (float(val[0]), float(val[1]))
        if len(limits) == 3:
            _logger.info("颈部限位加载自 %s: Yaw=%s, Pitch=%s",
                         _JOINT_LIMITS_PATH, limits.get("Joint_Neck_Yaw"),
                         limits.get("Joint_Neck_Pitch"))
            return limits
    except Exception as e:
        _logger.warning("加载颈部限位失败: %s，使用默认值", e)
    return dict(_DEFAULT_NECK_LIMITS)


class NeckTracker:
    """像素偏差 → 颈部关节指令（归一化映射，分辨率无关）

    符号约定（机器人标准右手系）:
      Yaw 正 = 头向左转 → 手在右侧(cx>0)需要 Yaw 减小 → 用负号
      Pitch 正 = 抬头 → 手在下方(cy>0)需要 Pitch 减小 → 用负号

    Parameters
    ----------
    enable_ros2:
        是否启用 ROS2 颈部控制。False 时仅计算角度不发布，用于无 ROS2 调试。
    limits:
        颈部三轴限位 dict，如 {"Joint_Neck_Yaw": (-45,45), "Joint_Neck_Pitch": (-20,20), ...}。
        默认: Yaw [-45,45], Pitch [-20,20], Roll [-25,25]。
    """

    def __init__(self, enable_ros2: bool = True,
                 limits: dict[str, tuple[float, float]] | None = None) -> None:
        self._enable_ros2 = enable_ros2
        self._limits = limits or _load_neck_limits()
        self._yaw = 0.0
        self._pitch = 0.0
        self._smoothed_yaw = 0.0
        self._smoothed_pitch = 0.0
        if enable_ros2:
            try:
                self._init_ros2()
            except Exception as e:
                _logger.warning("ROS2 颈部控制初始化失败 (%s)，降级到无 ROS2 模式", e)
                self._enable_ros2 = False

    def _init_ros2(self) -> None:
        for k in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
            os.environ.pop(k, None)
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        try:
            rclpy.init()
        except (RuntimeError, Exception) as e:
            if not rclpy.ok():
                raise RuntimeError(f"rclpy 不可用: {e}") from e

        self._rclpy = rclpy
        self._String = String
        self._node = Node(f"track_neck_{_NODE_ID}")
        self._pub = self._node.create_publisher(String, _TOPIC_TARGET, 10)
        self._node.create_subscription(String, _TOPIC_STATUS, self._status_cb, 10)
        _logger.info("ROS2 颈部控制就绪 (topic=%s)", _TOPIC_TARGET)
        self._read_current()

    def _status_cb(self, msg) -> None:
        try:
            data = json.loads(msg.data)
            neck = data.get("neck_joint_state", {})
            pos = neck.get("position", [])
            # neck_joint_state 顺序: [Yaw, Pitch, Roll]（匹配 JOINT_GROUPS["neck"]）
            if len(pos) >= 3:
                self._yaw = float(pos[0])    # Yaw = 左右扭头
                self._pitch = float(pos[1])  # Pitch = 上下点头
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

    def _read_current(self) -> None:
        if self._enable_ros2:
            self._rclpy.spin_once(self._node, timeout_sec=0.5)

    def update(self, centroid_x: int, centroid_y: int,
               frame_w: int, frame_h: int) -> dict[str, Any]:
        """根据目标质心计算颈部关节指令（归一化偏差 → 角度增量）

        Returns
        -------
        dict
            {yaw, pitch, norm_error: (cx_norm, cy_norm), in_dead_zone: bool}
        """
        cx_norm = (centroid_x - frame_w / 2.0) / (frame_w / 2.0)  # [-1, 1], 正=右
        cy_norm = (centroid_y - frame_h / 2.0) / (frame_h / 2.0)  # [-1, 1], 正=下

        in_dead_zone = abs(cx_norm) < _DEAD_ZONE and abs(cy_norm) < _DEAD_ZONE

        if not in_dead_zone:
            target_yaw = self._yaw - cx_norm * _YAW_GAIN
            target_pitch = self._pitch - cy_norm * _PITCH_GAIN

            target_yaw = float(np.clip(target_yaw, *self._limits["Joint_Neck_Yaw"]))
            target_pitch = float(np.clip(target_pitch, *self._limits["Joint_Neck_Pitch"]))

            delta_yaw = float(np.clip(target_yaw - self._smoothed_yaw, -_MAX_STEP, _MAX_STEP))
            delta_pitch = float(np.clip(target_pitch - self._smoothed_pitch, -_MAX_STEP, _MAX_STEP))

            self._smoothed_yaw += delta_yaw
            self._smoothed_pitch += delta_pitch

        self._smoothed_yaw = float(np.clip(self._smoothed_yaw, *self._limits["Joint_Neck_Yaw"]))
        self._smoothed_pitch = float(np.clip(self._smoothed_pitch, *self._limits["Joint_Neck_Pitch"]))

        if self._enable_ros2 and not in_dead_zone:
            self._publish_neck()

        return {
            "yaw": self._smoothed_yaw,
            "pitch": self._smoothed_pitch,
            "norm_error": (cx_norm, cy_norm),
            "in_dead_zone": in_dead_zone,
        }

    def _publish_neck(self) -> None:
        if not self._enable_ros2:
            return
        neck_target = {
            # neck_target_joints_position 顺序: [Yaw, Pitch, Roll]（匹配 JOINT_GROUPS["neck"]）
            "neck_target_joints_position": [self._smoothed_yaw, self._smoothed_pitch, 0.0],
            "left_arm_target_joints_position": [0.0] * 7,
            "right_arm_target_joints_position": [0.0] * 7,
            "leg_waist_target_joints_position": [0.0] * 4,
        }
        msg = self._String()
        msg.data = json.dumps(neck_target)
        self._pub.publish(msg)
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def home(self) -> None:
        """平滑归零颈部（3 秒过渡）"""
        if not self._enable_ros2:
            self._smoothed_yaw = 0.0
            self._smoothed_pitch = 0.0
            self._yaw = 0.0
            self._pitch = 0.0
            return
        _logger.info("颈部归零中...")
        steps = int(3.0 * 50)
        for i in range(steps):
            t = (i + 1) / steps
            self._smoothed_yaw = self._smoothed_yaw * (1 - t)
            self._smoothed_pitch = self._smoothed_pitch * (1 - t)
            self._publish_neck()
            time.sleep(1 / 50)
        self._smoothed_yaw = 0.0
        self._smoothed_pitch = 0.0
        self._yaw = 0.0
        self._pitch = 0.0
        _logger.info("颈部已归零")

    def reset_to_current(self) -> None:
        if self._enable_ros2:
            self._read_current()
        self._smoothed_yaw = self._yaw
        self._smoothed_pitch = self._pitch

    def shutdown(self) -> None:
        if self._enable_ros2 and hasattr(self, "_node"):
            self._node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
            _logger.info("ROS2 颈部控制已关闭")
