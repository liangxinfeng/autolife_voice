"""跟踪会话引擎 — 视觉循环 + 颈部控制 + 手势 TTS 回调

TrackSession 是重构后的核心引擎，与原 RobotTrackEngine 的区别:
  - 不再自持 ASR（由上层 robot_voice.py 的 ASR 回调驱动）
  - 不再自持 TTS（通过 on_speak 回调注入，复用上层 TTS 流水线）
  - 提供 start_tracking / stop_tracking 同步接口，便于上层路由 ASR 指令

用法（被 robot_voice.py 集成）:
    session = TrackSession(
        on_speak=robot_voice._speak_reply,  # 复用上层 TTS
        enable_ros2=True,
        enable_display=True,
    )
    await session.start()                  # 启动视觉循环（后台 task）
    msg = session.start_tracking(cmd)      # ASR 识别"跟踪手" → 启动跟踪
    if msg:
        await session.speak(msg)           # TTS 提示"正在识别手手势"
    ...
    await session.stop_tracking()          # ASR 识别"停止跟踪" → 停止
    await session.stop()                   # 退出时清理
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

import numpy as np

from autolife_s2_voice.run.track.camera import RgbdCamera
from autolife_s2_voice.run.track.detector import (
    GestureResponder,
    TrackCommand,
    UnifiedTracker,
)
from autolife_s2_voice.run.track.neck_tracker import NeckTracker

_logger = logging.getLogger("track.engine")

# TTS 回调类型：接收文本，返回 awaitable
SpeakCallback = Callable[[str], Awaitable[None]]


class TrackSession:
    """S2 机器人视觉跟踪会话 — 视觉循环 + 颈部控制

    Parameters
    ----------
    on_speak:
        TTS 回调，签名 async (text: str) -> None。用于播放手势回复和跟踪状态提示。
        由上层（robot_voice.py）注入，复用其 TTS 流水线。
    sdk_camera:
        RGBD 相机标签名，默认 "rgbd_head"。
    enable_ros2:
        是否启用 ROS2 颈部控制。False 时仅计算角度不发布。
    enable_display:
        是否启用 GUI 显示。无 cv2/GUI 环境自动降级到无 GUI。
    target_fps:
        视觉循环目标帧率（默认 15 fps）。
    gesture_config:
        手势回复配置文件路径（gestures.json）。
    """

    def __init__(
        self,
        *,
        on_speak: SpeakCallback | None = None,
        sdk_camera: str = "rgbd_head",
        enable_ros2: bool = True,
        enable_display: bool = True,
        target_fps: float = 15.0,
        gesture_config: str | None = None,
    ) -> None:
        self._sdk_camera = sdk_camera
        self._enable_ros2 = enable_ros2
        self._target_fps = max(1.0, target_fps)
        self._frame_interval = 1.0 / self._target_fps
        self._on_speak = on_speak
        self._gesture_config = gesture_config

        # 组件（延迟初始化）
        self._camera: RgbdCamera | None = None
        self._tracker: UnifiedTracker | None = None
        self._neck: NeckTracker | None = None

        # 异步控制
        self._stop_event = asyncio.Event()
        self._vision_task: asyncio.Task | None = None
        self._initialized = False

        # GUI（可选）
        self._window_name = "S2 Track"
        self._cv2 = None
        self._enable_display = enable_display
        if enable_display:
            try:
                import cv2
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                self._cv2 = cv2
            except Exception as e:
                _logger.warning("GUI 显示不可用 (%s)，降级到无 GUI 模式", e)
                self._enable_display = False

    @property
    def is_tracking(self) -> bool:
        """当前是否正在跟踪目标"""
        return self._tracker is not None and self._tracker.active

    @property
    def is_running(self) -> bool:
        """视觉循环是否在运行"""
        return self._vision_task is not None and not self._vision_task.done()

    # ------------------------------------------------------------------
    # 初始化（惰性，首次 start 时执行）
    # ------------------------------------------------------------------
    async def _init(self) -> None:
        if self._initialized:
            return

        # 手势回复器 + 跟踪器
        responder = GestureResponder(config_path=self._gesture_config) \
            if self._gesture_config else GestureResponder()
        self._tracker = UnifiedTracker(responder)
        _logger.info("跟踪器就绪 (%d 种手势回复)",
                     len(responder._gestures))

        # RGBD 相机
        try:
            self._camera = RgbdCamera(camera_name=self._sdk_camera)
        except Exception as e:
            _logger.error("RGBD 相机初始化失败: %s", e)
            raise

        # 颈部跟踪器
        self._neck = NeckTracker(enable_ros2=self._enable_ros2)
        self._neck.reset_to_current()
        _logger.info("颈部限位: Yaw=%s, Pitch=%s, Roll=%s",
                     self._neck._limits.get("Joint_Neck_Yaw", "?"),
                     self._neck._limits.get("Joint_Neck_Pitch", "?"),
                     self._neck._limits.get("Joint_Neck_Roll", "?"))

        self._initialized = True

    # ------------------------------------------------------------------
    # 跟踪控制（由上层 ASR 回调调用）
    # ------------------------------------------------------------------
    def start_tracking(self, cmd: TrackCommand) -> str:
        """启动/切换跟踪（同步，非阻塞）

        Returns
        -------
        str
            状态消息（用于 TTS 提示），空串表示无变化
        """
        if self._tracker is None:
            _logger.warning("TrackSession 未初始化，忽略跟踪指令")
            return ""
        msg = self._tracker.apply_command(cmd)
        return msg

    async def stop_tracking(self) -> str:
        """停止跟踪（颈部归零）

        Returns
        -------
        str
            状态消息（"跟踪已停止"）
        """
        if self._tracker is None:
            return ""
        cmd = TrackCommand(None, "stop")
        msg = self._tracker.apply_command(cmd)
        # 颈部归零
        if self._neck:
            await asyncio.to_thread(self._neck.home)
        return msg

    async def speak(self, text: str) -> None:
        """通过注入的 TTS 回调播放文本"""
        if self._on_speak and text:
            try:
                await self._on_speak(text)
            except Exception as e:
                _logger.warning("TTS 回调失败: %s", e)

    # ------------------------------------------------------------------
    # 视觉循环
    # ------------------------------------------------------------------
    async def _vision_loop(self) -> None:
        """主视觉循环：读帧 → 检测 → 头部跟踪 → 手势 TTS 回调"""
        _logger.info("视觉循环启动 (目标 FPS=%.0f)", self._target_fps)
        frame_count = 0
        start_time = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()

            # 读帧
            frame = await asyncio.to_thread(self._camera.read)
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1
            fh, fw = frame.shape[:2]

            # 检测
            detections, gesture_event = await asyncio.to_thread(
                self._tracker.detect, frame)

            # 头部跟踪
            ctrl = {"yaw": 0.0, "pitch": 0.0, "in_dead_zone": True,
                    "norm_error": (0.0, 0.0)}
            if detections:
                det = detections[0]
                cx, cy = det["centroid"]
                ctrl = await asyncio.to_thread(
                    self._neck.update, cx, cy, fw, fh)

            # 手势 TTS 回复（异步回调，不阻塞循环）
            if gesture_event and gesture_event.get("response"):
                response = gesture_event["response"]
                _logger.info("[TTS 触发] %s", response)
                asyncio.create_task(self.speak(response))

            # 显示
            if self._enable_display and self._cv2:
                await asyncio.to_thread(
                    self._draw_frame, frame, detections, ctrl,
                    frame_count, start_time)
                key = self._cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    _logger.info("用户退出视觉窗口")
                    self._stop_event.set()
                    break
            elif frame_count % 30 == 0 and detections:
                fps = frame_count / max(time.time() - start_time, 0.001)
                det_str = (f"gesture={detections[0].get('gesture_cn', '?')}"
                           if detections[0].get("method") == "gesture"
                           else f"area={detections[0]['area']:.0f}")
                _logger.info("  #%d fps=%.1f | %s | yaw=%.1f pitch=%.1f",
                             frame_count, fps, det_str,
                             ctrl["yaw"], ctrl["pitch"])

            # FPS 控制
            elapsed = time.time() - loop_start
            if elapsed < self._frame_interval:
                await asyncio.sleep(self._frame_interval - elapsed)

        _logger.info("视觉循环结束，共 %d 帧", frame_count)

    def _draw_frame(self, frame: np.ndarray, detections: list[dict],
                    ctrl: dict, frame_count: int, start_time: float) -> None:
        cv2 = self._cv2
        display = frame.copy()
        h, w = display.shape[:2]

        if detections:
            self._tracker.draw(display, detections)

        cx, cy = w // 2, h // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        if self._tracker.active:
            method_label = {"gesture": "GESTURE", "skin": "SKIN",
                            "hsv": "HSV"}.get(self._tracker.method, "?")
            cv2.putText(display, f"[{method_label}] {self._tracker.target_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "IDLE — 说 '跟踪我的手'",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        yaw = ctrl.get("yaw", 0)
        pitch = ctrl.get("pitch", 0)
        in_dz = ctrl.get("in_dead_zone", True)
        status = f"Yaw={yaw:+.1f}  Pitch={pitch:+.1f}  {'STABLE' if in_dz else 'TRACK'}"
        color = (0, 255, 0) if in_dz else (0, 255, 255)
        cv2.putText(display, status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = frame_count / max(time.time() - start_time, 0.001)
        cv2.putText(display, f"FPS:{fps:.0f} | q=quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(self._window_name, display)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """初始化并启动视觉循环（后台 task）"""
        await self._init()
        self._stop_event.clear()
        self._vision_task = asyncio.create_task(self._vision_loop())
        _logger.info("TrackSession 已启动")

    async def stop(self) -> None:
        """停止视觉循环并清理资源"""
        _logger.info("正在停止 TrackSession...")
        self._stop_event.set()

        if self._vision_task is not None:
            self._vision_task.cancel()
            try:
                await asyncio.wait_for(self._vision_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._vision_task = None

        # 清理资源
        if self._neck:
            try:
                await asyncio.to_thread(self._neck.home)
            except Exception as e:
                _logger.warning("颈部归零失败: %s", e)
            self._neck.shutdown()
            self._neck = None
        if self._tracker:
            self._tracker.close()
            self._tracker = None
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._enable_display and self._cv2:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass

        self._initialized = False
        _logger.info("TrackSession 已停止")
