"""视觉检测器 — 手势识别 + HSV/皮肤颜色跟踪 + 指令解析

包含:
  - TrackCommand: 跟踪指令数据模型
  - parse_track_command: 从 ASR 文本解析跟踪指令
  - GestureResponder: 手势 → TTS 回复查表
  - GestureDetector: MediaPipe GestureRecognizer + 规则引擎扩展
  - UnifiedTracker: 统一跟踪器（gesture / skin / hsv 三种方法切换）
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from autolife_s2_voice.paths import PACKAGE_ROOT, VOID_COG_ROOT

# Shared gesture service implementation remains in void-cog, while all
# robot-specific rules and replies live inside this ROS package.
_PROJECT_ROOT = VOID_COG_ROOT

# 核心手势引擎服务（项目级）
from void_cog.services.track import RuleBasedGestureEngine  # noqa: E402

_logger = logging.getLogger("track.detector")

# ── 手势配置路径 ──────────────────────────────────────────
# S2 机器人端 TTS 回复映射
_GESTURE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "gestures.json"
)
# 项目级手势规则配置
_RULES_CONFIG_PATH = PACKAGE_ROOT / "configs" / "gesture_rules.json"
_GESTURE_MODEL_DIR = Path(os.path.expanduser("~/models/mediapipe"))
_GESTURE_MODEL = _GESTURE_MODEL_DIR / "gesture_recognizer.task"

_GESTURE_CN: dict[str, str] = {
    "None": "", "Closed_Fist": "握拳", "Open_Palm": "张开手掌",
    "Pointing_Up": "食指指", "Thumb_Down": "拇指向下",
    "Thumb_Up": "竖大拇指", "Victory": "剪刀手", "ILoveYou": "摇滚",
}

# ── HSV 颜色映射 ──────────────────────────────────────────
_COLOR_HSV_MAP: dict[str, tuple[list[int], list[int]]] = {
    "红色": ([0, 100, 100], [10, 255, 255]), "红": ([0, 100, 100], [10, 255, 255]),
    "橙色": ([10, 80, 80], [25, 255, 255]),
    "黄色": ([25, 80, 80], [35, 255, 255]), "黄": ([25, 80, 80], [35, 255, 255]),
    "绿色": ([35, 80, 80], [85, 255, 255]), "绿": ([35, 80, 80], [85, 255, 255]),
    "蓝色": ([100, 100, 100], [130, 255, 255]), "蓝": ([100, 100, 100], [130, 255, 255]),
    "紫色": ([130, 80, 80], [160, 255, 255]),
}
_DEFAULT_HSV_LOW = np.array([15, 80, 80], dtype=np.uint8)
_DEFAULT_HSV_HIGH = np.array([40, 255, 255], dtype=np.uint8)
_SKIN_HSV_LOW = np.array([0, 40, 70], dtype=np.uint8)
_SKIN_HSV_HIGH = np.array([22, 180, 255], dtype=np.uint8)
_RED_LOW1 = np.array([0, 100, 100], dtype=np.uint8)
_RED_HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
_RED_LOW2 = np.array([170, 100, 100], dtype=np.uint8)
_RED_HIGH2 = np.array([179, 255, 255], dtype=np.uint8)

# 手势 TTS 回复冷却（秒）— 同一手势 3 秒内不重复触发
_GESTURE_COOLDOWN = 3.0


# ============================================================================
# 指令模型与解析
# ============================================================================
class TrackCommand:
    """跟踪指令（由 ASR 文本解析而来）

    Attributes
    ----------
    target:
        跟踪目标名称（如 "手" / "脸" / "瓶子"），停止指令时为 None
    method:
        跟踪方法: "gesture" / "skin" / "hsv" / "hsv_default" / "stop" / None
    color:
        颜色名（method=hsv 时有效），如 "红色"
    raw_text:
        原始 ASR 文本
    """
    __slots__ = ("target", "method", "color", "raw_text")

    def __init__(self, target: str | None, method: str | None,
                 color: str | None = None, raw_text: str = "") -> None:
        self.target = target
        self.method = method
        self.color = color
        self.raw_text = raw_text

    @property
    def is_stop(self) -> bool:
        return self.method == "stop"

    @property
    def is_valid(self) -> bool:
        return self.method is not None and self.method != "null"

    def __repr__(self) -> str:
        return (f"TrackCommand(target={self.target!r}, method={self.method!r}, "
                f"color={self.color!r})")


def parse_track_command(text: str) -> TrackCommand:
    """从 ASR 文本解析跟踪指令

    识别规则:
      - 含"停止/取消/停下/别跟踪/不跟踪" → 停止指令
      - 不含"跟踪" → 无效指令（None method）
      - 含"手/手掌/hand/手势" → 手势跟踪
      - 含"脸/面部/face/人脸" → 皮肤色跟踪
      - 含颜色词 → HSV 颜色跟踪
      - 其他 → 默认 HSV 物品跟踪
    """
    text = text.strip()
    if any(w in text for w in ("停止", "取消", "停下", "别跟踪", "不跟踪", "退出跟踪", "结束跟踪")):
        return TrackCommand(None, "stop", raw_text=text)
    if "跟踪" not in text:
        return TrackCommand(None, None, raw_text=text)
    # 手势
    if any(w in text for w in ("手", "手掌", "hand", "Hand", "手势")):
        return TrackCommand("手", "gesture", raw_text=text)
    # 脸（皮肤色跟踪）
    if any(w in text for w in ("脸", "面部", "face", "人脸")):
        return TrackCommand("脸", "skin", raw_text=text)
    # 颜色 + 物体
    for color_name in _COLOR_HSV_MAP:
        if color_name in text:
            remaining = text.replace(color_name, "").replace("跟踪", "").strip()
            target = remaining if remaining else "物体"
            return TrackCommand(target, "hsv", color=color_name, raw_text=text)
    # 默认物体跟踪
    target = text
    for w in ("跟踪", "那个", "这个", "一下"):
        target = target.replace(w, "")
    target = target.strip() or "物体"
    return TrackCommand(target, "hsv_default", raw_text=text)


# ============================================================================
# 手势回复器
# ============================================================================
class GestureResponder:
    """加载 gestures.json + gesture_rules.json，提供手势→回复查表"""

    def __init__(self, config_path: str | Path = str(_GESTURE_CONFIG_PATH)) -> None:
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self._config: dict = json.load(f)
        self._gestures: dict[str, dict] = self._config.get("gestures", {})

        # 合并项目级手势规则配置中的 name_cn / response
        rules_path = _RULES_CONFIG_PATH
        if rules_path.exists():
            with open(rules_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)
            for rule in rules_data.get("gestures", []):
                gid = rule["gesture_id"]
                if gid not in self._gestures:
                    self._gestures[gid] = {
                        "name_cn": rule.get("name_cn", gid),
                        "response": rule.get("response"),
                        "description": rule.get("description", ""),
                    }
            _logger.debug("手势回复表: %d 条（合并自 %s + %s）",
                         len(self._gestures), config_path.name, rules_path.name)

    def get_response(self, gesture_en: str) -> str | None:
        g = self._gestures.get(gesture_en)
        return g.get("response") if g else None

    def get_name_cn(self, gesture_en: str) -> str:
        g = self._gestures.get(gesture_en)
        return g.get("name_cn", gesture_en) if g else _GESTURE_CN.get(gesture_en, gesture_en)


# ============================================================================
# 手势检测器（MediaPipe GestureRecognizer + 规则引擎扩展）
# ============================================================================
class GestureDetector:
    """MediaPipe GestureRecognizer + void_cog 规则引擎兜底"""

    def __init__(self, max_hands: int = 1) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not _GESTURE_MODEL.exists():
            raise FileNotFoundError(f"手势模型未找到: {_GESTURE_MODEL}")

        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(_GESTURE_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._rule_engine = RuleBasedGestureEngine.from_config()

    def detect(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb,
        )
        result = self._recognizer.recognize(mp_image)

        hands: list[dict[str, Any]] = []
        h, w = frame_bgr.shape[:2]

        for i in range(len(result.hand_landmarks)):
            landmarks = result.hand_landmarks[i]
            info: dict[str, Any] = {}
            gesture_en = "None"
            gesture_score = 0.0
            gesture_cn = ""

            if i < len(result.gestures) and result.gestures[i]:
                top = result.gestures[i][0]
                gesture_en = top.category_name
                gesture_score = top.score

            # 规则引擎兜底：MediaPipe 返回 None/Unknown 时用几何规则判定
            if gesture_en in ("None", "Unknown"):
                custom = self._rule_engine.classify(landmarks)
                if custom:
                    gesture_en = custom.gesture_id
                    gesture_cn = custom.name_cn
                    gesture_score = 0.8

            info["gesture_en"] = gesture_en
            info["gesture_cn"] = gesture_cn
            info["gesture_score"] = gesture_score
            if i < len(result.handedness) and result.handedness[i]:
                info["handedness"] = result.handedness[i][0].category_name
            else:
                info["handedness"] = "?"
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            info["bbox"] = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            info["centroid"] = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
            info["landmarks"] = landmarks
            hands.append(info)
        return hands

    def draw(self, frame_bgr: np.ndarray, hand: dict[str, Any],
             label: str = "") -> np.ndarray:
        import cv2
        from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections

        drawing_utils.draw_landmarks(
            frame_bgr, hand["landmarks"], HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1),
        )
        if label:
            cx, cy = hand["centroid"]
            cv2.putText(frame_bgr, label, (cx - 40, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return frame_bgr

    def close(self) -> None:
        self._recognizer.close()


# ============================================================================
# 统一跟踪器（gesture / skin / hsv 三种方法切换）
# ============================================================================
class UnifiedTracker:
    """统一跟踪器 — 根据 TrackCommand 切换检测方法

    方法:
      - gesture: MediaPipe 手势识别 + 规则引擎扩展，触发 TTS 回复
      - skin:    HSV 皮肤色检测（跟踪脸/手区域）
      - hsv:     HSV 颜色检测（跟踪指定颜色物品）
    """

    def __init__(self, responder: GestureResponder) -> None:
        self._method: str = "none"
        self._active: bool = False
        self._target_label: str = ""
        self._hsv_low = _DEFAULT_HSV_LOW.copy()
        self._hsv_high = _DEFAULT_HSV_HIGH.copy()
        self._gesture_detector: GestureDetector | None = None
        self._responder = responder
        self._last_gesture: str = ""
        self._last_gesture_time: float = 0.0

    @property
    def method(self) -> str:
        return self._method

    @property
    def active(self) -> bool:
        return self._active

    @property
    def target_label(self) -> str:
        return self._target_label

    def apply_command(self, cmd: TrackCommand) -> str:
        """应用指令，返回状态消息（用于 TTS 提示）"""
        if cmd.is_stop:
            self._active = False
            self._target_label = ""
            self._last_gesture = ""
            if self._gesture_detector:
                self._gesture_detector.close()
                self._gesture_detector = None
            _logger.info("跟踪已停止")
            return "跟踪已停止"

        if not cmd.is_valid:
            return ""

        method = cmd.method
        if method == "gesture":
            self._method = "gesture"
            self._target_label = cmd.target or "手"
            self._active = True
            if self._gesture_detector is None:
                try:
                    self._gesture_detector = GestureDetector(max_hands=1)
                except Exception as e:
                    _logger.error("手势检测器初始化失败: %s", e)
                    self._active = False
                    return f"手势识别初始化失败: {e}"
            self._last_gesture = ""
            _logger.info("切换到手部检测: %s", self._target_label)
            return f"正在识别 {self._target_label} 手势"

        if method == "skin":
            self._method = "skin"
            self._target_label = cmd.target or "皮肤区域"
            self._active = True
            _logger.info("切换到皮肤检测: %s", self._target_label)
            return f"正在跟踪 {self._target_label}"

        if method in ("hsv", "hsv_default"):
            self._method = "hsv"
            self._target_label = cmd.target or "物体"
            if method == "hsv" and cmd.color and cmd.color in _COLOR_HSV_MAP:
                low, high = _COLOR_HSV_MAP[cmd.color]
                self._hsv_low = np.array(low, dtype=np.uint8)
                self._hsv_high = np.array(high, dtype=np.uint8)
            else:
                self._hsv_low = _DEFAULT_HSV_LOW.copy()
                self._hsv_high = _DEFAULT_HSV_HIGH.copy()
            self._active = True
            color_desc = f"（{cmd.color}）" if cmd.color else ""
            _logger.info("切换到 HSV 物品跟踪: %s%s", self._target_label, color_desc)
            return f"正在跟踪 {self._target_label}{color_desc}"
        return ""

    def detect(self, frame: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """检测并返回 (结果列表, 手势事件)

        手势事件: {"gesture_en": str, "gesture_cn": str, "response": str|None}
                  仅在触发新手势（且通过冷却）时返回，否则 None
        """
        if not self._active:
            return [], None

        results: list[dict[str, Any]] = []
        gesture_event: dict[str, Any] | None = None

        if self._method == "gesture" and self._gesture_detector:
            hands = self._gesture_detector.detect(frame)
            for hand in hands:
                hand["method"] = "gesture"
                g_en = hand.get("gesture_en", "None")
                # gesture_cn 优先用规则引擎返回的，否则查 responder
                if not hand.get("gesture_cn"):
                    hand["gesture_cn"] = self._responder.get_name_cn(g_en)
            # 手势变化触发事件
            if hands:
                g_en = hands[0].get("gesture_en", "None")
                now = time.time()
                if (g_en and g_en != "None"
                        and g_en != self._last_gesture
                        and (now - self._last_gesture_time) >= _GESTURE_COOLDOWN):
                    self._last_gesture = g_en
                    self._last_gesture_time = now
                    gesture_event = {
                        "gesture_en": g_en,
                        "gesture_cn": self._responder.get_name_cn(g_en),
                        "response": self._responder.get_response(g_en),
                    }
                    _logger.info("[手势] %s (%s) → 回复: %s",
                                 gesture_event["gesture_cn"], g_en,
                                 gesture_event["response"] or "(无)")
            else:
                if self._last_gesture and self._last_gesture != "None":
                    self._last_gesture = ""
            results = hands

        elif self._method == "skin":
            det = self._detect_hsv(frame, _SKIN_HSV_LOW, _SKIN_HSV_HIGH)
            if det:
                det["method"] = "skin"
                results = [det]

        elif self._method == "hsv":
            if self._hsv_low[0] == 0 and self._hsv_high[0] == 10:
                det = self._detect_red(frame)
            else:
                det = self._detect_hsv(frame, self._hsv_low, self._hsv_high)
            if det:
                det["method"] = "hsv"
                results = [det]

        return results, gesture_event

    @staticmethod
    def _detect_hsv(frame: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict | None:
        import cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, high)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 300:
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(largest)
        return {"centroid": (cx, cy), "area": area, "bbox": (x, y, w, h)}

    @staticmethod
    def _detect_red(frame: np.ndarray) -> dict | None:
        import cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, _RED_LOW1, _RED_HIGH1),
            cv2.inRange(hsv, _RED_LOW2, _RED_HIGH2),
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 300:
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(largest)
        return {"centroid": (cx, cy), "area": area, "bbox": (x, y, w, h)}

    def draw(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        import cv2
        for det in detections:
            method = det.get("method", "")
            if method == "gesture" and self._gesture_detector:
                label = det.get("gesture_cn", "")
                self._gesture_detector.draw(frame, det, label)
            elif method in ("skin", "hsv"):
                bx, by, bw, bh = det["bbox"]
                cx, cy = det["centroid"]
                color = (0, 255, 255) if method == "skin" else (0, 255, 0)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), (frame.shape[1] // 2, frame.shape[0] // 2),
                         (0, 255, 255), 1, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        if self._gesture_detector:
            self._gesture_detector.close()
            self._gesture_detector = None
