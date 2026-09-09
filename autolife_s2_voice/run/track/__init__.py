"""Autolife S2 视觉跟踪模块 — 整合到 robot_voice.py 使用

提供:
  - TrackSession: 跟踪会话引擎（视觉循环 + 颈部控制 + 手势 TTS 回调）
  - TrackCommand: 跟踪指令数据模型
  - parse_track_command: 从 ASR 文本解析跟踪指令

典型用法:
    from autolife_s2_voice.run.track import TrackSession, parse_track_command

    session = TrackSession(on_speak=engine._speak_reply, enable_ros2=True)
    await session.start()
    cmd = parse_track_command("跟踪我的手")
    msg = session.start_tracking(cmd)
    if msg:
        await session.speak(msg)
"""

from __future__ import annotations

from autolife_s2_voice.run.track.detector import (
    GestureDetector,
    GestureResponder,
    TrackCommand,
    UnifiedTracker,
    parse_track_command,
)
from autolife_s2_voice.run.track.track_engine import TrackSession

__all__ = [
    "TrackSession",
    "TrackCommand",
    "parse_track_command",
    "GestureDetector",
    "GestureResponder",
    "UnifiedTracker",
]
