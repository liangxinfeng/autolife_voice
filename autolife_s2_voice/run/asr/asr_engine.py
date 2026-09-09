"""ASR 引擎 — 基于 Silero VAD + Qwen3-ASR 的正式语音识别模块

参考: robots/autolife_s2/example/asr/vad_asr_demo.py

流水线:
  SHM 麦克风音频 → 降混/重采样 16kHz → Silero VAD → 语音段 → QwenASRService → 回调

用法:
    from autolife_s2_voice.run.asr_engine import AsrEngine

    async def on_text(text: str):
        print(f"识别到: {text}")

    engine = AsrEngine(on_text=on_text)
    await engine.start()      # 启动持续监听
    # ... Ctrl+C 后 ...
    await engine.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# SDK 导入
# ---------------------------------------------------------------------------
try:
    from autolife_robot_sdk.utils import SHMPCMAudioRingBufferConsumer
    from autolife_robot_sdk import SDK_DESCRIPTIONS_ROOT, GLOBAL_VARS
except ImportError:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "..", "sdk_python"))
    from autolife_robot_sdk.utils import SHMPCMAudioRingBufferConsumer
    from autolife_robot_sdk import SDK_DESCRIPTIONS_ROOT, GLOBAL_VARS

ROBOT_MODEL = GLOBAL_VARS.ACTIVE_ROBOT_MODEL
ROBOT_VERSION = GLOBAL_VARS.ACTIVE_ROBOT_VERSION

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
TARGET_SR = 16000
VAD_WINDOW_SEC = 10.0
VAD_CHECK_INTERVAL = 0.3
VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 500
MIN_SEGMENT_DURATION = 2.0
MIN_SKIP_DURATION = 0.3

DEFAULT_MODEL_PATH = "/home/ubuntu/models/Qwen3-ASR-1.7B"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("asr_engine")


# ============================================================================
# 工具函数
# ============================================================================
def _load_robot_config() -> dict:
    config_path = os.path.join(SDK_DESCRIPTIONS_ROOT, ROBOT_MODEL, "configs", f"{ROBOT_VERSION}.json")
    with open(config_path) as f:
        return json.load(f)[ROBOT_VERSION]


def _get_microphone_shm_configs() -> dict[str, dict]:
    config = _load_robot_config()
    result: dict[str, dict] = {}
    for mod_name, mod_val in config.get("mod_microphone_", {}).items():
        if not isinstance(mod_val, dict):
            continue
        shm = mod_val.get("settings", {}).get("shm", {})
        if shm.get("enabled", False):
            result[mod_name] = dict(shm)
    return result


def _find_default_microphone(all_cfgs: dict[str, dict]) -> str | None:
    if not all_cfgs:
        return None
    for mod_name, cfg in all_cfgs.items():
        dn = cfg.get("data_shm_name", "")
        if os.path.exists(f"/dev/shm/{dn.lstrip('/')}"):
            return mod_name
    return next(iter(all_cfgs.keys()))


def _downmix_to_mono(stereo: np.ndarray) -> np.ndarray:
    if stereo.ndim == 1:
        return stereo
    return stereo.mean(axis=1).astype(stereo.dtype)


def _resample_simple(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    ratio = src_rate / dst_rate
    out_len = int(len(audio) / ratio)
    indices = (np.arange(out_len) * ratio).astype(int)
    return audio[indices]


def _detect_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            _logger.info("检测到 GPU: %s", name)
            return "cuda:0"
    except Exception:
        pass
    _logger.info("未检测到 GPU，使用 CPU")
    return "cpu"


# ============================================================================
# Silero VAD 引擎
# ============================================================================
class _SileroDetector:
    """Silero VAD 语音段检测器"""

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = MIN_SPEECH_MS,
        min_silence_ms: int = MIN_SILENCE_MS,
    ):
        from silero_vad import load_silero_vad

        self._vad = load_silero_vad()
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._buffer_start_time: float = 0.0
        self._last_emitted_end_time: float = -999.0

    def feed(self, audio_chunk: np.ndarray, timestamp: float) -> list[_SpeechSegment]:
        from silero_vad import get_speech_timestamps

        if len(self._buffer) == 0:
            self._buffer_start_time = timestamp

        self._buffer = np.concatenate([self._buffer, audio_chunk])
        max_samples = int(TARGET_SR * VAD_WINDOW_SEC)
        if len(self._buffer) > max_samples:
            trim = len(self._buffer) - max_samples
            self._buffer = self._buffer[trim:]
            self._buffer_start_time += trim / TARGET_SR

        import torch
        audio_tensor = torch.from_numpy(self._buffer)
        timestamps = get_speech_timestamps(
            audio_tensor, self._vad, sampling_rate=TARGET_SR,
            threshold=self._threshold,
            min_speech_duration_ms=self._min_speech_ms,
            min_silence_duration_ms=self._min_silence_ms,
        )

        buffer_end_time = self._buffer_start_time + len(self._buffer) / TARGET_SR
        new_segments = []
        for ts in timestamps:
            abs_start = self._buffer_start_time + ts["start"] / TARGET_SR
            abs_end = self._buffer_start_time + ts["end"] / TARGET_SR
            if abs_end <= self._last_emitted_end_time + 0.1:
                continue
            sil_after = buffer_end_time - abs_end
            if sil_after < (self._min_silence_ms / 1000):
                continue

            segment_audio = self._buffer[ts["start"]:ts["end"]]
            new_segments.append(_SpeechSegment(
                start_time=abs_start, end_time=abs_end,
                duration=len(segment_audio) / TARGET_SR,
                audio=segment_audio.copy(),
            ))
            self._last_emitted_end_time = abs_end

        return new_segments

    def reset(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_start_time = 0.0
        self._last_emitted_end_time = -999.0


class _SpeechSegment:
    __slots__ = ("start_time", "end_time", "duration", "audio")

    def __init__(self, start_time: float, end_time: float, duration: float, audio: np.ndarray):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.audio = audio


# ============================================================================
# ASR 引擎
# ============================================================================
class AsrEngine:
    """ASR 语音识别引擎 — 持续监听麦克风 SHM 并回调识别结果

    用法::

        engine = AsrEngine(on_text=lambda text: print(f"识别: {text}"))
        await engine.start()
        # ... 持续运行 ...
        await engine.stop()
    """

    def __init__(
        self,
        on_text: Callable[[str], None] | None = None,
        model_path: str | None = None,
        device: str = "auto",
        mic_mod: str | None = None,
        threshold: float = VAD_THRESHOLD,
    ):
        self._on_text = on_text
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._device = device if device != "auto" else _detect_gpu()
        self._mic_mod = mic_mod
        self._threshold = threshold

        self._service = None
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._service is not None:
            return
        from void_cog.services.asr.asr_qwen import QwenASRService
        _logger.info("加载 QwenASRService (device=%s)", self._device)
        self._service = QwenASRService(
            model_path=self._model_path,
            device=self._device,
        )
        self._service._load_model()

    def recognize(self, audio_float32: np.ndarray) -> str:
        duration = len(audio_float32) / TARGET_SR
        if duration < MIN_SKIP_DURATION:
            return ""

        CHUNK_SEC = 2.0
        target_dur = max(CHUNK_SEC, ((int(duration / CHUNK_SEC) + 1) * CHUNK_SEC))
        if duration < target_dur:
            target_samples = int(target_dur * TARGET_SR)
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[:len(audio_float32)] = audio_float32
            audio_float32 = padded

        with self._lock:
            result = self._service.recognize(audio_float32)
            text = result.text if hasattr(result, 'text') else str(result)
            return text.strip() if text else ""

    # ------------------------------------------------------------------
    # 持续监听
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动持续监听（异步，非阻塞）"""
        if self._running:
            _logger.warning("ASR 引擎已在运行")
            return
        self._running = True
        self._stop_flag.clear()
        self._loop = asyncio.get_running_loop()  # 捕获事件循环引用，供工作线程使用

        # 加载模型
        await asyncio.to_thread(self.load_model)
        _logger.info("ASR 模型就绪")

        # 获取麦克风 SHM
        all_cfgs = _get_microphone_shm_configs()
        if not all_cfgs:
            _logger.error("未找到启用 SHM 的麦克风模块")
            self._running = False
            return
        mic_mod = self._mic_mod or _find_default_microphone(all_cfgs)
        if mic_mod not in all_cfgs:
            _logger.error("麦克风模块 '%s' 不存在", mic_mod)
            self._running = False
            return
        shm_cfg = all_cfgs[mic_mod]

        # 后台线程运行
        await asyncio.to_thread(self._run_vad_asr_loop, shm_cfg, mic_mod)

    def _run_vad_asr_loop(self, shm_cfg: dict, mod_name: str) -> None:
        """后台线程：SHM → VAD → ASR 主循环"""
        _logger.info("连接麦克风 '%s' SHM ...", mod_name)
        consumer = SHMPCMAudioRingBufferConsumer(shm_cfg)
        meta = consumer.get_audio_metadata()
        src_rate = meta["sample_rate"]
        src_channels = meta["sample_channels"]
        _logger.info("已连接: %d Hz / %d ch", src_rate, src_channels)

        detector = _SileroDetector(threshold=self._threshold)

        # ASR 推理队列
        segment_queue: queue.Queue[_SpeechSegment] = queue.Queue()

        def _asr_worker():
            while not self._stop_flag.is_set():
                try:
                    seg = segment_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    t0 = time.time()
                    text = self.recognize(seg.audio)
                    elapsed = time.time() - t0
                    if text:
                        _logger.info("识别: [%.1fs / %.1fs] %s", seg.duration, elapsed, text)
                        if self._on_text:
                            # 回到主线程回调
                            if asyncio.iscoroutinefunction(self._on_text):
                                asyncio.run_coroutine_threadsafe(
                                    self._on_text(text), self._loop
                                )
                            else:
                                self._on_text(text)
                except Exception as e:
                    _logger.error("ASR 异常: %s", e)
                finally:
                    segment_queue.task_done()

        asr_thread = threading.Thread(target=_asr_worker, daemon=True)
        asr_thread.start()

        # 音频攒流
        accumulator: list[np.ndarray] = []
        acc_samples = 0
        target_samples = int(src_rate * VAD_CHECK_INTERVAL)

        consumer.reset_read_position()
        t_start = time.time()
        last_vad = t_start

        def _process_vad(ts: float):
            nonlocal accumulator, acc_samples
            if acc_samples == 0:
                return
            chunk = np.concatenate(accumulator)
            accumulator.clear()
            acc_samples = 0

            raw_2d = chunk.reshape(-1, src_channels)
            mono = _downmix_to_mono(raw_2d).astype(np.float32) / 32768.0
            mono_16k = _resample_simple(mono, src_rate, TARGET_SR)
            chunk_dur = len(chunk) / (src_channels * src_rate)
            chunk_ts = ts - chunk_dur

            for seg in detector.feed(mono_16k, chunk_ts):
                if seg.duration < MIN_SKIP_DURATION:
                    continue
                segment_queue.put(seg)

        try:
            while not self._stop_flag.is_set():
                if not consumer.wait_for_data(timeout=0.3):
                    continue
                for audio_bytes, ts in consumer.get_available():
                    raw = np.frombuffer(audio_bytes, dtype=np.int16)
                    accumulator.append(raw)
                    acc_samples += len(raw)
                    if acc_samples >= target_samples:
                        _process_vad(time.time() - t_start)

                now = time.time()
                if now - last_vad >= VAD_CHECK_INTERVAL and acc_samples > 0:
                    _process_vad(time.time() - t_start)
                    last_vad = now
        except Exception as e:
            _logger.error("VAD 循环异常: %s", e)
        finally:
            if acc_samples > 0:
                _process_vad(time.time() - t_start)
            consumer.close()
            self._stop_flag.set()
            _logger.info("ASR 引擎已停止")

    async def stop(self) -> None:
        """停止持续监听"""
        if not self._running:
            return
        self._stop_flag.set()
        self._running = False
        _logger.info("正在停止 ASR 引擎...")
