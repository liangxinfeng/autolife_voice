"""TTS 引擎工厂 — 统一封装 Matcha / Piper / Sherpa(VITS MeloTTS) 三种离线 TTS 方案

通过 ``--tts-engine`` 选择后端（默认 matcha），通过 ``--playback`` 选择播放方式：
  - auto:  有机器人喇叭 SHM 则用 SHM，否则本地 sounddevice
  - shm:   机器人喇叭 SHM（需 autolife_robot_sdk + 喇叭模块）
  - local: 本地 sounddevice（开发机/调试用）

三种后端均来自 void_cog.services.tts，CPU 推理，无需 PyTorch/CUDA：
  - matcha: sherpa-onnx + Matcha-TTS（flow-matching + HiFi-GAN，默认，质量好）
  - piper:  piper-tts（最快，RTF~0.08，模型 61MB）
  - sherpa: sherpa-onnx + VITS MeloTTS（中英双语，44.1kHz）

统一接口（与原 TtsEngine 兼容）::

    from autolife_s2_voice.run.tts.tts_factory import create_tts_engine

    engine = create_tts_engine(engine="matcha", playback="auto")
    await engine.speak("你好，云蝶科技")          # 合成并播放
    audio = engine.synthesize("你好")             # 仅合成 -> float32 numpy
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Any, Callable

import numpy as np

from autolife_s2_voice.paths import VOID_COG_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("tts_factory")

# ---------------------------------------------------------------------------
# 引擎常量
# ---------------------------------------------------------------------------
ENGINE_MATCHA = "matcha"
ENGINE_PIPER = "piper"
ENGINE_SHERPA = "sherpa"
VALID_ENGINES = (ENGINE_MATCHA, ENGINE_PIPER, ENGINE_SHERPA)

DEFAULT_ENGINE = ENGINE_MATCHA


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def has_robot_sdk() -> bool:
    """是否可导入 autolife_robot_sdk"""
    try:
        import autolife_robot_sdk  # noqa: F401
        return True
    except Exception:
        return False


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """线性重采样（与 tts_engine 保持一致）"""
    if src_sr == dst_sr:
        return audio
    ratio = src_sr / dst_sr
    out_len = int(len(audio) / ratio)
    indices = (np.arange(out_len) * ratio).astype(int)
    return audio[indices]


def _float32_to_pcm(audio: np.ndarray, sample_width: int) -> bytes:
    if sample_width == 2:
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    elif sample_width == 4:
        return (np.clip(audio, -1, 1) * 2147483647).astype(np.int32).tobytes()
    raise ValueError(f"Unsupported sample_width={sample_width}")


# ---------------------------------------------------------------------------
# 统一 TTS 引擎
# ---------------------------------------------------------------------------
class UnifiedTtsEngine:
    """统一 TTS 引擎 — 包装三种后端服务，提供 speak/synthesize 统一接口

    playback 决定音频去向:
        - shm:   写入机器人喇叭 SHM
        - local: 调用服务自带的 sounddevice 播放
    """

    def __init__(
        self,
        service: Any,
        playback: str = "auto",
        volume: float = 1.0,
        on_result: Callable[[Any], Any] | None = None,
    ) -> None:
        self._service = service
        self._volume = float(volume)
        self._on_result = on_result

        # 探测 SHM 喇叭配置
        self._shm_cfg: dict | None = None
        if playback in ("shm", "auto"):
            self._shm_cfg = self._discover_speaker_shm()

        # 解析最终播放方式
        if playback == "auto":
            self._effective = "shm" if self._shm_cfg else "local"
        elif playback == "shm":
            if not self._shm_cfg:
                raise RuntimeError(
                    "SHM 播放不可用（无 autolife_robot_sdk 或喇叭 SHM），"
                    "请改用 --playback local"
                )
            self._effective = "shm"
        elif playback == "local":
            self._effective = "local"
        else:
            raise ValueError(f"未知 playback: {playback}，可选: auto/shm/local")

        _logger.info(
            "TTS 引擎就绪: %s | 后端播放: %s | 采样率: %d",
            type(service).__name__, self._effective, service.sample_rate,
        )

    # -- SHM 喇叭发现 ------------------------------------------------------

    def _discover_speaker_shm(self) -> dict | None:
        if not has_robot_sdk():
            return None
        try:
            from autolife_s2_voice.run.tts.tts_engine import (
                _find_default_speaker,
                _get_speaker_shm_configs,
            )
            cfgs = _get_speaker_shm_configs()
            mod = _find_default_speaker(cfgs)
            cfg = cfgs.get(mod) if mod else None
            if cfg:
                _logger.info("使用喇叭 SHM: %s", cfg.get("data_shm_name", "?"))
            return cfg
        except Exception as e:
            _logger.warning("SHM 喇叭发现失败，回退本地播放: %s", e)
            return None

    # -- 属性 --------------------------------------------------------------

    @property
    def service(self) -> Any:
        return self._service

    @property
    def engine_name(self) -> str:
        return type(self._service).__name__

    @property
    def playback_mode(self) -> str:
        return self._effective

    @property
    def sample_rate(self) -> int:
        return self._service.sample_rate

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def has_speaker(self) -> bool:
        return self._effective == "shm" or True  # local 始终可尝试

    # -- 合成 --------------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray | None:
        """仅合成，返回 float32 numpy 音频（失败返回 None）"""
        result = self._service.synthesize(text)
        if not getattr(result, "success", False) or result.audio is None:
            return None
        return np.asarray(result.audio, dtype=np.float32)

    # -- 播放 --------------------------------------------------------------

    def _play_shm(self, audio: np.ndarray, sample_rate: int) -> None:
        """写入机器人喇叭 SHM"""
        from autolife_robot_sdk.utils import SHMPCMAudioRingBufferProducer

        producer = SHMPCMAudioRingBufferProducer(
            self._shm_cfg, label="tts_factory", create_if_not_exists=False,
        )
        try:
            target_sr = producer.sample_rate
            target_ch = producer.sample_channels
            target_sw = producer.sample_width
            _logger.info(
                "[SHM] 喇叭: %dHz / %dch / %dbit | 音量: %.0f%%",
                target_sr, target_ch, target_sw * 8, self._volume * 100,
            )

            audio = np.asarray(audio, dtype=np.float32)
            if self._volume != 1.0:
                audio = audio * self._volume
            if sample_rate != target_sr:
                audio = _resample_audio(audio, sample_rate, target_sr)
            if target_ch == 2:
                audio = np.column_stack([audio, audio]).flatten()
            elif target_ch > 1:
                audio = np.tile(audio, target_ch)

            pcm_bytes = _float32_to_pcm(audio, target_sw)
            bytes_per_frame = target_ch * target_sw
            chunk_size = target_sr * bytes_per_frame // 5
            total_bytes = len(pcm_bytes)
            offset = 0
            while offset < total_bytes:
                end = min(offset + chunk_size, total_bytes)
                producer.append(pcm_bytes[offset:end])
                offset = end
                if offset < total_bytes:
                    chunk_dur = (end - offset + chunk_size) / (target_sr * bytes_per_frame)
                    time.sleep(chunk_dur * 0.7)
            tail_dur = total_bytes / (target_sr * bytes_per_frame)
            time.sleep(max(0, tail_dur + 0.3))
        finally:
            producer.close()

    async def speak(self, text: str) -> None:
        """合成并播放（异步，避免阻塞事件循环）"""
        result = await self._service.asynthesize(text)
        if not getattr(result, "success", False) or result.audio is None:
            _logger.warning("合成失败: %s", text[:30])
            return

        if self._on_result:
            if asyncio.iscoroutinefunction(self._on_result):
                await self._on_result(result)
            else:
                self._on_result(result)

        if self._effective == "shm":
            await asyncio.to_thread(self._play_shm, result.audio, result.sample_rate)
        else:
            await self._service.aplay_audio(result.audio, result.sample_rate)

    async def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """播放已合成的音频（用于流水线：合成和播放分离）"""
        if self._effective == "shm":
            await asyncio.to_thread(self._play_shm, audio, sample_rate)
        else:
            await self._service.aplay_audio(audio, sample_rate)

    async def play_audio_stream(self, audio_queue: "queue.Queue[tuple[np.ndarray, int] | None]") -> None:
        """从队列消费音频块，通过单一 SHM 连接连续播放（无间隙）

        队列中每个元素为 (audio_float32, sample_rate)，None 表示结束。
        用于流水线：合成线程持续入队 → SHM 播放线程连续出队写入。
        """
        if self._effective == "shm":
            await asyncio.to_thread(self._play_shm_stream, audio_queue)
        else:
            import queue as _q
            while True:
                try:
                    item = audio_queue.get(timeout=0.1)
                except _q.Empty:
                    continue
                if item is None:
                    break
                audio, sr = item
                await self._service.aplay_audio(audio, sr)

    def _play_shm_stream(self, audio_queue: "queue.Queue[tuple[np.ndarray, int] | None]") -> None:
        """SHM 后台线程：持续从队列取音频写入喇叭（单一 producer 连接）

        句首预滚 100ms 静音防止第一个音节被 SHM 缓冲区吞掉。
        """
        import queue as _q
        from autolife_robot_sdk.utils import SHMPCMAudioRingBufferProducer

        producer = SHMPCMAudioRingBufferProducer(
            self._shm_cfg, label="tts_factory", create_if_not_exists=False,
        )
        try:
            target_sr = producer.sample_rate
            target_ch = producer.sample_channels
            target_sw = producer.sample_width
            _logger.info(
                "[SHM Stream] 喇叭: %dHz / %dch / %dbit | 音量: %.0f%%",
                target_sr, target_ch, target_sw * 8, self._volume * 100,
            )
            bytes_per_frame = target_ch * target_sw
            chunk_size = target_sr * bytes_per_frame // 5  # ~200ms per chunk

            def _prepare(audio: np.ndarray, sr: int) -> bytes:
                audio = np.asarray(audio, dtype=np.float32)
                if self._volume != 1.0:
                    audio = audio * self._volume
                if sr != target_sr:
                    audio = _resample_audio(audio, sr, target_sr)
                if target_ch == 2:
                    audio = np.column_stack([audio, audio]).flatten()
                elif target_ch > 1:
                    audio = np.tile(audio, target_ch)
                return _float32_to_pcm(audio, target_sw)

            def _write_pcm_bytes(pcm: bytes) -> None:
                total_bytes = len(pcm)
                offset = 0
                while offset < total_bytes:
                    end = min(offset + chunk_size, total_bytes)
                    producer.append(pcm[offset:end])
                    offset = end
                    if offset < total_bytes:
                        time.sleep((chunk_size / (target_sr * bytes_per_frame)) * 0.7)

            # ── 句首静音预滚：100ms 零振幅，防止 SHM 缓冲区吞掉首个音节 ──
            pre_roll_samples = int(target_sr * target_ch * 0.1)
            silence = np.zeros(pre_roll_samples, dtype=np.float32)
            _write_pcm_bytes(_prepare(silence, target_sr))

            # 第一块：等合成完成再开始写入
            first = audio_queue.get()
            if first is None:
                return
            first_audio, first_sr = first
            _write_pcm_bytes(_prepare(first_audio, first_sr))

            # 持续消费后续块（无间隙：上一块写完立即写下块）
            while True:
                try:
                    item = audio_queue.get(timeout=0.3)
                except _q.Empty:
                    # 超时：检查是否还有数据
                    continue
                if item is None:
                    break
                audio_np, sr = item
                _write_pcm_bytes(_prepare(audio_np, sr))

            # 尾部缓冲，确保最后一块播放完
            time.sleep(0.3)
        finally:
            producer.close()


# ---------------------------------------------------------------------------
# 工具：构造最小 config dict 给 create_tts_service 使用
# ---------------------------------------------------------------------------
def _build_tts_cfg(
    engine: str,
    config_file: str | None,
    volume: float,
    speed: float | None,
    output_device: int | str | None,
) -> dict:
    """组装传递给 create_tts_service 的配置。

    优先加载 config/default.yaml（如果存在）作为基础，
    再用 engine 子段覆盖参数（volume/speed/output_device/config_file）。
    """
    from pathlib import Path

    import yaml as _yaml

    base_cfg: dict = {}
    default_yaml = VOID_COG_ROOT / "config" / "default.yaml"
    if default_yaml.exists():
        try:
            with open(default_yaml, encoding="utf-8") as f:
                base_cfg = _yaml.safe_load(f) or {}
        except Exception:
            base_cfg = {}
    services = base_cfg.setdefault("services", {})
    tts_cfg = services.setdefault("tts", {})
    tts_cfg.setdefault("provider", engine)
    engine_section = tts_cfg.setdefault(engine, {})
    if not isinstance(engine_section, dict):
        engine_section = {}
        tts_cfg[engine] = engine_section
    if config_file:
        engine_section["config_file"] = config_file
    engine_section["volume"] = volume
    if speed is not None:
        engine_section["speed"] = speed
    if output_device is not None:
        engine_section["output_device"] = output_device
    return base_cfg


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_tts_engine(
    engine: str = DEFAULT_ENGINE,
    playback: str = "auto",
    volume: float = 2.0,
    speed: float | None = None,
    output_device: int | str | None = None,
    on_result: Callable[[Any], Any] | None = None,
    config_file: str | None = None,
) -> UnifiedTtsEngine:
    """创建统一 TTS 引擎

    Args:
        engine: matcha / piper / sherpa（默认 matcha）
        playback: auto / shm / local（默认 auto）
        volume: 音量倍数
        speed: 语速（matcha/sherpa 直接生效；piper 内部转 length_scale）
        output_device: 本地播放设备（None=默认）
        on_result: 合成结果回调
        config_file: 指定 yaml 配置路径（默认各后端各自的 config/tts_<engine>.yaml）

    Returns:
        UnifiedTtsEngine
    """
    engine = (engine or DEFAULT_ENGINE).lower()
    if engine not in VALID_ENGINES:
        raise ValueError(f"未知 TTS 引擎: {engine!r}，可选: {VALID_ENGINES}")

    # 用统一工厂 create_tts_service 创建 AbstractTTSService 实例，保持与 create_asr_service /
    # create_llm_service 完全一致的策略模式链路；不再直接 new 具体 Service 类。
    from void_cog.services.tts import create_tts_service

    cfg = _build_tts_cfg(engine, config_file, volume, speed, output_device)
    service = create_tts_service(cfg, provider=engine)

    # piper 的 speed 通过 length_scale 反向控制；统一工厂已按配置文件加载，这里按需再 override
    if engine == ENGINE_PIPER and speed is not None:
        service.set_speed(speed)

    return UnifiedTtsEngine(
        service=service, playback=playback, volume=volume, on_result=on_result,
    )
