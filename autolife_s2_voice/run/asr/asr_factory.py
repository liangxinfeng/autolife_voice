"""ASR 引擎工厂 — SHM麦克风+VAD / 本地sounddevice 两种方案

当前技术栈: sherpa-onnx SenseVoice-Small（CPU 离线识别）

通过 ``--asr-engine`` 选择后端：
  - auto:   有 autolife_robot_sdk 且麦克风 SHM 运行中 → SHM 持续录音 + silero VAD 断句 → SenseVoice
            无 SDK/SHM → sounddevice 流式监听（开发机调试用）
  - sherpa: 强制 sounddevice 流式监听（VAD 内置在 service 层）

统一接口::

    from autolife_s2_voice.run.asr.asr_factory import create_asr_engine

    async def on_text(text: str):
        print(f"识别: {text}")

    engine = create_asr_engine(engine="auto", on_text=on_text)
    await engine.start()   # 启动持续监听
    await engine.stop()    # 停止
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from autolife_s2_voice.paths import VOID_COG_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("asr_factory")

ENGINE_SHERPA = "sherpa"
ENGINE_AUTO = "auto"
VALID_ENGINES = (ENGINE_AUTO, ENGINE_SHERPA)
DEFAULT_ENGINE = ENGINE_AUTO


# ---------------------------------------------------------------------------
# SDK / SHM 探测
# ---------------------------------------------------------------------------
def _has_robot_sdk() -> bool:
    try:
        import autolife_robot_sdk  # noqa: F401
        return True
    except Exception:
        return False


def _find_microphone_shm() -> dict | None:
    """探测可用的麦克风 SHM 配置"""
    if not _has_robot_sdk():
        return None
    try:
        import json
        from autolife_robot_sdk import SDK_DESCRIPTIONS_ROOT, GLOBAL_VARS

        robot_model = GLOBAL_VARS.ACTIVE_ROBOT_MODEL
        robot_version = GLOBAL_VARS.ACTIVE_ROBOT_VERSION
        config_path = os.path.join(
            SDK_DESCRIPTIONS_ROOT, robot_model, "configs", f"{robot_version}.json",
        )
        with open(config_path) as f:
            data = json.load(f)
        ver_data = data.get(robot_version, {})
        for _mod_name, mod_val in ver_data.get("mod_microphone_", {}).items():
            if not isinstance(mod_val, dict):
                continue
            shm = mod_val.get("settings", {}).get("shm", {})
            if not shm.get("enabled", False):
                continue
            data_name = shm.get("data_shm_name", "")
            if os.path.exists(f"/dev/shm/{data_name.lstrip('/')}"):
                return dict(shm)
        # fallback: any enabled microphone
        for _mod_name, mod_val in ver_data.get("mod_microphone_", {}).items():
            if not isinstance(mod_val, dict):
                continue
            shm = mod_val.get("settings", {}).get("shm", {})
            if shm.get("enabled", False):
                return dict(shm)
    except Exception as e:
        _logger.warning("探测麦克风 SHM 失败: %s", e)
    return None


# ---------------------------------------------------------------------------
# 引擎 1: SHM + VAD + SenseVoice（机器人模式）
# ---------------------------------------------------------------------------
class ShmAsrEngine:
    """SHM ASR 引擎 — SHM 持续录音 + silero VAD 断句 + SenseVoice 识别

    流水线:
      SHM 音频帧 → 重采样 16kHz → silero VAD 检测语音段 → SenseVoice 识别 → 回调
    """

    def __init__(
        self,
        shm_cfg: dict,
        on_text: Callable[[str], Any] | None = None,
        vad_threshold: float = 0.5,
        min_silence_dur: float = 0.6,
        max_speech_dur: float = 8.0,
        min_speech_dur: float = 0.15,
    ) -> None:
        self._shm_cfg = shm_cfg
        self._on_text = on_text
        self._vad_threshold = vad_threshold
        self._min_silence_dur = min_silence_dur
        self._max_speech_dur = max_speech_dur
        self._min_speech_dur = min_speech_dur

        self._svc = None
        self._stop_flag = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_svc(self):
        if self._svc is None:
            from void_cog.services.asr.asr_sherpa import SherpaASRService
            self._svc = SherpaASRService()
        return self._svc

    # -- 捕获线程 ------------------------------------------------------------

    def _capture_vad_loop(self, speech_queue: queue.Queue[np.ndarray]) -> None:
        """后台线程: SHM PCM → 重采样 16kHz → silero VAD → 语音段入队"""
        from autolife_robot_sdk.utils import SHMPCMAudioRingBufferConsumer

        try:
            consumer = SHMPCMAudioRingBufferConsumer(self._shm_cfg)
        except Exception as e:
            _logger.error("无法打开 SHM: %s", e)
            return

        try:
            meta = consumer.get_audio_metadata()
            src_sr = meta["sample_rate"]
            src_ch = meta["sample_channels"]
            src_sw = meta["sample_width"]

            _logger.info("SHM 音频: %dHz/%dch/%dbit → 16kHz 单声道",
                         src_sr, src_ch, src_sw * 8)

            # silero VAD
            import sherpa_onnx
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = str(Path.home() / "models" / "silero_vad.onnx")
            vad_config.silero_vad.threshold = self._vad_threshold
            vad_config.silero_vad.min_silence_duration = self._min_silence_dur
            vad_config.silero_vad.max_speech_duration = self._max_speech_dur
            vad_config.sample_rate = 16000
            vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
            _logger.info("VAD: threshold=%.2f min_silence=%.1fs max_speech=%.1fs",
                         self._vad_threshold, self._min_silence_dur, self._max_speech_dur)

            target_sr = 16000
            frame_samples = 512
            src_frame_bytes = src_sr * src_ch * src_sw * 32 // 1000

            consumer.reset_read_position()
            pcm_buffer = bytearray()

            while not self._stop_flag.is_set():
                if not consumer.wait_for_data(timeout=0.1):
                    continue
                for audio_bytes, _ts in consumer.get_available():
                    pcm_buffer.extend(audio_bytes)

                while len(pcm_buffer) >= src_frame_bytes:
                    raw = bytes(pcm_buffer[:src_frame_bytes])
                    pcm_buffer = pcm_buffer[src_frame_bytes:]

                    if src_sw == 2:
                        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    elif src_sw == 4:
                        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        continue

                    if src_ch > 1:
                        samples = samples.reshape(-1, src_ch).mean(axis=1)

                    if src_sr != target_sr:
                        import librosa
                        samples = librosa.resample(
                            samples.astype(np.float64), orig_sr=src_sr, target_sr=target_sr,
                        ).astype(np.float32)

                    for i in range(0, len(samples) - frame_samples + 1, frame_samples):
                        vad.accept_waveform(samples[i:i + frame_samples])

                while not vad.empty():
                    segment = vad.front.samples
                    vad.pop()
                    seg_np = np.array(segment, dtype=np.float32)
                    if len(seg_np) / target_sr < self._min_speech_dur:
                        continue
                    try:
                        if speech_queue.full():
                            try:
                                speech_queue.get_nowait()
                            except queue.Empty:
                                pass
                        speech_queue.put_nowait(seg_np)
                    except queue.Full:
                        pass

            # flush
            try:
                vad.flush()
                while not vad.empty():
                    segment = vad.front.samples
                    vad.pop()
                    seg_np = np.array(segment, dtype=np.float32)
                    if len(seg_np) / target_sr >= self._min_speech_dur:
                        try:
                            speech_queue.put_nowait(seg_np)
                        except queue.Full:
                            pass
            except Exception:
                pass

        except Exception as e:
            _logger.error("SHM VAD 异常: %s", e)
        finally:
            try:
                consumer.close()
            except Exception:
                pass

    # -- 公开 API ------------------------------------------------------------

    async def start(self) -> None:
        """启动持续监听"""
        self._loop = asyncio.get_running_loop()
        self._stop_flag.clear()
        speech_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)

        capture_thread = threading.Thread(
            target=self._capture_vad_loop,
            args=(speech_queue,),
            daemon=True,
        )
        capture_thread.start()

        svc = self._get_svc()
        _logger.info("SHM ASR 引擎已启动")

        # 识别循环
        while not self._stop_flag.is_set():
            try:
                audio = await asyncio.to_thread(speech_queue.get, True, 0.5)
            except queue.Empty:
                continue

            try:
                result = await asyncio.to_thread(svc.recognize, audio, 16000)
                text = (result.text or "").strip()
                if text and self._on_text:
                    _logger.info("识别: %s", text)
                    ret = self._on_text(text)
                    if asyncio.iscoroutine(ret):
                        await ret
            except Exception as e:
                _logger.error("识别失败: %s", e)

        capture_thread.join(timeout=3.0)
        _logger.info("SHM ASR 引擎已停止")

    async def stop(self) -> None:
        self._stop_flag.set()


# ---------------------------------------------------------------------------
# 引擎 2: Sounddevice 流式监听（开发机模式）
# ---------------------------------------------------------------------------
class SdAsrEngine:
    """Sounddevice ASR 引擎 — sounddevice 麦克风 → VAD → SenseVoice"""

    def __init__(
        self,
        on_text: Callable[[str], Any] | None = None,
        model_path: str | None = None,
        language: str | None = None,
        vad_threshold: float | None = None,
        min_silence_duration: float | None = None,
        max_speech_duration: float | None = None,
        input_device: int | str | None = None,
        config_file: str | None = None,
    ) -> None:
        self._on_text = on_text
        self._input_device = input_device
        self._svc = None

        # 构建配置
        cfg = self._build_cfg(
            model_path=model_path,
            language=language,
            vad_threshold=vad_threshold,
            min_silence_duration=min_silence_duration,
            max_speech_duration=max_speech_duration,
            config_file=config_file,
        )

        from void_cog.services.asr import create_asr_service
        self._svc = create_asr_service(cfg, provider="sherpa")
        self._svc.on_result = self._on_result

    def _build_cfg(self, **kw) -> dict:
        import yaml as _yaml
        base_cfg: dict = {}
        default_yaml = VOID_COG_ROOT / "config" / "default.yaml"
        if default_yaml.exists():
            try:
                with open(default_yaml, encoding="utf-8") as f:
                    base_cfg = _yaml.safe_load(f) or {}
            except Exception:
                pass
        services = base_cfg.setdefault("services", {})
        asr_cfg = services.setdefault("asr", {})
        asr_cfg.setdefault("provider", "sherpa")
        engine_section = asr_cfg.setdefault("sherpa", {})
        for k, v in kw.items():
            if v is not None:
                engine_section[k] = v
        return base_cfg

    async def _on_result(self, result: Any) -> None:
        text = getattr(result, "text", "") or ""
        text = text.strip()
        if not text or not self._on_text:
            return
        if asyncio.iscoroutinefunction(self._on_text):
            await self._on_text(text)
        else:
            self._on_text(text)

    async def start(self) -> None:
        _logger.info("Sounddevice ASR 启动监听")
        await self._svc.start_listening()

    async def stop(self) -> None:
        await self._svc.stop_listening()
        _logger.info("Sounddevice ASR 已停止")


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_asr_engine(
    engine: str = DEFAULT_ENGINE,
    on_text: Callable[[str], Any] | None = None,
    model_path: str | None = None,
    device: str = "auto",
    language: str | None = None,
    vad_threshold: float | None = None,
    min_silence_duration: float | None = None,
    max_speech_duration: float | None = None,
    input_device: int | str | None = None,
    config_file: str | None = None,
) -> ShmAsrEngine | SdAsrEngine:
    """创建 ASR 引擎

    Args:
        engine: auto / sherpa（默认 auto: SHM 优先，否则 sounddevice）
        on_text: 识别结果回调（接收 str，可 async）
        其他参数仅对 sounddevice 模式生效

    Returns:
        ShmAsrEngine 或 SdAsrEngine
    """
    engine = (engine or DEFAULT_ENGINE).lower()
    if engine not in VALID_ENGINES:
        raise ValueError(f"未知 ASR 引擎: {engine!r}，可选: {VALID_ENGINES}")

    shm_cfg = _find_microphone_shm() if engine == ENGINE_AUTO else None

    if shm_cfg:
        _logger.info("ASR 引擎: SHM+VAD (SenseVoice-Small)")
        return ShmAsrEngine(
            shm_cfg=shm_cfg,
            on_text=on_text,
        )

    _logger.info("ASR 引擎: Sounddevice (SenseVoice-Small)")
    return SdAsrEngine(
        on_text=on_text,
        model_path=model_path,
        language=language,
        vad_threshold=vad_threshold,
        min_silence_duration=min_silence_duration,
        max_speech_duration=max_speech_duration,
        input_device=input_device,
        config_file=config_file,
    )
