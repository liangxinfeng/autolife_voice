"""TTS 引擎 — 基于 Piper-TTS 的正式语音合成与 SHM 播放模块

参考: robots/autolife_s2/example/tts/piper_tts_demo.py

流水线:
  文本 → Piper (onnxruntime) → float32 音频 → PCM 重采样 → 喇叭 SHM → 机器人播放

用法:
    from autolife_s2_voice.run.tts_engine import TtsEngine

    engine = TtsEngine()
    await engine.speak("你好，欢迎来到云蝶。")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# SDK 导入
# ---------------------------------------------------------------------------
try:
    from autolife_robot_sdk.utils import SHMPCMAudioRingBufferProducer
    from autolife_robot_sdk import SDK_DESCRIPTIONS_ROOT, GLOBAL_VARS
except ImportError:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "..", "sdk_python"))
    from autolife_robot_sdk.utils import SHMPCMAudioRingBufferProducer
    from autolife_robot_sdk import SDK_DESCRIPTIONS_ROOT, GLOBAL_VARS

ROBOT_MODEL = GLOBAL_VARS.ACTIVE_ROBOT_MODEL
ROBOT_VERSION = GLOBAL_VARS.ACTIVE_ROBOT_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("tts_engine")


# ============================================================================
# Piper 模型扫描
# ============================================================================
def _find_piper_models() -> dict[str, str]:
    """扫描已知路径下的 Piper .onnx 模型"""
    models: dict[str, str] = {}
    scan_dirs = [
        "/home/ubuntu/miniconda3/envs/robot_env/lib/python3.12/site-packages/autolife_robot_vision/assets/tts/piper",
    ]
    try:
        import autolife_robot_vision
        pkg_dir = os.path.join(
            os.path.dirname(autolife_robot_vision.__file__),
            "assets", "tts", "piper",
        )
        scan_dirs.insert(0, pkg_dir)
    except ImportError:
        pass

    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname.endswith(".onnx"):
                key = fname.replace(".onnx", "")
                models.setdefault(key, os.path.join(scan_dir, fname))
    return models


# ============================================================================
# 配置加载（喇叭 SHM）
# ============================================================================
def _load_robot_config() -> dict:
    config_path = os.path.join(SDK_DESCRIPTIONS_ROOT, ROBOT_MODEL, "configs", f"{ROBOT_VERSION}.json")
    with open(config_path) as f:
        return json.load(f)[ROBOT_VERSION]


def _get_speaker_shm_configs() -> dict[str, dict]:
    config = _load_robot_config()
    result: dict[str, dict] = {}
    for mod_name, mod_val in config.get("mod_speaker_", {}).items():
        if not isinstance(mod_val, dict):
            continue
        settings = mod_val.get("settings", {})
        shm = dict(settings.get("input_shm") or settings.get("speak_shm") or {})
        if shm.get("enabled", False):
            result[mod_name] = shm
    return result


def _find_default_speaker(all_cfgs: dict[str, dict]) -> str | None:
    if not all_cfgs:
        return None
    for mod_name, cfg in all_cfgs.items():
        dn = cfg.get("data_shm_name", "")
        if os.path.exists(f"/dev/shm/{dn.lstrip('/')}"):
            return mod_name
    return next(iter(all_cfgs.keys()))


# ============================================================================
# 音频工具
# ============================================================================
def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
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


# ============================================================================
# Piper TTS Runner
# ============================================================================
class _PiperRunner:
    """Piper-TTS 封装（onnxruntime 推理）"""

    def __init__(self, model_path: str, use_cuda: bool = False):
        from piper import PiperVoice, SynthesisConfig

        config_path = model_path + ".json"
        self._model_path = model_path

        with open(config_path) as f:
            cfg = json.load(f)
        self._sample_rate: int = int(cfg["audio"]["sample_rate"])

        t0 = time.time()
        self._voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)
        self._load_time = time.time() - t0

        self._synth_cfg = SynthesisConfig(
            noise_scale=float(cfg["inference"].get("noise_scale", 0.667)),
            length_scale=float(cfg["inference"].get("length_scale", 1.0)),
            noise_w_scale=float(cfg["inference"].get("noise_w", 0.8)),
        )
        _logger.info("Piper 模型加载: %s (%s) %.1fs",
                      os.path.basename(model_path),
                      "GPU" if use_cuda else "CPU",
                      self._load_time)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        t0 = time.time()
        chunks = list(self._voice.synthesize(text, syn_config=self._synth_cfg))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        elapsed = time.time() - t0
        dur = len(audio) / self._sample_rate
        _logger.info("合成: %d字 → %.1fs | RTF %.3fx", len(text), dur, elapsed / dur if dur > 0 else 0)
        return audio, elapsed


# ============================================================================
# TTS 引擎
# ============================================================================
class TtsEngine:
    """TTS 语音合成与机器人喇叭播放引擎

    用法::

        engine = TtsEngine(model_name="zh_CN-huayan-medium")
        await engine.speak("你好，欢迎来到云蝶。")
    """

    def __init__(
        self,
        model_name: str = "zh_CN-huayan-medium",
        model_path: str | None = None,
        use_cuda: bool = False,
        speaker_mod: str | None = None,
        volume: float = 1.0,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._use_cuda = use_cuda
        self._speaker_mod = speaker_mod
        self._volume = volume

        self._runner: _PiperRunner | None = None
        self._speaker_cfgs: dict[str, dict] = {}
        self._effective_speaker_mod: str | None = None
        self._shm_cfg: dict | None = None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._runner is not None:
            return

        # 解析模型路径
        model_path = self._model_path
        if model_path is None:
            models = _find_piper_models()
            if self._model_name not in models:
                available = list(models.keys())
                raise FileNotFoundError(
                    f"模型 '{self._model_name}' 未找到，可用: {available}"
                )
            model_path = models[self._model_name]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self._runner = _PiperRunner(model_path, use_cuda=self._use_cuda)

        # 喇叭 SHM 配置
        self._speaker_cfgs = _get_speaker_shm_configs()
        self._effective_speaker_mod = self._speaker_mod or _find_default_speaker(self._speaker_cfgs)
        if self._effective_speaker_mod:
            self._shm_cfg = self._speaker_cfgs[self._effective_speaker_mod]
            _logger.info("喇叭: %s (SHM: %s)",
                          self._effective_speaker_mod,
                          self._shm_cfg.get("data_shm_name", "?"))

    # ------------------------------------------------------------------
    # 合成
    # ------------------------------------------------------------------
    def synthesize(self, text: str) -> np.ndarray:
        """合成文本为音频（float32 numpy 数组）"""
        self._ensure_loaded()
        audio, _elapsed = self._runner.synthesize(text)
        return audio

    # ------------------------------------------------------------------
    # 播放到机器人喇叭
    # ------------------------------------------------------------------
    def _play_shm(self, audio: np.ndarray) -> None:
        """将音频写入喇叭 SHM"""
        if not self._shm_cfg:
            _logger.warning("无可用喇叭 SHM，跳过播放")
            return

        producer = SHMPCMAudioRingBufferProducer(
            self._shm_cfg, label="tts_engine",
            create_if_not_exists=False,
        )
        try:
            target_sr = producer.sample_rate
            target_ch = producer.sample_channels
            target_sw = producer.sample_width

            # 音量
            if self._volume != 1.0:
                audio = audio * self._volume

            # 重采样
            if self._runner.sample_rate != target_sr:
                audio = _resample_audio(audio, self._runner.sample_rate, target_sr)

            # 声道
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
                chunk_dur = (end - offset + chunk_size) / (target_sr * bytes_per_frame)
                if offset < total_bytes:
                    time.sleep(chunk_dur * 0.7)

            tail_dur = total_bytes / (target_sr * bytes_per_frame)
            time.sleep(max(0, tail_dur + 0.3))
        finally:
            producer.close()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    async def speak(self, text: str) -> None:
        """合成并播放（异步包装，避免阻塞事件循环）"""
        self._ensure_loaded()
        audio = await asyncio.to_thread(self.synthesize, text)
        await asyncio.to_thread(self._play_shm, audio)

    @property
    def is_ready(self) -> bool:
        return self._runner is not None

    @property
    def has_speaker(self) -> bool:
        return self._shm_cfg is not None
