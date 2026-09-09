"""机器人语音交互主程序 — 三模式（展厅 / 聊天 / 动作）

通过 ASR 持续监听语音输入，根据模式执行不同处理逻辑：

【展厅模式 exhibition】
  仅匹配 configs/voice_interaction.json 中的特殊指令 → TTS 回复 + 动作执行。
  不启用 LLM，适合展会、展厅等固定话术场景。

【聊天模式 chat】
  完整 ASR → LLM → TTS 语音对话链路。ASR 持续监听、LLM 理解并生成回复、
  TTS 合成播放。纯对话，不触发动作。

【动作模式 action】
  在聊天模式的基础上叠加动作能力：优先匹配特殊指令（含动作），
  未命中则走 LLM → TTS 聊天链路。语音说"你好你好"会触发挥手动作，
  说其他内容则正常对话。

视觉跟踪（所有模式）:
  语音"跟踪我的手"/"跟踪人脸" → 启动 MediaPipe 视觉跟踪 + 头部跟随。
  "停止跟踪" → 退出跟踪。用 --no-track 可完全禁用。

中断支持（所有模式）:
  识别到"停停"/"停下"/"停" → 立即停止当前 LLM/TTS 处理。

参数:
  --mode, -m         运行模式: exhibition / chat / action (默认 action)
  --config, -c       指令配置文件路径 (默认 voice_interaction.json)
  --llm-provider     LLM 后端: openai_compat (默认) / mock (离线验证)
  --system-prompt    LLM 系统提示词
  --tts-engine       TTS 引擎: matcha (默认) / piper / sherpa
  --tts-playback     TTS 播放方式: auto (默认,机器人默认声卡) / local (开发机)
  --tts-speed        TTS 语速倍数 (默认 1.0)
  --tts-volume       TTS 音量倍数 (默认 2.0=200%)
  --asr-engine       ASR 引擎: auto (默认,sherpa本地) / sherpa
  --text, -t         一次性模式: 跳过 ASR 监听，直接处理指定文本后退出
  --no-track         禁用视觉跟踪（默认启用）
  --track-camera     跟踪用相机 (默认 rgbd_head)
  --no-ros2          禁用 ROS2 颈部控制（开发机调试用）
  --no-display       禁用跟踪 GUI 显示窗口（SSH 无头模式）

用法示例:
  # ── 基础模式 ──
  python robots/autolife_s2/run/robot_voice.py                     # 默认动作模式
  python robots/autolife_s2/run/robot_voice.py --mode exhibition    # 展厅模式
  python robots/autolife_s2/run/robot_voice.py --mode chat          # 纯聊天模式

  # ── 离线测试（无 LLM 网络） ──
  python robots/autolife_s2/run/robot_voice.py --llm-provider mock   # LLM mock 模式
  python robots/autolife_s2/run/robot_voice.py --mode chat --llm-provider mock

  # ── 一次性文本测试（跳过 ASR 麦克风） ──
  python robots/autolife_s2/run/robot_voice.py --text "你好"
  python robots/autolife_s2/run/robot_voice.py --text "握手"          # 测试动作指令
  python robots/autolife_s2/run/robot_voice.py --mode action --text "你好你好"
  python robots/autolife_s2/run/robot_voice.py --mode exhibition --text "归位"
  python3 robots/autolife_s2/run/robot_voice.py --mode action --text "比心" --llm-provider mock --no-track --no-display

  # ── TTS 引擎切换 ──
  python robots/autolife_s2/run/robot_voice.py --tts-engine piper
  python robots/autolife_s2/run/robot_voice.py --tts-engine piper --tts-speed 1.5
  python robots/autolife_s2/run/robot_voice.py --tts-volume 3.0      # 300% 音量

  # ── 开发机调试（无 ROS2 / 无显示） ──
  python robots/autolife_s2/run/robot_voice.py --tts-playback local   # 开发机声卡播放
  python robots/autolife_s2/run/robot_voice.py --no-ros2 --no-display # 纯语音，无视觉无关节
  python robots/autolife_s2/run/robot_voice.py --no-track             # 禁用视觉跟踪

模块结构:
  robots/autolife_s2/run/
  ├── act/act_engine.py     — 动作模块（PKL 回放 + 预置位姿 + 平滑插值）
  ├── asr/asr_factory.py    — ASR 工厂（sherpa-onnx SenseVoice）
  ├── tts/tts_factory.py    — TTS 工厂（matcha/piper/sherpa，SHM 输出）
  ├── voice_interaction.py  — 指令匹配引擎（voice_interaction.json）
  ├── robot_voice.py        — 本文件：三模式语音交互引擎
  ├── track/                — 视觉跟踪（MediaPipe 手势/人脸 + ROS2 头部跟踪）
  └── test_asr_demo.py / test_tts_demo.py / test_llm_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from autolife_s2_voice.paths import STATE_ROOT, VOID_COG_ROOT

_PROJECT_ROOT = VOID_COG_ROOT

import yaml

# jieba关键词提取（按需导入，离线场景可降级）
_JIEBA_AVAILABLE = False
try:
    import jieba
    import jieba.analyse as jieba_analyse
    _JIEBA_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("robot_voice")

# 视觉跟踪模块（按需导入，避免无 ROS2/SDK 环境导入失败）
# 无头环境必须设置 offscreen，否则 cv2 调用 Qt 时会 abort
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from autolife_s2_voice.run.track import TrackSession, parse_track_command
    _TRACK_AVAILABLE = True
except ImportError as _e:
    _TRACK_AVAILABLE = False
    _logger.debug("视觉跟踪模块不可用: %s", _e)

# ---------------------------------------------------------------------------
# 模式常量
# ---------------------------------------------------------------------------
MODE_EXHIBITION = "exhibition"  # 展厅：仅特殊指令匹配
MODE_CHAT = "chat"              # 聊天：ASR→LLM→TTS
MODE_ACTION = "action"          # 动作：指令优先 + LLM 对话 + 动作执行

VALID_MODES = (MODE_EXHIBITION, MODE_CHAT, MODE_ACTION)
WEB_READY_MARKER = "__WEB_RESULT_READY__"

# 跟踪触发/退出关键词（在任意模式下都拦截，不走模式分发）
_TRACK_STOP_KW = ("停止跟踪", "停下跟踪", "退出跟踪", "结束跟踪",
                  "别跟踪", "不跟踪", "取消跟踪")

# 语音打断关键词
INTERRUPT_KW = ("停停", "停下", "停", "闭嘴", "别说", "安静")



# ---------------------------------------------------------------------------
# 关键词提取配置
# ---------------------------------------------------------------------------
class KeywordConfig:
    """关键词提取配置，从 voice_interaction.json 的 keyword 节加载"""

    __slots__ = ("enabled", "topK", "merge_strategy",
                 "stopwords_path", "user_dict_path", "auto_dict_from_commands")

    def __init__(self, data: dict | None = None):
        d = data or {}
        self.enabled = d.get("enabled", True)
        self.topK = d.get("topK", 5)
        self.merge_strategy = d.get("merge_strategy", "union")
        self.stopwords_path = d.get("stopwords_path", "configs/jieba_stopwords.txt")
        self.user_dict_path = d.get("user_dict_path", "configs/jieba_dict.txt")
        self.auto_dict_from_commands = d.get("auto_dict_from_commands", True)
# ---------------------------------------------------------------------------
# 指令配置（复用 voice_interaction.json）
# ---------------------------------------------------------------------------
class CommandConfig:
    __slots__ = ("trigger", "reply", "action")

    def __init__(self, trigger: list[str], reply: str, action: str | None):
        self.trigger = trigger  # 关键词列表
        self.reply = reply
        self.action = action


def _normalize(text: str) -> str:
    """去除中文/英文标点和空白"""
    return re.sub(r"[^\u4e00-\u9fff\w]", "", text, flags=re.UNICODE)


_GREETING_WORDS = (
    "你好你好", "早上好", "你好", "您好", "哈喽", "hello",
    "云蝶机器人", "云蝶", "机器人", "小云",
)
_GREETING_FILLERS = ("呀", "啊", "喂", "在吗")


def _has_content_beyond_greeting(text: str) -> bool:
    """问候及机器人称呼之外是否还有需要继续理解的内容。"""
    remaining = _normalize(text).lower()
    for word in _GREETING_WORDS:
        remaining = remaining.replace(word, "")
    for word in _GREETING_FILLERS:
        remaining = remaining.replace(word, "")
    return bool(remaining)


def _is_greeting_command(cmd: "CommandConfig") -> bool:
    """识别 voice_interaction.json 中的固定问候指令。"""
    return any(
        marker in trigger
        for trigger in cmd.trigger
        for marker in ("你好", "哈喽", "早上好")
    )


class ModeSwitchConfig:
    """模式切换指令配置"""
    __slots__ = ("trigger", "mode", "reply")

    def __init__(self, trigger: list[str], mode: str, reply: str):
        self.trigger = trigger
        self.mode = mode
        self.reply = reply


def _match_triggers(text: str, triggers: list[str]) -> bool:
    """检查文本是否包含触发词列表中的任意一个

    支持多种匹配策略（容错性）:
    1. 精确子串匹配（已归一化）
    2. 关键词组合匹配：触发词中用 '|' 分隔表示"或"，用 '+' 分隔表示"与"
       例如: "重新复位|复位" 表示匹配 "重新复位" 或 "复位"
       例如: "你好+小云" 表示匹配同时包含 "你好" 和 "小云"
    3. 最小片段匹配：若触发词 >= 4 字，允许匹配其前 3 字
       例如: 触发词"展厅模式"，识别为"展厅"也可匹配
    """
    normalized = _normalize(text)
    if not normalized:
        return False

    for trigger_expr in triggers:
        trigger_expr = trigger_expr.strip()

        # 解析 '+' 组合匹配（AND 逻辑）
        if '+' in trigger_expr:
            keywords = [_normalize(k.strip()) for k in trigger_expr.split('+') if k.strip()]
            if keywords and all(k in normalized for k in keywords):
                return True
            continue

        # 解析 '|' 选择匹配（OR 逻辑）
        if '|' in trigger_expr:
            sub_triggers = [t.strip() for t in trigger_expr.split('|') if t.strip()]
            for sub in sub_triggers:
                if _match_single_trigger(normalized, _normalize(sub)):
                    return True
            continue

        # 普通单触发词
        if _match_single_trigger(normalized, _normalize(trigger_expr)):
            return True

    return False


def _match_single_triggers(normalized_text: str, trigger: str) -> bool:
    """单触发词匹配（支持长度容错）

    - 触发词长度 >= 4 字：允许匹配前 3 字（容错 ASR 截断）
    - 触发词长度 >= 6 字：允许匹配前 4 字
    - 否则：精确子串匹配
    """
    if not trigger:
        return False

    # 精确子串匹配
    if trigger in normalized_text:
        return True

    # 长度容错：长关键词允许匹配前缀
    trigger_len = len(trigger)
    if trigger_len >= 6:
        # 6字以上：允许4字前缀匹配
        prefix_len = min(4, trigger_len - 2)
        if prefix_len >= 3 and trigger[:prefix_len] in normalized_text:
            _logger.debug("模糊匹配: '%s' 命中 '%s' 的前%d字",
                          trigger[:prefix_len], trigger, prefix_len)
            return True
    elif trigger_len >= 4:
        # 4-5字：允许3字前缀匹配
        if trigger[:3] in normalized_text:
            _logger.debug("模糊匹配: '%s' 命中 '%s' 的前3字",
                          trigger[:3], trigger)
            return True

    return False


# 别名：保留原函数名以兼容
_match_single_trigger = _match_single_triggers


def load_commands(config_path: str | Path) -> tuple[list[ModeSwitchConfig], list[CommandConfig]]:
    """从 voice_interaction.json 加载指令列表和模式切换配置"""
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    # 模式切换指令
    mode_switches = []
    for item in data.get("mode_switches", []):
        triggers = item.get("trigger", [])
        if isinstance(triggers, str):
            triggers = [triggers]
        mode = item.get("mode", "")
        reply = item.get("reply", "")
        if triggers and mode:
            mode_switches.append(ModeSwitchConfig(
                trigger=triggers, mode=mode, reply=reply
            ))
    _logger.info("加载 %d 条模式切换指令", len(mode_switches))

    # 特殊指令
    commands = []
    for item in data.get("commands", []):
        trigger = item.get("trigger", [])
        if isinstance(trigger, str):
            trigger = [trigger]
        reply = item.get("reply", "").strip()
        action = item.get("action")
        if trigger:
            commands.append(CommandConfig(trigger=trigger, reply=reply, action=action))
    _logger.info("加载 %d 条指令", len(commands))
    return mode_switches, commands


# ---------------------------------------------------------------------------
# 语音交互引擎
# ---------------------------------------------------------------------------
class RobotVoiceInteraction:
    """机器人语音交互引擎 — 三模式统一入口"""

    def __init__(
        self,
        *,
        mode: str = MODE_ACTION,
        config_path: str | Path | None = None,
        tts_engine: str = "matcha",
        tts_playback: str = "auto",
        tts_speed: float = 1.0,
        tts_volume: float = 2.0,
        asr_engine: str = "auto",
        llm_provider: str = "openai_compat",
        system_prompt: str | None = None,
        enable_track: bool = True,
        track_enable_ros2: bool = True,
        track_enable_display: bool = True,
        track_camera: str = "rgbd_head",
        emit_web_ready_marker: bool = False,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"未知模式: {mode!r}，可选: {VALID_MODES}")

        self._mode = mode
        self._tts_engine = tts_engine
        self._tts_playback = tts_playback
        self._tts_speed = tts_speed
        self._tts_volume = tts_volume
        self._asr_engine = asr_engine
        self._llm_provider = llm_provider
        self._system_prompt = system_prompt or (
            "你是云蝶科技的机器人语音助手，回答简洁口语化，每次回答不超过两句话。")

        # 指令配置路径
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        self._config_path = Path(config_path or config_dir / "voice_interaction.json")

        # 组件
        self._mode_switches: list[ModeSwitchConfig] = []
        self._commands: list[CommandConfig] = []
        self._asr: Any = None
        self._tts: Any = None
        self._llm: Any = None

        # 视觉跟踪会话（惰性启动：首次识别到"跟踪XXX"时 start）
        self._track_enabled = enable_track and _TRACK_AVAILABLE
        self._track_session: Any = None  # TrackSession 实例
        self._track_enable_ros2 = track_enable_ros2
        self._track_enable_display = track_enable_display
        self._track_camera = track_camera
        self._emit_web_ready_marker = emit_web_ready_marker
        self._web_ready_emitted = False

        # 串行化 + 打断
        self._busy = asyncio.Lock()
        self._interrupt = asyncio.Event()

        # 关键词提取
        self._keyword_config: KeywordConfig | None = None
        self._last_keywords: list[str] = []
        self._keyword_mappings: list[dict] = []
        self._last_matches: list[dict] = []
        self._short_commands: list[dict] = []
        self._short_cmd_threshold: int = 10
        self._fixed_tasks: list[dict] = []

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    async def _init(self) -> None:
        # 指令配置
        if self._config_path.exists():
            self._mode_switches, self._commands = load_commands(self._config_path)

            try:
                _config_data = json.loads(
                    self._config_path.read_text(encoding="utf-8")
                )
            except Exception as _e:
                _logger.warning("语音配置读取失败: %s", _e)
                _config_data = {}

            # 加载关键词提取配置
            try:
                keyword_data = _config_data.get("keyword", {})
                self._keyword_config = KeywordConfig(keyword_data)
            except Exception as _e:
                _logger.warning("关键词配置读取失败: %s", _e)
                self._keyword_config = KeywordConfig({})

            # 加载纯地点/任务能力表。该文件不再承载关键词、提示语或动作回复。
            mapping_path = Path(__file__).resolve().parents[1] / "configs" / "keyword_mapping.json"
            _mdata = None
            if mapping_path.exists():
                try:
                    with open(mapping_path, encoding="utf-8") as _f:
                        _mdata = json.load(_f)
                    _logger.info("地点任务库已加载: %s", mapping_path)
                except Exception as _e:
                    _logger.warning("地点任务库加载失败: %s", _e)

            # 唯一支持的格式是 {"task_locations": {"地点": ["任务"]}}。
            # 不兼容旧 hints/reply/mappings/short_commands，避免配置职责混杂。
            if _mdata:
                self._fixed_tasks = []
                for _location, _tasks in _mdata.get("task_locations", {}).items():
                    if not isinstance(_location, str) or not isinstance(_tasks, list):
                        continue
                    for _item in _tasks:
                        if isinstance(_item, str) and _item.strip():
                            self._fixed_tasks.append({
                                "task": _item.strip(),
                                "location": _location.strip(),
                            })
                if self._fixed_tasks:
                    _logger.info("地点任务库包含 %d 个任务", len(self._fixed_tasks))

            # 短指令阈值（从 voice_interaction.json 读取）
            self._short_cmd_threshold = _config_data.get("short_cmd_threshold", 10)


        # TTS 引擎
        from autolife_s2_voice.run.tts.tts_factory import create_tts_engine
        self._tts = create_tts_engine(
            engine=self._tts_engine,
            playback=self._tts_playback,
            speed=self._tts_speed,
            volume=self._tts_volume,
        )
        _logger.info("TTS: %s (playback=%s, volume=%.0f%%)",
                     self._tts.engine_name, self._tts.playback_mode, self._tts_volume * 100)

        # LLM 服务（聊天/动作模式需要）
        if self._mode in (MODE_CHAT, MODE_ACTION):
            from void_cog.services.llm import create_llm_service
            cfg = _load_config()
            self._llm = create_llm_service(cfg, provider=self._llm_provider)
            _logger.info("LLM: %s", self._llm_provider)

        # ASR 引擎
        from autolife_s2_voice.run.asr.asr_factory import create_asr_engine
        self._asr = create_asr_engine(
            engine=self._asr_engine,
            on_text=self._on_asr_text,
        )
        _logger.info("ASR: %s", type(self._asr).__name__)

        # jieba关键词提取初始化
        if _JIEBA_AVAILABLE and self._keyword_config is not None and self._keyword_config.enabled:
            self._init_jieba()
        elif not _JIEBA_AVAILABLE:
            _logger.info("jieba未安装，关键词提取已禁用")

    # ------------------------------------------------------------------
    # ASR 回调
    # ------------------------------------------------------------------
    async def _on_asr_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        _logger.info("[ASR] 识别: %s", text)

        # 打断检查
        if self._is_interrupt(text):
            _logger.info("接收到打断指令，停止当前处理")
            self._interrupt.set()
            return

        # 关键词提取（打断检查之后、模式分发之前）
        if _JIEBA_AVAILABLE and self._keyword_config is not None and self._keyword_config.enabled:
            try:
                self._last_keywords = await asyncio.to_thread(
                    self._extract_keywords, text
                )
                if self._last_keywords:
                    _logger.info("[关键词] %s", self._last_keywords)
                    self._save_keyword_log(text, self._last_keywords)
                    self._last_matches = self._match_keyword_mapping(self._last_keywords)
                    if self._last_matches:
                        for m in self._last_matches:
                            _logger.info("[映射] %s -> %s (%s)", m["matched"], m["targets"], m["label"])
            except Exception as e:
                _logger.warning("关键词提取失败: %s", e)
                self._last_keywords = []

        # 1. 优先检查模式切换指令（所有模式都支持）
        for switch in self._mode_switches:
            if _match_triggers(text, switch.trigger):
                _logger.info("模式切换命中: %s → %s", switch.trigger, switch.mode)
                await self._switch_mode(switch)
                return

        # 2. 视觉跟踪指令拦截（任意模式下都识别"跟踪XXX"/"停止跟踪"）
        if self._track_enabled:
            handled = await self._handle_track_command(text)
            if handled:
                return

        # 3. 若跟踪会话正在运行，其他语音不触发 LLM/动作（避免干扰）
        if self._track_session is not None and self._track_session.is_tracking:
            _logger.debug("跟踪进行中，忽略非跟踪语音: %s", text)
            return

        # 4. 按当前模式分发
        if self._mode == MODE_EXHIBITION:
            await self._handle_exhibition(text)
        elif self._mode == MODE_CHAT:
            await self._handle_chat(text)
        elif self._mode == MODE_ACTION:
            await self._handle_action(text)


    # ------------------------------------------------------------------
    # 关键词提取（jieba.analyse）
    # ------------------------------------------------------------------
    def _init_jieba(self) -> None:
        """初始化jieba：加载自定义词典和停用词"""
        cfg = self._keyword_config
        if cfg is None:
            return

        config_dir = Path(__file__).resolve().parents[1] / "configs"

        # 1. 加载自定义词典
        dict_path = config_dir / cfg.user_dict_path.replace("configs/", "").replace("configs\\", "")
        if dict_path.exists():
            jieba.load_userdict(str(dict_path))
            _logger.info("jieba自定义词典已加载: %s", dict_path)
        else:
            _logger.info("jieba自定义词典不存在: %s", dict_path)

        # 2. 自动从 voice_interaction.json 的 commands.trigger 中提取词条
        if cfg.auto_dict_from_commands and self._commands:
            for cmd in self._commands:
                for trigger_expr in cmd.trigger:
                    for sub in re.split(r"[|+]", trigger_expr):
                        sub = sub.strip()
                        if sub and len(sub) >= 1:
                            jieba.add_word(sub, freq=8)
            _logger.info("已从%d条指令中自动注册jieba词典", len(self._commands))

        # 3. 加载停用词
        sw_path = config_dir / cfg.stopwords_path.replace("configs/", "").replace("configs\\", "")
        self._stopwords: set[str] = set()
        if sw_path.exists():
            with open(sw_path, encoding="utf-8") as f:
                self._stopwords = {line.strip() for line in f if line.strip() and not line.startswith("#")}
            _logger.info("jieba停用词已加载: %d个", len(self._stopwords))
        else:
            _logger.info("jieba停用词文件不存在: %s", sw_path)

    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取关键词（在后台线程中调用）

        组合TF-IDF和TextRank算法，按配置的merge_strategy合并：
          - "union": 并集，更全面（默认）
          - "intersection": 交集，更精确
          - 允许词性: n(名词), v(动词)及其子类
        """
        cfg = self._keyword_config
        if cfg is None:
            return []

        topK = cfg.topK
        merge = cfg.merge_strategy
        stopwords = getattr(self, "_stopwords", set())

        # TF-IDF提取（名词+动词词性过滤）
        tfidf_kws: set[str] = set()
        try:
            tfidf_raw = jieba_analyse.extract_tags(
                text, topK=topK,
                allowPOS=("n", "nr", "ns", "nt", "nz", "v", "vn")
            )
            tfidf_kws = {w for w in tfidf_raw if w not in stopwords}
        except Exception:
            pass

        # TextRank提取（名词+动词词性过滤）
        textrank_kws: set[str] = set()
        try:
            tr_raw = jieba_analyse.textrank(
                text, topK=topK,
                allowPOS=("n", "nr", "ns", "nt", "nz", "v", "vn")
            )
            textrank_kws = {w for w in tr_raw if w not in stopwords}
        except Exception:
            pass

        # 按策略合并
        if merge == "intersection":
            combined = tfidf_kws & textrank_kws
        else:
            combined = tfidf_kws | textrank_kws

        # 排序：TF-IDF结果排前面
        result: list[str] = []
        try:
            tfidf_ordered = jieba_analyse.extract_tags(text, topK=topK)
            for w in tfidf_ordered:
                if w in combined and w not in result:
                    result.append(w)
        except Exception:
            pass
        for w in combined:
            if w not in result:
                result.append(w)

        return result[:topK]

    # ------------------------------------------------------------------
    # 关键词日志持久化
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 关键词映射匹配
    # ------------------------------------------------------------------
    def _match_keyword_mapping(self, keywords: list[str]) -> list[dict]:
        """将提取的关键词与映射表进行匹配"""
        if not keywords or not self._keyword_mappings:
            return []

        results: list[dict] = []
        for rule in self._keyword_mappings:
            triggers = rule.get("triggers", [])
            targets = rule.get("targets", [])
            label = rule.get("label", "")

            hit = None
            for t in triggers:
                if t in keywords:
                    hit = t
                    break

            if hit:
                results.append({
                    "label": label,
                    "matched": hit,
                    "targets": targets,
                })

        return results

    def _match_fixed_task(self, text: str):
        """固定任务硬匹配：原始文本包含任一关键词即命中。

        返回 dict{task, location, reply} 或 None。
        在调用 LLM 之前拦截，保证固定任务不受 LLM 自由发挥影响。
        """
        if not text or not self._fixed_tasks:
            return None
        for rule in self._fixed_tasks:
            for t in rule.get("triggers", []):
                if t in text:
                    return {
                        "task": rule.get("task", ""),
                        "location": rule.get("location", ""),
                        "reply": rule.get("reply", ""),
                    }
        return None

    _KEYWORD_LOG_PATH = STATE_ROOT / "keyword_log.txt"

    @classmethod
    def _save_keyword_log(cls, text: str, keywords: list[str]) -> None:
        """将输入文本和关键词追加写入日志文件"""
        try:
            log_dir = cls._KEYWORD_LOG_PATH.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            kw_str = " ".join(keywords) if keywords else "(无)"
            line = timestamp + " | 输入: " + text + " | 关键词: " + kw_str
            # 附加映射匹配结果
            matches = getattr(cls, "_last_matches", [])
            if matches:
                match_strs = []
                for m in matches:
                    match_strs.append(m["label"] + ":" + "+".join(m["targets"]))
                line += " | 匹配: " + "; ".join(match_strs)
            line += chr(10)
            with open(cls._KEYWORD_LOG_PATH, "a", encoding="utf-8") as log_f:
                log_f.write(line)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _is_interrupt(self, text: str) -> bool:
        """是否语音打断指令"""
        norm = _normalize(text)
        return any(kw in norm for kw in INTERRUPT_KW)

    async def _busy_guard(self, coro):
        """串行化执行：同一时刻只处理一组 LLM+TTS"""
        if self._busy.locked():
            _logger.info("正在处理上一轮，跳过")
            return None
        async with self._busy:
            self._interrupt.clear()
            return await coro

    # ------------------------------------------------------------------
    # 视觉跟踪集成
    # ------------------------------------------------------------------
    async def _ensure_track_session(self) -> bool:
        """惰性创建并启动 TrackSession（首次跟踪时调用）

        Returns
        -------
        bool
            True 表示会话就绪，False 表示不可用或启动失败
        """
        if not self._track_enabled:
            _logger.warning("视觉跟踪未启用")
            return False
        if self._track_session is not None:
            if not self._track_session.is_running:
                await self._track_session.start()
            return True

        # 首次创建：注入 TTS 回调（复用本引擎的 _speak_reply 流水线）
        try:
            self._track_session = TrackSession(
                on_speak=self._speak_reply,
                sdk_camera=self._track_camera,
                enable_ros2=self._track_enable_ros2,
                enable_display=self._track_enable_display,
            )
            await self._track_session.start()
            _logger.info("视觉跟踪会话已启动 (camera=%s, ros2=%s)",
                         self._track_camera, self._track_enable_ros2)
            return True
        except Exception as e:
            _logger.error("视觉跟踪会话启动失败: %s", e)
            self._track_session = None
            return False

    async def _handle_track_command(self, text: str) -> bool:
        """处理跟踪指令，返回是否已拦截此条语音

        - "停止跟踪/退出跟踪/..." → 停止跟踪，TTS 提示
        - "跟踪XXX" → 启动/切换跟踪，TTS 提示
        - 其他文本 → 不拦截，返回 False

        Returns
        -------
        bool
            True 表示已识别为跟踪指令并处理，False 表示非跟踪指令
        """
        # 1. 停止跟踪指令
        if any(kw in text for kw in _TRACK_STOP_KW):
            if self._track_session is not None and self._track_session.is_tracking:
                msg = await self._track_session.stop_tracking()
                _logger.info("[跟踪] 停止: %s", msg)
                if msg:
                    await self._speak_reply(msg)
            else:
                _logger.info("[跟踪] 未在跟踪，忽略停止指令")
            return True

        # 2. 解析跟踪指令
        cmd = parse_track_command(text)
        if not cmd.is_valid and not cmd.is_stop:
            # 非跟踪指令（"跟踪"都没出现）→ 不拦截
            return False
        if cmd.method is None:
            return False

        # 3. 启动/切换跟踪
        ok = await self._ensure_track_session()
        if not ok:
            await self._speak_reply("视觉跟踪不可用")
            return True

        msg = self._track_session.start_tracking(cmd)
        _logger.info("[跟踪] 启动: %s → %s", cmd, msg)
        if msg:
            await self._speak_reply(msg)
        return True

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    async def _switch_mode(self, switch: ModeSwitchConfig) -> None:
        """执行模式切换"""
        target_mode = switch.mode
        if target_mode not in VALID_MODES:
            _logger.warning("未知目标模式: %s", target_mode)
            return

        reply_text = switch.reply
        old_mode = self._mode
        self._mode = target_mode
        _logger.info("模式切换: %s → %s", old_mode, target_mode)

        # 切换到动作模式时，执行复位（解锁安全锁 + 归零）
        if target_mode == MODE_ACTION and old_mode != MODE_ACTION:
            _logger.info("切换到动作模式，执行机器人复位...")
            try:
                from autolife_s2_voice.run.act.act_engine import reset_robot
                await asyncio.to_thread(reset_robot)
            except Exception as e:
                _logger.error("机器人复位失败: %s", e)

        # 切换到需要 LLM 的模式时，初始化 LLM（如果还没初始化）
        if target_mode in (MODE_CHAT, MODE_ACTION) and self._llm is None:
            try:
                from void_cog.services.llm import create_llm_service
                cfg = _load_config()
                self._llm = create_llm_service(cfg, provider=self._llm_provider)
                _logger.info("LLM 初始化完成: %s", self._llm_provider)
            except Exception as e:
                _logger.error("LLM 初始化失败: %s", e)

        # 用 TTS 播放切换回复
        if reply_text and self._tts:
            try:
                await self._tts.speak(reply_text)
            except Exception as e:
                _logger.error("TTS 播放失败: %s", e)

    # ------------------------------------------------------------------
    # 展厅模式：仅特殊指令匹配
    # ------------------------------------------------------------------
    async def _handle_exhibition(self, text: str) -> None:
        cmd = await self._match_command(text)
        if cmd is None:
            _logger.info("展厅模式：未命中指令: %s", text)
            return
        await self._busy_guard(self._execute_command(cmd))

    # ------------------------------------------------------------------
    # 聊天模式：短指令直连 + 长文本 LLM 聊天
    # ------------------------------------------------------------------
    async def _handle_chat(self, text: str) -> None:
        # 先匹配特殊指令
        cmd = await self._match_command(text)
        if cmd is not None:
            _logger.info("聊天模式：命中指令: %s", cmd.trigger)
            await self._busy_guard(self._execute_command(cmd))
            return
        if self._llm is None:
            _logger.warning("LLM 未初始化，无法聊天")
            return

        text_len = len(text.replace(" ", ""))
        if text_len < self._short_cmd_threshold:
            _logger.info("短指令(%d字): 走直接匹配", text_len)
            matched = await self._busy_guard(self._handle_short_cmd(text))
            if not matched:
                await self._busy_guard(self._do_chat(text))
        else:
            await self._busy_guard(self._do_chat(text))

    # ------------------------------------------------------------------
    # 动作模式：短指令直连 + 长文本 LLM 意图分发
    # ------------------------------------------------------------------
    async def _handle_action(self, text: str) -> None:
        """动作模式入口：短指令(<threshold字)直接匹配，长文本调LLM分析意图"""
        # 先匹配特殊指令（voice_interaction.json 的 commands）
        cmd = await self._match_command(text)
        if cmd is not None:
            _logger.info("动作模式：命中指令: %s", cmd.trigger)
            await self._busy_guard(self._execute_command(cmd))
            return

        if self._llm is None:
            _logger.warning("LLM 未初始化，无法聊天")
            return

        # ★ 短/长分流 ★
        text_len = len(text.replace(" ", ""))  # 去空格后的字数
        if text_len < self._short_cmd_threshold:
            # 短指令：先直接匹配动作，未命中再走 LLM 意图分析
            _logger.info("短指令(%d字 < %d): 走直接匹配", text_len, self._short_cmd_threshold)
            matched = await self._busy_guard(self._handle_short_cmd(text))
            if not matched:
                await self._busy_guard(self._do_chat_with_intent(text))
        else:
            # 长文本：LLM 意图分析 + 任务分发
            _logger.info("长文本(%d字 >= %d): 走 LLM 意图分析", text_len, self._short_cmd_threshold)
            await self._busy_guard(self._do_chat_with_intent(text))

    # ------------------------------------------------------------------
    # 短指令处理：keyword_mapping.json 的 short_commands 匹配
    # ------------------------------------------------------------------
    async def _handle_short_cmd(self, text: str) -> bool:
        """短指令匹配：遍历 short_commands，命中返回 True，未命中返回 False。

        未命中时不在这里兜底，由调用方决定是否走 LLM。
        """

        print(f"\n[你]   {text}")

        # 遍历 short_commands 规则
        for rule in self._short_commands:
            triggers = rule.get("triggers", [])
            for t in triggers:
                if t in text:
                    action = rule.get("action")
                    reply = rule.get("reply", "")
                    _logger.info("短指令命中: '%s' -> action=%s", t, action)

                    # 执行动作（如果有）
                    if action:
                        try:
                            from autolife_s2_voice.run.act.act_engine import execute_action
                            await asyncio.to_thread(execute_action, action)
                            _logger.info("短指令动作完成: %s", action)
                        except Exception as e:
                            _logger.error("短指令动作失败(%s): %s", action, e)

                    # TTS 回复
                    if reply and self._tts:
                        print(f"[助手] {reply}")
                        await self._speak_reply(reply)
                    return True

        # 未命中任何短指令
        _logger.info("短指令未命中: %s", text)
        return False

    # ------------------------------------------------------------------
    # 长文本：LLM 意图分析 + 任务/闲聊分发
    # ------------------------------------------------------------------
    async def _do_chat_with_intent(self, text: str) -> None:
        """长文本(>=10字)：调用LLM分析意图，判断闲聊/任务并分发"""

        print(f"\n[你]   {text}")

        if self._interrupt.is_set():
            return

        # 构建意图分析 Prompt。能力表只包含地点和任务，LLM 根据用户原句
        # 的语义拆分、排序，不再依赖易膨胀且难维护的 hints 列表。
        _fixed_lines = []
        for _rule in self._fixed_tasks:
            _task = _rule.get("task", "")
            _loc = _rule.get("location", "")
            _fixed_lines.append(f"- 地点：{_loc}；任务：{_task}")
        _fixed_table = "\n".join(_fixed_lines) if _fixed_lines else "- （无）"

        _examples = []
        for _rule in self._fixed_tasks[:2]:
            _task = _rule.get("task", "")
            _loc = _rule.get("location", "")
            if not _task or not _loc:
                continue
            _examples.append(
                f"输入：请帮我{_task}\n"
                f"输出：好的，我马上处理。[TASK:{_task}][LOCATION:{_loc}]"
            )
        if len(self._fixed_tasks) >= 2:
            _first, _second = self._fixed_tasks[:2]
            _examples.append(
                f"输入：先{_first['task']}，再{_second['task']}\n"
                f"输出：好的，我按顺序处理。"
                f"[TASK:{_first['task']}][LOCATION:{_first['location']}]"
                f"[TASK:{_second['task']}][LOCATION:{_second['location']}]"
            )
            _examples.append(
                f"输入：厨房桌面脏了，帮我擦干净，再{_second['task']}\n"
                f"输出：抱歉，我不能处理厨房桌面；但可以完成后一个任务。"
                f"[TASK:{_second['task']}][LOCATION:{_second['location']}]"
            )
        if self._fixed_tasks:
            _first = self._fixed_tasks[0]
            _examples.append(
                f"输入：电视旁的纸盒我自己处理，你只需要{_first['task']}\n"
                f"输出：好的，纸盒由您处理，我只完成您交代的任务。"
                f"[TASK:{_first['task']}][LOCATION:{_first['location']}]"
            )
        if len(self._fixed_tasks) >= 4:
            _chain = self._fixed_tasks[:4]
            _examples.append(
                "输入：请按顺序执行："
                + "，然后".join(_item["task"] for _item in _chain)
                + "\n"
                "输出：好的，我按顺序完成这四项任务。"
                + "".join(
                    f"[TASK:{_item['task']}][LOCATION:{_item['location']}]"
                    for _item in _chain
                )
            )
        if len(self._fixed_tasks) >= 2:
            _valid = self._fixed_tasks[0]
            _wrong_location = next(
                (
                    _item["location"]
                    for _item in self._fixed_tasks[1:]
                    if _item["location"] != _valid["location"]
                ),
                "其他地点",
            )
            _examples.append(
                f"输入：去{_wrong_location}{_valid['task']}\n"
                f"输出：抱歉，{_valid['task']}不能在{_wrong_location}执行；请确认是否改去{_valid['location']}。"
            )
        _catalog_pairs = {
            (_item["task"], _item["location"])
            for _item in self._fixed_tasks
        }
        if {
            ("放置配送物品", "配送区"),
            ("夹取配送物品", "物品区"),
        }.issubset(_catalog_pairs):
            _examples.extend([
                "输入：把这个包裹送到存放物品的地方\n"
                "输出：好的，我把包裹送过去放好。"
                "[TASK:放置配送物品][LOCATION:配送区]",
                "输入：去物品区把要配送的盒子夹起来\n"
                "输出：好的，我去把盒子夹起来。"
                "[TASK:夹取配送物品][LOCATION:物品区]",
            ])
        if ("拿水", "前台区") in _catalog_pairs and any(
            _item["location"] == "电视区" for _item in self._fixed_tasks
        ):
            _examples.append(
                "输入：去电视区给我拿杯水\n"
                "输出：拿水只能去前台区，不能在电视区执行；请确认是否改去前台区。"
            )
        _examples.extend([
            "输入：我自己去拿水，你不用过去\n输出：好的，有需要再叫我。",
            "输入：你会做什么？\n输出：我可以完成能力表里的服务任务。",
            "输入：今天天气不错\n输出：是呀，今天让人心情很好。",
        ])
        _example_text = "\n\n".join(_examples)

        intent_prompt = (
            f"{self._system_prompt}\n\n"
            f"你是服务机器人，需要理解用户语音输入的「真实意图」，判断是否需要执行任务以及去哪里执行。\n\n"
            f"【你能执行的任务及对应地点】\n"
            f"{_fixed_table}\n\n"
            f"【输出格式】\n"
            f"- 闲聊或无需执行任务时：直接友好回复，不要输出任何标记。\n"
            f"- 需要执行任务时：先用一句简短口语回复概括执行顺序，再在回复末尾按用户要求的先后顺序输出一组或多组 "
            f"[TASK:任务描述][LOCATION:地点]。每个任务必须紧跟自己的地点，任务和地点严格从上表选择，不要自创。\n\n"
            f"【判断原则】\n"
            f"1. 只提取用户明确要求机器人执行的任务；用户自己做、提问、能力咨询和被否定的动作都不是任务。回复中的主语也必须一致，绝不能把“用户自己做”说成“机器人做”，或把“机器人做”说成“用户做”。\n"
            f"2. 先按“然后、再、接着、最后”等连接关系拆成独立动作，再按原句先后逐项输出；后一步即使与前一步对象相同也不能合并或漏掉。用户明确要求重复执行时要保留重复项。\n"
            f"3. 用户表达不必与任务名称逐字相同，要按语义选择最符合的库内任务。特别注意“拿取/取出”和“送到/放置”的方向相反，必须依据动词选择；不唯一时先追问，不要猜。\n"
            f"4. 每项任务只能在表内与它绑定的地点执行。用户明确指定的地点与任务绑定地点不一致时，只能拒绝或追问，绝不能擅自改到正确地点继续执行，也不能为了迁就地点而替换成另一个任务；同句中的其他合法任务仍应保留。用户没有说地点时，才允许补全该任务的唯一绑定地点。\n"
            f"5. 闲聊和任务并存时先简短回应闲聊，再确认任务，两部分都保留。\n"
            f"6. 回复口语化且不超过两句话，标记只放在末尾。\n\n"
            f"【示例】\n{_example_text}\n"
        )

        messages = [
            {"role": "system", "content": intent_prompt},
            {"role": "user", "content": text},
        ]

        _llm_started = time.perf_counter()
        try:
            r = await self._llm.chat(messages, temperature=0.1, max_tokens=512)
        except Exception as e:
            _logger.error("LLM 调用失败: %s", e)
            await self._speak_reply("抱歉，我出了一点小问题。")
            return
        finally:
            _logger.info("LLM 意图分析耗时: %.3fs", time.perf_counter() - _llm_started)

        if not r.success or not r.content:
            _logger.warning("LLM 返回失败: %s", r.error)
            await self._speak_reply("抱歉，我没有理解你的意思。")
            return

        raw_reply = r.content.strip()
        _logger.info("LLM 原始回复: %s", raw_reply)

        # 按输出顺序解析全部 [TASK:xxx][LOCATION:xxx] 任务对。
        # LOCATION 与 TASK 绑定匹配，避免多个任务时地点发生错位。
        task_pairs = re.findall(
            r'\[TASK:\s*(.+?)\s*\]\s*\[LOCATION:\s*(.+?)\s*\]',
            raw_reply,
        )
        # LLM 输出属于不可信结构化输入：任务及地点组合必须原样存在于能力表。
        # 不再自动纠正地点，避免把用户的冲突地点悄悄改成另一个地点后执行。
        _locations_by_task: dict[str, set[str]] = {}
        for _rule in self._fixed_tasks:
            _task = str(_rule.get("task", "")).strip()
            _loc = str(_rule.get("location", "")).strip()
            if _task and _loc:
                _locations_by_task.setdefault(_task, set()).add(_loc)

        tasks = []
        _explicit_locations = {
            str(_rule.get("location", "")).strip()
            for _rule in self._fixed_tasks
            if str(_rule.get("location", "")).strip() in text
        }
        _single_explicit_location = (
            next(iter(_explicit_locations))
            if len(task_pairs) == 1 and len(_explicit_locations) == 1
            else None
        )
        for _task_raw, _location_raw in task_pairs:
            _task = _task_raw.strip()
            _location = _location_raw.strip()
            _allowed_locations = _locations_by_task.get(_task, set())
            if not _allowed_locations:
                _logger.warning("丢弃能力表外任务: %s (%s)", _task, _location)
                continue
            if _location not in _allowed_locations:
                _logger.warning("丢弃地点不匹配任务: %s (%s)", _task, _location)
                continue
            if (
                _single_explicit_location is not None
                and _location != _single_explicit_location
            ):
                _logger.warning(
                    "丢弃擅自改写用户地点的任务: %s (%s -> %s)",
                    _task,
                    _single_explicit_location,
                    _location,
                )
                continue
            tasks.append({"task": _task, "location": _location})
        for index, item in enumerate(tasks, start=1):
            _logger.info(
                "LLM 识别到任务%d: %s，地点: %s",
                index,
                item["task"],
                item["location"],
            )

        # 去除全部 TASK/LOCATION 标记，保留干净的口语回复。
        clean_reply = re.sub(
            r'\[(TASK|LOCATION):\s*.+?\s*\]', '', raw_reply
        ).strip()

        print(f"[助手] {clean_reply}")
        for index, item in enumerate(tasks, start=1):
            print(f"[任务{index}] {item['task']}")
            print(f"[地点{index}] {item['location']}")
        if self._interrupt.is_set():
            return
        await self._speak_reply(clean_reply)

    async def _do_chat(self, text: str) -> None:
        """执行一轮 LLM + TTS"""
        print(f"\n[你]   {text}")

        # LLM
        if self._interrupt.is_set():
            return
        reply = await self._llm_reply(text)
        if self._interrupt.is_set() or not reply:
            return

        print(f"[助手] {reply}")

        # TTS（分句播放，支持打断）
        await self._speak_reply(reply)

    async def _llm_reply(self, text: str) -> str:
        """调用 LLM 生成回复"""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text},
        ]
        try:
            r = await self._llm.chat(messages, temperature=0.7, max_tokens=256)
        except Exception as e:
            _logger.error("LLM 调用失败: %s", e)
            return "抱歉，我出了一点小问题。"
        if not r.success:
            _logger.warning("LLM 返回失败: %s", r.error)
            return "抱歉，我没有理解你的意思。"
        return (r.content or "").strip()

    # ------------------------------------------------------------------
    # 动作模式 — 见上方 _handle_action（短/长分流）
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 指令匹配（顺序匹配，命中即止；支持触发词列表）
    # ------------------------------------------------------------------
    async def _match_command(self, text: str) -> CommandConfig | None:
        """匹配语音指令（顺序遍历，命中第一个就返回）"""
        for cmd in self._commands:
            if _match_triggers(text, cmd.trigger):
                # 固定欢迎只处理纯问候；“你好，帮我……”应继续交给 LLM。
                if _is_greeting_command(cmd) and _has_content_beyond_greeting(text):
                    continue
                return cmd
        return None

    # ------------------------------------------------------------------
    # LLM 聊天（带 Function Call 支持）
    # ------------------------------------------------------------------
    async def _do_chat_with_tools(self, text: str) -> None:
        """动作模式：LLM 聊天 + Function Call 动作执行"""
        print(f"\n[你]   {text}")

        # 获取可用工具描述
        tools = self._get_action_tools()

        # LLM 调用
        if self._interrupt.is_set():
            return
        
        # 第一步：尝试让 LLM 决定是否调用工具
        messages = [
            {"role": "system", "content": self._system_prompt + " 你可以调用机器人动作工具来让机器人执行动作。"},
            {"role": "user", "content": text},
        ]
        
        tool_calls = await self._llm_get_tool_calls(messages, tools)
        
        if self._interrupt.is_set():
            return

        # 如果有工具调用，执行它们
        if tool_calls:
            for tc in tool_calls:
                action_name = tc.get("function", {}).get("name", "")
                _logger.info("LLM 请求执行动作: %s", action_name)
                await self._execute_llm_action(action_name)

        # 获取最终文本回复
        if self._interrupt.is_set():
            return
        reply = await self._llm_reply(text)
        if self._interrupt.is_set() or not reply:
            return

        print(f"[助手] {reply}")
        await self._speak_reply(reply)

    def _get_action_tools(self) -> list[dict]:
        """获取动作模式下可用的工具定义"""
        from autolife_s2_voice.run.act.act_engine import ACTION_REGISTRY
        
        tools = []
        for name, (_handler, desc) in ACTION_REGISTRY.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            })
        return tools

    async def _llm_get_tool_calls(self, messages: list, tools: list) -> list:
        """调用 LLM 获取 Function Call（支持原生 tools 和文本解析 fallback）"""
        try:
            # 先尝试原生 tool_calls（如果 LLM 支持）
            if hasattr(self._llm, 'chat'):
                r = await self._llm.chat(
                    messages, 
                    tools=tools,
                    temperature=0.7, 
                    max_tokens=256
                )
                if r.success and hasattr(r, 'tool_calls') and r.tool_calls:
                    return r.tool_calls
        except Exception as e:
            _logger.debug("LLM 原生 tool_calls 失败: %s", e)
        
        # Fallback: 尝试从文本回复中解析（Prompt 引导格式）
        tool_names = [t["function"]["name"] for t in tools]
        action_names_str = ", ".join(tool_names)
        
        parse_messages = [
            {"role": "system", "content": f"你可以调用以下动作: {action_names_str}。如果你想调用某个动作，请在回复中包含 [ACTION:动作名] 格式。例如 [ACTION:wave]。"},
            messages[-1]  # 只用最后一条用户消息
        ]
        
        try:
            r = await self._llm.chat(parse_messages, temperature=0.7, max_tokens=256)
            if r.success and r.content:
                # 正则匹配 [ACTION:xxx]
                match = re.search(r'\[ACTION:\s*(\w+)\s*\]', r.content)
                if match:
                    action_name = match.group(1)
                    if action_name in tool_names:
                        return [{
                            "function": {"name": action_name, "arguments": "{}"}
                        }]
        except Exception as e:
            _logger.warning("LLM 文本解析 fallback 失败: %s", e)
            
        return []

    async def _execute_llm_action(self, action_name: str) -> None:
        """执行 LLM 请求的动作"""
        try:
            from autolife_s2_voice.run.act.act_engine import execute_action
            await asyncio.to_thread(execute_action, action_name)
            _logger.info("LLM 动作执行完成: %s", action_name)
        except Exception as e:
            _logger.error("LLM 动作执行失败 (%s): %s", action_name, e)

    # ------------------------------------------------------------------
    # 指令执行（TTS 回复 + 动作）
    # ------------------------------------------------------------------
    async def _execute_command(self, cmd: CommandConfig) -> None:
        """执行语音指令：TTS 播报 + 动作并行执行

        TTS 播报与动作执行同时启动，并行进行：
          - TTS 在主事件循环中分句播放
          - 动作在后台线程执行（PKL 回放 / reset 复位）
          - 两者并行，谁先完成谁先结束，最后 gather 等待全部完成

        对于耗时动作（如 reset），动作在后台执行，不阻塞 TTS 播报。
        """
        tasks: list[asyncio.Task] = []

        # 1. TTS 播报回复（并行 task）
        async def _speak_task() -> None:
            if not (cmd.reply and self._tts):
                return
            try:
                print(f"[回复] {cmd.reply}")
                await self._speak_reply(cmd.reply)
                _logger.info("TTS 回复完成: %s", cmd.reply[:30])
            except Exception as e:
                _logger.error("TTS 播放失败: %s", e)

        # 2. 执行动作（并行 task）
        async def _action_task() -> None:
            if not cmd.action:
                return
            try:
                from autolife_s2_voice.run.act.act_engine import ACTION_REGISTRY
                handler_info = ACTION_REGISTRY.get(cmd.action)
                if handler_info is None:
                    _logger.warning("未知动作: %s", cmd.action)
                    return
                handler, desc = handler_info
                _logger.info("开始执行动作: %s (%s)", cmd.action, desc)

                if cmd.action == "reset":
                    # reset 耗时（~18s），后台 daemon 线程，不等待
                    thread = threading.Thread(target=handler, daemon=True)
                    thread.start()
                    _logger.info("reset 动作后台启动，不阻塞语音流程")
                else:
                    # PKL 动作：在 to_thread 中调用 handler，拿到返回的 Thread 后 join 等待完成
                    result = await asyncio.to_thread(handler)
                    if hasattr(result, "join"):
                        # PKL 动作返回后台播放线程，在 to_thread 中 join 等待播完
                        _logger.info("动作 '%s' 播放中...", cmd.action)
                        await asyncio.to_thread(result.join)
                        _logger.info("动作 '%s' 已完成", cmd.action)
                    else:
                        # 预置位姿（home/left_open 等）同步执行，已返回
                        _logger.info("动作 '%s' 已完成", cmd.action)
            except Exception as e:
                _logger.error("动作 '%s' 执行失败: %s", cmd.action, e)

        # 并行启动 TTS 和动作
        tasks.append(asyncio.create_task(_speak_task()))
        tasks.append(asyncio.create_task(_action_task()))
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # TTS 分句流水线播放（带打断检测）
    #
    # 设计：合成速度是播放速度的 11 倍（Matcha RTF≈0.09），
    # 所以上一句播放期间足够完成下一句的合成。
    # 流水线：播放 N 时后台合成 N+1，做到无缝衔接。
    # ------------------------------------------------------------------
    _SENT_SPLIT = re.compile(r"([^。！？!?；;\n]+[。！？!?；;\n]*)")
    _SENT_MIN_CHARS = 5      # 最少 5 字（约 1.25 秒语音），保证流水线有效
    _SENT_MAX_CHARS = 40     # 最长 40 字强制截断

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """按标点切分 + 合并短句"""
        text = (text or "").strip()
        if not text:
            return []
        parts = cls._SENT_SPLIT.findall(text)
        out: list[str] = []
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            buf += p
            if len(buf) >= cls._SENT_MIN_CHARS:
                # 超长截断
                while len(buf) > cls._SENT_MAX_CHARS:
                    out.append(buf[:cls._SENT_MAX_CHARS])
                    buf = buf[cls._SENT_MAX_CHARS:]
                out.append(buf)
                buf = ""
        if buf:
            # 剩余短句：合并到最后一句
            if out:
                out[-1] += buf
            else:
                out.append(buf)
        return out

    async def _speak_reply(self, reply: str) -> None:
        """流水线分句播放：合成与播放并行，单 SHM 连接无间隙"""
        # Web调用时结构化文字已经打印完成：先通知网页返回，TTS继续播放。
        # flush会同时冲出此前的[助手]/[任务]/[地点]行。
        if self._emit_web_ready_marker and not self._web_ready_emitted:
            print(WEB_READY_MARKER, flush=True)
            self._web_ready_emitted = True
        sentences = self._split_sentences(reply)
        if not sentences:
            return

        sr = self._tts.sample_rate
        _logger.info("TTS 分句: %d 句", len(sentences))

        # 线程安全队列：合成线程入队 → 播放线程出队
        audio_queue: queue.Queue = queue.Queue(maxsize=4)

        # 背景合成协程：逐句合成并入队
        async def _synthesizer():
            try:
                for i, s in enumerate(sentences):
                    if self._interrupt.is_set():
                        break
                    audio = await asyncio.to_thread(self._tts.synthesize, s)
                    if audio is not None:
                        audio_queue.put((audio, sr))
                audio_queue.put(None)  # 结束信号
            except Exception as e:
                _logger.warning("TTS 合成管线异常: %s", e)
                audio_queue.put(None)

        synth_task = asyncio.create_task(_synthesizer())

        # SHM 播放线程：单连接连续写入，无间隙
        try:
            await self._tts.play_audio_stream(audio_queue)
        except Exception as e:
            _logger.warning("TTS 播放管线异常: %s", e)
        finally:
            # 确保合成任务结束
            if not synth_task.done():
                synth_task.cancel()
                try:
                    await synth_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    async def run(self, text: str | None = None) -> None:
        """运行语音交互引擎

        Args:
            text: 一次性模式（非交互），跳过 ASR 直接处理 text
                  - 动作模式: 先匹配指令，未命中走 LLM 对话
                  - 展厅模式: 仅匹配指令
                  - 聊天模式: 直接 LLM 对话
        """
        await self._init()

        if text:
            # 一次性模式：按当前模式处理（指令匹配优先）
            mode_handler = {
                MODE_EXHIBITION: self._handle_exhibition,
                MODE_CHAT: self._handle_chat,
                MODE_ACTION: self._handle_action,
            }.get(self._mode, self._handle_chat)
            # 关键词提取（一次性文本模式也触发）
            if _JIEBA_AVAILABLE and self._keyword_config is not None and self._keyword_config.enabled:
                try:
                    self._last_keywords = await asyncio.to_thread(
                        self._extract_keywords, text
                    )
                    if self._last_keywords:
                        _logger.info("[关键词] %s", self._last_keywords)
                        self._save_keyword_log(text, self._last_keywords)
                        self._last_matches = self._match_keyword_mapping(self._last_keywords)
                        if self._last_matches:
                            for m in self._last_matches:
                                _logger.info("[映射] %s -> %s (%s)", m["matched"], m["targets"], m["label"])
                except Exception as e:
                    _logger.warning("关键词提取失败: %s", e)

            await mode_handler(text)
            if self._llm is not None:
                try:
                    await self._llm.close()
                except Exception:
                    pass
            _logger.info("一次性处理完成")
            return

        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _on_signal():
            _logger.info("收到中断信号，正在退出...")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_signal)
            except (NotImplementedError, RuntimeError):
                try:
                    signal.signal(sig, lambda *_: _on_signal())
                except (ValueError, OSError):
                    pass

        mode_desc = {
            MODE_EXHIBITION: "展厅模式 — 仅匹配特殊指令",
            MODE_CHAT: "聊天模式 — ASR → LLM → TTS 对话",
            MODE_ACTION: "动作模式 — 指令+对话+动作",
        }
        print()
        print("=" * 60)
        print(f"机器人语音交互引擎 — {mode_desc[self._mode]}")
        print(f"  ASR: {type(self._asr).__name__}")
        print(f"  TTS: {self._tts.engine_name} (playback={self._tts.playback_mode}, volume={self._tts_volume*100:.0f}%)")
        if self._llm is not None:
            print(f"  LLM: {self._llm_provider}")
        if self._track_enabled:
            print(f"  视觉跟踪: 已启用 (说 '跟踪我的手' 触发，'停止跟踪' 退出)")
        if self._commands:
            print(f"  指令: {[c.trigger for c in self._commands]}")
        print("按 Ctrl+C 退出")
        print("=" * 60)
        print()

        asr_task = asyncio.create_task(self._asr.start())

        try:
            await stop_event.wait()
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except Exception:
                    pass
            _logger.info("正在停止 ASR...")
            await self._asr.stop()
            asr_task.cancel()
            try:
                await asyncio.wait_for(asr_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            # 停止视觉跟踪会话（清理相机/颈部/ROS2）
            if self._track_session is not None:
                try:
                    await self._track_session.stop()
                except Exception as e:
                    _logger.warning("停止视觉跟踪失败: %s", e)
                self._track_session = None
            if self._llm is not None:
                try:
                    await self._llm.close()
                except Exception:
                    pass
            _logger.info("语音交互引擎已退出")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    cfg_path = _PROJECT_ROOT / "config" / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


async def amain():
    parser = argparse.ArgumentParser(
        description="Autolife S2 机器人语音交互引擎（三模式）",
    )
    parser.add_argument("--mode", "-m", type=str, default=MODE_ACTION,
                        choices=list(VALID_MODES),
                        help="运行模式: exhibition(展厅) / chat(聊天) / action(动作，默认)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="指令配置文件路径 (默认 voice_interaction.json)")
    parser.add_argument("--tts-engine", type=str, default="matcha",
                        choices=["matcha", "piper", "sherpa"],
                        help="TTS 引擎 (默认 matcha)")
    parser.add_argument("--tts-playback", type=str, default="auto",
                        choices=["auto", "local"],
                        help="TTS 播放方式 (开发机用 local)")
    parser.add_argument("--tts-speed", type=float, default=1.0,
                        help="TTS 语速")
    parser.add_argument("--tts-volume", type=float, default=2.0,
                        help="TTS 音量倍数 (默认 2.0=200%%)")
    parser.add_argument("--asr-engine", type=str, default="auto",
                        choices=["auto", "sherpa"],
                        help="ASR 引擎 (开发机默认 sherpa)")
    parser.add_argument("--llm-provider", type=str, default="openai_compat",
                        choices=["openai_compat", "mock"],
                        help="LLM provider (无网络时用 mock)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="LLM 系统提示词")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="一次性模式: 跳过 ASR，直接 text→LLM→TTS（非交互测试）")
    parser.add_argument("--no-track", action="store_true",
                        help="禁用视觉跟踪（默认启用，说 '跟踪我的手' 触发）")
    parser.add_argument("--track-camera", type=str, default="rgbd_head",
                        help="跟踪用相机 (默认 rgbd_head)")
    parser.add_argument("--no-ros2", action="store_true",
                        help="禁用 ROS2 颈部控制（开发机调试用）")
    parser.add_argument("--no-display", action="store_true",
                        help="禁用跟踪 GUI 显示（无头模式）")
    parser.add_argument("--web-ready-marker", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    engine = RobotVoiceInteraction(
        mode=args.mode,
        config_path=args.config,
        tts_engine=args.tts_engine,
        tts_playback=args.tts_playback,
        tts_speed=args.tts_speed,
        tts_volume=args.tts_volume,
        asr_engine=args.asr_engine,
        llm_provider=args.llm_provider,
        system_prompt=args.system_prompt,
        enable_track=not args.no_track,
        track_enable_ros2=not args.no_ros2,
        track_enable_display=not args.no_display,
        track_camera=args.track_camera,
        emit_web_ready_marker=args.web_ready_marker,
    )
    await engine.run(text=args.text)


def main():
    # 检查是否需要 source ROS2 环境
    try:
        import rclpy  # noqa: F401
        # rclpy 可以导入，直接运行
    except (ImportError, Exception):
        # rclpy 导入失败，需要 source ROS2 环境后重新运行
        ros_setup = "/opt/ros/jazzy/setup.bash"
        if os.path.exists(ros_setup):
            _logger.warning("rclpy 导入失败，source ROS2 环境后重新启动...")
            # 用 bash source 环境后 exec 当前脚本
            os.execvpe(
                "/bin/bash",
                ["/bin/bash", "-c", f"source {ros_setup} && exec \"$@\"", "--", sys.executable] + sys.argv,
                os.environ
            )
        else:
            _logger.error("找不到 ROS2 环境文件: %s，动作功能不可用", ros_setup)

    asyncio.run(amain())


if __name__ == "__main__":
    main()
