# 云蝶 S2 语音交互

从307机器人导出的独立语音功能包（2026-09-09）。支持机器人麦克风、浏览器录音、语音识别、意图/动作路由和语音合成；保留聊天、交互、展厅三类模式。

**这是语音链路，不是完整机器人系统。** ASR/LLM/TTS公共实现及模型依赖外部 `void-cog`，音频和动作执行依赖厂商SDK。

## 环境与安装

- Ubuntu 24.04、ROS 2 Jazzy、Python 3.12。
- 已配置的 `/home/ubuntu/miniconda3/envs/robot_env`，其中含厂商 `autolife_robot_sdk`。
- `/home/ubuntu/code/void-cog` 及其模型、依赖和模型服务配置。该目录不在仓库中，模型服务凭据需自行配置。

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/liangxinfeng/autolife_s2_voice.git
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src/autolife_s2_voice --ignore-src -r -y
colcon build --packages-select autolife_s2_voice
source install/setup.bash
```

## 启动

机器人麦克风交互（可能触发真机动作，使用前核对配置并清空运动空间）：

```bash
ros2 launch autolife_s2_voice voice.launch.py robot_id:=307
```

仅浏览器录音：

```bash
ros2 launch autolife_s2_voice voice.launch.py start_voice:=false start_web:=true
```

访问 `https://<机器人IP>:8765`。首次使用需先生成本机HTTPS证书，见下节。原机的 `~/s2voice` 是仓库外快捷入口，克隆不会自动生成。

## HTTPS证书

**公开仓库不包含原机器人私钥或证书。** 在编译前，为实际访问IP生成新证书：

```bash
cd ~/ros2_ws/src/autolife_s2_voice
robot_ip=192.168.8.68  # 改成实际IP
mkdir -p autolife_s2_voice/configs/https
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout autolife_s2_voice/configs/https/web.key \
  -out autolife_s2_voice/configs/https/web.crt \
  -subj "/CN=$robot_ip" \
  -addext "subjectAltName=IP:$robot_ip,DNS:localhost"
chmod 600 autolife_s2_voice/configs/https/web.key
```

生成后重新编译该包，浏览器信任证书后才能使用麦克风；正式部署建议使用终端信任的证书。不要提交私钥。

## 常用配置

| 内容 | 文件/参数 |
| --- | --- |
| 模式、动作与回复 | `autolife_s2_voice/configs/voice_interaction.json` |
| 地点/任务关键词 | `autolife_s2_voice/configs/keyword_mapping.json` |
| 动作、手势和关节约束 | 同目录的 `gestures.json`、`gesture_rules.json`、`joint_limits.yaml` |
| 动作数据 | `autolife_s2_voice/asset/`、`example/pkl/` |
| 编号、模型目录、解释器 | `launch/voice.launch.py` |

可通过 `robot_id:=...`、`void_cog_root:=...`、`robot_env_python:=...` 覆盖默认配置。

不要与旧版语音进程同时占用麦克风/喇叭；不要执行不可信PKL文件。导航任务编排界面位于独立的 [VR导航项目](https://github.com/liangxinfeng/openarmx_teleop_vr_navigation_306)，不是本包8765页面。

## 交付范围

本次保留307源代码和动作资源，仅增加说明及上传忽略规则。未随上传操作启动真机测试。保留原 `Proprietary` 声明，不为厂商SDK、模型和动作数据重新授予许可证。

[原始说明](docs/ORIGINAL_README.md)
