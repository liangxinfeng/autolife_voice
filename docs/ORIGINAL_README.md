# autolife_s2_voice（原始说明，部分历史内容）

Autolife S2 standalone voice chain for robot 306. It contains the local ASR,
three-mode intent router, configured voice actions, TTS pipeline and optional
browser recorder formerly embedded in `autolife_s2`.

## Start

```bash
~/s2voice
```

The helper refuses duplicate standalone instances. If the legacy
`robots/autolife_s2/run/robot_voice.py` is still running, it sends that voice
process `SIGTERM`, waits for it to exit, and only then starts this package. It
does not stop other `autolife_s2` processes or robot services.

Equivalent ROS 2 command:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash
ros2 launch autolife_s2_voice voice.launch.py
```

The default starts continuous robot-microphone voice interaction in `action`
mode. RGB-D tracking is deliberately disabled by default and can be enabled
with `enable_track:=true`.

Browser-only recording mode:

```bash
~/s2voice start_voice:=false start_web:=true
```

Then open `https://<robot-ip>:8765` and accept the local certificate.

## Shared runtime dependency

Models and common ASR/LLM/TTS service implementations remain in
`/home/ubuntu/code/void-cog`. Override this location with the launch argument
`void_cog_root:=...`. Robot audio and camera access still require
`autolife_robot_sdk` from `robot_env`.

Do not run this package together with the legacy
`robots/autolife_s2/run/robot_voice.py`; both consume the same microphone and
speaker resources.
