"""Launch the standalone S2 voice chain and optional browser recorder."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration


def _enabled(value: str) -> bool:
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _voice_process(context):
    if not _enabled(LaunchConfiguration('start_voice').perform(context)):
        return []
    python = LaunchConfiguration('robot_env_python').perform(context)
    cmd = [
        python,
        '-m', 'autolife_s2_voice.run.robot_voice',
        '--mode', LaunchConfiguration('mode').perform(context),
        '--tts-engine', LaunchConfiguration('tts_engine').perform(context),
        '--tts-playback', LaunchConfiguration('tts_playback').perform(context),
        '--tts-speed', LaunchConfiguration('tts_speed').perform(context),
        '--tts-volume', LaunchConfiguration('tts_volume').perform(context),
        '--asr-engine', LaunchConfiguration('asr_engine').perform(context),
        '--llm-provider', LaunchConfiguration('llm_provider').perform(context),
        '--track-camera', LaunchConfiguration('track_camera').perform(context),
    ]
    if not _enabled(LaunchConfiguration('enable_track').perform(context)):
        cmd.append('--no-track')
    if not _enabled(LaunchConfiguration('track_enable_ros2').perform(context)):
        cmd.append('--no-ros2')
    if not _enabled(LaunchConfiguration('track_enable_display').perform(context)):
        cmd.append('--no-display')
    return [ExecuteProcess(cmd=cmd, output='screen')]


def _web_process(context):
    if not _enabled(LaunchConfiguration('start_web').perform(context)):
        return []
    python = LaunchConfiguration('robot_env_python').perform(context)
    cmd = [
        python,
        '-m', 'autolife_s2_voice.run.web_voice',
        '--host', LaunchConfiguration('web_host').perform(context),
        '--port', LaunchConfiguration('web_port').perform(context),
    ]
    if _enabled(LaunchConfiguration('web_https').perform(context)):
        cmd.append('--https')
    return [ExecuteProcess(cmd=cmd, output='screen')]


def generate_launch_description():
    robot_id = LaunchConfiguration('robot_id')
    ros_domain_id = LaunchConfiguration('ros_domain_id')
    void_cog_root = LaunchConfiguration('void_cog_root')
    state_dir = LaunchConfiguration('state_dir')

    return LaunchDescription([
        DeclareLaunchArgument('robot_id', default_value='307'),
        DeclareLaunchArgument('ros_domain_id', default_value='0'),
        DeclareLaunchArgument(
            'robot_env_python',
            default_value='/home/ubuntu/ros2_ws/venvs/openarmx_v4_placo/bin/python'),
        DeclareLaunchArgument(
            'void_cog_root', default_value='/home/ubuntu/ros2_ws/external/void-cog'),
        DeclareLaunchArgument(
            'state_dir', default_value='/home/ubuntu/.ros/autolife_s2_voice'),
        DeclareLaunchArgument('start_voice', default_value='true'),
        DeclareLaunchArgument(
            'mode', default_value='action',
            description='exhibition, chat or action'),
        DeclareLaunchArgument('asr_engine', default_value='auto'),
        DeclareLaunchArgument('tts_engine', default_value='matcha'),
        DeclareLaunchArgument('tts_playback', default_value='auto'),
        DeclareLaunchArgument('tts_speed', default_value='1.0'),
        DeclareLaunchArgument('tts_volume', default_value='2.0'),
        DeclareLaunchArgument('llm_provider', default_value='openai_compat'),
        DeclareLaunchArgument(
            'enable_track', default_value='false',
            description='Enable voice-triggered RGB-D tracking.'),
        DeclareLaunchArgument('track_camera', default_value='rgbd_head'),
        DeclareLaunchArgument('track_enable_ros2', default_value='true'),
        DeclareLaunchArgument('track_enable_display', default_value='false'),
        DeclareLaunchArgument('start_web', default_value='false'),
        DeclareLaunchArgument('web_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('web_port', default_value='8765'),
        DeclareLaunchArgument('web_https', default_value='true'),
        SetEnvironmentVariable('ROBOT_ID', robot_id),
        SetEnvironmentVariable('ROS_DOMAIN_ID', ros_domain_id),
        SetEnvironmentVariable('AUTOLIFE_VOID_COG_ROOT', void_cog_root),
        SetEnvironmentVariable('AUTOLIFE_S2_VOICE_STATE_DIR', state_dir),
        OpaqueFunction(function=_voice_process),
        OpaqueFunction(function=_web_process),
    ])
