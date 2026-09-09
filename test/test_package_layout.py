from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_voice_sources_do_not_use_legacy_robot_imports():
    source_root = ROOT / 'autolife_s2_voice' / 'run'
    offenders = []
    for path in source_root.rglob('*.py'):
        if 'robots.autolife_s2.run' in path.read_text(encoding='utf-8'):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_required_voice_assets_are_packaged():
    package = ROOT / 'autolife_s2_voice'
    assert (package / 'configs' / 'voice_interaction.json').is_file()
    assert (package / 'configs' / 'keyword_mapping.json').is_file()
    assert (package / 'run' / 'asr' / 'asr_factory.py').is_file()
    assert (package / 'run' / 'tts' / 'tts_factory.py').is_file()
    assert (package / 'run' / 'act' / 'act_engine.py').is_file()


def test_launch_has_safe_defaults():
    launch = (ROOT / 'launch' / 'voice.launch.py').read_text(encoding='utf-8')
    assert "DeclareLaunchArgument('robot_id', default_value='307')" in launch
    assert "'enable_track', default_value='false'" in launch
    assert "'start_web', default_value='false'" in launch


def test_task_catalog_contains_only_locations_and_tasks():
    path = ROOT / 'autolife_s2_voice' / 'configs' / 'keyword_mapping.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    assert set(data) == {'task_locations'}
    assert all(isinstance(location, str) and location for location in data['task_locations'])
    assert all(
        isinstance(tasks, list)
        and tasks
        and all(isinstance(task, str) and task for task in tasks)
        for tasks in data['task_locations'].values()
    )
    serialized = json.dumps(data, ensure_ascii=False)
    for removed_key in ('hints', 'reply', 'mappings', 'short_commands', 'triggers'):
        assert removed_key not in serialized


def test_laundry_steps_are_separate_catalog_entries():
    path = ROOT / 'autolife_s2_voice' / 'configs' / 'keyword_mapping.json'
    locations = json.loads(path.read_text(encoding='utf-8'))['task_locations']
    assert '将毛巾放入洗衣篮' in locations['毛巾区']
    assert '送毛巾篮子送到洗衣房桌面' in locations['洗衣区']
