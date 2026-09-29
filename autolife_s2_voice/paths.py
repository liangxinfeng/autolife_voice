"""Stable package and shared-service paths for robot deployment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
VOID_COG_ROOT = Path(
    os.environ.get('AUTOLIFE_VOID_COG_ROOT', '/home/ubuntu/ros2_ws/external/void-cog')
).expanduser().resolve()
STATE_ROOT = Path(
    os.environ.get(
        'AUTOLIFE_S2_VOICE_STATE_DIR',
        str(Path.home() / '.ros' / 'autolife_s2_voice'),
    )
).expanduser().resolve()


def ensure_void_cog_importable() -> None:
    """Expose the shared model/service framework without cwd assumptions."""
    root = str(VOID_COG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_void_cog_importable()
