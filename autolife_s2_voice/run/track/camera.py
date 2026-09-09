"""RGBD 相机源 — 通过 SDK SHM 获取机器人头部相机 BGR 图像"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_logger = logging.getLogger("track.camera")


class RgbdCamera:
    """通过 SDK SHM 获取机器人 RGBD 头部相机 BGR 图像

    Parameters
    ----------
    camera_name:
        相机标签名，会自动映射到 SDK module_name。
        可选: head_left / head_right / head_rear / head_binocular
              hand_left / hand_right / rgbd_head (默认)
    """

    # 标签名 → SDK module_name 映射
    _LABEL_TO_MOD = {
        "head_left": "mod_camera_head_left",
        "head_right": "mod_camera_head_right",
        "head_rear": "mod_camera_head_rear",
        "head_binocular": "mod_camera_head_binocular",
        "hand_left": "mod_camera_hand_left",
        "hand_right": "mod_camera_hand_right",
        "rgbd_head": "mod_camera_rgbd_head",
    }

    def __init__(self, camera_name: str = "rgbd_head") -> None:
        from autolife_robot_sdk.utils import (
            list_camera_shm_outputs,
            open_camera_shm_consumer,
        )

        self._camera_name = camera_name
        outputs = list_camera_shm_outputs("autolife_s1", "robot_v2_2")

        target_mod = self._LABEL_TO_MOD.get(camera_name, camera_name)

        output = None
        for o in outputs:
            if (o.module_name == target_mod
                    and o.output_name in ("decoded", "color")
                    and o.pixel_format in ("BGR", "RGB")):
                output = o
                break
        if output is None:
            available = sorted({o.module_name for o in outputs})
            raise RuntimeError(
                f"找不到相机 '{camera_name}' (module={target_mod})，"
                f"可用模块: {available}")

        self._consumer = open_camera_shm_consumer(output, name=camera_name)
        dims = f"{output.width}x{output.height}" if output.width else "?"
        _logger.info("RGBD 相机已打开: %s %s %s",
                     camera_name, dims, output.pixel_format)

    def read(self) -> np.ndarray | None:
        """读取最新一帧 BGR 图像（非阻塞），无帧返回 None"""
        result = self._consumer.get_latest(nonblock=True, with_meta=True)
        if result is None:
            return None
        img, _fid, _meta = result
        return img

    def release(self) -> None:
        try:
            self._consumer.close()
        except Exception:
            pass
        _logger.info("RGBD 相机已关闭")
