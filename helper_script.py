# by Shadab Alam <md_shadab_alam@outlook.com>

from __future__ import annotations
import logging
import threading
from typing import Optional
import pandas as pd
import torch
import common
from config_utils import ConfigUtils
from custom_logger import CustomLogger

configutils_class = ConfigUtils()
logger = CustomLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Consts / defaults
LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False

UPGRADE_LOG_FILE = "upgrade_log.json"

# Thread-local storage for per-thread model instances
_TLS = threading.local()


class Youtube_Helper:
    """
    Helper class: download/trim/video utils + tracking.
    """

    def __init__(self, video_title: Optional[str] = None):
        self.video_title = video_title

        # Snellius/HPC flag(s)
        self.snellius_mode = bool(configutils_class._safe_get_config("snellius_mode", False))
        # Default behavior on HPC: do not download from internet/FTP on compute nodes
        self.snellius_disable_downloads = bool(
            configutils_class._safe_get_config("snellius_disable_downloads", True if self.snellius_mode else False)
        )
        # Optional TF32 (usually beneficial on A100/H100; can be disabled if you prefer)
        self.snellius_tf32 = bool(configutils_class._safe_get_config("snellius_tf32", True))

        if self.snellius_mode and torch.cuda.is_available() and self.snellius_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Available in newer PyTorch; best-effort.
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
                logger.info("Snellius mode: TF32 enabled (matmul/cudnn) for performance.")
            except Exception as e:
                logger.warning(f"Snellius mode: TF32 enable failed (continuing): {e!r}")

        # Models / trackers (same keys you already use)
        self.tracking_model = common.get_configs("tracking_model")
        self.segment_model = common.get_configs("segment_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")

        # Tracking behavior
        try:
            self.confidence = float(common.get_configs("confidence") or 0.0)
        except Exception:
            self.confidence = 0.0

        try:
            self.track_buffer_sec = float(common.get_configs("track_buffer_sec"))
        except Exception:
            self.track_buffer_sec = 2.0

        # Existing flags (kept for compatibility; not required by thread-safe path)
        self.display_frame_tracking = bool(configutils_class._safe_get_config("display_frame_tracking", False))
        self.display_frame_segmentation = bool(configutils_class._safe_get_config("display_frame_segmentation", False))
        self.output_path = configutils_class._safe_get_config("videos", ".")
        self.save_annoted_img = bool(configutils_class._safe_get_config("save_annoted_img", False))
        self.save_tracked_img = bool(configutils_class._safe_get_config("save_tracked_img", False))
        self.delete_labels = bool(configutils_class._safe_get_config("delete_labels", False))
        self.delete_frames = bool(configutils_class._safe_get_config("delete_frames", False))
        self.update_package = bool(configutils_class._safe_get_config("update_package", False))
        self.need_authentication = bool(configutils_class._safe_get_config("need_authentication", False))
        self.client = configutils_class._safe_get_config("client", None)

        # Snellius: never do runtime self-upgrades on compute nodes
        if self.snellius_mode:
            self.update_package = False

        # Mapping (some pipelines still use this)
        try:
            self.mapping_path = common.get_configs("mapping")
            self.mapping = pd.read_csv(self.mapping_path)
        except Exception:
            self.mapping_path = None
            self.mapping = None

    # -------------------------------------------------------------------------
    # General utilities
    # -------------------------------------------------------------------------

    def set_video_title(self, title: str) -> None:
        self.video_title = title
