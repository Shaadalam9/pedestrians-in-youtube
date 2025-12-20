"""HPC and Slurm utilities.

This module contains helpers for running the pipeline on HPC environments
(e.g., Snellius) where jobs may be sharded across tasks and may stage data
to node-local storage.
"""

import os
import threading
import zlib
from types import SimpleNamespace
from typing import Tuple

import torch

from custom_logger import CustomLogger

# Module-scoped logger for consistent, centrally configured logging output.
logger = CustomLogger(__name__)


class HPC:
    """HPC/Slurm helper functions for Snellius-mode runs.

    The methods in this class are designed to be safe no-ops when the process is
    not running under Slurm or when snellius_mode is disabled.

    Notes:
        - This class intentionally has no persistent state; it only provides helpers.
        - Several methods read and/or write environment variables (process-wide side effects).
    """

    def __init__(self) -> None:
        """Initializes the HPC helper.

        This class is stateless; it provides utility methods only.
        """
        # Nothing to initialize; present for interface symmetry and future extension.
        pass

    def _resolve_tmp_root(self, config: SimpleNamespace) -> str:
        """Resolves tmp root for Snellius staging.

        Preference order:
          1) $TMPDIR (if set by Slurm)
          2) config.snellius_tmp_root (expanded; may include "$TMPDIR")
          3) "" (empty string meaning no temp root is available)

        Args:
          config: Runtime config namespace (SimpleNamespace or similar).

        Returns:
          Temp root path string, or "" when no staging root is available.
        """
        # Slurm often exports TMPDIR pointing to node-local fast storage.
        # If present, it is generally the most appropriate staging location.
        env_tmp = os.getenv("TMPDIR", "") or ""
        if env_tmp:
            return env_tmp

        # Allow explicit configuration override (may contain environment variables).
        cfg_tmp = str(getattr(config, "snellius_tmp_root", "") or "").strip()
        if cfg_tmp:
            # Expand environment variables like "$TMPDIR" if present.
            return os.path.expandvars(cfg_tmp)

        # No tmp root found.
        return ""

    def _maybe_bind_gpu_from_slurm_localid(self, config: SimpleNamespace) -> None:
        """Binds one GPU per Slurm task if Slurm did not set CUDA_VISIBLE_DEVICES.

        If Snellius mode is enabled and CUDA is available, this sets
        CUDA_VISIBLE_DEVICES from SLURM_LOCALID to help each task use a distinct
        GPU on the node.

        This function does nothing when:
          - snellius_mode is disabled,
          - CUDA is not available,
          - CUDA_VISIBLE_DEVICES is already set.

        Args:
          config: Runtime config namespace.

        Returns:
          None.
        """
        # Only apply this behavior when explicitly running in Snellius mode.
        # This avoids surprising GPU binding behavior in non-HPC/local runs.
        if not bool(getattr(config, "snellius_mode", False)):
            return

        # If there is no CUDA device, binding has no effect.
        if not torch.cuda.is_available():
            return

        # If Slurm already set CUDA_VISIBLE_DEVICES, do not override it.
        # Many clusters configure this explicitly; overriding could break scheduling policy.
        cur = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cur:
            return

        # SLURM_LOCALID is typically an integer per node (0..N-1).
        # Default to "0" if missing or malformed.
        try:
            local_id = int(os.environ.get("SLURM_LOCALID", "0"))
        except Exception:
            local_id = 0

        # Expose only one GPU to this process.
        # This environment variable is consumed by CUDA runtime and most ML frameworks.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_id)

        # Log for traceability and debugging of multi-GPU behavior.
        logger.info(
            "Set CUDA_VISIBLE_DEVICES={} from SLURM_LOCALID={}".format(
                os.environ["CUDA_VISIBLE_DEVICES"],
                local_id,
            )
        )

    def _snellius_task_identity(self) -> Tuple[int, int]:
        """Determines (task_id, task_count) for sharding.

        Preference order:
          1) SLURM_PROCID / SLURM_NTASKS (MPI-style multi-task jobs)
          2) SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT (array jobs)
          3) Fallback: (0, 1)

        Returns:
          (task_id, task_count) as integers.
        """
        # MPI-style multi-task jobs (common with srun and multiple tasks).
        # This is the preferred sharding signal when multiple ranks are launched together.
        procid = os.getenv("SLURM_PROCID", None)
        ntasks = os.getenv("SLURM_NTASKS", None)
        if procid is not None and ntasks is not None:
            try:
                return int(procid), int(ntasks)
            except Exception:
                # If parsing fails, fall back to array semantics below.
                pass

        # Array job semantics (common with sbatch --array).
        # In this case, each array element is typically a distinct job instance.
        array_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
        array_count = os.getenv("SLURM_ARRAY_TASK_COUNT", "1")
        try:
            return int(array_id), int(array_count)
        except Exception:
            # Conservative fallback: treat as single task.
            return 0, 1

    def _shard_to_task(self, key: str, task_count: int) -> int:
        """Maps a stable string key to a task index deterministically.

        Uses CRC32 for speed and stability across runs.

        Args:
          key: Stable identifier (e.g., video_id).
          task_count: Total number of tasks available.

        Returns:
          Task index in [0, task_count - 1]. If task_count <= 1, returns 0.
        """
        # Single-task case: everything belongs to task 0.
        if task_count <= 1:
            return 0

        # Compute a stable 32-bit hash for the key.
        # Mask ensures the result is treated as an unsigned 32-bit value.
        #
        # CRC32 is intentionally used here because it is:
        # - deterministic across runs/platforms for the same input bytes,
        # - fast, and
        # - sufficient for even sharding (not security-sensitive).
        h = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF

        # Modulo assigns the key to one of the tasks.
        return int(h % task_count)

    def _log_run_banner(self, config: SimpleNamespace) -> None:
        """Logs a run configuration banner for debugging/reproducibility.

        Args:
          config: Runtime config namespace.

        Returns:
          None.
        """
        # Determine whether we can use CUDA and which GPU is visible.
        # Note: get_device_name(0) assumes at least one visible CUDA device.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

        # Make the banner visually distinct in log files.
        logger.info("============================================================")
        logger.info("Pipeline run configuration")

        # Log the runtime device context.
        logger.info("device={} gpu={}".format(device, gpu_name))

        # Log CPU and torch thread settings (useful for performance debugging).
        # Thread configuration can materially affect throughput on shared nodes.
        logger.info(
            "cpu_count={} torch_num_threads={}".format(
                os.cpu_count(),
                torch.get_num_threads(),
            )
        )

        # Log HPC mode and Slurm environment details.
        # These values are frequently used to diagnose sharding and resource binding issues.
        logger.info("snellius_mode={}".format(bool(getattr(config, "snellius_mode", False))))
        logger.info(
            "slurm_task_id={} slurm_task_count={} slurm_localid={} cuda_visible={} tmpdir={}".format(
                getattr(config, "slurm_task_id", 0),
                getattr(config, "slurm_task_count", 1),
                os.getenv("SLURM_LOCALID", ""),
                os.getenv("CUDA_VISIBLE_DEVICES", ""),
                getattr(config, "tmpdir", ""),
            )
        )

        # Log sharding and fallback controls.
        # These flags impact which assets can be downloaded/processed on compute nodes.
        logger.info(
            "snellius_shard_mode={}".format(getattr(config, "snellius_shard_mode", "video"))
        )
        logger.info(
            "snellius_disable_youtube_fallback={}".format(
                bool(getattr(config, "snellius_disable_youtube_fallback", False))
            )
        )

        # Log concurrency configuration.
        # Concurrency settings can lead to oversubscription if not tuned for node limits.
        logger.info(
            "max_workers={} (segment ThreadPoolExecutor)".format(getattr(config, "max_workers", None))
        )
        logger.info(
            "download_workers={} (download ThreadPoolExecutor)".format(
                getattr(config, "download_workers", None)
            )
        )
        logger.info(
            "prefetch_videos={} (max in-flight videos)".format(getattr(config, "prefetch_videos", None))
        )
        logger.info(
            "max_active_segments_per_video={}".format(
                getattr(config, "max_active_segments_per_video", None)
            )
        )
        logger.info("active_threads_now={}".format(threading.active_count()))

        # Log model and tracker settings.
        # Keeping these in logs simplifies reproducibility and debugging across reruns.
        logger.info(
            "tracking_mode={} segmentation_mode={}".format(
                getattr(config, "tracking_mode", None),
                getattr(config, "segmentation_mode", None),
            )
        )
        logger.info("tracking_model={}".format(getattr(config, "tracking_model", None)))
        logger.info("segment_model={}".format(getattr(config, "segment_model", None)))
        logger.info("bbox_tracker={}".format(getattr(config, "bbox_tracker", None)))
        logger.info("seg_tracker={}".format(getattr(config, "seg_tracker", None)))
        logger.info("track_buffer_sec={}".format(getattr(config, "track_buffer_sec", None)))

        # Log effective confidence configured on the helper.
        logger.info("confidence={}".format(getattr(config, "confidence", 0.0)))

        # Log storage and download policy flags.
        # These are particularly relevant when diagnosing missing files on compute nodes.
        logger.info(
            "external_ssd={}".format(
                getattr(config, "external_ssd", False)
            )
        )
        logger.info("delete_youtube_video={}".format(getattr(config, "delete_youtube_video", False)))

        logger.info("============================================================")
