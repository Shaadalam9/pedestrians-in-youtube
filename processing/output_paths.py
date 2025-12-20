"""Output path and directory utilities.

This module centralizes filesystem layout and housekeeping for output artifacts.

Key behaviors:
  - Ensures required output directories exist (bbox/ and seg/).
  - Creates temp directories used for "write-then-commit" output strategy.
  - Performs best-effort cleanup of stale partial files left by crashes.

Design notes:
  - These helpers are intentionally forgiving: failures in cleanup should not
    crash the pipeline.
  - The directory conventions should remain stable to avoid breaking downstream
    consumers.
"""

import glob
import os
from typing import Tuple


class OutputPaths:
    """Manages output directory creation and partial-file cleanup.

    The pipeline writes outputs under a "data_path" directory using the
    following structure:

      data_path/
        bbox/            Final bbox CSVs
        seg/             Final seg CSVs
        __tmp__/         Temporary staging area for atomic commits
          bbox/
          seg/

    The __tmp__ staging area is used to implement write-then-commit semantics:
      - Write output to __tmp__/.../*.partial
      - Atomically move into bbox/ or seg/ via os.replace()

    Notes:
        - All methods are designed to be idempotent and safe to call repeatedly.
        - Cleanup is best-effort: it must never prevent the pipeline from running.
    """

    def __init__(self) -> None:
        """Initializes the output path helper.

        This class is stateless; it only provides filesystem helpers.
        """
        # Nothing to initialize; present for future extension and interface symmetry.
        pass

    def _ensure_dirs(self, data_path: str) -> None:
        """Ensure that the main output directories exist.

        This method creates:
          - data_path/
          - data_path/bbox/
          - data_path/seg/

        Args:
            data_path: Base output directory.

        Returns:
            None.
        """
        # Create the base folder first so subfolder creation is reliable.
        # exist_ok=True makes directory creation idempotent.
        os.makedirs(data_path, exist_ok=True)

        # Create per-mode output directories; these are stable conventions.
        os.makedirs(os.path.join(data_path, "bbox"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "seg"), exist_ok=True)

    def _ensure_tmp_dirs(self, data_path: str) -> Tuple[str, str]:
        """Create temp dirs used for "write-then-commit" output staging.

        The pipeline stages files under __tmp__/ to ensure "all-or-nothing"
        visibility:
          - If the pipeline crashes, partially written files remain in __tmp__.
          - Final output folders contain only completed CSVs.

        Args:
            data_path: Base output directory.

        Returns:
            Tuple (tmp_bbox_dir, tmp_seg_dir) representing the per-mode staging paths.
        """
        # Keep temp paths inside the data directory so cleanup is straightforward
        # and does not require tracking separate OS temp locations.
        tmp_root = os.path.join(data_path, "__tmp__")

        # Separate directories per mode avoid collisions and simplify cleanup.
        tmp_bbox_dir = os.path.join(tmp_root, "bbox")
        tmp_seg_dir = os.path.join(tmp_root, "seg")

        # Ensure directories exist; creation is idempotent.
        os.makedirs(tmp_bbox_dir, exist_ok=True)
        os.makedirs(tmp_seg_dir, exist_ok=True)

        return tmp_bbox_dir, tmp_seg_dir

    def _cleanup_stale_partials(self, tmp_dir: str) -> None:
        """Best-effort cleanup of stale '*.partial' files from previous crashes.

        Stale partial files can appear if the pipeline terminates mid-write.
        Removing them:
          - reduces confusion when inspecting __tmp__ folders,
          - avoids accidental consumption of incomplete artifacts.

        This cleanup is intentionally best-effort:
          - Any errors (permissions, races, missing files) are ignored.
          - Cleanup must never crash the pipeline.

        Args:
            tmp_dir: Temp directory to scan for *.partial files.

        Returns:
            None.
        """
        # Guard glob operations: unusual filesystem states should not be fatal.
        try:
            # Only target "*.partial" files. We do not remove any other files to
            # avoid accidentally deleting debugging artifacts or completed outputs.
            pattern = os.path.join(tmp_dir, "*.partial")

            # Iterate over any partial files and try to remove them.
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                except Exception:
                    # Ignore individual file failures (e.g., concurrent removal, permission issues).
                    pass

        except Exception:
            # Ignore unexpected failures (e.g., directory disappeared, glob errors).
            pass
