"""Maintenance utilities for controlled runtime upgrades and filesystem cleanup.

This module provides:
- A minimal upgrade log mechanism to avoid repeated pip upgrades within a single day.
- A guarded runtime package-upgrade helper that is disabled in HPC/Snellius mode.
- Convenience filesystem operations (rename/delete) and cleanup of derived video outputs.

Operational constraints:
- In Snellius/HPC environments, runtime self-upgrades (pip install --upgrade) are
  intentionally disabled to avoid modifying shared environments on compute nodes.
"""

import os
import json
import datetime
from custom_logger import CustomLogger
import subprocess
import sys
import re
import shutil
from config_utils import ConfigUtils

# JSON file storing "package -> date" markers for attempted upgrades.
UPGRADE_LOG_FILE = "upgrade_log.json"

# Module-level logger for consistent logging configuration.
logger = CustomLogger(__name__)

# Shared config accessor used to read runtime flags (best-effort defaults).
config_utils = ConfigUtils()


class Maintenance:
    """Encapsulates upgrade and cleanup helpers with HPC-safe defaults.

    The primary responsibility is to provide a safe, idempotent way to attempt
    package upgrades at most once per day, while avoiding such upgrades entirely
    on Snellius/HPC compute nodes.

    Additional helpers include basic directory rename/delete operations and a
    targeted cleanup for derived YouTube "_mod.mp4" artifacts.
    """

    def __init__(self) -> None:
        """Initialize maintenance flags from configuration.

        Configuration keys (best-effort):
            - snellius_mode: Whether running under HPC constraints.
            - update_package: Whether runtime pip upgrades are allowed.

        Notes:
            If snellius_mode is enabled, update_package is forced off regardless of
            configuration to prevent runtime environment mutation on compute nodes.
        """
        # Snellius/HPC flag(s)
        self.snellius_mode = bool(config_utils._safe_get_config("snellius_mode", False))
        self.update_package = bool(config_utils._safe_get_config("update_package", False))
        # Snellius: never do runtime self-upgrades on compute nodes
        if self.snellius_mode:
            self.update_package = False

    def load_upgrade_log(self) -> dict:
        """Load the daily-upgrade marker log from disk.

        Returns:
            A dict mapping package names to ISO date strings.

        Notes:
            - Returns an empty dict if the log file does not exist.
            - Returns an empty dict if parsing fails (defensive behavior).
        """
        # If no log exists, treat as no upgrades performed.
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Defensive: corrupted/partial log should not break the pipeline.
            return {}

    def save_upgrade_log(self, log_data: dict) -> None:
        """Persist the upgrade log to disk.

        Args:
            log_data: Dict mapping package names to ISO date strings.

        Returns:
            None.
        """
        # Overwrite the log file to keep a simple single-source-of-truth state.
        with open(UPGRADE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f)

    def was_upgraded_today(self, package_name: str) -> bool:
        """Check whether an upgrade for a package was attempted today.

        Args:
            package_name: Name of the package to check (pip package name).

        Returns:
            True if the upgrade log indicates the package was attempted today,
            otherwise False.
        """
        log_data = self.load_upgrade_log()
        today = datetime.date.today().isoformat()
        return log_data.get(package_name) == today

    def mark_as_upgraded(self, package_name: str) -> None:
        """Record that an upgrade attempt for a package occurred today.

        This is marked even when the upgrade attempt fails, to prevent repeated
        attempts within the same day.

        Args:
            package_name: Name of the package to mark.

        Returns:
            None.
        """
        log_data = self.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        self.save_upgrade_log(log_data)

    def upgrade_package_if_needed(self, package_name: str) -> None:
        """Attempt to upgrade a pip package at most once per day.

        Guardrails:
            - No-op when running in Snellius/HPC mode.
            - No-op when update_package is disabled by configuration.
            - Skips if an attempt was already made today.

        Args:
            package_name: Pip package name to upgrade.

        Returns:
            None.
        """
        # Snellius/HPC: never attempt pip upgrades at runtime
        if self.snellius_mode or (not self.update_package):
            return

        # Enforce once-per-day upgrade attempts for this package.
        if self.was_upgraded_today(package_name):
            logger.debug(f"{package_name} upgrade already attempted today. Skipping.")
            return

        try:
            # Use the current interpreter to ensure the correct environment is targeted.
            logger.info(f"Upgrading {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            logger.info(f"{package_name} upgraded successfully.")
            self.mark_as_upgraded(package_name)
        except subprocess.CalledProcessError as e:
            # Mark as attempted even on failure to avoid repeated failures in one day.
            logger.error(f"Failed to upgrade {package_name}: {e}")
            self.mark_as_upgraded(package_name)

    def rename_folder(self, old_name: str, new_name: str) -> None:
        """Rename a folder (thin wrapper around os.rename with logging).

        Args:
            old_name: Existing folder name/path.
            new_name: Target folder name/path.

        Returns:
            None.

        Notes:
            This method logs common filesystem errors but does not raise them.
        """
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

    def delete_folder(self, folder_path: str) -> bool:
        """Delete a folder recursively if it exists.

        Args:
            folder_path: Path to the directory to delete.

        Returns:
            True if the folder was deleted, otherwise False.

        Notes:
            - If the path does not exist, returns False and logs informationally.
            - If deletion fails, logs the exception and returns False.
        """
        # Only delete directories; avoid accidental deletion of files.
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Folder '{folder_path}' deleted successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to delete folder '{folder_path}': {e}")
                return False
        logger.info(f"Folder '{folder_path}' does not exist.")
        return False

    def delete_youtube_mod_videos(self, folders: list[str]) -> None:
        """Delete derived YouTube "*_mod.mp4" files from one or more folders.

        A "mod" video filename is expected to match:
            <11-char YouTube ID>_<anything>_mod.mp4

        Args:
            folders: List of directories to scan for matching files.

        Returns:
            None.
        """
        # YouTube IDs are typically 11 chars composed of [A-Za-z0-9_-].
        pattern = re.compile(r"^[A-Za-z0-9_-]{11}_.*_mod\.mp4$")
        for folder in folders:
            # Skip missing directories to keep cleanup idempotent and robust.
            if not os.path.exists(folder):
                logger.info(f"Skipping missing folder: {folder}")
                continue
            for fn in os.listdir(folder):
                if pattern.match(fn):
                    fp = os.path.join(folder, fn)
                    try:
                        os.remove(fp)
                        logger.info(f"Deleted: {fp}")
                    except Exception as e:
                        # Intentionally info-level to avoid noisy logs in cleanup phases.
                        logger.info(f"Failed to delete {fp}: {e}")
