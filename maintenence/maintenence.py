import os
import json
import datetime
from custom_logger import CustomLogger
import subprocess
import sys
import re
import shutil
from config_utils import ConfigUtils

UPGRADE_LOG_FILE = "upgrade_log.json"
logger = CustomLogger(__name__)
config_utils = ConfigUtils()


class Maintenance:
    def __init__(self) -> None:
        # Snellius/HPC flag(s)
        self.snellius_mode = bool(config_utils._safe_get_config("snellius_mode", False))
        self.update_package = bool(config_utils._safe_get_config("update_package", False))
        # Snellius: never do runtime self-upgrades on compute nodes
        if self.snellius_mode:
            self.update_package = False

    def load_upgrade_log(self) -> dict:
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_upgrade_log(self, log_data: dict) -> None:
        with open(UPGRADE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f)

    def was_upgraded_today(self, package_name: str) -> bool:
        log_data = self.load_upgrade_log()
        today = datetime.date.today().isoformat()
        return log_data.get(package_name) == today

    def mark_as_upgraded(self, package_name: str) -> None:
        log_data = self.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        self.save_upgrade_log(log_data)

    def upgrade_package_if_needed(self, package_name: str) -> None:
        # Snellius/HPC: never attempt pip upgrades at runtime
        if self.snellius_mode or (not self.update_package):
            return

        if self.was_upgraded_today(package_name):
            logger.debug(f"{package_name} upgrade already attempted today. Skipping.")
            return

        try:
            logger.info(f"Upgrading {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            logger.info(f"{package_name} upgraded successfully.")
            self.mark_as_upgraded(package_name)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade {package_name}: {e}")
            self.mark_as_upgraded(package_name)

    def rename_folder(self, old_name: str, new_name: str) -> None:
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            logger.error(f"Error: Folder '{old_name}' not found.")
        except FileExistsError:
            logger.error(f"Error: Folder '{new_name}' already exists.")

    def delete_folder(self, folder_path: str) -> bool:
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
        pattern = re.compile(r"^[A-Za-z0-9_-]{11}_.*_mod\.mp4$")
        for folder in folders:
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
                        logger.info(f"Failed to delete {fp}: {e}")
