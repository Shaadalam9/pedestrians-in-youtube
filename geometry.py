import numpy as np
import common
from custom_logger import CustomLogger
from logmod import logs
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class Geometry():
    def __init__(self) -> None:
        pass
