# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

for filename in os.listdir(common.get_configs('data')):
    if filename.endswith(".csv"):
        file_path = os.path.join(common.get_configs('data'), filename)
        try:
            df = pd.read_csv(file_path)

            if len(df.columns) == 6:
                os.remove(file_path)
                logger.info(f"Deleted file: {filename}.")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}.")
