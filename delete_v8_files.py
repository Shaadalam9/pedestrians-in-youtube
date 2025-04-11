# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

folders = common.get_configs('data')

for filename in os.listdir(folders[-1]):  # use last folder
    if filename.endswith(".csv"):
        file_path = os.path.join(folders[-1], filename)
        try:
            df = pd.read_csv(file_path)

            if len(df.columns) == 6:
                os.remove(file_path)
                logger.info(f"Deleted file: {filename}.")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}.")
