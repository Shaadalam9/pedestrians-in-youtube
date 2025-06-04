# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import csv
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# file path
filename = "mapping.csv"

# columns to check
videos_col = 'videos'
channels_col = 'channel'
id_col = 'id'


def parse_list(raw):
    raw = raw.strip()[1:-1]  # remove brackets
    if not raw:
        return []
    return [item.strip() for item in raw.split(',')]


with open(filename, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row_num, row in enumerate(reader, start=1):
        try:
            videos = parse_list(row[videos_col])
            channels = parse_list(row[channels_col])
        except Exception as e:
            print(f'error parsing row {row_num}: {e}')
            continue

        len_v = len(videos)
        len_c = len(channels)

        if len_v != len_c:
            logger.info(f'mismatch at row {row[id_col]}: {len_v} videos vs {len_c} channels')