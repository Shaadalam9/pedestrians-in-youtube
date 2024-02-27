"""Contain function to display or store logging messages."""
import logging
import sys
import os
import datetime as dt
from typing import Union, Optional
import common


def logs(
        show_level: Optional[Union[int, str]] = None,
        save_level: Optional[Union[int, str]] = None,
        program_name: Optional[str] = None,
        path: Optional[str] = None,
        threads: bool = False,
        multiproc: bool = False,
        show_color: bool = True
) -> None:
    """
    Initialize the logger.

    Parameters
    ----------
    show_level : int or {'debug', 'info', warning', 'error', 'exception'},
        optional. Pass a log level to display logs of that level and above.
    save_level : int or {'debug', 'info', warning', 'error', 'exception'},
        optional. Pass a log level to store logs of that level and above to
        a file on disk. If no path is passed the default `_cache` folder will
        be used.
    program_name : str, optional
        When saving logs to disk this string will be used in the filename.
    path : str, optional
        Path where log files will be stored. Defaults to the `_cache` folder.
    threads : bool, default False
        Add the thread name to log messages. Useful when using threading.
    multiproc : bool, default False
        Add the process name to log messages. Useful when using
        multiprocessing.
    show_color : bool, default True
        If you have the coloredlogs package installed the messages will be
        colored.

    Note that log levels can be one of the listed strings or an integer between
    1 and 100. If you want to get all possible log messages, use a log level of
    1.
    """
    logger_root = logging.getLogger()
    fmt_items = ('%(asctime)s',
                 '%(levelname)-8s',
                 '%(threadName)s' if threads else None,
                 '%(processName)s' if multiproc else None,
                 '%(name)s',
                 '%(message)s')
    fmt = ' - '.join((item for item in fmt_items if item is not None))
    formatter = logging.Formatter(fmt)
    logging.addLevelName(5, "VERBOSE")
    logger_root.setLevel(5)

    if show_level and show_color:
        try:
            import coloredlogs
        except ImportError:
            show_color = False
        else:
            coloredlogs.install(fmt=fmt,
                                level=_convert_logging_level(show_level),
                                stream=sys.stdout)
    if show_level and not show_color:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(_convert_logging_level(show_level))
        stream_handler.setFormatter(formatter)
        logger_root.addHandler(stream_handler)
    if save_level:
        if program_name is None:
            program_name = 'noname'
        date_str = dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = 'log_{}_{}.log'.format(program_name, date_str)
        if path is None:
            path = common.log_dir
        file_handler = logging.FileHandler(filename=os.path.join(path,
                                                                 log_filename))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_convert_logging_level(save_level))
        logger_root.addHandler(file_handler)
    _logging_level_threshold()


def _logging_level_threshold():
    """
    Set the level threshold for a couple of internal and external modules.
    """
    for mod_name in ['requests', 'matplotlib', 'numexpr.utils',
                     'urllib3.connectionpool', 'PIL.TiffImagePlugin']:
        logging.getLogger(mod_name).setLevel(logging.WARNING)


def _convert_logging_level(level: Union[int, str]) -> int:
    """Convert the user-provided level to a logging level integer."""
    if isinstance(level, int):
        assert 0 < level <= 50, 'Logging level must be between 0 and 50.'
        return level
    if not hasattr(logging, level.upper()):
        raise ValueError('Unknown logging level name: {}.'.format(level))
    return getattr(logging, level.upper())
