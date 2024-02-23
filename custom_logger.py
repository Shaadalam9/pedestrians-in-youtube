import logging


class CustomLogger:
    """Logger that handles string formatting.

    Contains a logging.Logger object. Copies the various logging.Logger
    methods. The purpose is to accept str.format() style formatting.
    With this custom class, messages may contain '{}' where the next
    arguments will be placed. Doesn't work with keyword arguments for the
    formatting.

    Examples
    --------
    >>> CustomLogger(__name__)
    <gazes.CustomLogger object at 0x00000AB32390>
    """

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(5)

    def debug(self, msg, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        if self.logger.isEnabledFor(level):
            msg = msg.format(*args)
            self.logger._log(level, msg, args=(), **kwargs)
