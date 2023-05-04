import os
import copy
import logging
from pathlib import Path
from typing import Any, Optional, Union
from termcolor import colored


def get_logger(
    filepath: Optional[Union[Path, str]] = None,
    detail_level: int = 1
) -> logging.Logger:
    logger = logging.getLogger("uphill")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(
        stream_handler(
            level=logging.INFO,
            detail_level=detail_level
        )
    )
    if filepath is not None:
        logger.addHandler(
            file_handler(
                filepath=filepath,
                level=logging.DEBUG,
                detail_level=detail_level
            )
        )
    return logger


###############
# Handler
###############
def stream_handler(level=logging.INFO, detail_level: int = 1):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(to_file=False, detail_level=detail_level))
    return handler

def file_handler(filepath, level=logging.INFO, detail_level: int = 1):
    handler = logging.FileHandler(filepath)
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(to_file=True, detail_level=detail_level))
    return handler


###############
# Formatter
###############
class ColoredFormatter(logging.Formatter):
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    RESET_SEQ = "\033[0m"

    def __init__(
        self,
        to_file: bool = False,
        detail_level: int = 1
    ):
        self.to_file = to_file
        self.colors = {
            'WARNING': self.YELLOW,
            'INFO': self.CYAN,
            'DEBUG': self.WHITE,
            'CRITICAL': self.YELLOW,
            'ERROR': self.RED
        }
        assert detail_level in [1,2,3], "`detail_level` shoul be one of 1, 2, 3, {detail_level} not allowed"
        if detail_level == 1:
            msg = "$BOLD%(levelname)8s$RESET: %(message)s"
        elif detail_level == 2:
            msg = "$BOLD%(levelname)8s:[%(filename)s@%(lineno)d:%(funcName)s]$RESET: %(message)s"
        else:
            msg = "%(asctime)s $BOLD%(levelname)8s:[%(filename)s@%(lineno)d:%(funcName)s]$RESET: %(message)s"
        if self.to_file:
            msg = msg.replace("$RESET", '').replace("$BOLD", '')
        else:
            msg = msg.replace("$RESET", self.RESET_SEQ).replace("$BOLD", self.BOLD_SEQ)
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        format_record = record
        levelname = record.levelname
        if not self.to_file and levelname in self.colors:
            levelname_color = self.COLOR_SEQ % (30 + self.colors[levelname]) + levelname
            format_record = copy.copy(record)
            format_record.levelname = levelname_color
        return logging.Formatter.format(self, format_record)


###############
# LoggerX
###############
class LoggerX(object):
    def __init__(self):
        self._initialized = False
        self.detail_level = int(os.environ.get("DETAIL_LEVEL", 1))

    def initialize(
        self, 
        logdir: Optional[Union[str, Path]] = None, 
        to_file: bool = False, 
        detail_level: Optional[int] = None
    ):
        '''
        Initialize LoggerX

        inputs
        - logdir - where to write logfiles
        - to_file - whether to write log to files
        - debug - write full details to log record
        '''
        if self._initialized:
            self.logger.warning(
                "loggerx has been initialized, cannot initialize again", 
                stacklevel=3
            )
            return
        self.detail_level = self.detail_level if detail_level is None else detail_level
        log_path = None
        if to_file:
            assert logdir is not None, "write to file but no place decleared, "\
                "set `to_file` to False or set `logdir` a valid path"
            logdir = Path(logdir) if isinstance(logdir, str) else logdir
            # confirm target log directory exists
            if not logdir.exists():
                logdir.mkdir(parents=True, exist_ok=True)
            from uphill.core.utils.timing import current_datetime
            log_path = logdir / f'uphill.{current_datetime()}.log'
        self.logger = get_logger(filepath=log_path, detail_level=self.detail_level)
        self._initialized = True

    def __del__(self):
        return

    def __check_msg(self, msg: Any=""):
        if msg is None or len(msg) == 0:
            self.logger.warning(
                "empty message for logger is not recommended, skip", 
                stacklevel=3
            )
            return False
        return True
    
    def __check_initialized(self):
        if not self._initialized:
            print(
                colored(
                    "using loggerx before `initialize` is not recommended, "\
                    "here initialize to console logger automatically",
                    "red"
                )
            )
            self.initialize()
        return
    
    #### logger functions
    def debug(self, msg: Any="", *args, **kwargs):
        self.__check_initialized()
        if not self.__check_msg(msg):
            return
        self.logger.debug(msg, stacklevel=2, *args, **kwargs)
    
    def info(self, msg: Any="", *args, **kwargs):
        self.__check_initialized()
        if not self.__check_msg(msg):
            return
        self.logger.info(msg, stacklevel=2, *args, **kwargs)
    
    def warning(self, msg: Any="", *args, **kwargs):
        self.__check_initialized()
        if not self.__check_msg(msg):
            return
        self.logger.warning(msg, stacklevel=2, *args, **kwargs)
    
    def error(self, msg: Any="", *args, **kwargs):
        self.__check_initialized()
        if not self.__check_msg(msg):
            return
        self.logger.error(msg, stacklevel=2, *args, **kwargs)
    
    def critical(self, msg: Any="", *args, **kwargs):
        self.__check_initialized()
        if not self.__check_msg(msg):
            return
        self.logger.critical(msg, stacklevel=2, *args, **kwargs)


global loggerx 
loggerx = LoggerX()
