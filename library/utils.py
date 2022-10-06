"""Utils module with display_info about function"""
import functools
import inspect
import os
import time

from . import parameter
from .logger import logger


def display_info(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
        if log_level == "TRACE":
            inner_frame = inspect.currentframe()
            called_frame = inspect.getouterframes(inner_frame)[1].frame

            try:
                module = inspect.getmodule(called_frame)
                info = inspect.getframeinfo(called_frame)
            finally:
                del inner_frame
                del called_frame

            bind_arguments = inspect.signature(func).bind(*args, **kwargs)
            bind_arguments.apply_defaults()
            arguments = dict(bind_arguments.arguments)

            info_filename = os.path.splitext(os.path.basename(info.filename))[0]

            module_info = [module.__package__]
            frame_info = [info_filename, info.function]
            module_frame_info = module_info + frame_info

            info_path = ".".join([info for info in module_frame_info if info])
            info_path += f":{info.lineno}"

            logger.debug(f"{info_path} Executing {func.__name__}({arguments})")

            start_time = time.monotonic()

        value = func(*args, **kwargs)

        if log_level == "TRACE":
            end_time = time.monotonic()
            elapsed_time = round((end_time - start_time) * 1000, 6)
            logger.debug(f"{info_path} Finished {func.__name__} in {elapsed_time} miliseconds -> {value}")

        return value

    return inner
