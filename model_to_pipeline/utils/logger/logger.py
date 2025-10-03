# Copyright (c) 2025 SiMa.ai
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import io
import logging
import threading
import time
from loguru import logger
from contextlib import contextmanager
from rich.console import Console


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


spinner_running = False
spinner_message = ""
spinner_lock = threading.Lock()
green = "\033[32m"
reset = "\033[0m"
max_len = 20  # Maximum length of the step_name shown for cosmetic purpose

# Console that bypasses redirected sys.stdout
rich_console = Console(file=sys.__stdout__)


def start_spinner():
    def spinner():
        symbols = ["∙∙∙", "●∙∙", "∙●∙", "∙∙●", "∙∙∙"]
        idx = 0
        start_time = time.time()
        while spinner_running:
            with spinner_lock:
                msg = f"\r\033[K{green}{symbols[idx % len(symbols)]} {reset} {spinner_message} : {int(time.time() - start_time)} sec"
                sys.__stdout__.write(msg)
                sys.__stdout__.flush()
            idx += 1
            time.sleep(0.1)

    thread = threading.Thread(target=spinner)
    thread.start()
    return thread


@contextmanager
def step_logger(
    step_name: str,
    log_dir: str = "logs",
    verbose: bool = False,
    spinner_message_prefix: str = "Running Step: ",
):
    global spinner_running, spinner_message
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{step_name}_{time.time()}.log")

    logger.remove()
    logger.add(
        log_file, level="DEBUG", format="[{time}][{level}] - {message}", enqueue=True
    )

    if verbose:
        logger.add(
            sys.stdout,
            level="DEBUG",
            format="[{time}][{level}] - {message}",
            enqueue=True,
        )
        sys.stdout = Tee(sys.__stdout__)
        sys.stderr = Tee(sys.__stdout__, filter_func=lambda msg: False)
    else:
        log_file_stream = open(log_file, "a")
        sys.stdout = Tee(log_file_stream)
        sys.stderr = Tee(log_file_stream)

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.info(f"Step '{step_name}' started")
    spinner_message = f"{spinner_message_prefix} {green}{step_name:<{max_len}}{reset}"
    spinner_running = True
    start_time = time.time()
    spinner_thread = start_spinner()

    success = True
    try:
        yield
    except Exception as e:
        success = False
        logger.exception(f"Step '{step_name}' failed: {e}")
    finally:
        spinner_running = False
        spinner_thread.join()
        if success:
            sys.__stdout__.write(f"\r\033[K")
            rich_console.print(
                f"✅ Completed : [bold green]{step_name:<{max_len}}[/bold green]: {time.time() - start_time:.2f} sec"
            )
            logger.info(f"Step '{step_name}' completed")
        else:
            sys.__stdout__.write(f"\r\033[K")
            rich_console.print(
                f"❌ Failed :    [bold red]{step_name:<{max_len}}[/bold red]: {time.time() - start_time:.2f} sec"
            )
            logger.error(f"Step '{step_name}' failed")
