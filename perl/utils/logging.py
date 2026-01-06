# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE from mika: torchtitan logger is used here

import logging
import os
import sys


logger = logging.getLogger()


def init_logger(rank: int = 0) -> None:
    """Initialize logger with rank information.
    
    Args:
        rank: The process rank. Only rank 0 logs INFO level, others log WARNING+.
    """
    # Only main process logs INFO, others only log WARNING and above
    log_level = logging.INFO if rank == 0 else logging.WARNING
    
    logger.setLevel(log_level)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        f"[PeRL][Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


_logged: set[str] = set()


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)