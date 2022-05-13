"""
BEGIN_COPYRIGHT

IBM Confidential
OCO Source Materials

5727-I17
(C) Copyright IBM Corp. 2021 All Rights Reserved.
 
The source code for this program is not published or otherwise
divested of its trade secrets, irrespective of what has been
deposited with the U.S. Copyright Office.

END_COPYRIGHT
"""
import signal
import sys
from functools import partial
from queue import Queue
from types import FrameType
from typing import Optional

SENTINEL = None


def checkpoint_on_exit(rank: int):
    if rank < 1:
        queue = Queue()
        handler = partial(_trigger_checkpoint, queue=queue)
    else:
        queue = None
        handler = _exit

    signal.signal(signal.SIGTERM, handler)
    return queue


def _trigger_checkpoint(signo: int, frame: Optional[FrameType], queue: Queue):
    queue.put(SENTINEL)


def _exit(signo: int, frame: Optional[FrameType]):
    sys.exit(0)
