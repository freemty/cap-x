"""Host-local robosuite overrides installed by ``exp/exp01b/run.py``.

The vendored robosuite ignores ``macros_private.py``.  Shared machines can
already contain another user's mode-0600 ``/tmp/robosuite.log``; disabling the
optional file logger prevents that unrelated file from making imports fail.
"""

from robosuite.macros import *  # noqa: F403
import robosuite.macros as _macros


FILE_LOGGING_LEVEL = None
_macros.FILE_LOGGING_LEVEL = None
