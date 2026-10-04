"""Write JSON checkpoints without truncating the last successful snapshot."""

import errno
import json
import os
from pathlib import Path
import tempfile
import time


def write_json_atomic(path: str | Path, value) -> None:
    """Replace a checkpoint atomically, retrying temporary Windows file errors."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, ensure_ascii=False, indent=2)
    retryable = {errno.EACCES, errno.EPERM, errno.EBUSY, errno.EAGAIN, errno.EINVAL}
    for attempt in range(5):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent,
                prefix=f".{target.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            return
        except OSError as error:
            if error.errno not in retryable or attempt == 4:
                raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        time.sleep(0.1 * 2 ** attempt)
