"""Optional worker diagnostics, separate from scientific result identities."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

from andrew_mlmdp.validation import _atomic_write_json, _json_value

_last_write = 0.0
_record = {}


def report_progress(*, throttle=False, **updates):
    path = os.environ.get("REGRESSION_PROGRESS")
    if not path:
        return
    global _last_write
    now = time.monotonic()
    if throttle and now - _last_write < 5:
        return
    stamp = datetime.now(UTC).isoformat()
    _record.setdefault("started_at", stamp)
    _record.update(updates, updated_at=stamp)
    _atomic_write_json(Path(path), _json_value(_record))
    _last_write = now


def report_fit_step(step):
    report_progress(
        throttle=True,
        phase="optimizer",
        iteration=step.evaluation,
        loss=step.loss,
        learning_rate=step.lr,
    )
