"""Production `Clock` (architecture §5): wall time in UTC and a monotonic millisecond counter."""

from __future__ import annotations

import time
from datetime import UTC, datetime


class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def monotonic_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000
