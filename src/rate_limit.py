import asyncio
import time
from typing import Optional


class RateLimiter:
    """단순 토큰 버킷/슬리핑 기반 비동기 레이트 리미터.

    - calls_per_sec: 초당 허용 호출 수
    - burst: 버스트 허용 수(기본 calls_per_sec)
    """

    def __init__(self, calls_per_sec: float = 20.0, burst: Optional[int] = None):
        self.rate = max(0.1, float(calls_per_sec))
        self.tokens = burst if burst is not None else int(self.rate)
        self.capacity = self.tokens
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        delta = now - self.updated
        self.updated = now
        add = delta * self.rate
        self.tokens = min(self.capacity, self.tokens + add)

    async def acquire(self):
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # sleep until one token likely available
                need = 1.0 - self.tokens
                await asyncio.sleep(max(need / self.rate, 0.01))

