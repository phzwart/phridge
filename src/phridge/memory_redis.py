"""In-process Redis stand-in for Bridge(memory=True).

Implements the subset of redis-py used by RedisStore and Protocol.
Not a network server and not shared across processes.
"""

from __future__ import annotations

from typing import Any, Optional


class MemoryRedis:
    """Duck-typed redis.Redis for same-process jobs without redis-server."""

    def __init__(self) -> None:
        self._kv: dict[str, bytes] = {}
        self._lists: dict[str, list[bytes]] = {}
        self._streams: dict[str, list[tuple[bytes, dict[bytes, bytes]]]] = {}
        self._stream_seq = 0

    def set(self, name: str, value: Any, ex: Optional[int] = None) -> bool:
        del ex  # TTL is a no-op in memory mode
        if isinstance(value, str):
            value = value.encode("utf-8")
        elif not isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value)
        else:
            value = bytes(value)
        self._kv[name] = value
        return True

    def get(self, name: str) -> Optional[bytes]:
        return self._kv.get(name)

    def rpush(self, name: str, *values: Any) -> int:
        bucket = self._lists.setdefault(name, [])
        for value in values:
            if isinstance(value, str):
                value = value.encode("utf-8")
            else:
                value = bytes(value)
            bucket.append(value)
        return len(bucket)

    def expire(self, name: str, time: int) -> bool:
        del name, time
        return True

    def blpop(self, keys: Any, timeout: int = 0) -> Optional[tuple[str, bytes]]:
        del timeout  # jobs are processed before wait in memory mode
        if isinstance(keys, (str, bytes)):
            keys = [keys]
        for key in keys:
            name = key.decode("utf-8") if isinstance(key, bytes) else key
            bucket = self._lists.get(name)
            if bucket:
                return name, bucket.pop(0)
        return None

    def xadd(self, name: str, fields: dict[Any, Any], id: str = "*") -> bytes:
        del id
        self._stream_seq += 1
        entry_id = f"0-{self._stream_seq}".encode("utf-8")
        encoded: dict[bytes, bytes] = {}
        for key, value in fields.items():
            k = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            if isinstance(value, str):
                v = value.encode("utf-8")
            elif isinstance(value, bytes):
                v = value
            else:
                v = str(value).encode("utf-8")
            encoded[k] = v
        self._streams.setdefault(name, []).append((entry_id, encoded))
        return entry_id
