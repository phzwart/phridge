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

    def delete(self, *names: Any) -> int:
        count = 0
        for name in names:
            n = name.decode("utf-8") if isinstance(name, bytes) else str(name)
            if n in self._kv:
                del self._kv[n]
                count += 1
            if n in self._lists:
                del self._lists[n]
                count += 1
            if n in self._streams:
                del self._streams[n]
                count += 1
        return count

    def xlen(self, name: str) -> int:
        return len(self._streams.get(name, []))

    def xgroup_create(
        self,
        name: str,
        groupname: str,
        id: str = "$",
        mkstream: bool = False,
    ) -> bool:
        del groupname, id
        if mkstream and name not in self._streams:
            self._streams[name] = []
        return True

    def xadd(
        self,
        name: str,
        fields: dict[Any, Any],
        id: str = "*",
        maxlen: Optional[int] = None,
        approximate: bool = False,
    ) -> bytes:
        del id, approximate
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
        stream = self._streams.setdefault(name, [])
        stream.append((entry_id, encoded))
        if maxlen is not None and len(stream) > maxlen:
            del stream[:-maxlen]
        return entry_id
