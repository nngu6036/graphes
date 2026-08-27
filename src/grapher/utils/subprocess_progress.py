"""Terminal progress reporting for subprocesses whose output is persisted to a log.

The baseline wrappers keep complete upstream logs on disk for reproducibility.
This helper optionally mirrors newly appended log bytes to stderr and emits a
bounded heartbeat when the subprocess is quiet.  It never changes stdout, so
scripts may continue to reserve stdout for machine-readable JSON summaries.
"""

from __future__ import annotations

import codecs
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


def _format_elapsed(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _last_log_fragment(path: Path, *, limit: int = 240) -> str:
    """Return one compact printable fragment from the end of ``path``."""

    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(size - 8192, 0))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    fragments = [
        fragment.strip()
        for fragment in text.replace("\r", "\n").splitlines()
        if fragment.strip()
    ]
    if not fragments:
        return ""
    compact = " ".join(fragments[-2:])
    return compact[-limit:]


class SubprocessLogReporter:
    """Mirror a growing subprocess log and report liveness to stderr."""

    def __init__(
        self,
        *,
        label: str,
        log_path: Path,
        enabled: bool = False,
        stream_output: bool = False,
        interval_seconds: float = 30.0,
        output: TextIO | None = None,
        prefix: str = "GraphER/DeFoG",
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("progress interval_seconds must be positive.")
        self.label = str(label)
        self.log_path = Path(log_path)
        self.enabled = bool(enabled)
        self.stream_output = bool(stream_output)
        self.interval_seconds = float(interval_seconds)
        self.output = output or sys.stderr
        normalized_prefix = str(prefix).strip()
        if not normalized_prefix or any(ch in normalized_prefix for ch in "\x00\r\n[]"):
            raise ValueError("progress prefix must be a non-empty bracket-safe string.")
        self.prefix = normalized_prefix
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._offset = 0

    def _write_line(self, message: str) -> None:
        print(message, file=self.output, flush=True)

    def start(self, *, start_offset: int | None = None) -> None:
        if not self.enabled:
            return
        self._started = time.monotonic()
        if start_offset is None:
            try:
                start_offset = self.log_path.stat().st_size
            except OSError:
                start_offset = 0
        self._offset = max(int(start_offset), 0)
        mode = "streaming output" if self.stream_output else "heartbeat only"
        self._write_line(
            f"[{self.prefix}] {self.label} started ({mode}); "
            f"log={self.log_path.resolve()}"
        )
        self._thread = threading.Thread(
            target=self._follow,
            name=f"grapher-progress-{self.label}",
            daemon=True,
        )
        self._thread.start()

    def _follow(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        next_heartbeat = time.monotonic() + self.interval_seconds
        while not self._stop.wait(0.20):
            self._drain(decoder)
            now = time.monotonic()
            if now >= next_heartbeat:
                self._heartbeat(now)
                next_heartbeat = now + self.interval_seconds
        self._drain(decoder)
        remainder = decoder.decode(b"", final=True)
        if remainder and self.stream_output:
            self.output.write(remainder)
            self.output.flush()

    def _drain(self, decoder: codecs.IncrementalDecoder) -> None:
        if not self.stream_output or not self.log_path.is_file():
            return
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(self._offset)
                data = handle.read()
                self._offset = handle.tell()
        except OSError:
            return
        if not data:
            return
        text = decoder.decode(data, final=False)
        if text:
            self.output.write(text)
            self.output.flush()

    def _heartbeat(self, now: float) -> None:
        try:
            size = self.log_path.stat().st_size
        except OSError:
            size = 0
        fragment = _last_log_fragment(self.log_path)
        suffix = f"; last_output={fragment!r}" if fragment else ""
        self._write_line(
            f"[{self.prefix}] {self.label} still running; "
            f"elapsed={_format_elapsed(now - self._started)}; "
            f"log_bytes={size}{suffix}"
        )

    def stop(self, *, status: str) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            # The follower polls every 0.2 seconds, so shutdown should not
            # inherit a potentially very large heartbeat interval.
            self._thread.join(timeout=2.0)
        elapsed = _format_elapsed(time.monotonic() - self._started)
        self._write_line(
            f"[{self.prefix}] {self.label} {status}; elapsed={elapsed}; "
            f"log={self.log_path.resolve()}"
        )
