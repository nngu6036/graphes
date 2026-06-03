from __future__ import annotations

import os
import platform
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any


def _bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def hardware_summary() -> dict[str, Any]:
    torch = _try_import_torch()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpus = []
    if cuda_available:
        try:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append({
                    "index": idx,
                    "name": props.name,
                    "total_memory_mib": _bytes_to_mib(int(props.total_memory)),
                })
        except Exception:
            gpus = []
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cuda_available": cuda_available,
        "gpus": gpus,
        "torch": getattr(torch, "__version__", None) if torch is not None else None,
    }


def hardware_label(summary: dict[str, Any] | None = None) -> str:
    summary = summary or hardware_summary()
    gpus = summary.get("gpus") or []
    if gpus:
        names = ", ".join(str(g.get("name")) for g in gpus if g.get("name"))
        return names or "CUDA GPU"
    cpu = summary.get("processor") or "CPU"
    count = summary.get("cpu_count_logical")
    return f"{cpu} ({count} logical CPUs)" if count else str(cpu)


@dataclass
class PeakMemoryMonitor:
    interval_seconds: float = 0.2
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _process: Any = field(default=None, init=False)
    _torch: Any = field(default=None, init=False)
    peak_rss_bytes: int | None = field(default=None, init=False)
    peak_python_traced_bytes: int | None = field(default=None, init=False)
    peak_cuda_allocated_bytes: int | None = field(default=None, init=False)
    peak_cuda_reserved_bytes: int | None = field(default=None, init=False)
    _started_tracemalloc: bool = field(default=False, init=False)

    def __enter__(self):
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self.peak_rss_bytes = int(self._process.memory_info().rss)
        except Exception:
            self._process = None

        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True

        self._torch = _try_import_torch()
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                self._torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        self._thread = threading.Thread(target=self._sample_loop, name="peak-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 5))
        self._sample_once()
        try:
            _, peak = tracemalloc.get_traced_memory()
            self.peak_python_traced_bytes = int(peak)
        except Exception:
            pass
        if self._started_tracemalloc:
            tracemalloc.stop()
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                self.peak_cuda_allocated_bytes = int(self._torch.cuda.max_memory_allocated())
                self.peak_cuda_reserved_bytes = int(self._torch.cuda.max_memory_reserved())
            except Exception:
                pass

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        if self._process is None:
            return
        try:
            rss = int(self._process.memory_info().rss)
            self.peak_rss_bytes = max(int(self.peak_rss_bytes or 0), rss)
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        candidates = [self.peak_rss_bytes, self.peak_cuda_reserved_bytes, self.peak_python_traced_bytes]
        peak = max([int(x) for x in candidates if x is not None], default=None)
        return {
            "peak_memory_bytes": peak,
            "peak_memory_mib": _bytes_to_mib(peak),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": _bytes_to_mib(self.peak_rss_bytes),
            "peak_python_traced_bytes": self.peak_python_traced_bytes,
            "peak_python_traced_mib": _bytes_to_mib(self.peak_python_traced_bytes),
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_allocated_mib": _bytes_to_mib(self.peak_cuda_allocated_bytes),
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "peak_cuda_reserved_mib": _bytes_to_mib(self.peak_cuda_reserved_bytes),
            "memory_note": (
                "peak_rss is sampled process resident memory when psutil is installed; "
                "peak_python_traced is tracemalloc memory; CUDA fields use torch.cuda peak counters."
            ),
        }


def compute_report(*, operation: str, runtime_seconds: float, num_graphs: int | None = None, memory: dict[str, Any] | None = None) -> dict[str, Any]:
    hardware = hardware_summary()
    report: dict[str, Any] = {
        "operation": operation,
        "hardware": hardware,
        "hardware_label": hardware_label(hardware),
        "runtime_seconds": float(runtime_seconds),
        "runtime_minutes": float(runtime_seconds) / 60.0,
    }
    if num_graphs is not None:
        n = max(int(num_graphs), 1)
        report["num_graphs"] = int(num_graphs)
        report["seconds_per_graph"] = float(runtime_seconds) / n
        report["seconds_per_128_graphs"] = float(runtime_seconds) * 128.0 / n
    if memory:
        report["memory"] = memory
        report["peak_memory_mib"] = memory.get("peak_memory_mib")
    return report
