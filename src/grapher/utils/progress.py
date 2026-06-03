from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

ProgressCallback = Callable[[int], None]


def update_progress(progress_callback: ProgressCallback | None, amount: int) -> None:
    """Safely advance an optional integer progress callback."""
    if progress_callback is None:
        return
    amount = int(amount)
    if amount <= 0:
        return
    progress_callback(amount)


@contextmanager
def progress_bar(
    *,
    total: int,
    desc: str = "Progress",
    unit: str = "it",
    enabled: bool = True,
) -> Iterator[ProgressCallback]:
    """Yield a callback that advances a tqdm bar when available.

    The fallback prints coarse progress messages so long-running sampling remains
    observable even in minimal environments where tqdm is not installed.
    """
    total = max(0, int(total))
    completed = 0
    if not enabled or total == 0:
        yield lambda amount=1: None
        return

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover - only used without optional tqdm.
        tqdm = None

    if tqdm is None:
        print(f"{desc}: 0/{total} {unit}", flush=True)

        def _print_update(amount: int = 1) -> None:
            nonlocal completed
            amount = max(0, int(amount))
            if amount <= 0:
                return
            inc = min(amount, total - completed)
            if inc <= 0:
                return
            completed += inc
            print(f"{desc}: {completed}/{total} {unit}", flush=True)

        yield _print_update
        return

    bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)

    def _tqdm_update(amount: int = 1) -> None:
        nonlocal completed
        amount = max(0, int(amount))
        if amount <= 0:
            return
        inc = min(amount, total - completed)
        if inc <= 0:
            return
        completed += inc
        bar.update(inc)

    try:
        yield _tqdm_update
    finally:
        bar.close()
