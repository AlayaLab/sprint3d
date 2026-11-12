"""CUDA stream and host-thread helpers used to overlap texture and mesh work.

* :class:`SideStream` runs a callable on a dedicated CUDA stream and hands the
  result back to the caller's stream through an event (the caller's stream
  waits for the event; the result tensors are ``record_stream``-ed so the
  caching allocator does not recycle them early). The texture flow is sampled
  this way, so its kernels can run concurrently with the mesh work the worker
  thread queues on the default stream.
* :class:`RequestExecutor` is a small per-request thread pool for work whose
  result is only needed later (PNG encoding of the baked textures, allocator
  cache releases). Everything submitted is joined by :meth:`RequestExecutor.join`.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict, List, Optional


class SideStream:
    """One cached side stream per device; ``run(fn, *args)`` executes ``fn`` on it."""

    _streams: Dict[Any, Any] = {}

    def __init__(self, device=None):
        import torch

        key = torch.device(device or "cuda").index
        stream = self._streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream()
            self._streams[key] = stream
        self.stream = stream

    def run(self, fn: Callable, *args, **kwargs):
        import torch

        with torch.cuda.stream(self.stream):
            out = fn(*args, **kwargs)
        event = torch.cuda.Event()
        event.record(self.stream)
        default_stream = torch.cuda.current_stream()
        default_stream.wait_event(event)
        for attr in ("feats", "coords"):
            tensor = getattr(out, attr, None)
            if tensor is not None and hasattr(tensor, "record_stream"):
                tensor.record_stream(default_stream)
        return out


class RequestExecutor:
    """Thread pool scoped to one request. ``defer_empty_cache`` turns allocator
    cache releases into background calls; :meth:`join` waits for everything."""

    def __init__(self, max_workers: int = 2, defer_empty_cache: bool = False):
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures: List[concurrent.futures.Future] = []
        self.defer_empty_cache = defer_empty_cache

    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        future = self._pool.submit(fn, *args, **kwargs)
        self._futures.append(future)
        return future

    def empty_cache(self) -> None:
        import torch

        if self.defer_empty_cache:
            self.submit(torch.cuda.empty_cache)
        else:
            torch.cuda.empty_cache()

    def join(self) -> None:
        try:
            for future in self._futures:
                future.result()
        finally:
            self._pool.shutdown(wait=True)
            self._futures.clear()


def run_in_thread(fn: Callable[[], Any], name: str):
    """Start ``fn`` on a new thread (under ``torch.no_grad``); returns ``(thread, holder)``.

    ``holder`` receives ``value`` or ``error``/``traceback`` and ``wall_s``."""
    import threading
    import time
    import traceback

    holder: Dict[str, Any] = {}

    def target():
        import torch

        started = time.perf_counter()
        try:
            with torch.no_grad():
                holder["value"] = fn()
        except BaseException as error:  # re-raised by the caller after join()
            holder["error"] = error
            holder["traceback"] = traceback.format_exc()
        finally:
            holder["wall_s"] = round(time.perf_counter() - started, 6)

    thread = threading.Thread(target=target, name=name)
    thread.start()
    return thread, holder


def join_thread(thread, holder: Dict[str, Any]) -> Optional[Any]:
    thread.join()
    if "error" in holder:
        raise holder["error"]
    return holder.get("value")
