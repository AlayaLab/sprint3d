"""Static CUDA graph for the dense sparse-structure flow forward.

The structure student calls the same dense transformer four times per request
with fixed shapes (batch 1, ``x`` [1, 8, 16, 16, 16] fp32, ``t`` [1] fp32,
``cond = {'global': [1, Ng, 1024], 'proj': [1, 4096, P]}``). :class:`GraphRunner`
captures that forward once per process and replays it, which removes the
per-kernel launch overhead. Two parts of the eager forward are handled
explicitly so that the replay is bit-identical to the eager forward:

* the timestep embedding builds its frequency table on the CPU on every call
  (an H2D copy that cannot be captured); during warm-up and capture it is
  replaced by the same expression with the table computed once on the device;
* the per-block context projections (``to_kv(global)`` of the cross attention
  and ``proj_linear(proj)`` of the projection attention, LoRA-wrapped for the
  student) depend only on the conditioning, so they are computed once per
  request by the leaf modules themselves into static buffers that the graph
  reads (``use_kv=True``).

After capture, one eager forward on the capture inputs is compared with a
replay (after scribbling NaNs into freshly allocated memory) with
``torch.equal``; a mismatch raises. Calls that do not match the captured
contract (other shapes, grad enabled, autocast) run eagerly and are counted.
"""
from __future__ import annotations

import contextlib
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

T_EMBED_MAX_PERIOD = 10000


def _version(t: torch.Tensor) -> Optional[int]:
    try:
        return int(t._version)
    except Exception:
        return None


def manual_cast_for(model: nn.Module) -> Callable:
    """The ``manual_cast`` function the model's own module uses (resolved at call time)."""
    module = sys.modules.get(type(model).__module__)
    fn = getattr(module, "manual_cast", None) if module is not None else None
    if fn is None:
        from pixal3d.modules.utils import manual_cast as fn  # type: ignore
    return fn


class Leaf:
    __slots__ = ("module", "kind", "index")

    def __init__(self, module: nn.Module, kind: str, index: int):
        self.module, self.kind, self.index = module, kind, index

    @property
    def key(self) -> Tuple[str, int]:
        return (self.kind, self.index)


def find_context_leaves(model: nn.Module) -> List[Leaf]:
    """Per block: ``cross_attn.cross_attn_block.to_kv`` (kind ``kv``) and
    ``cross_attn.proj_linear`` (kind ``proj``) of the projection attention."""
    blocks = model._modules.get("blocks")
    if blocks is None:
        raise TypeError(f"{type(model).__name__} has no registered 'blocks' ModuleList")
    leaves: List[Leaf] = []
    for index, block in enumerate(blocks):
        cross = getattr(block, "cross_attn", None)
        if cross is None:
            continue
        inner = getattr(cross, "cross_attn_block", None)
        to_kv = getattr(inner if inner is not None else cross, "to_kv", None)
        if isinstance(to_kv, nn.Module):
            leaves.append(Leaf(to_kv, "kv", index))
        proj = getattr(cross, "proj_linear", None)
        if isinstance(proj, nn.Module):
            leaves.append(Leaf(proj, "proj", index))
    return leaves


@contextlib.contextmanager
def static_leaves(leaves: List[Leaf], buffers: Dict[Tuple[str, int], torch.Tensor]):
    """Make every leaf return its static buffer (used during capture)."""
    installed = []
    try:
        for leaf in leaves:
            module = leaf.module
            had = "forward" in module.__dict__
            old = module.__dict__.get("forward")
            buffer = buffers[leaf.key]

            def forward(x, *args, __buffer=buffer, **kwargs):
                return __buffer

            module.forward = forward
            installed.append((module, had, old))
        yield
    finally:
        for module, had, old in reversed(installed):
            if had:
                module.forward = old
            else:
                del module.__dict__["forward"]


class GraphRunner:
    """Replaces ``model.forward`` while installed; graphs persist for the process."""

    def __init__(self, model: nn.Module, use_kv: bool = True, strict: bool = True, verify: bool = True,
                 max_graphs: int = 4):
        if not all(hasattr(model, a) for a in ("resolution", "in_channels", "t_embedder", "dtype")):
            raise TypeError("GraphRunner expects a dense SparseStructureFlowModel")
        self.model = model
        self.res = int(model.resolution)
        self.in_channels = int(model.in_channels)
        self.leaves = find_context_leaves(model) if use_kv else []
        self.use_kv = bool(use_kv and self.leaves)
        self.strict = bool(strict)
        self.verify = bool(verify)
        self.max_graphs = int(max_graphs)
        self.graphs: Dict[Any, Dict[str, Any]] = {}
        self.stats: Dict[str, Any] = {"captures": [], "replays": 0, "fallbacks": 0, "kv_fills": 0,
                                      "kv_reuses": 0, "disabled": False}
        self.orig_forward: Optional[Callable] = None
        self.leaf_originals: Dict[Tuple[str, int], Callable] = {}
        self.request: Optional[Dict[str, Any]] = None
        # Device-resident frequency table read by every replay: it must outlive the graphs.
        self._freqs: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}

    # -- contract -------------------------------------------------------------------------
    def _applicable(self, x, t, cond):
        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            return False, None, None
        if tuple(x.shape) != (1, self.in_channels, self.res, self.res, self.res):
            return False, None, None
        if not isinstance(t, torch.Tensor) or t.numel() != 1 or not t.is_cuda:
            return False, None, None
        if isinstance(cond, dict):
            if set(cond.keys()) != {"global", "proj"}:
                return False, None, None
            g, p = cond["global"], cond["proj"]
        elif isinstance(cond, (tuple, list)) and len(cond) == 2:
            g, p = cond
        else:
            return False, None, None
        if not (isinstance(g, torch.Tensor) and isinstance(p, torch.Tensor) and g.is_cuda and p.is_cuda):
            return False, None, None
        if g.dim() != 3 or int(g.shape[0]) != 1 or p.dim() != 3 or tuple(p.shape[:2]) != (1, self.res ** 3):
            return False, None, None
        if torch.is_grad_enabled() or torch.is_autocast_enabled():
            return False, None, None
        return True, g, p

    def _signature(self, x, t, g, p):
        try:
            from pixal3d.modules.attention import config as attn_config  # type: ignore
            backend = attn_config.BACKEND
        except Exception:
            backend = None
        return (tuple(x.shape), str(x.dtype), tuple(t.shape), str(t.dtype), tuple(g.shape), str(g.dtype),
                tuple(p.shape), str(p.dtype), x.device.index, backend, self.use_kv)

    # -- exact helpers --------------------------------------------------------------------
    def _freqs_for(self, device) -> torch.Tensor:
        device = torch.device(device)
        key = (device.type, device.index)
        freqs = self._freqs.get(key)
        if freqs is None:
            half = int(self.model.t_embedder.frequency_embedding_size) // 2
            freqs = torch.exp(
                -np.log(T_EMBED_MAX_PERIOD) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=device)
            self._freqs[key] = freqs
        return freqs

    @contextlib.contextmanager
    def _timestep_embedding_on_device(self, device):
        te = self.model.t_embedder
        dim = int(te.frequency_embedding_size)
        freqs = self._freqs_for(device)

        def timestep_embedding(t, dim_, max_period=T_EMBED_MAX_PERIOD):
            if int(dim_) != dim or max_period != T_EMBED_MAX_PERIOD:
                raise RuntimeError("timestep embedding contract changed")
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

        had = "timestep_embedding" in te.__dict__
        old = te.__dict__.get("timestep_embedding")
        te.__dict__["timestep_embedding"] = timestep_embedding
        try:
            yield
        finally:
            if had:
                te.__dict__["timestep_embedding"] = old
            else:
                del te.__dict__["timestep_embedding"]

    def _compute_kv(self, g, p) -> Dict[Tuple[str, int], torch.Tensor]:
        cast = manual_cast_for(self.model)
        gc = cast(g, self.model.dtype)
        pc = cast(p, self.model.dtype)
        out = {}
        with torch.no_grad():
            for leaf in self.leaves:
                out[leaf.key] = self.leaf_originals[leaf.key](gc if leaf.kind == "kv" else pc)
        return out

    @staticmethod
    def _kv_key(g, p):
        return (id(g), id(p), _version(g), _version(p))

    def _ensure_kv(self, entry, g, p) -> None:
        key = self._kv_key(g, p)
        pin = entry.get("kv_pin")
        if entry.get("kv_key") == key and pin is not None and pin[0] is g and pin[1] is p:
            self.stats["kv_reuses"] += 1
            self.request["kv_reuses"] += 1
            return
        fresh = self._compute_kv(g, p)
        for k, buffer in entry["static_kv"].items():
            buffer.copy_(fresh[k])
        entry["kv_key"] = key
        entry["kv_pin"] = (g, p)
        self.stats["kv_fills"] += 1
        self.request["kv_fills"] += 1

    # -- capture / replay -----------------------------------------------------------------
    def _capture(self, sig, x, t, g, p) -> Dict[str, Any]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        static = {"x": x.detach().clone(), "t": t.detach().clone(), "g": g.detach().clone(), "p": p.detach().clone()}
        cond = {"global": static["g"], "proj": static["p"]}
        static_kv = self._compute_kv(g, p) if self.use_kv else None
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            stack.enter_context(self._timestep_embedding_on_device(x.device))
            if static_kv is not None:
                stack.enter_context(static_leaves(self.leaves, static_kv))
            side = torch.cuda.Stream(device=x.device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.orig_forward(static["x"], static["t"], cond)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side, capture_error_mode="thread_local"):
                out = self.orig_forward(static["x"], static["t"], cond)
        torch.cuda.synchronize()
        static["freqs"] = self._freqs_for(x.device)
        entry = {"graph": graph, "static": static, "out": out, "static_kv": static_kv, "kv_key": None,
                 "kv_pin": None, "signature": sig}
        if static_kv is not None:
            entry["kv_key"] = self._kv_key(g, p)
            entry["kv_pin"] = (g, p)
        record = {"wall_s": None, "verified": None}
        if self.verify:
            with torch.no_grad():
                reference = self.orig_forward(x, t, {"global": g, "proj": p}).detach().clone()
                torch.cuda.synchronize()
                junk = [torch.empty(n, device=x.device, dtype=torch.float32)
                        for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
                        for _ in range(16)]
                for j in junk:
                    j.fill_(float("nan"))
                static["x"].copy_(x)
                static["t"].copy_(t)
                static["g"].copy_(g)
                static["p"].copy_(p)
                graph.replay()
                torch.cuda.synchronize()
                replayed = out.detach().clone()
                del junk
            equal = bool(torch.equal(replayed, reference))
            record["verified"] = equal
            record["verify_max_abs_diff"] = float((replayed.float() - reference.float()).abs().max())
            if not equal:
                if self.strict:
                    raise RuntimeError("structure CUDA graph replay differs from the eager forward "
                                       f"(max_abs_diff={record['verify_max_abs_diff']})")
                self.stats["disabled"] = True
        record["wall_s"] = round(time.perf_counter() - started, 4)
        self.stats["captures"].append(record)
        return entry

    def _fallback(self, x, t, cond, reason: str):
        self.stats["fallbacks"] += 1
        if self.request is not None:
            self.request["fallbacks"] += 1
            self.request.setdefault("fallback_reasons", []).append(reason)
        return self.orig_forward(x, t, cond)

    def __call__(self, x, t, cond, *args, **kwargs):
        if args or kwargs:
            return self.orig_forward(x, t, cond, *args, **kwargs)
        if self.request is None or self.stats["disabled"]:
            return self._fallback(x, t, cond, "inactive")
        ok, g, p = self._applicable(x, t, cond)
        if not ok:
            return self._fallback(x, t, cond, "off_contract")
        sig = self._signature(x, t, g, p)
        entry = self.graphs.get(sig)
        if entry is None:
            if len(self.graphs) >= self.max_graphs:
                return self._fallback(x, t, cond, "max_graphs")
            entry = self._capture(sig, x, t, g, p)
            if self.stats["disabled"]:
                return self._fallback(x, t, cond, "capture_verification_failed")
            self.graphs[sig] = entry
            self.request["graph_captured"] = True
            self.request["capture_s"] = self.stats["captures"][-1]["wall_s"]
            self.request["capture_verified"] = self.stats["captures"][-1].get("verified")
            if self.use_kv:
                self.stats["kv_fills"] += 1
                self.request["kv_fills"] += 1
        elif self.use_kv:
            self._ensure_kv(entry, g, p)
        static = entry["static"]
        static["x"].copy_(x)
        static["t"].copy_(t)
        static["g"].copy_(g)
        static["p"].copy_(p)
        entry["graph"].replay()
        self.stats["replays"] += 1
        self.request["replays"] += 1
        return entry["out"].clone()

    def begin_request(self) -> Dict[str, Any]:
        self.request = {"graph_captured": False, "capture_s": None, "capture_verified": None, "replays": 0,
                        "fallbacks": 0, "kv_fills": 0, "kv_reuses": 0}
        self.leaf_originals = {leaf.key: leaf.module.forward for leaf in self.leaves}
        return self.request

    def end_request(self) -> Dict[str, Any]:
        request = self.request or {}
        request["graphs_resident"] = len(self.graphs)
        for entry in self.graphs.values():  # never keep a request's conditioning alive
            entry["kv_pin"] = None
            entry["kv_key"] = None
        self.request = None
        return request


_RUNNERS: Dict[int, Tuple[nn.Module, GraphRunner]] = {}


def runner_for(model: nn.Module, use_kv: bool = True, strict: bool = True, verify: bool = True) -> GraphRunner:
    hit = _RUNNERS.get(id(model))
    if hit is not None and hit[0] is model:
        return hit[1]
    runner = GraphRunner(model, use_kv=use_kv, strict=strict, verify=verify)
    _RUNNERS[id(model)] = (model, runner)
    return runner


@contextlib.contextmanager
def graphed_forward(model: nn.Module, use_kv: bool = True, strict: bool = True, verify: bool = True):
    """Shadow ``model.forward`` with its process-resident :class:`GraphRunner` inside the block.
    Yields the per-request record (replays, fallbacks, capture info), filled at exit."""
    runner = runner_for(model, use_kv=use_kv, strict=strict, verify=verify)
    had = "forward" in model.__dict__
    old = model.__dict__.get("forward")
    runner.orig_forward = model.forward
    record = runner.begin_request()
    model.forward = runner
    try:
        yield record
    finally:
        if had:
            model.forward = old
        else:
            del model.__dict__["forward"]
        record.update(runner.end_request())
