"""Texture latent sampling.

``texture_student`` on: two Euler steps (``rescale_t`` 3, no guidance) of the
texture flow with the student's LoRA folded into its bf16 weights, through
Pixal3D's own texture sampler. Off: the 12-step teacher. The texture noise is
drawn from the global CPU generator seeded with ``4242 + seed`` inside a
forked RNG scope.

Levers acting on the sampling call (all leave the sampling arithmetic alone):

* ``fa3_texture_flow``: FlashAttention-3 for the texture flow's attention.
* ``texture_kv_reuse``: request-scoped caches inside the flow (identical values):
  the conditioning cast to bf16 is memoised, each block's cross-attention K/V
  and projected image features are computed once and reused in later steps,
  ``cu_seqlens`` of the varlen attention are cached (FlashAttention-2 path), and
  batch-1 broadcasts in the modulation use a zero-copy view instead of a gather.
* ``texture_projection_reuse``: the per-block context projections are also
  cached at the leaf modules for one sampling call, with a forward pre-hook that
  memoises the conditioning cast.
* ``texture_side_stream``: the sampling call runs on a separate CUDA stream (see
  :class:`fastpixal3d.runtime.streams.SideStream`).
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..runtime.cuda_graph import find_context_leaves, manual_cast_for


# ---------------------------------------------------------------------------
# texture_kv_reuse: request-scoped caches inside the flow transformer
# ---------------------------------------------------------------------------
class _Targets:
    """Live Pixal3D classes / module namespaces, discovered from the model and latent."""

    def __init__(self, flow_model, slat):
        cross_cls = proj_cls = None
        for _, module in flow_model.named_modules():
            if cross_cls is None and getattr(module, "_type", None) == "cross" and hasattr(module, "to_kv"):
                cross_cls = type(module)
            if proj_cls is None and hasattr(module, "proj_linear") and hasattr(module, "cross_attn_block"):
                proj_cls = type(module)
            if cross_cls is not None and proj_cls is not None:
                break
        if cross_cls is None or proj_cls is None:
            raise RuntimeError("texture_kv_reuse needs projection cross-attention blocks")
        self.flow_model_module = sys.modules[type(flow_model).__module__]
        self.cross_attn_cls = cross_cls
        self.proj_attn_cls = proj_cls
        self.modules_module = sys.modules[cross_cls.__module__]
        self.full_attn_module = sys.modules[self.modules_module.sparse_scaled_dot_product_attention.__module__]
        self.sparse_cls = type(slat)
        self.varlen_cls = self.sparse_cls.__bases__[0]


class _KVCache:
    def __init__(self):
        self.manual_cast: Dict[Tuple[int, Any], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.kv: Dict[Tuple[int, int], Tuple[Any, Any]] = {}
        self.proj: Dict[Tuple[int, int], Tuple[Any, Any]] = {}
        self.seqlen: Dict[Any, Tuple[Any, Any, Any]] = {}
        self.hits = {"manual_cast": 0, "kv": 0, "proj_linear": 0, "seqlen": 0}


@contextlib.contextmanager
def _memo_manual_cast(owner_module, cache: _KVCache):
    original = owner_module.manual_cast

    def memoized(tensor, dtype):
        if not isinstance(tensor, torch.Tensor):
            return original(tensor, dtype)
        key = (id(tensor), dtype)
        hit = cache.manual_cast.get(key)
        if hit is not None and hit[0] is tensor:
            cache.hits["manual_cast"] += 1
            return hit[1]
        result = original(tensor, dtype)
        cache.manual_cast[key] = (tensor, result)
        return result

    owner_module.manual_cast = memoized
    try:
        yield
    finally:
        owner_module.manual_cast = original


@contextlib.contextmanager
def _cache_context_projections(t: _Targets, cache: _KVCache):
    cross_cls, proj_cls, sparse_cls = t.cross_attn_cls, t.proj_attn_cls, t.sparse_cls
    original_cross, original_proj = cross_cls.forward, proj_cls.forward
    modules_module = t.modules_module

    def cross_forward(self, x, context=None):
        if getattr(self, "_type", "self") != "cross" or context is None:
            return original_cross(self, x, context)
        q = self._linear(self.to_q, x)
        q = self._reshape_chs(q, (self.num_heads, -1))
        key = (id(self), id(context))
        hit = cache.kv.get(key)
        if hit is not None and hit[0] is context:
            kv = hit[1]
            cache.hits["kv"] += 1
        else:
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            cache.kv[key] = (context, kv)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k, v = kv.unbind(dim=-3)
            k = self.k_rms_norm(k)
            h = modules_module.sparse_scaled_dot_product_attention(q, k, v)
        else:
            h = modules_module.sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        return self._linear(self.to_out, h)

    def proj_forward(self, x, context):
        if isinstance(context, dict):
            global_context, proj_context = context["global"], context["proj"]
        else:
            global_context, proj_context = context
        global_out = self.cross_attn_block(x, global_context)
        key = (id(self), id(proj_context))
        hit = cache.proj.get(key)
        if hit is not None and hit[0] is proj_context:
            proj_feats = hit[1]
            cache.hits["proj_linear"] += 1
        else:
            source = proj_context.feats if isinstance(proj_context, sparse_cls) else proj_context
            proj_feats = self.proj_linear(source)
            cache.proj[key] = (proj_context, proj_feats)
        return global_out.replace(proj_feats + global_out.feats)

    cross_cls.forward, proj_cls.forward = cross_forward, proj_forward
    try:
        yield
    finally:
        cross_cls.forward, proj_cls.forward = original_cross, original_proj


def _layout(value):
    layout = getattr(value, "layout", None)
    return layout if isinstance(layout, (list, tuple)) else None


def _cu_seqlens(cache: _KVCache, value, seqlen: List[int], device):
    layout = _layout(value)
    key = ("layout", id(layout), device) if layout is not None else ("value", tuple(seqlen), device)
    hit = cache.seqlen.get(key)
    if hit is not None:
        cache.hits["seqlen"] += 1
        return hit[1], hit[2]
    cu = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlen), dim=0)]).int().to(device)
    max_len = max(seqlen)
    cache.seqlen[key] = (layout, cu, max_len)
    return cu, max_len


@contextlib.contextmanager
def _cache_seqlens(t: _Targets, cache: _KVCache):
    """Reuse ``cu_seqlens`` / ``max_seqlen`` of the FlashAttention-2 varlen calls (other backends untouched)."""
    full_attn_module, modules_module = t.full_attn_module, t.modules_module
    original = full_attn_module.sparse_scaled_dot_product_attention

    def seqlens(v):
        return [s.stop - s.start for s in _layout(v)]

    def cached_sdpa(*args, **kwargs):
        if full_attn_module.config.ATTN != "flash_attn":
            return original(*args, **kwargs)
        names = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
        n = len(args) + len(kwargs)
        if n not in names or any(name not in kwargs for name in names[n][len(args):]):
            return original(*args, **kwargs)
        import flash_attn

        if n == 1:
            qkv = args[0] if args else kwargs["qkv"]
            cu, mx = _cu_seqlens(cache, qkv, seqlens(qkv), qkv.device)
            return qkv.replace(flash_attn.flash_attn_varlen_qkvpacked_func(qkv.feats, cu, mx))
        if n == 2:
            q = args[0] if args else kwargs["q"]
            kv = args[1] if len(args) > 1 else kwargs["kv"]
            if _layout(q) is None or _layout(kv) is None:
                return original(*args, **kwargs)
            cu_q, max_q = _cu_seqlens(cache, q, seqlens(q), q.device)
            cu_kv, max_kv = _cu_seqlens(cache, kv, seqlens(kv), q.device)
            return q.replace(flash_attn.flash_attn_varlen_kvpacked_func(q.feats, kv.feats, cu_q, cu_kv, max_q, max_kv))
        q = args[0] if args else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        if _layout(q) is None:
            return original(*args, **kwargs)
        if _layout(k) is not None:
            kv_seqlen, k_feats, v_feats, kv_key = seqlens(k), k.feats, v.feats, k
        else:
            if k.dim() != 4 or v.dim() != 4:
                return original(*args, **kwargs)
            N, L, H, C = k.shape
            kv_seqlen = [L] * N
            k_feats, v_feats, kv_key = k.reshape(N * L, H, C), v.reshape(N * L, v.shape[2], v.shape[3]), None
        cu_q, max_q = _cu_seqlens(cache, q, seqlens(q), q.device)
        cu_kv, max_kv = _cu_seqlens(cache, kv_key, kv_seqlen, q.device)
        return q.replace(flash_attn.flash_attn_varlen_func(q.feats, k_feats, v_feats, cu_q, cu_kv, max_q, max_kv))

    modules_module.sparse_scaled_dot_product_attention = cached_sdpa
    full_attn_module.sparse_scaled_dot_product_attention = cached_sdpa
    try:
        yield
    finally:
        modules_module.sparse_scaled_dot_product_attention = original
        full_attn_module.sparse_scaled_dot_product_attention = original


@contextlib.contextmanager
def _broadcast_views(t: _Targets):
    """Batch-1 broadcast in elementwise ops: zero-copy ``expand`` instead of a gather."""
    varlen_cls, sparse_cls = t.varlen_cls, t.sparse_cls
    original_varlen = varlen_cls.__elemwise__
    original_sparse = sparse_cls.__elemwise__ if sparse_cls is not varlen_cls else None

    def varlen_elemwise(self, other, op):
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                if len(self.layout) == 1 and other.shape[0] == 1:
                    other = other.expand(self.feats.shape[0], *other.shape[1:])
                else:
                    other = other[self.batch_boardcast_map]
            except Exception:
                pass
        if hasattr(other, "feats") and not isinstance(other, torch.Tensor):
            other = other.feats
        return self.replace(op(self.feats, other))

    def sparse_elemwise(self, other, op):
        if isinstance(other, torch.Tensor):
            try:
                broadcast = torch.broadcast_to(other, self.shape)
                if len(self.layout) == 1 and broadcast.shape[0] == 1:
                    other = broadcast.expand(self.feats.shape[0], *broadcast.shape[1:])
                else:
                    other = broadcast[self.batch_boardcast_map]
            except Exception:
                pass
        merge_other = other if hasattr(other, "feats") else None
        if hasattr(other, "feats"):
            other = other.feats
        new_tensor = self.replace(op(self.feats, other))
        if merge_other is not None and hasattr(self, "_SparseTensor__merge_sparse_cache"):
            new_tensor._spatial_cache = self._SparseTensor__merge_sparse_cache(merge_other)
        return new_tensor

    varlen_cls.__elemwise__ = varlen_elemwise
    if original_sparse is not None:
        sparse_cls.__elemwise__ = sparse_elemwise
    try:
        yield
    finally:
        varlen_cls.__elemwise__ = original_varlen
        if original_sparse is not None:
            sparse_cls.__elemwise__ = original_sparse


@contextlib.contextmanager
def texture_kv_reuse(flow_model, slat):
    """All four caches of ``texture_kv_reuse`` for one sampling call; yields the hit counters."""
    t = _Targets(flow_model, slat)
    cache = _KVCache()
    t.cross_attn_cls.modules_module = t.modules_module
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_memo_manual_cast(t.flow_model_module, cache))
            stack.enter_context(_cache_seqlens(t, cache))
            stack.enter_context(_cache_context_projections(t, cache))
            stack.enter_context(_broadcast_views(t))
            yield cache.hits
    finally:
        del t.cross_attn_cls.modules_module


# ---------------------------------------------------------------------------
# texture_projection_reuse: leaf-level projection cache + conditioning cast memo
# ---------------------------------------------------------------------------
def _version(t):
    try:
        return int(t._version)
    except Exception:
        return None


@contextlib.contextmanager
def texture_projection_reuse(flow_model, max_bytes: int = 8 * 1024 ** 3):
    """Cache each block's ``to_kv(global)`` and ``proj_linear(proj)`` outputs (identity-keyed,
    version-checked; at most ``max_bytes``) and memoise the conditioning cast, for one sampling call."""
    leaves = find_context_leaves(flow_model)
    entries: Dict[Tuple[int, int], Tuple[Any, Any, Any]] = {}
    cast_memo: Dict[Tuple[int, int], Tuple[Any, Any, Any]] = {}
    stats = {"hits": 0, "misses": 0, "bytes": 0}
    cast = manual_cast_for(flow_model)
    dtype = flow_model.dtype

    def hook(module, args, kwargs):
        if torch.is_autocast_enabled():
            return None
        positional = len(args) > 2
        cond = args[2] if positional else kwargs.get("cond")
        if not isinstance(cond, dict) or set(cond.keys()) != {"global", "proj"}:
            return None
        g, p = cond["global"], cond["proj"]
        if not isinstance(g, torch.Tensor) or not (isinstance(p, torch.Tensor) or hasattr(p, "feats")):
            return None
        key = (id(g), id(p))
        ent = cast_memo.get(key)
        if ent is None or ent[0] is not g or ent[1] is not p:
            ent = (g, p, {"global": cast(g, dtype), "proj": cast(p, dtype)})
            cast_memo[key] = ent
        new_cond = ent[2]
        if positional:
            return (*args[:2], new_cond, *args[3:]), kwargs
        return args, {**kwargs, "cond": new_cond}

    installed = []
    handle = flow_model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for leaf in leaves:
            module = leaf.module
            had = "forward" in module.__dict__
            old = module.__dict__.get("forward")
            original = module.forward

            def forward(x, *args, __original=original, __leaf_id=id(module), **kwargs):
                if args or kwargs or not isinstance(x, torch.Tensor) or torch.is_grad_enabled():
                    return __original(x, *args, **kwargs)
                ent = entries.get((__leaf_id, id(x)))
                if ent is not None and ent[0] is x and ent[2] == _version(x):
                    stats["hits"] += 1
                    return ent[1]
                out = __original(x)
                stats["misses"] += 1
                size = int(out.numel() * out.element_size())
                if stats["bytes"] + size <= max_bytes:
                    entries[(__leaf_id, id(x))] = (x, out, _version(x))
                    stats["bytes"] += size
                return out

            module.forward = forward
            installed.append((module, had, old))
        yield stats
    finally:
        for module, had, old in reversed(installed):
            if had:
                module.forward = old
            else:
                del module.__dict__["forward"]
        handle.remove()
        entries.clear()
        cast_memo.clear()
