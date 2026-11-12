"""YAML configuration files of the training and cache-building scripts.

Every script defines its defaults as a nested mapping. A configuration file may only set keys that
exist in the defaults (a misspelt key is an error, not a silently ignored value), and
``--set section.key=value`` overrides one key on the command line (the value is parsed as YAML, so
``--set optim.updates=10`` gives an int and ``--set data.views=null`` gives None).

Paths: relative paths written in a configuration file are resolved against the directory of that
file; relative paths given with ``--set`` are resolved against the current directory. Keys listed
as *optional paths* (for example the Pixal3D weights, which may also be a Hugging Face repository
id) are only resolved when the resolved path exists.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

ANY = object()  # default marker: the key accepts any YAML value (lists, strings, mappings)


class ConfigError(ValueError):
    """The configuration file or an override does not match the expected keys or values."""


def _merge(defaults: Mapping[str, Any], user: Mapping[str, Any], where: str) -> dict:
    out = copy.deepcopy(dict(defaults))
    for key, value in user.items():
        here = f"{where}{key}"
        if key not in defaults:
            raise ConfigError(f"unknown configuration key {here!r} (known: {sorted(defaults)})")
        default = defaults[key]
        if isinstance(default, Mapping):
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ConfigError(f"{here!r} must be a mapping, got {type(value).__name__}")
            out[key] = _merge(default, value, here + ".")
        else:
            out[key] = value
    return out


def _strip_markers(cfg: Any) -> Any:
    if isinstance(cfg, Mapping):
        return {k: _strip_markers(v) for k, v in cfg.items()}
    return None if cfg is ANY else cfg


def parse_override(text: str) -> Tuple[Tuple[str, ...], Any]:
    """``"optim.updates=10"`` -> ``(("optim", "updates"), 10)``."""
    import yaml

    if "=" not in text:
        raise ConfigError(f"override {text!r} is not of the form section.key=value")
    key, raw = text.split("=", 1)
    keys = tuple(k for k in key.strip().split(".") if k)
    if not keys:
        raise ConfigError(f"override {text!r} has an empty key")
    return keys, yaml.safe_load(raw)


def get(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        node = node[key]
    return node


def _set(cfg: MutableMapping[str, Any], keys: Sequence[str], value: Any, defaults: Mapping[str, Any]) -> None:
    node, dnode = cfg, defaults
    for i, key in enumerate(keys):
        if not isinstance(dnode, Mapping) or key not in dnode:
            raise ConfigError(f"unknown configuration key {'.'.join(keys[:i + 1])!r}")
        if i == len(keys) - 1:
            if isinstance(dnode[key], Mapping) and not isinstance(value, Mapping):
                raise ConfigError(f"{'.'.join(keys)!r} is a section; set one of its keys instead")
            node[key] = _merge(dnode[key], value, ".".join(keys) + ".") if isinstance(dnode[key], Mapping) else value
            return
        node, dnode = node[key], dnode[key]


def _resolve(value: Any, base: Path, only_if_exists: bool) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_resolve(v, base, only_if_exists) for v in value]
    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    resolved = (base / path).resolve()
    if only_if_exists and not resolved.exists():
        return value
    return str(resolved)


def load_config(path: Optional[str | Path], defaults: Mapping[str, Any], *,
                path_keys: Iterable[str] = (), optional_path_keys: Iterable[str] = (),
                overrides: Iterable[str] = (), cwd: Optional[Path] = None) -> dict:
    """Defaults <- configuration file <- ``--set`` overrides, with relative paths resolved."""
    import yaml

    path_keys, optional_path_keys = list(path_keys), list(optional_path_keys)
    cfg = _merge(defaults, {}, "")
    if path is not None:
        path = Path(path)
        user = yaml.safe_load(path.read_text()) or {}
        if not isinstance(user, Mapping):
            raise ConfigError(f"{path}: the top level must be a mapping")
        cfg = _merge(defaults, user, "")
        base = path.resolve().parent
        for dotted in path_keys + optional_path_keys:
            keys = dotted.split(".")
            if _present(user, keys):
                _set(cfg, keys, _resolve(get(cfg, dotted), base, dotted in optional_path_keys), defaults)
    cwd = Path.cwd() if cwd is None else Path(cwd)
    for text in overrides:
        keys, value = parse_override(text)
        dotted = ".".join(keys)
        if dotted in path_keys or dotted in optional_path_keys:
            value = _resolve(value, cwd, dotted in optional_path_keys)
        _set(cfg, keys, value, defaults)
    return _strip_markers(cfg)


def _present(user: Mapping[str, Any], keys: Sequence[str]) -> bool:
    node: Any = user
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return False
        node = node[key]
    return True


def dump(cfg: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(_strip_markers(dict(cfg)), sort_keys=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)
