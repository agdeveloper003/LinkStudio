"""Global block registry.

Blocks register themselves with :func:`register_block`, which makes them
constructible by name (scenario files, CLI, future UI) and discoverable through
:func:`block_catalogue`.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

from linkstudio.core.block import Block, BlockError

B = TypeVar("B", bound=type[Block])

_REGISTRY: dict[str, type[Block]] = {}
_BUILTIN_PACKAGES = ("linkstudio.phy",)
_builtins_loaded = False


def register_block(type_name: str, category: str, subcategory: str = "", label: str = "") -> Callable[[B], B]:
    """Class decorator registering a block under ``type_name``.

    ``category`` is the functional class of the block in a communication chain
    (``encoder``, ``modulator``, ``medium``...), ``subcategory`` the specific kind
    within it and ``label`` the readable name shown in the UI (defaults to the
    title-cased ``type_name``).
    """

    def decorator(cls: B) -> B:
        existing = _REGISTRY.get(type_name)
        if existing is not None and existing is not cls:
            raise BlockError(f"Block type '{type_name}' is already registered by {existing.__name__}.")
        cls.type_name = type_name
        cls.category = category
        cls.subcategory = subcategory
        cls.label = label or type_name.replace("_", " ").title()
        _REGISTRY[type_name] = cls
        return cls

    return decorator


def _ensure_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        _builtins_loaded = True
        for package in _BUILTIN_PACKAGES:
            importlib.import_module(package)


def get_block_class(type_name: str) -> type[Block]:
    _ensure_builtins()
    try:
        return _REGISTRY[type_name]
    except KeyError:
        raise BlockError(f"Unknown block type '{type_name}'. Available: {available_blocks()}") from None


def create_block(type_name: str, **params: Any) -> Block:
    """Instantiate a registered block by name."""
    return get_block_class(type_name)(**params)


def available_blocks(category: str | None = None) -> list[str]:
    _ensure_builtins()
    return sorted(n for n, c in _REGISTRY.items() if category is None or c.category == category)


def block_catalogue() -> dict[str, dict[str, Any]]:
    """Describe every registered block: category, docs, ports and parameter JSON Schema."""
    _ensure_builtins()
    catalogue: dict[str, dict[str, Any]] = {}
    for name in available_blocks():
        cls = _REGISTRY[name]
        catalogue[name] = {
            "category": cls.category,
            "subcategory": cls.subcategory,
            "label": cls.label,
            "doc": (cls.__doc__ or "").strip(),
            "inputs": {p: {"kinds": port.kinds.label(), "required": port.required, "doc": port.doc}
                       for p, port in cls.inputs.items()},
            "outputs": {p: {"kinds": f"same as '{port.same_as}'" if port.same_as else port.kinds.label(),
                            "doc": port.doc}
                        for p, port in cls.outputs.items()},
            "params": cls.Params.model_json_schema(),
        }
    return catalogue
