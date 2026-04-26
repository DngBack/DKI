"""Schema flattening, path get/set, and JSON assembly for SEF-ICL."""
from __future__ import annotations

from typing import Any, Dict, List


def empty_template_like(schema: Any) -> Any:
    """Same nested shape as schema; all leaves become empty strings."""
    if isinstance(schema, dict):
        return {k: empty_template_like(v) for k, v in schema.items()}
    if isinstance(schema, list):
        if not schema:
            return []
        return [empty_template_like(schema[0])]
    return ""


def flatten_schema_paths(schema: Any, prefix: str = "") -> List[str]:
    """Leaf dotted paths; list rows use `key[]` segments (same convention as schema_eval)."""
    paths: List[str] = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.extend(flatten_schema_paths(v, new_prefix))
        return paths
    if isinstance(schema, list):
        if not schema:
            return [prefix]
        return flatten_schema_paths(schema[0], f"{prefix}[]")
    return [prefix]


def get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for p in path.split("."):
        if p.endswith("[]"):
            key = p[:-2]
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
            if not isinstance(cur, list) or not cur:
                return None
            cur = cur[0]
            continue
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def set_by_path(obj: Any, path: str, value: Any) -> None:
    cur = obj
    parts = path.split(".")
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        if p.endswith("[]"):
            key = p[:-2]
            if not isinstance(cur, dict) or key not in cur:
                return
            lst = cur[key]
            if not isinstance(lst, list) or not lst:
                return
            cur = lst[0]
            if last:
                raise ValueError("Invalid path: cannot assign at list traversal token without leaf key")
            continue
        if not isinstance(cur, dict) or p not in cur:
            return
        if last:
            cur[p] = value
            return
        cur = cur[p]


def assemble_json(schema: Dict[str, Any], path_to_value: Dict[str, str]) -> Dict[str, Any]:
    out = empty_template_like(schema)
    for path, val in path_to_value.items():
        set_by_path(out, path, val)
    return out


def leaf_key_from_path(path: str) -> str:
    """Last human-readable segment (after final `.`, strip `[]`)."""
    tail = path.split(".")[-1]
    return tail[:-2] if tail.endswith("[]") else tail
