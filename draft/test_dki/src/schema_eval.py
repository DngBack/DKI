import json
import unicodedata
from typing import Any, Dict, List, Tuple


def align_to_schema(schema: Any, pred: Any) -> Any:
    """Force prediction into exactly the same structure as schema."""
    if isinstance(schema, dict):
        pred_dict = pred if isinstance(pred, dict) else {}
        return {k: align_to_schema(v, pred_dict.get(k)) for k, v in schema.items()}
    if isinstance(schema, list):
        if not schema:
            return []
        item_schema = schema[0]
        pred_list = pred if isinstance(pred, list) else []
        if not pred_list:
            # Keep one template row for stable form OCR output.
            return [align_to_schema(item_schema, {})]
        return [align_to_schema(item_schema, x) for x in pred_list]
    if pred is None:
        return ""
    if isinstance(pred, (dict, list)):
        return ""
    return str(pred)


def _collect_leaf_paths(obj: Any, prefix: str = "") -> List[str]:
    paths: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.extend(_collect_leaf_paths(v, new_prefix))
        return paths
    if isinstance(obj, list):
        if not obj:
            return [prefix]
        # Evaluate first row schema as table shape representative.
        return _collect_leaf_paths(obj[0], f"{prefix}[]")
    return [prefix]


def _get_by_path(obj: Any, path: str):
    cur = obj
    parts = path.split(".")
    for p in parts:
        if p.endswith("[]"):
            k = p[:-2]
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
            if not isinstance(cur, list) or not cur:
                return None
            cur = cur[0]
            continue
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _norm_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.strip().split())


def schema_metrics(schema_obj: Dict[str, Any], raw_pred_obj: Dict[str, Any] | None, aligned_obj: Dict[str, Any]) -> Dict[str, Any]:
    leaf_paths = _collect_leaf_paths(schema_obj)
    total = len(leaf_paths)
    existed = 0
    non_empty = 0
    for p in leaf_paths:
        raw_v = _get_by_path(raw_pred_obj, p) if raw_pred_obj is not None else None
        if raw_v is not None:
            existed += 1
        aligned_v = _get_by_path(aligned_obj, p)
        if _norm_text(aligned_v) != "":
            non_empty += 1
    return {
        "leaf_total": total,
        "leaf_present_in_raw_pred": existed,
        "leaf_presence_rate": (existed / total) if total else 0.0,
        "leaf_non_empty_in_aligned": non_empty,
        "leaf_non_empty_rate": (non_empty / total) if total else 0.0,
    }


def value_match_metrics(gt_obj: Dict[str, Any], pred_aligned_obj: Dict[str, Any], schema_obj: Dict[str, Any]) -> Dict[str, Any]:
    leaf_paths = _collect_leaf_paths(schema_obj)
    total = len(leaf_paths)
    exact = 0
    for p in leaf_paths:
        gt_v = _norm_text(_get_by_path(gt_obj, p))
        pr_v = _norm_text(_get_by_path(pred_aligned_obj, p))
        if gt_v == pr_v:
            exact += 1
    return {
        "leaf_exact_match_count": exact,
        "leaf_exact_match_rate": (exact / total) if total else 0.0,
        "leaf_total": total,
    }


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_largest_valid_json(raw_text: str) -> Dict[str, Any] | None:
    """
    Extract largest valid JSON object from raw text.
    Handles nested braces and ignores braces inside quoted strings.
    """
    candidates: List[str] = []
    stack: List[int] = []
    in_string = False
    escape = False

    for i, ch in enumerate(raw_text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append(i)
            continue
        if ch == "}":
            if not stack:
                continue
            start = stack.pop()
            candidates.append(raw_text[start : i + 1])

    best_obj = None
    best_len = -1
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if len(c) > best_len and isinstance(obj, dict):
            best_obj = obj
            best_len = len(c)
    return best_obj
