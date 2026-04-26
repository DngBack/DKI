import json
from pathlib import Path
from typing import Dict, Any, List

import torch

from dki_kv import (
    DKIKVController,
    build_dki_kv_memory,
    patch_qwen3_attention_layers,
    run_dki_kv_inference,
)


def _to_serializable_memory(memory_by_layer: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {str(k): {"key": v["key"].cpu(), "value": v["value"].cpu()} for k, v in memory_by_layer.items()}


def _from_serializable_memory(obj: Dict[str, Dict[str, torch.Tensor]]) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, v in obj.items():
        out[int(k)] = {"key": v["key"], "value": v["value"]}
    return out


def save_prefix_cache(
    cache_path: str,
    model_name: str,
    schema_obj: Dict[str, Any],
    target_layers: List[int],
    top_n: int,
    top_n_text: int,
    memory_by_layer: Dict[int, Dict[str, torch.Tensor]],
) -> None:
    payload = {
        "version": 1,
        "model_name": model_name,
        "schema_text": json.dumps(schema_obj, ensure_ascii=False, sort_keys=True),
        "target_layers": target_layers,
        "top_n": top_n,
        "top_n_text": top_n_text,
        "memory_by_layer": _to_serializable_memory(memory_by_layer),
    }
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_prefix_cache(
    cache_path: str,
    model_name: str,
    schema_obj: Dict[str, Any],
    target_layers: List[int],
    top_n: int,
    top_n_text: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    payload = torch.load(cache_path, map_location="cpu")
    if payload.get("version") != 1:
        raise ValueError("Unsupported prefix cache version")
    if payload.get("model_name") != model_name:
        raise ValueError("Prefix cache model_name mismatch")
    if payload.get("schema_text") != json.dumps(schema_obj, ensure_ascii=False, sort_keys=True):
        raise ValueError("Prefix cache schema mismatch")
    if payload.get("target_layers") != target_layers:
        raise ValueError("Prefix cache target_layers mismatch")
    if payload.get("top_n") != top_n or payload.get("top_n_text") != top_n_text:
        raise ValueError("Prefix cache token selection config mismatch")
    return _from_serializable_memory(payload["memory_by_layer"])


def build_or_load_prefix_memory(
    runner,
    demos: List[Dict[str, str]],
    schema_obj: Dict[str, Any],
    target_layers: List[int],
    top_n: int,
    top_n_text: int,
    cache_path: str = "",
    rebuild_cache: bool = False,
) -> Dict[int, Dict[str, torch.Tensor]]:
    if cache_path and (not rebuild_cache) and Path(cache_path).exists():
        return load_prefix_cache(
            cache_path=cache_path,
            model_name=runner.model.config._name_or_path,
            schema_obj=schema_obj,
            target_layers=target_layers,
            top_n=top_n,
            top_n_text=top_n_text,
        )

    memory_by_layer = build_dki_kv_memory(
        wrapper=runner,
        demos=demos,
        target_layers=target_layers,
        schema_obj=schema_obj,
        top_n=top_n,
        top_n_text=top_n_text,
    )
    if cache_path:
        save_prefix_cache(
            cache_path=cache_path,
            model_name=runner.model.config._name_or_path,
            schema_obj=schema_obj,
            target_layers=target_layers,
            top_n=top_n,
            top_n_text=top_n_text,
            memory_by_layer=memory_by_layer,
        )
    return memory_by_layer


def run_dyna_prefix_inference(
    runner,
    query_image: str,
    schema_obj: Dict[str, Any],
    target_layers: List[int],
    memory_by_layer: Dict[int, Dict[str, torch.Tensor]],
    beta: float,
    max_new_tokens: int,
):
    controller = DKIKVController()
    controller.set_memory(memory_by_layer, beta=beta)
    restore_patch = patch_qwen3_attention_layers(runner.model, controller, target_layers=target_layers)
    try:
        return run_dki_kv_inference(
            wrapper=runner,
            query_image=query_image,
            schema_obj=schema_obj,
            controller=controller,
            max_new_tokens=max_new_tokens,
        )
    finally:
        restore_patch()
