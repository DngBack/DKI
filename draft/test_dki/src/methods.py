from typing import Dict, Any, List

import torch
from torch.nn import ModuleList


SYSTEM_PROMPT = (
    "You are a structured OCR system for Vietnamese banking forms. "
    "Read the document and return OCR result as JSON only."
)


def _json_structure_signature(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_structure_signature(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        return [_json_structure_signature(obj[0])]
    return "<string>"


def _build_json_instruction(schema_obj: Dict[str, Any]) -> str:
    schema_hint = _json_structure_signature(schema_obj)
    schema_text = str(schema_hint).replace("'", '"')
    return (
        f"{SYSTEM_PROMPT}\n"
        "Requirements:\n"
        "1) Return one valid JSON object only.\n"
        "2) Keep key names exactly in Vietnamese as the template.\n"
        "3) Preserve nested structure/table arrays like template.\n"
        "4) If a value is missing, return empty string.\n"
        f"5) Output schema template:\n{schema_text}\n\n"
    )


def build_naive_messages(
    demos: List[Dict[str, str]],
    query_image: str,
    schema_obj: Dict[str, Any],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, str]] = [{"type": "text", "text": _build_json_instruction(schema_obj)}]
    for i, demo in enumerate(demos, start=1):
        content.append({"type": "text", "text": f"Example {i} image:\n"})
        content.append({"type": "image", "image": demo["image"]})
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nExample {i} answer (valid JSON):\n"
                    f"{demo['text']}\n\n"
                ),
            }
        )
    content.append({"type": "text", "text": "Now read this handwritten image:\n"})
    content.append({"type": "image", "image": query_image})
    content.append({"type": "text", "text": "\nReturn JSON answer only:"})
    return [{"role": "user", "content": content}]


def build_demo_teacher_forcing_messages(
    image_path: str,
    answer_text: str,
    schema_obj: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _build_json_instruction(schema_obj)},
                {"type": "image", "image": image_path},
                {"type": "text", "text": "\nReturn JSON answer only:"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
    ]


def build_dki_query_messages(query_image: str, schema_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _build_json_instruction(schema_obj)},
                {"type": "image", "image": query_image},
                {"type": "text", "text": "\nReturn JSON answer only:"},
            ],
        }
    ]


class DKILiteController:
    """
    Quick DKI-style approximation:
    - Extract a per-layer "memory vector" from demo hidden states.
    - Inject it into chosen mid-layer hidden states at query-time.
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.memory_by_layer: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def set_memory(self, memory_by_layer: Dict[int, torch.Tensor]) -> None:
        self.memory_by_layer = memory_by_layer

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def attach(self, model, target_layers: List[int]) -> None:
        self.clear()
        layers = get_decoder_layers(model)
        for layer_idx in target_layers:
            if layer_idx not in self.memory_by_layer:
                continue
            mem = self.memory_by_layer[layer_idx]

            def hook_fn(_module, _inputs, output, _mem=mem):
                hidden = output[0] if isinstance(output, tuple) else output
                mem_local = _mem.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                hidden = hidden + self.alpha * mem_local
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            self._hooks.append(layers[layer_idx].register_forward_hook(hook_fn))


def get_decoder_layers(model) -> ModuleList:
    # Try common paths first across Qwen2.5-VL/Qwen3-VL variants.
    candidate_paths = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
    ]
    for path in candidate_paths:
        cur = model
        ok = True
        for name in path:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and isinstance(cur, ModuleList) and len(cur) > 0:
            return cur

    # Fallback: scan named modules and pick the largest `.layers` ModuleList found.
    best = None
    for _name, module in model.named_modules():
        if hasattr(module, "layers"):
            layers = getattr(module, "layers")
            if isinstance(layers, ModuleList) and len(layers) > 0:
                if best is None or len(layers) > len(best):
                    best = layers
    if best is None:
        raise RuntimeError("Cannot locate decoder layers ModuleList in this model architecture.")
    return best


def extract_dki_lite_memory(
    wrapper,
    demos: List[Dict[str, str]],
    target_layers: List[int],
    schema_obj: Dict[str, Any],
) -> Dict[int, torch.Tensor]:
    memories: Dict[int, List[torch.Tensor]] = {idx: [] for idx in target_layers}
    for demo in demos:
        messages = build_demo_teacher_forcing_messages(
            image_path=demo["image"],
            answer_text=demo["text"],
            schema_obj=schema_obj,
        )
        _, outputs = wrapper.forward_teacher_forcing(messages)
        for layer_idx in target_layers:
            # hidden_states contains embedding output at index 0, decoder blocks from index 1.
            hs = outputs.hidden_states[layer_idx + 1][0]  # [seq, hidden]
            mem_vec = hs.mean(dim=0).detach().cpu()
            memories[layer_idx].append(mem_vec)

    return {
        idx: torch.stack(vecs, dim=0).mean(dim=0)
        for idx, vecs in memories.items()
        if vecs
    }
