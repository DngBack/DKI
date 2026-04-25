import types
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from methods import (
    build_demo_teacher_forcing_messages,
    build_dki_query_messages,
    get_decoder_layers,
)


def _extract_token_ids(tokenizer, candidates: List[str]) -> List[int]:
    ids = []
    for tok in candidates:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            continue
        ids.append(tok_id)
    return sorted(set(ids))


def get_image_token_indices(wrapper, input_ids: torch.Tensor) -> List[int]:
    tokenizer = wrapper.processor.tokenizer
    token_ids = _extract_token_ids(
        tokenizer,
        # Keep only dense visual slots; marker tokens are noisy for memory.
        candidates=["<|image_pad|>"],
    )
    ids = input_ids[0]
    positions: List[int] = []
    for tok_id in token_ids:
        pos = (ids == tok_id).nonzero(as_tuple=False).flatten().tolist()
        positions.extend(pos)
    return sorted(set(positions))


def _collect_schema_keys(schema_obj: Any, out: List[str]) -> None:
    if isinstance(schema_obj, dict):
        for k, v in schema_obj.items():
            out.append(k)
            _collect_schema_keys(v, out)
        return
    if isinstance(schema_obj, list):
        if schema_obj:
            _collect_schema_keys(schema_obj[0], out)


def get_answer_token_indices(wrapper, input_ids: torch.Tensor, answer_text: str) -> List[int]:
    tok = wrapper.processor.tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].tolist()
    seq = input_ids[0].tolist()
    n = len(tok)
    if n == 0:
        return []
    for start in range(len(seq) - n, -1, -1):
        if seq[start : start + n] == tok:
            return list(range(start, start + n))
    # Fallback to a small tail span.
    tail = min(32, input_ids.shape[1])
    return list(range(input_ids.shape[1] - tail, input_ids.shape[1]))


def select_top_visual_tokens_from_attention(
    attentions,
    layer_idx: int,
    image_indices: List[int],
    answer_indices: List[int],
    top_n: int,
):
    if len(image_indices) == 0:
        return []
    if len(answer_indices) == 0:
        return image_indices[: min(top_n, len(image_indices))]
    attn = attentions[layer_idx][0]  # [heads, seq, seq]
    answer_to_image = attn[:, answer_indices, :][:, :, image_indices]  # [heads, answer_len, image_len]
    scores = answer_to_image.mean(dim=(0, 1))
    k = min(top_n, scores.numel())
    local_top = torch.topk(scores, k=k).indices.tolist()
    return [image_indices[i] for i in local_top]


def patch_qwen3_kv_capture_layers(model, target_layers: List[int], capture_store: Dict[int, Dict[str, torch.Tensor]]):
    layers = get_decoder_layers(model)
    originals = {}
    for layer_idx in target_layers:
        attn = layers[layer_idx].self_attn
        originals[layer_idx] = attn.forward

        def capture_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            __layer_idx=layer_idx,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            capture_store[__layer_idx] = {
                "key": key_states.detach().cpu(),
                "value": value_states.detach().cpu(),
            }

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attn.forward = types.MethodType(capture_forward, attn)

    def restore():
        for idx in target_layers:
            layers[idx].self_attn.forward = originals[idx]

    return restore


def build_dki_kv_memory(
    wrapper,
    demos: List[Dict[str, str]],
    target_layers: List[int],
    schema_obj: Dict[str, Any],
    top_n: int = 32,
    top_n_text: int = 16,
) -> Dict[int, Dict[str, torch.Tensor]]:
    memories = {layer_idx: {"k": [], "v": []} for layer_idx in target_layers}
    capture_store: Dict[int, Dict[str, torch.Tensor]] = {}
    restore_capture = patch_qwen3_kv_capture_layers(wrapper.model, target_layers, capture_store)
    try:
        for demo in demos:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Keep extraction sequence shorter to reduce memory for attention maps.
            answer_text = demo["text"]
            try:
                answer_text = json_minify(answer_text)
            except Exception:
                pass
            msgs = build_demo_teacher_forcing_messages(
                image_path=demo["image"],
                answer_text=answer_text,
                schema_obj=schema_obj,
            )
            use_attentions = True
            try:
                inputs, outputs = wrapper.forward_teacher_forcing(msgs, output_attentions=True)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                # Fallback: still extract KV, but skip heavy attention map computation.
                torch.cuda.empty_cache()
                use_attentions = False
                inputs, outputs = wrapper.forward_teacher_forcing(msgs, output_attentions=False)

            image_indices = get_image_token_indices(wrapper, inputs.input_ids)
            answer_indices = get_answer_token_indices(wrapper, inputs.input_ids, answer_text)
            schema_keys: List[str] = []
            _collect_schema_keys(schema_obj, schema_keys)
            key_token_positions: List[int] = []
            token_seq = inputs.input_ids[0].tolist()
            for key_text in schema_keys:
                key_tok = wrapper.processor.tokenizer(
                    key_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids[0].tolist()
                if not key_tok:
                    continue
                n = len(key_tok)
                for start in range(len(token_seq) - n + 1):
                    if token_seq[start : start + n] == key_tok:
                        key_token_positions.extend(range(start, start + n))
            key_token_positions = sorted(set(key_token_positions))
            for layer_idx in target_layers:
                if use_attentions and outputs.attentions is not None:
                    selected = select_top_visual_tokens_from_attention(
                        outputs.attentions,
                        layer_idx=layer_idx,
                        image_indices=image_indices,
                        answer_indices=answer_indices,
                        top_n=top_n,
                    )
                else:
                    selected = image_indices[: min(top_n, len(image_indices))]
                if len(selected) == 0:
                    continue
                if layer_idx not in capture_store:
                    continue
                k = capture_store[layer_idx]["key"].to(dtype=torch.bfloat16)
                v = capture_store[layer_idx]["value"].to(dtype=torch.bfloat16)
                sel = torch.tensor(selected, device=k.device, dtype=torch.long)
                memories[layer_idx]["k"].append(k[:, :, sel, :].detach().cpu())
                memories[layer_idx]["v"].append(v[:, :, sel, :].detach().cpu())
                # Additional text-template memory branch from schema key tokens.
                if key_token_positions:
                    text_sel = key_token_positions[: min(top_n_text, len(key_token_positions))]
                    sel_txt = torch.tensor(text_sel, device=k.device, dtype=torch.long)
                    memories[layer_idx]["k"].append(k[:, :, sel_txt, :].detach().cpu())
                    memories[layer_idx]["v"].append(v[:, :, sel_txt, :].detach().cpu())

        final_memory: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx, kv_lists in memories.items():
            if not kv_lists["k"]:
                continue
            final_memory[layer_idx] = {
                "key": torch.cat(kv_lists["k"], dim=2),
                "value": torch.cat(kv_lists["v"], dim=2),
            }
        return final_memory
    finally:
        restore_capture()


def json_minify(text: str) -> str:
    obj = json.loads(text)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass
class DKIKVController:
    enabled: bool = False
    memory_by_layer: Dict[int, Dict[str, torch.Tensor]] = None
    beta: float = 1.0

    def __post_init__(self):
        if self.memory_by_layer is None:
            self.memory_by_layer = {}

    def set_memory(self, memory_by_layer: Dict[int, Dict[str, torch.Tensor]], beta: float = 1.0):
        self.memory_by_layer = memory_by_layer
        self.beta = beta

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


def patch_qwen3_attention_layers(model, controller: DKIKVController, target_layers: List[int]):
    layers = get_decoder_layers(model)
    originals = {}
    for layer_idx in target_layers:
        attn = layers[layer_idx].self_attn
        original_forward = attn.forward
        originals[layer_idx] = original_forward

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

            mem_len = 0
            # Gated injection: only apply during token-by-token decoding.
            is_decode_step = past_key_values is not None and hidden_states.shape[1] == 1
            if controller.enabled and is_decode_step and layer_idx in controller.memory_by_layer:
                mem = controller.memory_by_layer[layer_idx]
                mem_k = mem["key"].to(device=key_states.device, dtype=key_states.dtype)
                mem_v = mem["value"].to(device=value_states.device, dtype=value_states.dtype)
                bsz = key_states.shape[0]
                if mem_k.shape[0] == 1 and bsz > 1:
                    mem_k = mem_k.expand(bsz, -1, -1, -1)
                    mem_v = mem_v.expand(bsz, -1, -1, -1)
                mem_len = mem_k.shape[2]
                key_states = torch.cat([mem_k, key_states], dim=2)
                value_states = torch.cat([mem_v, value_states], dim=2)

                if attention_mask is not None:
                    prefix_mask = torch.zeros(
                        attention_mask.shape[0],
                        attention_mask.shape[1],
                        attention_mask.shape[2],
                        mem_len,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

                    if controller.beta != 0.0:
                        attention_mask = attention_mask.clone()
                        attention_mask[:, :, :, :mem_len] += controller.beta

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attn.forward = types.MethodType(patched_forward, attn)
        attn._dki_original_forward = original_forward

    def restore():
        for layer_idx in target_layers:
            if layer_idx in originals:
                layers[layer_idx].self_attn.forward = originals[layer_idx]

    return restore


def run_dki_kv_inference(
    wrapper,
    query_image: str,
    schema_obj: Dict[str, Any],
    controller: DKIKVController,
    max_new_tokens: int = 1024,
):
    msgs = build_dki_query_messages(query_image, schema_obj=schema_obj)
    controller.enable()
    out = wrapper.generate(msgs, max_new_tokens=max_new_tokens)
    controller.disable()
    return out
