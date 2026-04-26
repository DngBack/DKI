import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from methods import build_demo_teacher_forcing_messages, build_dki_query_messages, get_decoder_layers


def _extract_token_ids(tokenizer, candidates: List[str]) -> List[int]:
    ids = []
    for tok in candidates:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            continue
        ids.append(tok_id)
    return sorted(set(ids))


def _get_image_token_indices(wrapper, input_ids: torch.Tensor) -> List[int]:
    tokenizer = wrapper.processor.tokenizer
    token_ids = _extract_token_ids(tokenizer, candidates=["<|image_pad|>"])
    ids = input_ids[0]
    positions: List[int] = []
    for tok_id in token_ids:
        pos = (ids == tok_id).nonzero(as_tuple=False).flatten().tolist()
        positions.extend(pos)
    return sorted(set(positions))


def _get_answer_token_indices(wrapper, input_ids: torch.Tensor, answer_text: str) -> List[int]:
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
    return list(range(max(0, input_ids.shape[1] - 32), input_ids.shape[1]))


def _select_top_visual_tokens_from_attention(
    attentions,
    layer_idx: int,
    image_indices: List[int],
    answer_indices: List[int],
    top_n: int,
) -> List[int]:
    if len(image_indices) == 0:
        return []
    if len(answer_indices) == 0:
        return image_indices[: min(top_n, len(image_indices))]
    attn = attentions[layer_idx][0]
    answer_to_image = attn[:, answer_indices, :][:, :, image_indices]
    scores = answer_to_image.mean(dim=(0, 1))
    k = min(top_n, scores.numel())
    local_top = torch.topk(scores, k=k).indices.tolist()
    return [image_indices[i] for i in local_top]


def _compute_cross_modal_prefix(
    visual_k: torch.Tensor,
    visual_v: torch.Tensor,
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    prefix_len: int,
) -> Dict[str, torch.Tensor]:
    if visual_k.shape[2] == 0:
        return {"key": visual_k, "value": visual_v}
    if text_k.shape[2] == 0:
        return {
            "key": visual_k[:, :, : min(prefix_len, visual_k.shape[2]), :],
            "value": visual_v[:, :, : min(prefix_len, visual_v.shape[2]), :],
        }

    v_ctx = visual_k.mean(dim=1).squeeze(0)
    t_ctx = text_k.mean(dim=1).squeeze(0)
    affinity = (v_ctx @ t_ctx.transpose(0, 1)) / (v_ctx.shape[-1] ** 0.5)
    # Cross-modal routing: visual slots attend to text template slots.
    routing = torch.softmax(affinity, dim=-1)
    text_guided_visual = routing @ t_ctx
    gate = torch.sigmoid((v_ctx * text_guided_visual).sum(dim=-1))

    k = min(prefix_len, gate.numel())
    keep = torch.topk(gate, k=k).indices
    keep = keep.sort().values
    return {
        "key": visual_k[:, :, keep, :].detach().cpu(),
        "value": visual_v[:, :, keep, :].detach().cpu(),
    }


def _compute_separate_prefix(
    visual_k: torch.Tensor,
    visual_v: torch.Tensor,
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    prefix_len: int,
) -> Dict[str, torch.Tensor]:
    half = max(1, prefix_len // 2)
    v_len = min(half, visual_k.shape[2])
    t_len = min(prefix_len - v_len, text_k.shape[2])
    k_chunks = []
    v_chunks = []
    if v_len > 0:
        k_chunks.append(visual_k[:, :, :v_len, :])
        v_chunks.append(visual_v[:, :, :v_len, :])
    if t_len > 0:
        k_chunks.append(text_k[:, :, :t_len, :])
        v_chunks.append(text_v[:, :, :t_len, :])
    if not k_chunks:
        return {"key": visual_k[:, :, :0, :].detach().cpu(), "value": visual_v[:, :, :0, :].detach().cpu()}
    return {
        "key": torch.cat(k_chunks, dim=2).detach().cpu(),
        "value": torch.cat(v_chunks, dim=2).detach().cpu(),
    }


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


def build_dlpc_prefix_memory(
    wrapper,
    demos: List[Dict[str, str]],
    target_layers: List[int],
    schema_obj: Dict[str, Any],
    top_n_visual: int = 32,
    top_n_text: int = 16,
    prefix_len: int = 8,
    compressor_mode: str = "cross",
) -> Dict[int, Dict[str, torch.Tensor]]:
    if compressor_mode not in {"cross", "separate"}:
        raise ValueError(f"Unsupported compressor_mode: {compressor_mode}")

    memories = {layer_idx: {"k": [], "v": []} for layer_idx in target_layers}
    capture_store: Dict[int, Dict[str, torch.Tensor]] = {}
    restore_capture = patch_qwen3_kv_capture_layers(wrapper.model, target_layers, capture_store)
    try:
        for demo in demos:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            msgs = build_demo_teacher_forcing_messages(
                image_path=demo["image"],
                answer_text=demo["text"],
                schema_obj=schema_obj,
            )
            inputs, outputs = wrapper.forward_teacher_forcing(msgs, output_attentions=True)
            image_indices = _get_image_token_indices(wrapper, inputs.input_ids)
            answer_indices = _get_answer_token_indices(wrapper, inputs.input_ids, demo["text"])
            text_indices = answer_indices[: min(top_n_text, len(answer_indices))]
            for layer_idx in target_layers:
                selected_visual = _select_top_visual_tokens_from_attention(
                    outputs.attentions,
                    layer_idx=layer_idx,
                    image_indices=image_indices,
                    answer_indices=answer_indices,
                    top_n=top_n_visual,
                )
                if len(selected_visual) == 0:
                    continue
                if layer_idx not in capture_store:
                    continue
                k = capture_store[layer_idx]["key"].to(dtype=torch.bfloat16)
                v = capture_store[layer_idx]["value"].to(dtype=torch.bfloat16)
                sel_v = torch.tensor(selected_visual, device=k.device, dtype=torch.long)
                visual_k = k[:, :, sel_v, :]
                visual_v = v[:, :, sel_v, :]
                if text_indices:
                    sel_t = torch.tensor(text_indices, device=k.device, dtype=torch.long)
                    text_k = k[:, :, sel_t, :]
                    text_v = v[:, :, sel_t, :]
                else:
                    text_k = k[:, :, :0, :]
                    text_v = v[:, :, :0, :]

                if compressor_mode == "cross":
                    prefix = _compute_cross_modal_prefix(
                        visual_k=visual_k,
                        visual_v=visual_v,
                        text_k=text_k,
                        text_v=text_v,
                        prefix_len=prefix_len,
                    )
                else:
                    prefix = _compute_separate_prefix(
                        visual_k=visual_k,
                        visual_v=visual_v,
                        text_k=text_k,
                        text_v=text_v,
                        prefix_len=prefix_len,
                    )
                memories[layer_idx]["k"].append(prefix["key"])
                memories[layer_idx]["v"].append(prefix["value"])
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


@dataclass
class DLPCController:
    enabled: bool = False
    memory_by_layer: Dict[int, Dict[str, torch.Tensor]] = None
    beta: float = 1.0
    value_blend_lambda: float = 0.05
    inject_value_prefix: bool = True
    value_beta_scale: float = 0.1
    key_phase_tokens: int = 48
    positional_reset_prefix: bool = True
    decode_step: int = 0

    def __post_init__(self):
        if self.memory_by_layer is None:
            self.memory_by_layer = {}

    def set_memory(
        self,
        memory_by_layer: Dict[int, Dict[str, torch.Tensor]],
        beta: float = 1.0,
        value_blend_lambda: float = 0.05,
        inject_value_prefix: bool = True,
        value_beta_scale: float = 0.1,
        key_phase_tokens: int = 48,
        positional_reset_prefix: bool = True,
    ):
        self.memory_by_layer = memory_by_layer
        self.beta = beta
        self.value_blend_lambda = value_blend_lambda
        self.inject_value_prefix = inject_value_prefix
        self.value_beta_scale = value_beta_scale
        self.key_phase_tokens = key_phase_tokens
        self.positional_reset_prefix = positional_reset_prefix

    def enable(self):
        self.enabled = True
        self.decode_step = 0

    def disable(self):
        self.enabled = False

    def current_beta(self) -> float:
        # Generation-aware gating:
        # - Early decode tends to output JSON keys/structure -> stronger guidance.
        # - Later decode tends to output OCR values -> weaker guidance.
        if self.decode_step < self.key_phase_tokens:
            return self.beta
        return self.beta * self.value_beta_scale


def patch_qwen3_attention_layers(model, controller: DLPCController, target_layers: List[int]):
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
            is_decode_step = past_key_values is not None and hidden_states.shape[1] == 1
            if controller.enabled and is_decode_step and layer_idx in controller.memory_by_layer:
                mem = controller.memory_by_layer[layer_idx]
                mem_k = mem["key"].to(device=key_states.device, dtype=key_states.dtype)
                mem_v = mem["value"].to(device=value_states.device, dtype=value_states.dtype)
                bsz = key_states.shape[0]
                if mem_k.shape[0] == 1 and bsz > 1:
                    mem_k = mem_k.expand(bsz, -1, -1, -1)
                    mem_v = mem_v.expand(bsz, -1, -1, -1)
                if controller.positional_reset_prefix:
                    # Reduce absolute-position bias carried from support examples.
                    # This keeps prefix as a relative hint rather than a hard spatial anchor.
                    mem_k = mem_k - mem_k.mean(dim=2, keepdim=True)
                mem_len = mem_k.shape[2]
                key_states = torch.cat([mem_k, key_states], dim=2)
                if controller.inject_value_prefix:
                    lam = max(0.0, min(1.0, float(controller.value_blend_lambda)))
                    # Soft gating: keep prefix as a weak hint, not a hard override.
                    mem_v = mem_v * lam
                    value_states = torch.cat([mem_v, value_states], dim=2)
                else:
                    # Key-only injection: preserve query content path in values.
                    value_pad = torch.zeros_like(mem_v)
                    value_states = torch.cat([value_pad, value_states], dim=2)
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
                    cur_beta = controller.current_beta()
                    if cur_beta != 0.0:
                        attention_mask = attention_mask.clone()
                        attention_mask[:, :, :, :mem_len] += cur_beta

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
            if controller.enabled and is_decode_step and layer_idx == target_layers[-1]:
                controller.decode_step += 1
            return attn_output, attn_weights

        attn.forward = types.MethodType(patched_forward, attn)

    def restore():
        for idx in target_layers:
            layers[idx].self_attn.forward = originals[idx]

    return restore


def run_dlpc_inference(
    wrapper,
    query_image: str,
    schema_obj: Dict[str, Any],
    target_layers: List[int],
    memory_by_layer: Dict[int, Dict[str, torch.Tensor]],
    beta: float,
    value_blend_lambda: float,
    inject_value_prefix: bool,
    value_beta_scale: float,
    key_phase_tokens: int,
    positional_reset_prefix: bool,
    max_new_tokens: int,
):
    controller = DLPCController()
    controller.set_memory(
        memory_by_layer,
        beta=beta,
        value_blend_lambda=value_blend_lambda,
        inject_value_prefix=inject_value_prefix,
        value_beta_scale=value_beta_scale,
        key_phase_tokens=key_phase_tokens,
        positional_reset_prefix=positional_reset_prefix,
    )
    restore_patch = patch_qwen3_attention_layers(wrapper.model, controller, target_layers=target_layers)
    try:
        msgs = build_dki_query_messages(query_image, schema_obj=schema_obj)
        controller.enable()
        out = wrapper.generate(msgs, max_new_tokens=max_new_tokens)
        controller.disable()
        return out
    finally:
        restore_patch()


def save_prefix_cache(
    cache_path: str,
    model_name: str,
    schema_obj: Dict[str, Any],
    target_layers: List[int],
    top_n_visual: int,
    top_n_text: int,
    prefix_len: int,
    compressor_mode: str,
    memory_by_layer: Dict[int, Dict[str, torch.Tensor]],
) -> None:
    payload = {
        "version": 1,
        "model_name": model_name,
        "schema_text": json.dumps(schema_obj, ensure_ascii=False, sort_keys=True),
        "target_layers": target_layers,
        "top_n_visual": top_n_visual,
        "top_n_text": top_n_text,
        "prefix_len": prefix_len,
        "compressor_mode": compressor_mode,
        "memory_by_layer": {str(k): {"key": v["key"].cpu(), "value": v["value"].cpu()} for k, v in memory_by_layer.items()},
    }
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_prefix_cache(
    cache_path: str,
    model_name: str,
    schema_obj: Dict[str, Any],
    target_layers: List[int],
    top_n_visual: int,
    top_n_text: int,
    prefix_len: int,
    compressor_mode: str,
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
    if payload.get("top_n_visual") != top_n_visual or payload.get("top_n_text") != top_n_text:
        raise ValueError("Prefix cache token selection mismatch")
    if payload.get("prefix_len") != prefix_len or payload.get("compressor_mode") != compressor_mode:
        raise ValueError("Prefix cache compressor config mismatch")
    return {int(k): {"key": v["key"], "value": v["value"]} for k, v in payload["memory_by_layer"].items()}

