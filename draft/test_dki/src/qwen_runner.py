import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image


@dataclass
class GenerationResult:
    text: str
    elapsed_sec: float


class QwenVLRunner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        max_pixels: int = 512 * 28 * 28,
    ) -> None:
        dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)

    @staticmethod
    def _messages_with_list_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Qwen-VL expects multimodal `content` as a list of blocks; string content breaks path checks."""
        out: List[Dict[str, Any]] = []
        for msg in messages:
            c = msg.get("content")
            if isinstance(c, str):
                out.append({**msg, "content": [{"type": "text", "text": c}]})
            else:
                out.append(msg)
        return out

    @staticmethod
    def _check_paths(messages: List[Dict[str, Any]]) -> None:
        for msg in messages:
            for item in msg.get("content", []):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    img_path = item.get("image")
                    if img_path and isinstance(img_path, str):
                        if not os.path.exists(img_path):
                            raise FileNotFoundError(f"Image not found: {img_path}")

    @staticmethod
    def _render_pdf_first_page(pdf_path: str) -> Image.Image:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ImportError(
                "PDF input detected but pypdfium2 is not installed. "
                "Install with: pip install pypdfium2"
            ) from exc
        doc = pdfium.PdfDocument(pdf_path)
        page = doc[0]
        # Scale 2.0 for better OCR readability.
        pil_image = page.render(scale=2.0).to_pil()
        page.close()
        doc.close()
        return pil_image.convert("RGB")

    def _normalize_media_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            new_content: List[Dict[str, Any]] = []
            for item in msg.get("content", []):
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "image":
                    new_content.append(item)
                    continue
                img = item.get("image")
                if isinstance(img, str) and img.lower().endswith(".pdf"):
                    copied = dict(item)
                    copied["image"] = self._render_pdf_first_page(img)
                    new_content.append(copied)
                else:
                    new_content.append(item)
            copied_msg = dict(msg)
            copied_msg["content"] = new_content
            normalized.append(copied_msg)
        return normalized

    def _prepare_inputs(self, messages: List[Dict[str, Any]], add_generation_prompt: bool):
        messages = self._messages_with_list_content(messages)
        self._check_paths(messages)
        messages = self._normalize_media_messages(messages)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128) -> GenerationResult:
        import time

        inputs = self._prepare_inputs(messages, add_generation_prompt=True)
        start = time.time()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        elapsed = time.time() - start
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return GenerationResult(text=text, elapsed_sec=elapsed)

    @torch.no_grad()
    def forward_teacher_forcing(self, messages: List[Dict[str, Any]], output_attentions: bool = True):
        inputs = self._prepare_inputs(messages, add_generation_prompt=False)
        outputs = self.model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return inputs, outputs
