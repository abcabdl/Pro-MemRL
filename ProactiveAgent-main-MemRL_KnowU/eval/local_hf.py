from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype: str) -> torch.dtype | str:
    norm = dtype.strip().lower()
    if norm in {"auto", ""}:
        return "auto"
    if norm in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if norm in {"fp16", "float16"}:
        return torch.float16
    if norm in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


class LocalHFGenerator:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer_name_or_path: str | None,
        dtype: str,
        trust_remote_code: bool,
        attn_implementation: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        use_fast: bool,
    ) -> None:
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(dtype),
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def _build_prompt_inputs(self, messages: Sequence[dict[str, str]]) -> dict[str, torch.Tensor]:
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        # Qwen3-family chat templates may enable reasoning by default.
        # Disable it when supported so the model emits plain JSON.
        try:
            inputs = self.tokenizer.apply_chat_template(
                list(messages),
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                list(messages),
                **template_kwargs,
            )
        return {key: value.to(self.model.device) for key, value in inputs.items()}

    def generate_json(self, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
        raw_text = self.generate_text(messages)
        return self._extract_json_payload(raw_text)

    def generate_text(self, messages: Sequence[dict[str, str]]) -> str:
        inputs = self._build_prompt_inputs(messages)
        prompt_length = inputs["input_ids"].shape[-1]
        do_sample = self.temperature > 0.0
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(self.temperature, 1e-5)
            generate_kwargs["top_p"] = self.top_p

        generation = self.model.generate(
            **generate_kwargs,
        )
        output_ids = generation[0][prompt_length:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def score_continuation(
        self,
        messages: Sequence[dict[str, str]],
        continuation: str,
        *,
        normalize_by_length: bool = True,
    ) -> dict[str, float]:
        prompt_inputs = self._build_prompt_inputs(messages)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        cont_ids = self.tokenizer(
            continuation,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.model.device)

        if cont_ids.numel() == 0:
            raise ValueError("Continuation must contain at least one token.")

        input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
        attention_mask = torch.cat(
            [prompt_mask, torch.ones_like(cont_ids, device=self.model.device)],
            dim=1,
        )
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        token_count = int(cont_ids.shape[1])
        avg_nll = float(output.loss.item())
        total_nll = avg_nll * float(token_count)
        score = avg_nll if normalize_by_length else total_nll
        return {
            "score": float(score),
            "avg_nll": float(avg_nll),
            "total_nll": float(total_nll),
            "ppl": float(math.exp(min(avg_nll, 20.0))),
            "token_count": float(token_count),
        }

    def score_label_options(
        self,
        messages: Sequence[dict[str, str]],
        options: Sequence[str],
    ) -> list[dict[str, float | str]]:
        if not options:
            return []

        scored: list[dict[str, float | str]] = []
        raw_scores: list[float] = []
        for option in options:
            metrics = self.score_continuation(messages, option, normalize_by_length=True)
            raw_scores.append(-float(metrics["score"]))
            scored.append({"label": option, **metrics})

        max_score = max(raw_scores)
        weights = [math.exp(score - max_score) for score in raw_scores]
        total = max(1e-12, sum(weights))
        for item, weight in zip(scored, weights):
            item["prob"] = float(weight / total)
        return scored

    @staticmethod
    def _extract_json_payload(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE)
        candidates: list[str] = []

        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            candidates.append(fence_match.group(1).strip())
        candidates.append(text)

        decoder = json.JSONDecoder()
        for candidate in candidates:
            for idx, ch in enumerate(candidate):
                if ch != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(candidate[idx:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload

        preview = text[:200].replace("\n", "\\n")
        raise ValueError(f"Model output must contain a JSON object. Raw preview: {preview}")
