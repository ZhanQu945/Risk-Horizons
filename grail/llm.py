import os
from typing import Dict, List, Tuple

import torch


class LLMReranker:
    def __init__(self, backend: str = "none", model_name: str = None, api_key_env: str = "OPENAI_API_KEY"):
        self.backend = backend
        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env)
        self._tokenizer = None
        self._model = None

    def _load_hf(self):
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers not installed") from exc
        model_name = self.model_name or "meta-llama/Llama-3.1-8B-Instruct"
        cache_dir = os.environ.get("TRANSFORMERS_CACHE")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", cache_dir=cache_dir, token=hf_token
        )
        self._model.eval()

    def score(self, prompt: str, candidates: List[str]) -> List[float]:
        if self.backend == "none":
            return [0.0 for _ in candidates]
        if self.backend == "heuristic":
            prompt_lower = prompt.lower()
            return [1.0 if cand.lower() in prompt_lower else 0.0 for cand in candidates]
        if self.backend == "hf":
            self._load_hf()
            scores = []
            for cand in candidates:
                full_prompt = f"{prompt}\n\nNext event: "
                prompt_ids = self._tokenizer(full_prompt, return_tensors="pt")
                cand_ids = self._tokenizer(cand, add_special_tokens=False, return_tensors="pt")
                input_ids = torch.cat([prompt_ids["input_ids"], cand_ids["input_ids"]], dim=-1).to(self._model.device)
                labels = input_ids.clone()
                labels[:, : prompt_ids["input_ids"].shape[1]] = -100
                with torch.no_grad():
                    outputs = self._model(input_ids=input_ids, labels=labels)
                scores.append(float(-outputs.loss.item()))
            return scores
        if self.backend == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package not installed") from exc
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            client = OpenAI(api_key=self.api_key)
            scores = []
            for cand in candidates:
                response = client.responses.create(
                    model=self.model_name or "gpt-4.1-mini",
                    input=f"{prompt}\n\nCandidate: {cand}\nScore likelihood of being next event (0-1).",
                )
                try:
                    text = response.output[0].content[0].text
                    scores.append(float(text.strip()))
                except Exception:
                    scores.append(0.0)
            return scores
        raise ValueError(f"Unknown backend: {self.backend}")
