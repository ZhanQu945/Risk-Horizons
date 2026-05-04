import os
import json
import re
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


class LLMReranker:
    def __init__(
        self,
        backend: str = "none",
        model_name: str = None,
        api_key_env: str = "OPENAI_API_KEY",
        scoring_mode: str = "pointwise",
    ):
        self.backend = backend
        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env)
        self.scoring_mode = str(scoring_mode)
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
        if self._tokenizer.pad_token is None:
            # Required for batched candidate scoring with padding=True.
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype_name = str(os.environ.get("GRAIL_HF_DTYPE", "bfloat16")).lower()
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            cache_dir=cache_dir,
            token=hf_token,
            torch_dtype=dtype,
        )
        self._model.eval()

    def _hf_pairwise_scores(self, prompt: str, candidates: List[str], batched: bool = True) -> List[float]:
        if not candidates:
            return []
        full_prompt = f"{prompt}\n\nCurrent candidate: "
        prompt_tok = self._tokenizer(full_prompt, return_tensors="pt")
        prompt_ids = prompt_tok["input_ids"].to(self._model.device)
        prompt_attn = prompt_tok["attention_mask"].to(self._model.device)
        p_len = int(prompt_ids.shape[1])

        batch_size = int(os.environ.get("GRAIL_HF_SCORE_BATCH_SIZE", "4")) if batched else 1
        scores: List[float] = []
        start = 0
        while start < len(candidates):
            cur_bs = min(batch_size, len(candidates) - start)
            batch = candidates[start:start + cur_bs]
            try:
                cand_tok = self._tokenizer(batch, add_special_tokens=False, return_tensors="pt", padding=True)
                cand_ids = cand_tok["input_ids"].to(self._model.device)
                cand_attn = cand_tok["attention_mask"].to(self._model.device)
                bsz = cand_ids.size(0)

                rep_prompt_ids = prompt_ids.expand(bsz, -1)
                rep_prompt_attn = prompt_attn.expand(bsz, -1)
                input_ids = torch.cat([rep_prompt_ids, cand_ids], dim=1)
                attention_mask = torch.cat([rep_prompt_attn, cand_attn], dim=1)

                labels = input_ids.clone()
                labels[:, :p_len] = -100
                labels[:, p_len:][cand_attn == 0] = -100

                with torch.no_grad():
                    logits = self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.size(0), shift_labels.size(1))
                valid = (shift_labels != -100).float()
                seq_nll = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                scores.extend([float(-x.item()) for x in seq_nll])
                start += cur_bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs == 1:
                    raise
                batch_size = max(1, cur_bs // 2)
        return scores

    def _parse_listwise_scores(self, text: str, n: int) -> List[float]:
        # Accept either JSON list or plain numeric list.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "scores" in obj:
                vals = obj["scores"]
            else:
                vals = obj
            if isinstance(vals, list):
                out = []
                for x in vals[:n]:
                    try:
                        out.append(float(x))
                    except Exception:
                        out.append(0.0)
                if len(out) < n:
                    out.extend([0.0] * (n - len(out)))
                return out[:n]
        except Exception:
            pass
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        out = [float(x) for x in nums[:n]]
        if len(out) < n:
            out.extend([0.0] * (n - len(out)))
        return out[:n]

    def _hf_listwise_scores(self, prompt: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        list_lines = [f"{i+1}. {c}" for i, c in enumerate(candidates)]
        prompt_all = (
            f"{prompt}\n\nCandidate list:\n"
            + "\n".join(list_lines)
            + "\n\nOutput format: JSON array of "
            + str(len(candidates))
            + " floating-point scores in [0,1], ordered by candidate index."
            + "\nReturn ONLY the JSON array."
        )
        tok = self._tokenizer(prompt_all, return_tensors="pt").to(self._model.device)
        max_new = int(os.environ.get("GRAIL_HF_LISTWISE_MAX_NEW_TOKENS", "256"))
        with torch.no_grad():
            out = self._model.generate(
                **tok,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = out[0][tok["input_ids"].shape[1]:]
        txt = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        scores = self._parse_listwise_scores(txt, len(candidates))
        # Fallback to robust pairwise when listwise output is malformed/uniformly empty.
        if len(scores) != len(candidates) or sum(abs(s) for s in scores) == 0.0:
            return self._hf_pairwise_scores(prompt, candidates, batched=True)
        return scores

    def score(self, prompt: str, candidates: List[str]) -> List[float]:
        if self.backend == "none":
            return [0.0 for _ in candidates]
        if self.backend == "heuristic":
            prompt_lower = prompt.lower()
            return [1.0 if cand.lower() in prompt_lower else 0.0 for cand in candidates]
        if self.backend == "hf":
            self._load_hf()
            mode = self.scoring_mode
            if mode == "pointwise":
                return self._hf_pairwise_scores(prompt, candidates, batched=False)
            if mode == "batched_pairwise":
                return self._hf_pairwise_scores(prompt, candidates, batched=True)
            if mode == "listwise":
                return self._hf_listwise_scores(prompt, candidates)
            raise ValueError(f"Unknown llm scoring mode: {mode}")
        if self.backend == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package not installed") from exc
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            client = OpenAI(api_key=self.api_key)
            if self.scoring_mode == "listwise" and candidates:
                list_lines = [f"{i+1}. {c}" for i, c in enumerate(candidates)]
                req = (
                    f"{prompt}\n\nCandidate list:\n"
                    + "\n".join(list_lines)
                    + "\n\nReturn ONLY a JSON array of "
                    + str(len(candidates))
                    + " float scores in [0,1], ordered by candidate index."
                )
                response = client.responses.create(
                    model=self.model_name or "gpt-4.1-mini",
                    input=req,
                )
                txt = ""
                try:
                    txt = response.output[0].content[0].text
                except Exception:
                    txt = ""
                return self._parse_listwise_scores(txt, len(candidates))
            scores = []
            for cand in candidates:
                response = client.responses.create(
                    model=self.model_name or "gpt-4.1-mini",
                    input=f"{prompt}\n\nCurrent candidate: {cand}\nScore:",
                )
                try:
                    text = response.output[0].content[0].text
                    scores.append(float(text.strip()))
                except Exception:
                    scores.append(0.0)
            return scores
        raise ValueError(f"Unknown backend: {self.backend}")
