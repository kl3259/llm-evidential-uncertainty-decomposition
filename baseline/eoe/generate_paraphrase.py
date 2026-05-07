from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_PARAPHRASE_MODEL = "eugenesiow/bart-paraphrase"


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


@dataclass
class ParaphraseConfig:
    model_name: str = DEFAULT_PARAPHRASE_MODEL
    max_new_tokens: int = 64
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 120
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    batch_size: int = 4
    num_return_sequences: int = 8
    max_attempt_multiplier: int = 12


class BartParaphraser:
    def __init__(self, config: ParaphraseConfig, device: str | None = None):
        self.config = config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _fork_rng_devices(self) -> List[int]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return []
        if self.device.index is not None:
            return [int(self.device.index)]
        return [torch.cuda.current_device()]

    def _add_unique(
        self,
        pool: List[str],
        normalized_to_text: Dict[str, str],
        text: str,
    ) -> None:
        clean = _normalize_text(text)
        if not clean:
            return
        key = clean.lower()
        if key in normalized_to_text:
            return
        normalized_to_text[key] = clean
        pool.append(clean)

    def generate_pool(
        self,
        question: str,
        pool_size: int = 500,
        include_original: bool = True,
        seed: int = 42,
    ) -> List[str]:
        if pool_size <= 0:
            return []

        question_clean = _normalize_text(question)
        paraphrase_pool: List[str] = []
        normalized_to_text: Dict[str, str] = {}
        if include_original:
            self._add_unique(paraphrase_pool, normalized_to_text, question_clean)
            if len(paraphrase_pool) >= pool_size:
                return paraphrase_pool[:pool_size]

        max_attempts = max(
            8,
            (pool_size * max(1, self.config.max_attempt_multiplier))
            // max(1, self.config.batch_size * self.config.num_return_sequences),
        )
        with torch.random.fork_rng(devices=self._fork_rng_devices()):
            torch.manual_seed(int(seed))

            attempts = 0
            while len(paraphrase_pool) < pool_size and attempts < max_attempts:
                attempts += 1
                prompts = [f"paraphrase: {question_clean} </s>"] * max(1, self.config.batch_size)
                encoded = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **encoded,
                        do_sample=self.config.do_sample,
                        temperature=float(self.config.temperature),
                        top_k=int(self.config.top_k),
                        top_p=float(self.config.top_p),
                        repetition_penalty=float(self.config.repetition_penalty),
                        max_new_tokens=int(self.config.max_new_tokens),
                        num_beams=8,
                        num_return_sequences=int(self.config.num_return_sequences),
                    )
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                before = len(paraphrase_pool)
                for text in decoded:
                    self._add_unique(paraphrase_pool, normalized_to_text, text)
                    if len(paraphrase_pool) >= pool_size:
                        break
                if len(paraphrase_pool) == before and attempts >= 2:
                    # Stop early if we stopped getting any new paraphrases.
                    break

        if include_original:
            self._add_unique(paraphrase_pool, normalized_to_text, question_clean)

        return paraphrase_pool[:pool_size]
