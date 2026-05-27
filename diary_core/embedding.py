from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


BatchProgress = Callable[[int, int, int, int], None]


def default_device(torch_module) -> str:
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def default_dtype(torch_module, device: str):
    return torch_module.float16 if device in {"cuda", "mps"} else torch_module.float32


def last_token_pool(last_hidden_states, attention_mask):
    """Pool Qwen-style embeddings from the final valid token."""
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def split_token_ids(
    token_ids: list[int], max_tokens: int, chunk_overlap: int = 0
) -> list[list[int]]:
    if len(token_ids) <= max_tokens:
        return [token_ids]
    if chunk_overlap >= max_tokens:
        raise ValueError("chunk_overlap must be smaller than max_tokens")

    chunks = []
    start = 0
    step = max_tokens - chunk_overlap
    while start < len(token_ids):
        chunks.append(token_ids[start : start + max_tokens])
        if start + max_tokens >= len(token_ids):
            break
        start += step
    return chunks


class QwenEmbedder:
    """Small reusable wrapper for local Qwen embedding inference.

    Defaults are intentionally conservative for a 16GB MacBook Pro: MPS uses
    float16, batches are controlled by the caller, and MPS cache is cleared
    after each batch.
    """

    def __init__(self, model_dir: str | Path, device: str | None = None, torch_dtype=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.device = device or default_device(torch)
        self.torch_dtype = torch_dtype if torch_dtype is not None else default_dtype(torch, self.device)

        print(f"Loading embedding model on {self.device}: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, padding_side="left")
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

    def clear_cache(self) -> None:
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()

    def embed_texts(
        self,
        texts: list[str],
        *,
        max_tokens: int,
        batch_size: int,
        sort_by_length: bool = False,
        progress_label: str | None = None,
        progress_start: int = 0,
        progress_total: int | None = None,
        on_batch: BatchProgress | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        indexed = list(enumerate(texts))
        if sort_by_length:
            indexed.sort(key=lambda item: len(item[1]))

        vectors: list[np.ndarray | None] = [None] * len(texts)
        total = progress_total if progress_total is not None else len(texts)
        processed = 0

        for start in range(0, len(indexed), batch_size):
            batch_items = indexed[start : start + batch_size]
            batch = [text for _, text in batch_items]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            ).to(self.device)

            with self.torch.no_grad():
                outputs = self.model(**encoded)

            embs = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embs = self.torch.nn.functional.normalize(embs, p=2, dim=1)
            embs_np = embs.cpu().float().numpy()

            for (orig_idx, _), vec in zip(batch_items, embs_np):
                vectors[orig_idx] = vec

            processed += len(batch_items)
            token_count = int(encoded["input_ids"].shape[1])
            if on_batch is not None:
                on_batch(progress_start + processed, total, len(batch_items), token_count)
            elif progress_label:
                print(
                    f"Embedded {progress_label}: {progress_start + processed}/{total} "
                    f"(batch={len(batch_items)}, tokens={token_count})"
                )

            self.clear_cache()

        return np.vstack(vectors).astype(np.float32)

    def embed_token_chunks(self, token_chunks: list[list[int]]) -> np.ndarray:
        if not token_chunks:
            return np.empty((0, 0), dtype=np.float32)

        encoded = self.tokenizer.pad(
            {"input_ids": token_chunks},
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**encoded)

        embs = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embs = self.torch.nn.functional.normalize(embs, p=2, dim=1)
        out = embs.cpu().float().numpy()
        self.clear_cache()
        return out.astype(np.float32)

    def embed_long_texts(
        self,
        texts: list[str],
        *,
        max_tokens: int,
        chunk_overlap: int,
        chunk_batch_size: int = 1,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            token_ids = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            chunks = split_token_ids(token_ids, max_tokens, chunk_overlap)
            chunk_parts = []
            for start in range(0, len(chunks), chunk_batch_size):
                chunk_parts.append(self.embed_token_chunks(chunks[start : start + chunk_batch_size]))
            chunk_vecs = np.vstack(chunk_parts)

            if len(chunk_vecs) == 1:
                vectors.append(chunk_vecs[0])
            else:
                weights = np.asarray([len(chunk) for chunk in chunks], dtype=np.float32)
                vec = np.average(chunk_vecs, axis=0, weights=weights)
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                vectors.append(vec.astype(np.float32))

        return np.vstack(vectors).astype(np.float32)
