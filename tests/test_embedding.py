from __future__ import annotations

import unittest

import numpy as np

from diary_core.embedding import QwenEmbedder, split_token_ids


class SplitTokenIdsTest(unittest.TestCase):
    def test_short_input_is_single_chunk(self) -> None:
        self.assertEqual(split_token_ids([1, 2, 3], max_tokens=5), [[1, 2, 3]])

    def test_long_input_uses_overlap(self) -> None:
        chunks = split_token_ids(list(range(10)), max_tokens=4, chunk_overlap=1)
        self.assertEqual(chunks, [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]])

    def test_overlap_must_be_smaller_than_max_tokens(self) -> None:
        with self.assertRaises(ValueError):
            split_token_ids(list(range(10)), max_tokens=4, chunk_overlap=4)


class QwenEmbedderShapeTest(unittest.TestCase):
    def test_empty_helpers_return_empty_float32_arrays(self) -> None:
        embedder = QwenEmbedder.__new__(QwenEmbedder)

        texts = QwenEmbedder.embed_texts(
            embedder,
            [],
            max_tokens=16,
            batch_size=2,
        )
        chunks = QwenEmbedder.embed_token_chunks(embedder, [])

        self.assertEqual(texts.shape, (0, 0))
        self.assertEqual(chunks.shape, (0, 0))
        self.assertEqual(texts.dtype, np.float32)
        self.assertEqual(chunks.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
