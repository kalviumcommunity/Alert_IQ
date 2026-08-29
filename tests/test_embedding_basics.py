"""Unit tests for embedding fundamentals and vector similarity."""
import unittest

from src.embedding_basics import cosine_similarity, compare_text_pairs


class TestEmbeddingBasics(unittest.TestCase):
    def test_embedding_dimension_is_consistent(self):
        embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
        self.assertEqual(len(embeddings[0]), 3)
        self.assertEqual(len(embeddings[0]), len(embeddings[1]))

    def test_cosine_similarity_identical_vectors_is_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_similar_pair_scores_higher_than_dissimilar_pair(self):
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = compare_text_pairs(embeddings)
        self.assertGreater(result["similarity_score"], result["dissimilarity_score"])
        self.assertTrue(result["similar_pair_scores_higher"])

    def test_mismatched_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_zero_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
