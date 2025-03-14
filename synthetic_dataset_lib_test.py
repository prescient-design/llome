import pprint
import torch
import unittest

from synthetic_dataset_lib import format_fewshot, ranked_fft


class TestFarthestFirstTraversal(unittest.TestCase):
    def test_ranked_fft_euclidean(self):
        library = torch.Tensor(
            [
                [1, 2, 3],  # lowest score, will be selected first
                [1, 2, 4],
                [3, 2, 3],  # furthest from the first row, will be selected 2nd
            ]
        )
        scores = torch.Tensor([-1.0, -0.5, 0.01])
        n = 2
        selected_idx = ranked_fft(library, scores, n=n)
        selected_idx = selected_idx.tolist()
        self.assertEqual(selected_idx, [0, 2])

    def test_ranked_fft_manhattan(self):
        library = torch.Tensor(
            [
                [3, 4, 3],  # furthest by l1 distance, not by l2 distance
                [1, 2, 3],  # lowest score, will be selected first
                [3.75, 2, 2],
            ]
        )
        scores = torch.Tensor([-0.5, -1.0, 0.01])
        n = 2
        distance_fn = lambda x, y: torch.norm(x - y, p=1).item()
        selected_idx = ranked_fft(library, scores, n=n, distance_fn=distance_fn)
        selected_idx = selected_idx.tolist()
        self.assertEqual(selected_idx, [1, 0])

    def test_ranked_fft_descending(self):
        library = torch.Tensor(
            [
                [3, 4, 3],  # furthest by l1 distance, not by l2 distance
                [1, 2, 3],  # highest score, will be selected first
                [3.75, 2, 2],  # furthest by l2 distance
            ]
        )
        scores = torch.Tensor([-0.1, 1.0, 0.01])
        n = 2
        selected_idx = ranked_fft(library, scores, n=n, descending=True)
        selected_idx = selected_idx.tolist()
        self.assertEqual(selected_idx, [1, 2])

class TestFormatting(unittest.TestCase):
    def test_format_fewshot_disallow_same_score(self):
        X = torch.FloatTensor([
            [1.0, 4.0, 2.0, 3.0],
            [2.0, 1.0, 3.0, 5.0],
            [3.0, 3.0, 2.0, 4.0],
        ])
        scores = torch.FloatTensor([-0.5, 0.25, -0.5])
        ks = [1]  # only do one-shot
        outputs = format_fewshot(X, scores, ks, allow_same_score=False, seed=0)
        self.assertEqual(X.shape[0], len(outputs))
        for o in outputs:
            self.assertEqual(o["input"].count("Score: "), ks[0] + 1)
            self.assertEqual(o["input"].count("Particle: "), ks[0] + 1)
        self.assertEqual(outputs[0]["input"], "Score: 0.250\nParticle: [2, 1, 3, 5]\n\nScore: -0.500\nParticle: ")
        self.assertEqual(outputs[0]["target"], "[1, 4, 2, 3]")
        self.assertEqual(outputs[1]["target"], "[2, 1, 3, 5]")
        self.assertEqual(outputs[2]["input"], "Score: 0.250\nParticle: [2, 1, 3, 5]\n\nScore: -0.500\nParticle: ")
        self.assertEqual(outputs[2]["target"], "[3, 3, 2, 4]")


if __name__ == "__main__":
    unittest.main()
