"""Jalankan: python -m unittest discover -s tests -v."""
import unittest
import numpy as np
from src.lapis4 import forward_lapis4, update_consequents_lse
from src.anfis import ANFIS


class TestLapis4(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.X = self.rng.uniform(size=(40, 5))
        self.w = self.rng.uniform(size=(40, 32))
        self.w /= self.w.sum(axis=1, keepdims=True)
        self.y = self.rng.uniform(size=40)

    def test_manual_example(self):
        X = [[0.5, 0.25, 0.4, 0.5, 0.0]]
        weights = np.zeros((1, 32))
        weights[0, :2] = [0.6, 0.4]
        params = np.zeros((32, 6))
        params[0] = [0.2, 0.3, -0.1, 0.15, 0.05, 0.1]
        params[1] = [0.1, 0.4, -0.05, 0.2, 0.1, 0.05]
        f, o4 = forward_lapis4(X, weights, params)
        np.testing.assert_allclose(f[0, :2], [0.31, 0.28])
        np.testing.assert_allclose(o4[0, :2], [0.186, 0.112])
        self.assertAlmostEqual(o4.sum(), 0.298)

    def test_ridge_matches_augmented_least_squares(self):
        reg = 0.01
        # Reference assembled independently with explicit per-rule blocks.
        A = np.column_stack([
            self.w[:, r] * feature
            for r in range(32)
            for feature in [*self.X.T, np.ones(40)]
        ])
        reference = np.linalg.lstsq(
            np.vstack([A, np.sqrt(reg) * np.eye(192)]),
            np.r_[self.y, np.zeros(192)], rcond=None
        )[0]
        result = update_consequents_lse(self.X, self.y, self.w, reg)
        np.testing.assert_allclose(result.ravel(), reference, atol=1e-10)
        _, o4 = forward_lapis4(self.X, self.w, result)
        np.testing.assert_allclose(o4.sum(axis=1), A @ reference, atol=1e-10)

    def test_unregularized_singular_design(self):
        X = np.ones((3, 5))
        w = np.full((3, 32), 1 / 32)
        theta = update_consequents_lse(X, np.full(3, 0.7), w, reg=0)
        _, o4 = forward_lapis4(X, w, theta)
        np.testing.assert_allclose(o4.sum(axis=1), 0.7)

    def test_zero_weights_and_no_mutation(self):
        X, w, y = self.X.copy(), self.w.copy(), self.y.copy()
        update_consequents_lse(X, y, w)
        np.testing.assert_array_equal(X, self.X)
        np.testing.assert_array_equal(w, self.w)
        np.testing.assert_array_equal(y, self.y)
        theta = update_consequents_lse(X, y, np.zeros_like(w))
        np.testing.assert_array_equal(theta, np.zeros((32, 6)))

    def test_invalid_inputs(self):
        cases = [
            (self.X[:, :4], self.y, self.w, 0.001),
            (self.X, self.y[:, None], self.w, 0.001),
            (self.X, self.y, self.w[:, :31], 0.001),
            (self.X, self.y, -self.w, 0.001),
            (self.X, self.y, self.w, -1),
            (self.X, self.y, self.w, np.nan),
            (self.X, np.full(40, np.inf), self.w, 0.001),
            (np.empty((0, 5)), np.empty(0), np.empty((0, 32)), 0.001),
        ]
        for args in cases:
            with self.subTest(shapes=[np.shape(a) for a in args]):
                with self.assertRaises(ValueError):
                    update_consequents_lse(*args)
        with self.assertRaises(ValueError):
            forward_lapis4(self.X, self.w, np.zeros((32, 5)))

    def test_real_anfis_hybrid_integration(self):
        model = ANFIS(lr=0.01)
        target = 0.1 + 0.3 * self.X[:, 0] + 0.2 * self.X[:, 1]
        initial = np.sqrt(np.mean(target ** 2))
        history = model.fit(self.X, target, epochs=3)
        self.assertEqual(len(history), 3)
        self.assertTrue(np.all(np.isfinite(history)))
        prediction = model.forward_layers(self.X)[-1]
        self.assertLess(np.sqrt(np.mean((prediction - target) ** 2)), initial)
        self.assertFalse(np.array_equal(model.c, model.init_c))


if __name__ == "__main__":
    unittest.main()
