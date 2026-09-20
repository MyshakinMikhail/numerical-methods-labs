import unittest

from common.matrix_utils import identity, matmul, max_abs_matrix_difference, transpose
from lab_1_5_qr.qr import householder_qr, qr_eigenvalues


VARIANT_A = [
    [2.0, -4.0, 5.0],
    [-5.0, -2.0, -3.0],
    [1.0, -8.0, -3.0],
]


class QRTests(unittest.TestCase):
    def test_householder_reconstructs_variant(self) -> None:
        result = householder_qr(VARIANT_A)
        self.assertLess(max_abs_matrix_difference(matmul(result.q, result.r), VARIANT_A), 1e-9)
        self.assertLess(
            max_abs_matrix_difference(matmul(transpose(result.q), result.q), identity(3)),
            1e-9,
        )

    def test_complex_block_returns_conjugate_pair(self) -> None:
        matrix = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        values = qr_eigenvalues(matrix, 0.01, 10_000).eigenvalues
        self.assertTrue(any(abs(value - 1j) < 0.01 for value in values))
        self.assertTrue(any(abs(value + 1j) < 0.01 for value in values))
        self.assertTrue(any(abs(value - 2.0) < 0.01 for value in values))

    def test_variant_values_satisfy_characteristic_equation(self) -> None:
        values = qr_eigenvalues(VARIANT_A, 0.01, 10_000).eigenvalues
        self.assertEqual(len(values), 3)
        for value in values:
            self.assertLess(abs(value**3 + 3 * value**2 - 53 * value - 246), 0.5)

    def test_one_by_one_matrix(self) -> None:
        decomposition = householder_qr([[5.0]])
        self.assertEqual(decomposition.q, [[1.0]])
        self.assertEqual(decomposition.r, [[5.0]])
        self.assertEqual(qr_eigenvalues([[5.0]], 0.01, 10).eigenvalues, [5.0])

    def test_iteration_limit_is_reported(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "не сошёлся"):
            qr_eigenvalues(VARIANT_A, 1e-14, 1)


if __name__ == "__main__":
    unittest.main()
