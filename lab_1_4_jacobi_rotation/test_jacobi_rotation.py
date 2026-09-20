import unittest

from common.matrix_utils import identity, matmul, max_abs_matrix_difference, transpose
from lab_1_4_jacobi_rotation.jacobi_rotation import jacobi_eigen


VARIANT_A = [
    [-7.0, -5.0, -9.0],
    [-5.0, 5.0, 2.0],
    [-9.0, 2.0, 9.0],
]


class JacobiRotationTests(unittest.TestCase):
    def test_variant_satisfies_eigenpair_identity(self) -> None:
        result = jacobi_eigen(VARIANT_A, 0.01, 10_000)
        diagonal = [
            [result.eigenvalues[row] if row == column else 0.0 for column in range(3)]
            for row in range(3)
        ]
        self.assertLess(
            max_abs_matrix_difference(
                matmul(VARIANT_A, result.eigenvectors),
                matmul(result.eigenvectors, diagonal),
            ),
            0.02,
        )
        self.assertLess(
            max_abs_matrix_difference(
                matmul(transpose(result.eigenvectors), result.eigenvectors), identity(3)
            ),
            1e-9,
        )
        self.assertLessEqual(result.off_diagonal_norm, 0.01)

    def test_known_two_by_two_matrix(self) -> None:
        result = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]], 1e-10, 100)
        self.assertAlmostEqual(result.eigenvalues[0], 1.0)
        self.assertAlmostEqual(result.eigenvalues[1], 3.0)

    def test_one_by_one_matrix(self) -> None:
        result = jacobi_eigen([[7.0]], 0.01, 10)
        self.assertEqual(result.eigenvalues, [7.0])
        self.assertEqual(result.eigenvectors, [[1.0]])

    def test_already_diagonal_matrix_returns_every_eigenpair(self) -> None:
        result = jacobi_eigen(
            [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]],
            0.01,
            10,
        )
        self.assertEqual(result.eigenvalues, [-1.0, 2.0, 3.0])
        self.assertEqual(len(result.eigenvectors), 3)
        self.assertTrue(all(len(row) == 3 for row in result.eigenvectors))

    def test_non_symmetric_matrix_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "симметричной"):
            jacobi_eigen([[1.0, 2.0], [0.0, 1.0]], 0.01, 10)

    def test_iteration_limit_is_reported(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "не сошёлся"):
            jacobi_eigen(VARIANT_A, 1e-15, 1)


if __name__ == "__main__":
    unittest.main()
