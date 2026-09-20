import unittest

from common.matrix_utils import identity, matmul, max_abs_matrix_difference, residual_inf_norm
from lab_1_1_lu.lu import (
    determinant_from_lup,
    inverse_from_lup,
    lup_decompose,
    solve,
    solve_lup,
)


VARIANT_A = [
    [-1.0, -3.0, -4.0, 0.0],
    [3.0, 7.0, -8.0, 3.0],
    [1.0, -6.0, 2.0, 5.0],
    [-8.0, -4.0, -1.0, -1.0],
]
VARIANT_B = [-3.0, 30.0, -90.0, 12.0]


class LUPTests(unittest.TestCase):
    def test_pivoting_reconstructs_permuted_matrix(self) -> None:
        matrix = [[0.0, 2.0], [1.0, 3.0]]
        result = lup_decompose(matrix)
        self.assertLess(
            max_abs_matrix_difference(matmul(result.l, result.u), matmul(result.p, matrix)),
            1e-10,
        )
        self.assertAlmostEqual(determinant_from_lup(result), -2.0)

    def test_variant_solution_inverse_and_determinant(self) -> None:
        result = lup_decompose(VARIANT_A)
        solution = solve_lup(result, VARIANT_B)
        inverse = inverse_from_lup(result)
        self.assertLess(residual_inf_norm(VARIANT_A, solution, VARIANT_B), 1e-9)
        self.assertLess(max_abs_matrix_difference(matmul(VARIANT_A, inverse), identity(4)), 1e-9)
        self.assertNotEqual(determinant_from_lup(result), 0.0)

    def test_singular_matrix_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "вырождена"):
            lup_decompose([[1.0, 2.0], [2.0, 4.0]])

    def test_one_by_one_matrix(self) -> None:
        self.assertEqual(solve([[4.0]], [8.0]), [2.0])

    def test_rhs_length_is_checked(self) -> None:
        decomposition = lup_decompose([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "правой части"):
            solve_lup(decomposition, [1.0])


if __name__ == "__main__":
    unittest.main()
