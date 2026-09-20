import unittest

from common.matrix_utils import vector_inf_norm
from lab_1_1_lu.lu import solve
from lab_1_3_iterations.iterations import (
    gauss_seidel_method,
    iteration_matrix_inf_norm,
    jacobi_method,
)


VARIANT_A = [
    [-22.0, -2.0, -6.0, 6.0],
    [3.0, -17.0, -3.0, 7.0],
    [2.0, 6.0, -17.0, 5.0],
    [-1.0, -8.0, 8.0, 23.0],
]
VARIANT_B = [96.0, -26.0, 35.0, -234.0]


class IterationTests(unittest.TestCase):
    def test_both_methods_solve_variant(self) -> None:
        reference = solve(VARIANT_A, VARIANT_B)
        jacobi = jacobi_method(VARIANT_A, VARIANT_B, 0.01, 10_000)
        seidel = gauss_seidel_method(VARIANT_A, VARIANT_B, 0.01, 10_000)
        self.assertLessEqual(jacobi.delta_norm, 0.01)
        self.assertLessEqual(seidel.delta_norm, 0.01)
        jacobi_error = vector_inf_norm(
            [actual - expected for actual, expected in zip(jacobi.solution, reference)]
        )
        seidel_error = vector_inf_norm(
            [actual - expected for actual, expected in zip(seidel.solution, reference)]
        )
        self.assertLess(jacobi_error, 0.05)
        self.assertLess(seidel_error, 0.05)

    def test_iteration_matrix_norm_for_dominant_system_is_below_one(self) -> None:
        self.assertLess(iteration_matrix_inf_norm(VARIANT_A), 1.0)

    def test_zero_diagonal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "диагональный элемент"):
            jacobi_method([[0.0, 1.0], [1.0, 2.0]], [1.0, 1.0], 0.01, 10)

    def test_iteration_limit_is_reported(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "не сошёлся"):
            jacobi_method([[1.0, 2.0], [2.0, 1.0]], [1.0, 1.0], 1e-12, 3)

    def test_initial_guess_length_is_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "начального приближения"):
            gauss_seidel_method([[2.0]], [4.0], 0.01, 10, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
