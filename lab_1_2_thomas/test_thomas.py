import unittest

from lab_1_2_thomas.thomas import thomas_solve, tridiagonal_residual_inf_norm


LOWER = [7.0, -9.0, 7.0, -4.0]
DIAGONAL = [-1.0, -17.0, 19.0, -20.0, 12.0]
UPPER = [-1.0, -8.0, 8.0, 4.0]
RHS = [-4.0, 132.0, -59.0, -193.0, -40.0]


class ThomasTests(unittest.TestCase):
    def test_variant_is_solved_from_three_diagonals(self) -> None:
        result = thomas_solve(LOWER, DIAGONAL, UPPER, RHS)
        self.assertEqual(len(result.alpha), 5)
        self.assertEqual(len(result.beta), 5)
        self.assertLess(
            tridiagonal_residual_inf_norm(
                LOWER, DIAGONAL, UPPER, result.solution, RHS
            ),
            1e-9,
        )

    def test_known_three_by_three_system(self) -> None:
        result = thomas_solve([1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0], [5.0, 6.0, 5.0])
        for actual, expected in zip(result.solution, [2.0, 1.0, 2.0]):
            self.assertAlmostEqual(actual, expected)

    def test_one_equation(self) -> None:
        result = thomas_solve([], [4.0], [], [12.0])
        self.assertEqual(result.solution, [3.0])
        self.assertEqual(result.alpha, [0.0])

    def test_invalid_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "длины диагоналей"):
            thomas_solve([], [1.0, 2.0], [], [1.0, 2.0])

    def test_zero_denominator_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "(?i)нулевой знаменатель"):
            thomas_solve([], [0.0], [], [1.0])


if __name__ == "__main__":
    unittest.main()
