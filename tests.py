import unittest

from common import identity, matmul, max_matrix_difference, residual_norm
from lab1_lu import inverse, lup, permutation_matrix, solve_lup
from lab2_thomas import thomas, tridiagonal_residual
from lab3_iterations import iteration_form, jacobi, seidel
from lab4_jacobi import jacobi_eigen
from lab5_qr import qr_decompose, qr_eigenvalues


class NumericalMethodsTests(unittest.TestCase):
    def test_lup(self):
        matrix = [[-1, -3, -4, 0], [3, 7, -8, 3], [1, -6, 2, 5], [-8, -4, -1, -1]]
        rhs = [-3, 30, -90, 12]
        lower, upper, permutation, _ = lup(matrix)
        solution = solve_lup(lower, upper, permutation, rhs)

        self.assertLess(max_matrix_difference(matmul(lower, upper), matmul(permutation_matrix(permutation), matrix)), 1e-9)
        self.assertLess(residual_norm(matrix, solution, rhs), 1e-9)
        self.assertLess(max_matrix_difference(matmul(matrix, inverse(lower, upper, permutation)), identity(4)), 1e-9)

    def test_thomas(self):
        lower, diagonal, upper = [7, -9, 7, -4], [-1, -17, 19, -20, 12], [-1, -8, 8, 4]
        rhs = [-4, 132, -59, -193, -40]
        solution, _, _ = thomas(lower, diagonal, upper, rhs)

        self.assertEqual([round(value) for value in solution], [6, -2, -7, 7, -1])
        self.assertLess(tridiagonal_residual(lower, diagonal, upper, solution, rhs), 1e-9)

    def test_iterative_methods(self):
        matrix = [[-22, -2, -6, 6], [3, -17, -3, 7], [2, 6, -17, 5], [-1, -8, 8, 23]]
        rhs = [96, -26, 35, -234]
        alpha, beta = iteration_form(matrix, rhs)
        simple, _, _ = jacobi(alpha, beta, 1e-5, 10000)
        gauss_seidel, _, _ = seidel(alpha, beta, 1e-5, 10000)

        self.assertTrue(all(alpha[index][index] == 0 for index in range(4)))
        self.assertLess(residual_norm(matrix, simple, rhs), 1e-3)
        self.assertLess(residual_norm(matrix, gauss_seidel, rhs), 1e-3)

    def test_jacobi_eigen(self):
        matrix = [[-7, -5, -9], [-5, 5, 2], [-9, 2, 9]]
        values, vectors, _, _ = jacobi_eigen(matrix, 1e-8, 10000)
        diagonal = [[values[row] if row == column else 0.0 for column in range(3)] for row in range(3)]

        self.assertLess(max_matrix_difference(matmul(matrix, vectors), matmul(vectors, diagonal)), 1e-6)

    def test_qr(self):
        matrix = [[2, -4, 5], [-5, -2, -3], [1, -8, -3]]
        orthogonal, upper = qr_decompose(matrix)
        values, _, _ = qr_eigenvalues(matrix, 1e-8, 10000)

        self.assertLess(max_matrix_difference(matmul(orthogonal, upper), matrix), 1e-9)
        self.assertTrue(all(abs(value**3 + 3 * value**2 - 53 * value - 246) < 1e-4 for value in values))

    def test_qr_complex_pair(self):
        values, _, _ = qr_eigenvalues([[0, -1, 0], [1, 0, 0], [0, 0, 2]], 1e-8, 100)

        self.assertTrue(any(abs(value - 2) < 1e-8 for value in values))
        self.assertTrue(any(abs(value - 1j) < 1e-8 for value in values))
        self.assertTrue(any(abs(value + 1j) < 1e-8 for value in values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
