import json
import tempfile
import unittest
from pathlib import Path

from common.formatting import format_matrix, format_number, format_vector
from common.io_utils import (
    load_json,
    require_positive_float,
    require_positive_int,
    require_square_matrix,
    require_vector,
)
from common.matrix_utils import (
    identity,
    is_symmetric,
    matmul,
    matrix_subtract,
    matvec,
    max_abs_matrix,
    max_abs_matrix_difference,
    off_diagonal_norm,
    residual_inf_norm,
    transpose,
    vector_inf_norm,
)


class CommonUtilitiesTests(unittest.TestCase):
    def test_matrix_multiplication_and_transpose(self) -> None:
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(matmul(matrix, identity(2)), matrix)
        self.assertEqual(transpose(matrix), [[1.0, 3.0], [2.0, 4.0]])

    def test_rejects_empty_and_ragged_matrices(self) -> None:
        with self.assertRaisesRegex(ValueError, "не должна быть пустой"):
            require_square_matrix([], "matrix")
        with self.assertRaisesRegex(ValueError, "одинаковую длину"):
            require_square_matrix([[1, 2], [3]], "matrix")

    def test_load_json_requires_object_at_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON должен быть объектом"):
                load_json(path)

    def test_validates_vectors_and_positive_numbers(self) -> None:
        self.assertEqual(require_vector([1, 2.5], 2, "rhs"), [1.0, 2.5])
        self.assertEqual(require_positive_float(0.01, "epsilon"), 0.01)
        self.assertEqual(require_positive_int(10, "max_iterations"), 10)
        with self.assertRaises(ValueError):
            require_vector([1], 2, "rhs")
        with self.assertRaises(ValueError):
            require_positive_float(0, "epsilon")
        with self.assertRaises(ValueError):
            require_positive_int(1.5, "max_iterations")

    def test_matrix_helpers_calculate_independent_values(self) -> None:
        matrix = [[2.0, 1.0], [1.0, 3.0]]
        self.assertEqual(matvec(matrix, [2.0, -1.0]), [3.0, -1.0])
        self.assertEqual(matrix_subtract(matrix, identity(2)), [[1.0, 1.0], [1.0, 2.0]])
        self.assertEqual(vector_inf_norm([-3.0, 2.0]), 3.0)
        self.assertEqual(max_abs_matrix(matrix), 3.0)
        self.assertEqual(max_abs_matrix_difference(matrix, identity(2)), 2.0)
        self.assertEqual(residual_inf_norm(matrix, [2.0, -1.0], [3.0, -1.0]), 0.0)
        self.assertTrue(is_symmetric(matrix))
        self.assertAlmostEqual(off_diagonal_norm(matrix), 2.0**0.5)

    def test_formats_complex_number(self) -> None:
        self.assertEqual(format_number(complex(2, -1)), "2.000000 - 1.000000i")
        self.assertEqual(format_vector([1.0, -2.0]), "[1.000000, -2.000000]")
        self.assertEqual(format_matrix([[1.0, 0.0]]), "[1.000000  0.000000]")


if __name__ == "__main__":
    unittest.main()
