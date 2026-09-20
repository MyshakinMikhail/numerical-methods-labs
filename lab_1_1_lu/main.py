"""Запуск лабораторной 1.1."""

from __future__ import annotations

import sys
from pathlib import Path

from common.formatting import format_matrix, format_number, format_vector
from common.io_utils import load_json, require_square_matrix, require_vector
from common.matrix_utils import identity, matmul, max_abs_matrix_difference, residual_inf_norm
from lab_1_1_lu.lu import determinant_from_lup, inverse_from_lup, lup_decompose, solve_lup


def run(path: str | Path) -> None:
    data = load_json(path)
    matrix = require_square_matrix(data.get("matrix"))
    rhs = require_vector(data.get("rhs"), len(matrix), "rhs")

    decomposition = lup_decompose(matrix)
    solution = solve_lup(decomposition, rhs)
    inverse = inverse_from_lup(decomposition)
    lu_product = matmul(decomposition.l, decomposition.u)
    pa_product = matmul(decomposition.p, matrix)
    inverse_check = matmul(matrix, inverse)

    print("Матрица L:")
    print(format_matrix(decomposition.l))
    print("\nМатрица U:")
    print(format_matrix(decomposition.u))
    print("\nМатрица перестановок P:")
    print(format_matrix(decomposition.p))
    print("\nL · U:")
    print(format_matrix(lu_product))
    print("\nP · A:")
    print(format_matrix(pa_product))
    print(f"\nmax|L·U - P·A| = {format_number(max_abs_matrix_difference(lu_product, pa_product))}")
    print(f"\nРешение x = {format_vector(solution)}")
    print(f"Норма невязки ||Ax-b||∞ = {format_number(residual_inf_norm(matrix, solution, rhs))}")
    print("\nОбратная матрица A⁻¹:")
    print(format_matrix(inverse))
    print("\nПроверка A · A⁻¹:")
    print(format_matrix(inverse_check))
    print(
        "max|A·A⁻¹ - E| = "
        + format_number(max_abs_matrix_difference(inverse_check, identity(len(matrix))))
    )
    print(f"\nОпределитель det(A) = {format_number(determinant_from_lup(decomposition))}")


def main() -> None:
    default_path = Path(__file__).with_name("input.json")
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else default_path)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Ошибка: {error}") from error


if __name__ == "__main__":
    main()

