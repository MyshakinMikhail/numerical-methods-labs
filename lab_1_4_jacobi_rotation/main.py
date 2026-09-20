"""Запуск лабораторной 1.4."""

from __future__ import annotations

import sys
from pathlib import Path

from common.formatting import format_matrix, format_number, format_vector
from common.io_utils import (
    load_json,
    require_positive_float,
    require_positive_int,
    require_square_matrix,
)
from common.matrix_utils import matmul, max_abs_matrix_difference
from lab_1_4_jacobi_rotation.jacobi_rotation import jacobi_eigen


def run(path: str | Path) -> None:
    data = load_json(path)
    matrix = require_square_matrix(data.get("matrix"))
    epsilon = require_positive_float(data.get("epsilon"), "epsilon")
    max_iterations = require_positive_int(data.get("max_iterations"), "max_iterations")
    result = jacobi_eigen(matrix, epsilon, max_iterations)
    size = len(matrix)
    diagonal = [
        [result.eigenvalues[row] if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    left = matmul(matrix, result.eigenvectors)
    right = matmul(result.eigenvectors, diagonal)

    print(f"Заданная точность ε = {format_number(epsilon)}")
    print(f"Количество итераций: {result.iterations}")
    print(f"Норма внедиагональной части: {format_number(result.off_diagonal_norm)}")
    print(f"\nСобственные значения: {format_vector(result.eigenvalues)}")
    print("\nСобственные векторы:")
    for column in range(size):
        vector = [result.eigenvectors[row][column] for row in range(size)]
        print(f"h{column + 1} = {format_vector(vector)}")
    print("\nМатрица собственных векторов V:")
    print(format_matrix(result.eigenvectors))
    print("\nДиагональная матрица Λ:")
    print(format_matrix(diagonal))
    print("\nA · V:")
    print(format_matrix(left))
    print("\nV · Λ:")
    print(format_matrix(right))
    print(f"\nmax|A·V - V·Λ| = {format_number(max_abs_matrix_difference(left, right))}")


def main() -> None:
    default_path = Path(__file__).with_name("input.json")
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else default_path)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка: {error}") from error


if __name__ == "__main__":
    main()

