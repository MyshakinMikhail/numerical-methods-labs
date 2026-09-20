"""Запуск лабораторной 1.5."""

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
from lab_1_5_qr.qr import householder_qr, qr_eigenvalues


def read_input(path: str | Path) -> tuple[list[list[float]], float, int]:
    data = load_json(path)
    return (
        require_square_matrix(data.get("matrix")),
        require_positive_float(data.get("epsilon"), "epsilon"),
        require_positive_int(data.get("max_iterations"), "max_iterations"),
    )


def run(path: str | Path) -> None:
    matrix, epsilon, max_iterations = read_input(path)
    decomposition = householder_qr(matrix)
    reconstructed = matmul(decomposition.q, decomposition.r)
    eigen_result = qr_eigenvalues(matrix, epsilon, max_iterations)

    print("Исходная матрица A:")
    print(format_matrix(matrix))
    print("\nМатрица Q:")
    print(format_matrix(decomposition.q))
    print("\nМатрица R:")
    print(format_matrix(decomposition.r))
    print("\nQ · R:")
    print(format_matrix(reconstructed))
    print(f"\nmax|Q·R-A| = {format_number(max_abs_matrix_difference(reconstructed, matrix))}")
    print(f"Заданная точность ε = {format_number(epsilon)}")
    print(f"Количество QR-итераций: {eigen_result.iterations}")
    print("\nИтоговая вещественная квазитреугольная матрица:")
    print(format_matrix(eigen_result.schur_form))
    print(f"\nСобственные значения: {format_vector(eigen_result.eigenvalues)}")


def main() -> None:
    default_path = Path(__file__).with_name("input.json")
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else default_path)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка: {error}") from error


if __name__ == "__main__":
    main()

