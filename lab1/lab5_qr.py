"""Лабораторная 1.5: QR-разложение Хаусхолдера и QR-алгоритм."""

from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

from common import (
    Matrix,
    Number,
    format_number,
    identity,
    load_json,
    matmul,
    max_matrix_difference,
    print_matrix,
    print_vector,
    square_matrix,
)


def qr_decompose(matrix: Matrix, tolerance: float = 1e-12) -> tuple[Matrix, Matrix]:
    """Построить A=Q·R отражениями Хаусхолдера."""
    upper = square_matrix(matrix)
    size = len(upper)
    orthogonal = identity(size)

    for column in range(size - 1):
        householder = [upper[row][column] for row in range(column, size)]
        length = math.sqrt(sum(value * value for value in householder))
        if length <= tolerance:
            continue
        householder[0] += math.copysign(length, householder[0])
        norm = math.sqrt(sum(value * value for value in householder))
        householder = [value / norm for value in householder]

        for target_column in range(column, size):
            projection = sum(
                householder[index] * upper[column + index][target_column]
                for index in range(len(householder))
            )
            for index, value in enumerate(householder):
                upper[column + index][target_column] -= 2.0 * value * projection

        for row in range(size):
            projection = sum(
                orthogonal[row][column + index] * householder[index]
                for index in range(len(householder))
            )
            for index, value in enumerate(householder):
                orthogonal[row][column + index] -= 2.0 * projection * value

    return orthogonal, upper


def _quasi_triangular(matrix: Matrix, epsilon: float) -> bool:
    for row in range(len(matrix)):
        for column in range(row - 1):
            if abs(matrix[row][column]) > epsilon:
                return False
    previous_large = False
    for row in range(1, len(matrix)):
        current_large = abs(matrix[row][row - 1]) > epsilon
        if previous_large and current_large:
            return False
        previous_large = current_large
    return True


def _block_values(a: float, b: float, c: float, d: float) -> tuple[Number, Number]:
    trace = a + d
    root = cmath.sqrt(trace * trace - 4.0 * (a * d - b * c))
    values = ((trace + root) / 2.0, (trace - root) / 2.0)
    return tuple(float(value.real) if abs(value.imag) <= 1e-12 else value for value in values)  # type: ignore[return-value]


def _eigenvalues(matrix: Matrix, epsilon: float) -> list[Number]:
    values: list[Number] = []
    index = 0
    while index < len(matrix):
        if index == len(matrix) - 1 or abs(matrix[index + 1][index]) <= epsilon:
            values.append(matrix[index][index])
            index += 1
        else:
            values.extend(
                _block_values(
                    matrix[index][index],
                    matrix[index][index + 1],
                    matrix[index + 1][index],
                    matrix[index + 1][index + 1],
                )
            )
            index += 2
    return values


def qr_eigenvalues(
    matrix: Matrix, epsilon: float, max_iterations: int
) -> tuple[list[Number], Matrix, int]:
    current = square_matrix(matrix)
    for iteration in range(max_iterations + 1):
        if _quasi_triangular(current, epsilon):
            return _eigenvalues(current, epsilon), current, iteration
        orthogonal, upper = qr_decompose(current)
        current = [[float(value) for value in row] for row in matmul(upper, orthogonal)]
    raise RuntimeError(f"QR-алгоритм не сошёлся за {max_iterations} итераций")


def run(path: str | Path) -> None:
    data = load_json(path)
    matrix = square_matrix(data["matrix"])
    orthogonal, upper = qr_decompose(matrix)
    values, final_matrix, iterations = qr_eigenvalues(
        matrix, float(data["epsilon"]), int(data["max_iterations"])
    )

    print_matrix("Матрица Q:", orthogonal)
    print_matrix("Матрица R:", upper)
    print_matrix("Q · R:", matmul(orthogonal, upper))
    print(f"max|Q·R-A| = {format_number(max_matrix_difference(matmul(orthogonal, upper), matrix))}")
    print(f"Количество QR-итераций: {iterations}")
    print_matrix("Итоговая квазитреугольная матрица:", final_matrix)
    print_vector("Собственные значения:", values)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "inputs/5.json")
