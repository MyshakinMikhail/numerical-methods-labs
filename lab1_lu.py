"""Лабораторная 1.1: LUP-разложение, решение СЛАУ, обратная матрица и определитель."""

from __future__ import annotations

import sys
from math import prod
from pathlib import Path

from common import (
    Matrix,
    Vector,
    format_number,
    identity,
    load_json,
    matmul,
    max_matrix_difference,
    print_matrix,
    print_vector,
    residual_norm,
    square_matrix,
    vector,
)


def lup(matrix: Matrix, tolerance: float = 1e-12) -> tuple[Matrix, Matrix, list[int], int]:
    """Разложить матрицу так, что P·A=L·U, с частичным выбором главного элемента."""
    upper = square_matrix(matrix)
    size = len(upper)
    lower = identity(size)
    permutation = list(range(size))
    swaps = 0

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(upper[row][column]))
        if abs(upper[pivot][column]) <= tolerance:
            raise ValueError("Матрица вырождена")
        if pivot != column:
            upper[column], upper[pivot] = upper[pivot], upper[column]
            permutation[column], permutation[pivot] = permutation[pivot], permutation[column]
            for previous in range(column):
                lower[column][previous], lower[pivot][previous] = (
                    lower[pivot][previous],
                    lower[column][previous],
                )
            swaps += 1

        for row in range(column + 1, size):
            factor = upper[row][column] / upper[column][column]
            lower[row][column] = factor
            upper[row][column] = 0.0
            for index in range(column + 1, size):
                upper[row][index] -= factor * upper[column][index]

    return lower, upper, permutation, swaps


def permutation_matrix(permutation: list[int]) -> Matrix:
    return [[1.0 if column == permutation[row] else 0.0 for column in range(len(permutation))] for row in range(len(permutation))]


def solve_lup(lower: Matrix, upper: Matrix, permutation: list[int], rhs: Vector) -> Vector:
    size = len(lower)
    if len(rhs) != size:
        raise ValueError("Неверная длина правой части")

    permuted_rhs = [rhs[index] for index in permutation]
    intermediate = [0.0] * size
    for row in range(size):
        intermediate[row] = permuted_rhs[row] - sum(
            lower[row][column] * intermediate[column] for column in range(row)
        )

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        known = sum(upper[row][column] * solution[column] for column in range(row + 1, size))
        solution[row] = (intermediate[row] - known) / upper[row][row]
    return solution


def solve(matrix: Matrix, rhs: Vector) -> Vector:
    lower, upper, permutation, _ = lup(matrix)
    return solve_lup(lower, upper, permutation, rhs)


def inverse(lower: Matrix, upper: Matrix, permutation: list[int]) -> Matrix:
    size = len(lower)
    result = [[0.0] * size for _ in range(size)]
    for column in range(size):
        basis = [1.0 if row == column else 0.0 for row in range(size)]
        solution = solve_lup(lower, upper, permutation, basis)
        for row in range(size):
            result[row][column] = solution[row]
    return result


def run(path: str | Path) -> None:
    data = load_json(path)
    matrix = square_matrix(data["matrix"])
    rhs = vector(data["rhs"], len(matrix))
    lower, upper, permutation, swaps = lup(matrix)
    p = permutation_matrix(permutation)
    solution = solve_lup(lower, upper, permutation, rhs)
    inverse_matrix = inverse(lower, upper, permutation)

    print_matrix("Матрица L:", lower)
    print_matrix("Матрица U:", upper)
    print_matrix("Матрица P:", p)
    print_matrix("L · U:", matmul(lower, upper))
    print_matrix("P · A:", matmul(p, matrix))
    print(f"max|L·U-P·A| = {format_number(max_matrix_difference(matmul(lower, upper), matmul(p, matrix)))}")
    print_vector("Решение:", solution)
    print(f"Норма невязки: {format_number(residual_norm(matrix, solution, rhs))}")
    print_matrix("Обратная матрица:", inverse_matrix)
    print_matrix("A · A⁻¹:", matmul(matrix, inverse_matrix))
    determinant = (-1.0 if swaps % 2 else 1.0) * prod(upper[index][index] for index in range(len(matrix)))
    print(f"Определитель: {format_number(determinant)}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else Path("inputs/1.json"))
