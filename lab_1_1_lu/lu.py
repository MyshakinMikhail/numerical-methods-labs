"""LUP-разложение с частичным выбором главного элемента."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from common.io_utils import require_square_matrix
from common.matrix_utils import Matrix, Vector, identity


@dataclass(frozen=True)
class LUPResult:
    l: Matrix
    u: Matrix
    p: Matrix
    permutation: list[int]
    swap_count: int


def lup_decompose(matrix: Matrix, tolerance: float = 1e-12) -> LUPResult:
    """Разложить квадратную матрицу так, что P·A = L·U."""
    u = require_square_matrix(matrix)
    size = len(u)
    l = [[0.0] * size for _ in range(size)]
    p = identity(size)
    permutation = list(range(size))
    swap_count = 0

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(u[row][column]))
        if abs(u[pivot_row][column]) <= tolerance:
            raise ValueError("Матрица вырождена: опорный элемент численно равен нулю")

        if pivot_row != column:
            u[column], u[pivot_row] = u[pivot_row], u[column]
            p[column], p[pivot_row] = p[pivot_row], p[column]
            permutation[column], permutation[pivot_row] = permutation[pivot_row], permutation[column]
            for previous_column in range(column):
                l[column][previous_column], l[pivot_row][previous_column] = (
                    l[pivot_row][previous_column],
                    l[column][previous_column],
                )
            swap_count += 1

        l[column][column] = 1.0
        for row in range(column + 1, size):
            factor = u[row][column] / u[column][column]
            l[row][column] = factor
            u[row][column] = 0.0
            for index in range(column + 1, size):
                u[row][index] -= factor * u[column][index]

    return LUPResult(l, u, p, permutation, swap_count)


def solve_lup(decomposition: LUPResult, rhs: Vector) -> Vector:
    """Решить систему по готовому LUP-разложению."""
    size = len(decomposition.l)
    if len(rhs) != size:
        raise ValueError(f"Вектор правой части должен иметь длину {size}")

    permuted_rhs = [float(rhs[index]) for index in decomposition.permutation]
    intermediate = [0.0] * size
    for row in range(size):
        intermediate[row] = permuted_rhs[row] - sum(
            decomposition.l[row][column] * intermediate[column] for column in range(row)
        )

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        remainder = sum(
            decomposition.u[row][column] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = (intermediate[row] - remainder) / decomposition.u[row][row]
    return solution


def solve(matrix: Matrix, rhs: Vector, tolerance: float = 1e-12) -> Vector:
    return solve_lup(lup_decompose(matrix, tolerance), rhs)


def inverse_from_lup(decomposition: LUPResult) -> Matrix:
    """Найти обратную матрицу решением систем для единичных столбцов."""
    size = len(decomposition.l)
    inverse = [[0.0] * size for _ in range(size)]
    basis = identity(size)
    for column in range(size):
        solution = solve_lup(decomposition, [basis[row][column] for row in range(size)])
        for row in range(size):
            inverse[row][column] = solution[row]
    return inverse


def determinant_from_lup(decomposition: LUPResult) -> float:
    sign = -1.0 if decomposition.swap_count % 2 else 1.0
    return sign * prod(decomposition.u[index][index] for index in range(len(decomposition.u)))

