"""QR-разложение Хаусхолдера и QR-алгоритм для собственных значений."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from math import copysign, sqrt

from common.io_utils import require_positive_float, require_positive_int, require_square_matrix
from common.matrix_utils import Matrix, Number, identity, matmul


@dataclass(frozen=True)
class QRDecomposition:
    q: Matrix
    r: Matrix


@dataclass(frozen=True)
class QREigenResult:
    eigenvalues: list[Number]
    schur_form: Matrix
    iterations: int


def householder_qr(matrix: Matrix, tolerance: float = 1e-12) -> QRDecomposition:
    """Разложить квадратную матрицу A=Q·R отражениями Хаусхолдера."""
    r = require_square_matrix(matrix)
    size = len(r)
    q = identity(size)

    for column in range(size - 1):
        vector = [r[row][column] for row in range(column, size)]
        norm = sqrt(sum(value * value for value in vector))
        if norm <= tolerance:
            continue
        vector[0] += copysign(norm, vector[0])
        vector_norm = sqrt(sum(value * value for value in vector))
        vector = [value / vector_norm for value in vector]

        for target_column in range(column, size):
            projection = sum(
                vector[index] * r[column + index][target_column]
                for index in range(len(vector))
            )
            for index, value in enumerate(vector):
                r[column + index][target_column] -= 2.0 * value * projection

        for row in range(size):
            projection = sum(
                q[row][column + index] * vector[index]
                for index in range(len(vector))
            )
            for index, value in enumerate(vector):
                q[row][column + index] -= 2.0 * projection * value

    for row in range(size):
        for column in range(min(row, size)):
            if abs(r[row][column]) <= tolerance:
                r[row][column] = 0.0
    return QRDecomposition(q, r)


def _hessenberg(matrix: Matrix, tolerance: float = 1e-12) -> Matrix:
    """Привести матрицу к верхней форме Хессенберга подобием."""
    result = [row.copy() for row in matrix]
    size = len(result)
    for column in range(size - 2):
        vector = [result[row][column] for row in range(column + 1, size)]
        norm = sqrt(sum(value * value for value in vector))
        if norm <= tolerance:
            continue
        vector[0] += copysign(norm, vector[0])
        vector_norm = sqrt(sum(value * value for value in vector))
        vector = [value / vector_norm for value in vector]

        for target_column in range(column, size):
            projection = sum(
                vector[index] * result[column + 1 + index][target_column]
                for index in range(len(vector))
            )
            for index, value in enumerate(vector):
                result[column + 1 + index][target_column] -= 2.0 * value * projection

        for row in range(size):
            projection = sum(
                result[row][column + 1 + index] * vector[index]
                for index in range(len(vector))
            )
            for index, value in enumerate(vector):
                result[row][column + 1 + index] -= 2.0 * projection * value

        for row in range(column + 2, size):
            if abs(result[row][column]) <= tolerance:
                result[row][column] = 0.0
    return result


def _subdiagonal_is_small(matrix: Matrix, row: int, epsilon: float) -> bool:
    return abs(matrix[row][row - 1]) <= epsilon


def _clean_subdiagonal(matrix: Matrix, epsilon: float) -> None:
    for row in range(1, len(matrix)):
        if _subdiagonal_is_small(matrix, row, epsilon):
            matrix[row][row - 1] = 0.0


def _is_quasi_upper_triangular(matrix: Matrix, epsilon: float) -> bool:
    size = len(matrix)
    for row in range(size):
        for column in range(row - 1):
            if abs(matrix[row][column]) > epsilon:
                return False
    index = 0
    while index < size:
        if index == size - 1 or matrix[index + 1][index] == 0.0:
            index += 1
            continue
        if index + 2 < size and matrix[index + 2][index + 1] != 0.0:
            return False
        index += 2
    return True


def _block_eigenvalues(a: float, b: float, c: float, d: float) -> tuple[Number, Number]:
    trace = a + d
    determinant = a * d - b * c
    root = cmath.sqrt(trace * trace - 4.0 * determinant)
    values = ((trace + root) / 2.0, (trace - root) / 2.0)
    return tuple(
        float(value.real) if abs(value.imag) <= 1e-12 else value for value in values
    )  # type: ignore[return-value]


def _extract_eigenvalues(matrix: Matrix) -> list[Number]:
    values: list[Number] = []
    index = 0
    while index < len(matrix):
        if index == len(matrix) - 1 or matrix[index + 1][index] == 0.0:
            values.append(matrix[index][index])
            index += 1
        else:
            values.extend(
                _block_eigenvalues(
                    matrix[index][index],
                    matrix[index][index + 1],
                    matrix[index + 1][index],
                    matrix[index + 1][index + 1],
                )
            )
            index += 2
    return values


def _wilkinson_or_real_shift(matrix: Matrix) -> float:
    size = len(matrix)
    a = matrix[size - 2][size - 2]
    b = matrix[size - 2][size - 1]
    c = matrix[size - 1][size - 2]
    d = matrix[size - 1][size - 1]
    discriminant = (a + d) ** 2 - 4.0 * (a * d - b * c)
    if discriminant < 0.0:
        return d
    root = sqrt(discriminant)
    first = (a + d + root) / 2.0
    second = (a + d - root) / 2.0
    return first if abs(first - d) < abs(second - d) else second


def qr_eigenvalues(
    matrix: Matrix, epsilon: float, max_iterations: int
) -> QREigenResult:
    """Найти собственные значения через сдвинутые QR-итерации."""
    checked = require_square_matrix(matrix)
    epsilon = require_positive_float(epsilon, "epsilon")
    max_iterations = require_positive_int(max_iterations, "max_iterations")
    if len(checked) == 1:
        return QREigenResult([checked[0][0]], checked, 0)

    current = _hessenberg(checked)
    for iteration in range(max_iterations + 1):
        _clean_subdiagonal(current, epsilon)
        if _is_quasi_upper_triangular(current, epsilon):
            return QREigenResult(_extract_eigenvalues(current), current, iteration)
        if iteration == max_iterations:
            break

        shift = _wilkinson_or_real_shift(current)
        shifted = [
            [
                current[row][column] - (shift if row == column else 0.0)
                for column in range(len(current))
            ]
            for row in range(len(current))
        ]
        decomposition = householder_qr(shifted)
        current = [
            [float(value) for value in row]
            for row in matmul(decomposition.r, decomposition.q)
        ]
        for index in range(len(current)):
            current[index][index] += shift
        for row in range(len(current)):
            for column in range(row - 1):
                if abs(current[row][column]) <= epsilon:
                    current[row][column] = 0.0

    raise RuntimeError(f"QR-алгоритм не сошёлся за {max_iterations} итераций")
