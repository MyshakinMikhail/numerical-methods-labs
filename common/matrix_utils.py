"""Элементарные операции над матрицами и векторами без NumPy."""

from __future__ import annotations

from math import sqrt
from typing import Sequence, TypeAlias

Number: TypeAlias = float | complex
Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[list[float]]
NumberMatrix: TypeAlias = list[list[Number]]


def identity(size: int) -> Matrix:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def transpose(matrix: Sequence[Sequence[Number]]) -> NumberMatrix:
    if not matrix:
        return []
    return [[matrix[row][column] for row in range(len(matrix))] for column in range(len(matrix[0]))]


def matmul(left: Sequence[Sequence[Number]], right: Sequence[Sequence[Number]]) -> NumberMatrix:
    if not left or not right or not left[0] or not right[0]:
        raise ValueError("Матрицы для умножения не должны быть пустыми")
    if len(left[0]) != len(right):
        raise ValueError("Несогласованные размеры матриц для умножения")
    return [
        [
            sum((left[row][index] * right[index][column] for index in range(len(right))), 0.0)
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def matvec(matrix: Sequence[Sequence[Number]], vector: Sequence[Number]) -> list[Number]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("Несогласованные размеры матрицы и вектора")
    return [sum((value * vector[column] for column, value in enumerate(row)), 0.0) for row in matrix]


def matrix_subtract(left: Sequence[Sequence[Number]], right: Sequence[Sequence[Number]]) -> NumberMatrix:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("Несогласованные размеры матриц")
    return [[a - b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]


def vector_inf_norm(vector: Sequence[Number]) -> float:
    return max((abs(value) for value in vector), default=0.0)


def max_abs_matrix(matrix: Sequence[Sequence[Number]]) -> float:
    return max((abs(value) for row in matrix for value in row), default=0.0)


def max_abs_matrix_difference(
    left: Sequence[Sequence[Number]], right: Sequence[Sequence[Number]]
) -> float:
    return max_abs_matrix(matrix_subtract(left, right))


def residual_inf_norm(matrix: Matrix, solution: Sequence[Number], rhs: Sequence[Number]) -> float:
    product = matvec(matrix, solution)
    if len(product) != len(rhs):
        raise ValueError("Несогласованные размеры системы")
    return vector_inf_norm([value - expected for value, expected in zip(product, rhs)])


def is_symmetric(matrix: Matrix, tolerance: float = 1e-12) -> bool:
    return all(
        abs(matrix[row][column] - matrix[column][row]) <= tolerance
        for row in range(len(matrix))
        for column in range(row + 1, len(matrix))
    )


def off_diagonal_norm(matrix: Matrix) -> float:
    return sqrt(
        sum(
            matrix[row][column] ** 2
            for row in range(len(matrix))
            for column in range(len(matrix))
            if row != column
        )
    )

