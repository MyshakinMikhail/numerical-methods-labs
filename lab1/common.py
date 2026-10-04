"""Небольшой набор общих операций над матрицами и векторами."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, TypeAlias

Number: TypeAlias = float | complex
Matrix: TypeAlias = list[list[float]]
Vector: TypeAlias = list[float]


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Корень JSON должен быть объектом")
    return data


def square_matrix(values: Sequence[Sequence[float]]) -> Matrix:
    matrix = [[float(value) for value in row] for row in values]
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Матрица должна быть непустой и квадратной")
    return matrix


def vector(values: Sequence[float], size: int) -> Vector:
    result = [float(value) for value in values]
    if len(result) != size:
        raise ValueError(f"Вектор должен содержать {size} элементов")
    return result


def identity(size: int) -> Matrix:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def transpose(matrix: Sequence[Sequence[Number]]) -> list[list[Number]]:
    return [list(column) for column in zip(*matrix)]


def matmul(
    left: Sequence[Sequence[Number]], right: Sequence[Sequence[Number]]
) -> list[list[Number]]:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("Несогласованные размеры матриц")
    columns = transpose(right)
    return [[sum(a * b for a, b in zip(row, column)) for column in columns] for row in left]


def matvec(matrix: Sequence[Sequence[Number]], values: Sequence[Number]) -> list[Number]:
    return [sum(a * b for a, b in zip(row, values)) for row in matrix]


def vector_norm(values: Sequence[Number]) -> float:
    return max((abs(value) for value in values), default=0.0)


def matrix_norm(matrix: Sequence[Sequence[Number]]) -> float:
    return max((sum(abs(value) for value in row) for row in matrix), default=0.0)


def max_matrix_difference(
    left: Sequence[Sequence[Number]], right: Sequence[Sequence[Number]]
) -> float:
    return max(
        (abs(a - b) for left_row, right_row in zip(left, right) for a, b in zip(left_row, right_row)),
        default=0.0,
    )


def residual_norm(matrix: Matrix, solution: Sequence[Number], rhs: Sequence[Number]) -> float:
    return vector_norm([value - expected for value, expected in zip(matvec(matrix, solution), rhs)])


def format_number(value: Number, precision: int = 6) -> str:
    if isinstance(value, complex) and abs(value.imag) > 10 ** (-precision):
        sign = "+" if value.imag >= 0 else "-"
        return f"{value.real:.{precision}f} {sign} {abs(value.imag):.{precision}f}i"
    real = value.real if isinstance(value, complex) else value
    if abs(real) < 0.5 * 10 ** (-precision):
        real = 0.0
    return f"{real:.{precision}f}"


def print_vector(title: str, values: Sequence[Number]) -> None:
    print(title, "[" + ", ".join(format_number(value) for value in values) + "]")


def print_matrix(title: str, matrix: Sequence[Sequence[Number]]) -> None:
    print(title)
    for row in matrix:
        print("  [" + "  ".join(format_number(value) for value in row) + "]")
