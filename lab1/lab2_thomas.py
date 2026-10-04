"""Лабораторная 1.2: метод прогонки для трёхдиагональной СЛАУ."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from common import format_number, load_json, print_vector, vector_norm


def thomas(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
    tolerance: float = 1e-12,
) -> tuple[list[float], list[float], list[float]]:
    size = len(diagonal)
    if size == 0 or len(lower) != size - 1 or len(upper) != size - 1 or len(rhs) != size:
        raise ValueError("Неверные размеры диагоналей или правой части")

    alpha = [0.0] * size
    beta = [0.0] * size
    if abs(diagonal[0]) <= tolerance:
        raise ValueError("Нулевой знаменатель на строке 1")
    if size > 1:
        alpha[0] = -upper[0] / diagonal[0]
    beta[0] = rhs[0] / diagonal[0]

    for row in range(1, size):
        denominator = diagonal[row] + lower[row - 1] * alpha[row - 1]
        if abs(denominator) <= tolerance:
            raise ValueError(f"Нулевой знаменатель на строке {row + 1}")
        if row < size - 1:
            alpha[row] = -upper[row] / denominator
        beta[row] = (rhs[row] - lower[row - 1] * beta[row - 1]) / denominator

    solution = [0.0] * size
    solution[-1] = beta[-1]
    for row in range(size - 2, -1, -1):
        solution[row] = alpha[row] * solution[row + 1] + beta[row]
    return solution, alpha, beta


def tridiagonal_residual(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    solution: Sequence[float],
    rhs: Sequence[float],
) -> float:
    residuals = []
    for row in range(len(diagonal)):
        value = diagonal[row] * solution[row]
        if row > 0:
            value += lower[row - 1] * solution[row - 1]
        if row + 1 < len(diagonal):
            value += upper[row] * solution[row + 1]
        residuals.append(value - rhs[row])
    return vector_norm(residuals)


def run(path: str | Path) -> None:
    data = load_json(path)
    lower = data["lower"]
    diagonal = data["diagonal"]
    upper = data["upper"]
    rhs = data["rhs"]
    solution, alpha, beta = thomas(lower, diagonal, upper, rhs)

    print("Прогоночные коэффициенты:")
    for index, (a, b) in enumerate(zip(alpha, beta), start=1):
        print(f"i={index}: α={format_number(a)}, β={format_number(b)}")
    print_vector("Решение:", solution)
    print(f"Норма невязки: {format_number(tridiagonal_residual(lower, diagonal, upper, solution, rhs))}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "inputs/2.json")
