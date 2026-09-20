"""Метод прогонки без хранения плотной матрицы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from common.matrix_utils import Vector, vector_inf_norm


@dataclass(frozen=True)
class ThomasResult:
    solution: Vector
    alpha: Vector
    beta: Vector


def _as_float_vector(values: Sequence[float], name: str) -> Vector:
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} должна содержать только числа") from error


def _validate_lengths(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> int:
    size = len(diagonal)
    if size == 0:
        raise ValueError("Главная диагональ не должна быть пустой")
    if len(lower) != size - 1 or len(upper) != size - 1 or len(rhs) != size:
        raise ValueError(
            "Неверные длины диагоналей: ожидаются n-1, n, n-1 и n элементов"
        )
    return size


def thomas_solve(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
    tolerance: float = 1e-12,
) -> ThomasResult:
    """Решить трёхдиагональную систему методом прогонки."""
    size = _validate_lengths(lower, diagonal, upper, rhs)
    lower_values = _as_float_vector(lower, "Нижняя диагональ")
    diagonal_values = _as_float_vector(diagonal, "Главная диагональ")
    upper_values = _as_float_vector(upper, "Верхняя диагональ")
    rhs_values = _as_float_vector(rhs, "Правая часть")

    alpha = [0.0] * size
    beta = [0.0] * size
    if abs(diagonal_values[0]) <= tolerance:
        raise ValueError("Нулевой знаменатель прогонки на строке 1")

    alpha[0] = -upper_values[0] / diagonal_values[0] if size > 1 else 0.0
    beta[0] = rhs_values[0] / diagonal_values[0]

    for row in range(1, size):
        denominator = diagonal_values[row] + lower_values[row - 1] * alpha[row - 1]
        if abs(denominator) <= tolerance:
            raise ValueError(f"Нулевой знаменатель прогонки на строке {row + 1}")
        if row < size - 1:
            alpha[row] = -upper_values[row] / denominator
        beta[row] = (
            rhs_values[row] - lower_values[row - 1] * beta[row - 1]
        ) / denominator

    solution = [0.0] * size
    solution[-1] = beta[-1]
    for row in range(size - 2, -1, -1):
        solution[row] = alpha[row] * solution[row + 1] + beta[row]
    return ThomasResult(solution, alpha, beta)


def tridiagonal_residual_inf_norm(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    solution: Sequence[float],
    rhs: Sequence[float],
) -> float:
    """Вычислить невязку, не создавая плотную матрицу."""
    size = _validate_lengths(lower, diagonal, upper, rhs)
    if len(solution) != size:
        raise ValueError(f"Решение должно иметь длину {size}")
    residuals: Vector = []
    for row in range(size):
        value = diagonal[row] * solution[row]
        if row > 0:
            value += lower[row - 1] * solution[row - 1]
        if row < size - 1:
            value += upper[row] * solution[row + 1]
        residuals.append(float(value - rhs[row]))
    return vector_inf_norm(residuals)

