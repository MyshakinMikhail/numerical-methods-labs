"""Метод простых итераций (Якоби) и метод Гаусса—Зейделя."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from common.io_utils import require_positive_float, require_positive_int, require_square_matrix, require_vector
from common.matrix_utils import Matrix, Vector, residual_inf_norm, vector_inf_norm


@dataclass(frozen=True)
class IterationResult:
    solution: Vector
    iterations: int
    delta_norm: float
    residual_norm: float


def _prepare(
    matrix: Matrix,
    rhs: Sequence[float],
    epsilon: float,
    max_iterations: int,
    initial_guess: Sequence[float] | None,
    tolerance: float = 1e-12,
) -> tuple[Matrix, Vector, float, int, Vector]:
    checked_matrix = require_square_matrix(matrix)
    size = len(checked_matrix)
    checked_rhs = require_vector(list(rhs), size, "rhs")
    checked_epsilon = require_positive_float(epsilon, "epsilon")
    checked_max_iterations = require_positive_int(max_iterations, "max_iterations")
    for row in range(size):
        if abs(checked_matrix[row][row]) <= tolerance:
            raise ValueError(f"Нулевой диагональный элемент в строке {row + 1}")
    if initial_guess is None:
        current = [0.0] * size
    else:
        if len(initial_guess) != size:
            raise ValueError(f"Вектор начального приближения должен иметь длину {size}")
        current = [float(value) for value in initial_guess]
    return checked_matrix, checked_rhs, checked_epsilon, checked_max_iterations, current


def iteration_matrix_inf_norm(matrix: Matrix, tolerance: float = 1e-12) -> float:
    """Норма матрицы α после преобразования x = β + αx."""
    checked_matrix = require_square_matrix(matrix)
    row_sums: Vector = []
    for row in range(len(checked_matrix)):
        diagonal = checked_matrix[row][row]
        if abs(diagonal) <= tolerance:
            raise ValueError(f"Нулевой диагональный элемент в строке {row + 1}")
        row_sums.append(
            sum(
                abs(checked_matrix[row][column] / diagonal)
                for column in range(len(checked_matrix))
                if column != row
            )
        )
    return max(row_sums)


def jacobi_method(
    matrix: Matrix,
    rhs: Sequence[float],
    epsilon: float,
    max_iterations: int,
    initial_guess: Sequence[float] | None = None,
) -> IterationResult:
    """Решить систему методом простых итераций с одновременным обновлением."""
    checked_matrix, checked_rhs, epsilon, max_iterations, current = _prepare(
        matrix, rhs, epsilon, max_iterations, initial_guess
    )
    size = len(checked_matrix)
    for iteration in range(1, max_iterations + 1):
        next_solution = [
            (
                checked_rhs[row]
                - sum(
                    checked_matrix[row][column] * current[column]
                    for column in range(size)
                    if column != row
                )
            )
            / checked_matrix[row][row]
            for row in range(size)
        ]
        delta = vector_inf_norm(
            [next_solution[index] - current[index] for index in range(size)]
        )
        if delta <= epsilon:
            return IterationResult(
                next_solution,
                iteration,
                delta,
                residual_inf_norm(checked_matrix, next_solution, checked_rhs),
            )
        current = next_solution
    raise RuntimeError(f"Метод простых итераций не сошёлся за {max_iterations} итераций")


def gauss_seidel_method(
    matrix: Matrix,
    rhs: Sequence[float],
    epsilon: float,
    max_iterations: int,
    initial_guess: Sequence[float] | None = None,
) -> IterationResult:
    """Решить систему методом Гаусса—Зейделя."""
    checked_matrix, checked_rhs, epsilon, max_iterations, current = _prepare(
        matrix, rhs, epsilon, max_iterations, initial_guess
    )
    size = len(checked_matrix)
    for iteration in range(1, max_iterations + 1):
        next_solution = current.copy()
        for row in range(size):
            lower_sum = sum(
                checked_matrix[row][column] * next_solution[column]
                for column in range(row)
            )
            upper_sum = sum(
                checked_matrix[row][column] * current[column]
                for column in range(row + 1, size)
            )
            next_solution[row] = (
                checked_rhs[row] - lower_sum - upper_sum
            ) / checked_matrix[row][row]
        delta = vector_inf_norm(
            [next_solution[index] - current[index] for index in range(size)]
        )
        if delta <= epsilon:
            return IterationResult(
                next_solution,
                iteration,
                delta,
                residual_inf_norm(checked_matrix, next_solution, checked_rhs),
            )
        current = next_solution
    raise RuntimeError(f"Метод Зейделя не сошёлся за {max_iterations} итераций")

