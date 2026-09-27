"""Лабораторная 1.3: методы простых итераций и Гаусса—Зейделя."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

from common import (
    Matrix,
    Vector,
    format_number,
    load_json,
    matrix_norm,
    print_matrix,
    print_vector,
    residual_norm,
    square_matrix,
    vector,
    vector_norm,
)
from lab1_lu import solve


def iteration_form(matrix: Matrix, rhs: Vector) -> tuple[Matrix, Vector]:
    """Привести Ax=b к виду x=β+αx."""
    matrix = square_matrix(matrix)
    rhs = vector(rhs, len(matrix))
    size = len(matrix)
    alpha = [[0.0] * size for _ in range(size)]
    beta = [0.0] * size

    for row in range(size):
        diagonal = matrix[row][row]
        if abs(diagonal) <= 1e-12:
            raise ValueError(f"Нулевой диагональный элемент в строке {row + 1}")
        beta[row] = rhs[row] / diagonal
        for column in range(size):
            if column != row:
                alpha[row][column] = -matrix[row][column] / diagonal
    return alpha, beta


def jacobi(
    alpha: Matrix,
    beta: Vector,
    epsilon: float,
    max_iterations: int,
    initial: Sequence[float] | None = None,
) -> tuple[Vector, int, float]:
    current = list(initial) if initial is not None else [0.0] * len(beta)
    for iteration in range(1, max_iterations + 1):
        next_solution = [
            beta[row] + sum(alpha[row][column] * current[column] for column in range(len(beta)))
            for row in range(len(beta))
        ]
        if not all(math.isfinite(value) for value in next_solution):
            raise RuntimeError("Метод простых итераций расходится")
        delta = vector_norm([new - old for new, old in zip(next_solution, current)])
        if delta <= epsilon:
            return next_solution, iteration, delta
        current = next_solution
    raise RuntimeError(f"Метод простых итераций не сошёлся за {max_iterations} итераций")


def seidel(
    alpha: Matrix,
    beta: Vector,
    epsilon: float,
    max_iterations: int,
    initial: Sequence[float] | None = None,
) -> tuple[Vector, int, float]:
    current = list(initial) if initial is not None else [0.0] * len(beta)
    for iteration in range(1, max_iterations + 1):
        next_solution = current.copy()
        for row in range(len(beta)):
            lower = sum(alpha[row][column] * next_solution[column] for column in range(row))
            upper = sum(alpha[row][column] * current[column] for column in range(row + 1, len(beta)))
            next_solution[row] = beta[row] + lower + upper
        if not all(math.isfinite(value) for value in next_solution):
            raise RuntimeError("Метод Зейделя расходится")
        delta = vector_norm([new - old for new, old in zip(next_solution, current)])
        if delta <= epsilon:
            return next_solution, iteration, delta
        current = next_solution
    raise RuntimeError(f"Метод Зейделя не сошёлся за {max_iterations} итераций")


def run(path: str | Path) -> None:
    data = load_json(path)
    matrix = square_matrix(data["matrix"])
    rhs = vector(data["rhs"], len(matrix))
    epsilon = float(data["epsilon"])
    max_iterations = int(data["max_iterations"])
    initial = vector(data.get("initial_guess", [0.0] * len(matrix)), len(matrix))
    alpha, beta = iteration_form(matrix, rhs)
    simple, simple_iterations, simple_delta = jacobi(alpha, beta, epsilon, max_iterations, initial)
    gauss_seidel, seidel_iterations, seidel_delta = seidel(
        alpha, beta, epsilon, max_iterations, initial
    )
    reference = solve(matrix, rhs)

    print_matrix("Матрица α:", alpha)
    print_vector("Вектор β:", beta)
    print(f"||α||∞ = {format_number(matrix_norm(alpha))}")
    print_vector("Эталонное решение LUP:", reference)
    print("\nМетод простых итераций:")
    print_vector("Решение:", simple)
    print(f"Итераций: {simple_iterations}; последнее изменение: {format_number(simple_delta)}")
    print(f"Невязка: {format_number(residual_norm(matrix, simple, rhs))}")
    print("\nМетод Зейделя:")
    print_vector("Решение:", gauss_seidel)
    print(f"Итераций: {seidel_iterations}; последнее изменение: {format_number(seidel_delta)}")
    print(f"Невязка: {format_number(residual_norm(matrix, gauss_seidel, rhs))}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else Path("inputs/3.json"))
