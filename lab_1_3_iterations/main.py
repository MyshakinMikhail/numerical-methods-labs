"""Запуск лабораторной 1.3."""

from __future__ import annotations

import sys
from pathlib import Path

from common.formatting import format_number, format_vector
from common.io_utils import (
    load_json,
    require_positive_float,
    require_positive_int,
    require_square_matrix,
    require_vector,
)
from common.matrix_utils import vector_inf_norm
from lab_1_1_lu.lu import solve
from lab_1_3_iterations.iterations import (
    gauss_seidel_method,
    iteration_matrix_inf_norm,
    jacobi_method,
)


def run(path: str | Path) -> None:
    data = load_json(path)
    matrix = require_square_matrix(data.get("matrix"))
    rhs = require_vector(data.get("rhs"), len(matrix), "rhs")
    epsilon = require_positive_float(data.get("epsilon"), "epsilon")
    max_iterations = require_positive_int(data.get("max_iterations"), "max_iterations")
    initial_guess = require_vector(
        data.get("initial_guess", [0.0] * len(matrix)), len(matrix), "initial_guess"
    )

    contraction_norm = iteration_matrix_inf_norm(matrix)
    reference = solve(matrix, rhs)
    simple = jacobi_method(matrix, rhs, epsilon, max_iterations, initial_guess)
    seidel = gauss_seidel_method(matrix, rhs, epsilon, max_iterations, initial_guess)
    simple_error = vector_inf_norm(
        [value - expected for value, expected in zip(simple.solution, reference)]
    )
    seidel_error = vector_inf_norm(
        [value - expected for value, expected in zip(seidel.solution, reference)]
    )

    print(f"Заданная точность ε = {format_number(epsilon)}")
    print(f"Норма матрицы итераций ||α||∞ = {format_number(contraction_norm)}")
    if contraction_norm >= 1.0:
        print("Предупреждение: достаточное условие сходимости ||α||∞ < 1 не выполнено.")
    print(f"\nЭталонное решение (LUP): {format_vector(reference)}")

    print("\nМетод простых итераций:")
    print(f"x = {format_vector(simple.solution)}")
    print(f"Количество итераций: {simple.iterations}")
    print(f"||x(k)-x(k-1)||∞ = {format_number(simple.delta_norm)}")
    print(f"||Ax-b||∞ = {format_number(simple.residual_norm)}")
    print(f"Ошибка относительно эталона = {format_number(simple_error)}")

    print("\nМетод Зейделя:")
    print(f"x = {format_vector(seidel.solution)}")
    print(f"Количество итераций: {seidel.iterations}")
    print(f"||x(k)-x(k-1)||∞ = {format_number(seidel.delta_norm)}")
    print(f"||Ax-b||∞ = {format_number(seidel.residual_norm)}")
    print(f"Ошибка относительно эталона = {format_number(seidel_error)}")

    if simple.iterations == seidel.iterations:
        comparison = "Оба метода выполнили одинаковое число итераций."
    elif simple.iterations < seidel.iterations:
        comparison = "Метод простых итераций сошёлся быстрее."
    else:
        comparison = "Метод Зейделя сошёлся быстрее."
    print(f"\nСравнение: {comparison}")


def main() -> None:
    default_path = Path(__file__).with_name("input.json")
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else default_path)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка: {error}") from error


if __name__ == "__main__":
    main()

