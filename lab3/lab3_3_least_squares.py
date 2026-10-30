"""3.3. МНК для степеней 1, 2 и 3 через нормальную систему."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

from common import (load_json, parser_for, polynomial_text, polynomial_value,
                    print_nodes, print_table, read_table, sample, validate_table)
from plotting import Series, save_plot


def solve_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Метод Гаусса с выбором главного элемента по столбцу."""
    size = len(rhs)
    if not size or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("Несогласованные размеры системы")
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    scale = max(abs(value) for row in matrix for value in row)
    tolerance = sys.float_info.epsilon * size * scale
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("Нормальная система вырождена или плохо обусловлена")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, size):
            multiplier = augmented[row][column] / augmented[column][column]
            augmented[row][column] = 0.0
            for j in range(column + 1, size + 1):
                augmented[row][j] -= multiplier * augmented[column][j]
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        result[row] = (augmented[row][-1] - math.fsum(augmented[row][j] * result[j] for j in range(row + 1, size))) / augmented[row][row]
    return result


def normal_equations(x: Sequence[float], y: Sequence[float], degree: int) -> tuple[list[list[float]], list[float]]:
    validate_table(x, y)
    if not isinstance(degree, int) or not 0 <= degree < len(x):
        raise ValueError("Степень должна быть целой, от 0 до числа узлов минус 1")
    moments = [math.fsum(xi**power for xi in x) for power in range(2 * degree + 1)]
    matrix = [[moments[i + j] for j in range(degree + 1)] for i in range(degree + 1)]
    rhs = [math.fsum(yi * xi**power for xi, yi in zip(x, y)) for power in range(degree + 1)]
    return matrix, rhs


def least_squares(x: Sequence[float], y: Sequence[float], degree: int) -> list[float]:
    return solve_system(*normal_equations(x, y, degree))


def squared_error(x: Sequence[float], y: Sequence[float], coefficients: Sequence[float]) -> float:
    return math.fsum((polynomial_value(coefficients, xi) - yi)**2 for xi, yi in zip(x, y))


def run(path: str | Path, plot_path: Path | None = None) -> None:
    data = load_json(path, ["x", "y"])
    x, y = read_table(data, minimum=4)
    print("3.3. Метод наименьших квадратов, вариант 14")
    print_nodes(x, y)
    print("\nКоэффициенты перечисляются от a_0 к a_m: P_m(x) = a_0 + a_1*x + ... + a_m*x^m.")
    results = []
    for degree in (1, 2, 3):
        matrix, rhs = normal_equations(x, y, degree)
        coefficients = solve_system(matrix, rhs)
        sse = squared_error(x, y, coefficients)
        mse = sse / len(x)
        results.append((degree, coefficients, sse, mse))
        print(f"\nСтепень {degree}. Нормальная система Aᵀ*A*a = Aᵀ*y:")
        print_table([f"a_{i}" for i in range(degree + 1)] + ["правая часть"], [
            [f"{value:.10f}" for value in row] + [f"{value:.10f}"] for row, value in zip(matrix, rhs)
        ])
        print("Коэффициенты: " + ", ".join(f"a_{i} = {value:.12f}" for i, value in enumerate(coefficients)))
        print(f"P_{degree}(x) = {polynomial_text(coefficients)}")
        print(f"SSE{degree} = Σ(P_{degree}(x_i) - y_i)² = {sse:.12f}")
        print(f"MSE{degree} = SSE{degree}/{len(x)} = {mse:.12f}")
    print("\nСравнение ошибок по всем исходным точкам:")
    print_table(["степень", "SSE", "MSE"], [[degree, f"{sse:.12f}", f"{mse:.12f}"] for degree, _, sse, mse in results])
    best = min(results, key=lambda result: result[3])
    print(f"Лучшее приближение исходных данных по MSE: P_{best[0]}(x), MSE = {best[3]:.12f}.")
    if plot_path is not None:
        colors = ["#2563eb", "#ea580c", "#15803d"]
        curves = [Series(f"P_{degree}(x), MSE = {mse:.6f}", sample(lambda t, c=coefficients: polynomial_value(c, t), x[0], x[-1]), colors[degree - 1])
                  for degree, coefficients, _, mse in results]
        save_plot(plot_path, "3.3. Приближение методом наименьших квадратов, вариант 14", curves + [
            Series("Исходные данные", list(zip(x, y)), "#18212f", markers=True),
        ])


if __name__ == "__main__":
    parser = parser_for("3.3. Метод наименьших квадратов", 3, "least_squares.svg")
    args = parser.parse_args()
    try:
        run(args.input, None if args.no_plot else args.plot)
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
        parser.error(str(error))
