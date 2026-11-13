"""3.3. МНК: нормальные системы для многочленов первой, второй и третьей степени."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from common import (load_json, parser_for, polynomial_text, polynomial_value,
                    print_nodes, print_table, read_table, sample, validate_table)
from plotting import Series, save_plot


def normal_system(x: Sequence[float], y: Sequence[float], degree: int) -> tuple[list[list[float]], list[float]]:
    """A[j,k] = sum(x_i^(j+k)), b[j] = sum(y_i*x_i^j)."""
    if type(degree) is not int or degree < 0:
        raise ValueError("Степень должна быть неотрицательным целым числом")
    validate_table(x, y, minimum=degree + 1)
    moments = [math.fsum(xi ** power for xi in x) for power in range(2 * degree + 1)]
    matrix = [[moments[j + k] for k in range(degree + 1)] for j in range(degree + 1)]
    rhs = [math.fsum(yi * xi ** j for xi, yi in zip(x, y)) for j in range(degree + 1)]
    if not all(math.isfinite(v) for row in matrix for v in row) or not all(math.isfinite(v) for v in rhs):
        raise ValueError("Переполнение при построении нормальной системы")
    return matrix, rhs


def solve_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Метод Гаусса с масштабированным выбором главного элемента по столбцу."""
    size = len(rhs)
    if not size or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("Размеры матрицы и правой части не согласованы")
    if not all(math.isfinite(v) for row in matrix for v in row) or not all(math.isfinite(v) for v in rhs):
        raise ValueError("Система должна содержать конечные числа")
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    scales = [max(abs(value) for value in row) for row in matrix]
    if any(scale == 0 for scale in scales):
        raise ValueError("Вырожденная система")
    for column in range(size):
        pivot = max(range(column, size), key=lambda i: abs(a[i][column]) / scales[i])
        if abs(a[pivot][column]) <= 1e-14 * scales[pivot]:
            raise ValueError("Вырожденная или численно плохо обусловленная нормальная система")
        a[column], a[pivot] = a[pivot], a[column]
        scales[column], scales[pivot] = scales[pivot], scales[column]
        for row in range(column + 1, size):
            factor = a[row][column] / a[column][column]
            a[row][column] = 0.0
            for j in range(column + 1, size + 1):
                a[row][j] -= factor * a[column][j]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        solution[row] = (a[row][-1] - math.fsum(a[row][j] * solution[j] for j in range(row + 1, size))) / a[row][row]
    return solution


def least_squares(x: Sequence[float], y: Sequence[float], degree: int) -> list[float]:
    """Коэффициенты a_0..a_degree в исходной переменной x."""
    return solve_system(*normal_system(x, y, degree))


def errors(x: Sequence[float], y: Sequence[float], coefficients: Sequence[float]) -> tuple[float, float]:
    validate_table(x, y, minimum=1)
    squared_sum = math.fsum((polynomial_value(coefficients, xi) - yi) ** 2 for xi, yi in zip(x, y))
    return squared_sum, squared_sum / len(x)


def run(path: str | Path, plot_path: Path | None = None) -> None:
    data = load_json(path, ["x", "y"])
    x, y = read_table(data, minimum=4)
    print("3.3. Метод наименьших квадратов")
    if data.get("note"):
        print(data["note"])
    print_nodes(x, y)
    fits, metrics = {}, {}
    for degree in (1, 2, 3):
        matrix, rhs = normal_system(x, y, degree)
        coefficients = solve_system(matrix, rhs)
        fits[degree] = coefficients
        metrics[degree] = errors(x, y, coefficients)
        print(f"\nНормальная система для степени {degree}: A*a = b")
        print_table([f"A[:,{i}]" for i in range(degree + 1)] + ["b"], [
            [f"{value:.12g}" for value in row] + [f"{right:.12g}"] for row, right in zip(matrix, rhs)
        ])
        print(f"Коэффициенты многочлена степени {degree} (от свободного к старшему):")
        for i, coefficient in enumerate(coefficients):
            print(f"a_{i} = {coefficient:.12f}")
        print(f"P_{degree}(x) = {polynomial_text(coefficients)}")

    print("\nЗначения приближающих многочленов во всех исходных точках:")
    print_table(["i", "x_i", "y_i", "P_1(x_i)", "P_2(x_i)", "P_3(x_i)"], [
        [i, f"{xi:.6f}", f"{yi:.8f}"] + [f"{polynomial_value(fits[d], xi):.12f}" for d in (1, 2, 3)]
        for i, (xi, yi) in enumerate(zip(x, y))
    ])
    print("\nSSE = sum((P(x_i) - y_i)^2); MSE = SSE / N, N =", len(x))
    print_table(["Многочлен", "SSE", "MSE"], [
        [f"P_{degree}", f"{metrics[degree][0]:.12e}", f"{metrics[degree][1]:.12e}"] for degree in (1, 2, 3)
    ])
    for degree in (1, 2, 3):
        print(f"MSE{degree} = {metrics[degree][1]:.12e}")
    best = min(metrics, key=lambda d: metrics[d][1])
    ties = [d for d in metrics if math.isclose(metrics[d][1], metrics[best][1], rel_tol=1e-12, abs_tol=1e-24)]
    if len(ties) == 1:
        print(f"Наименьшая MSE у P_{best}(x): многочлен степени {best} лучше приближает исходные данные.")
    else:
        print("Минимальные MSE совпадают в пределах округления у: " + ", ".join(f"P_{d}" for d in ties))
    print("Сравнение относится к заданным точкам; это не оценка ошибки на новых данных.")

    if plot_path is not None:
        colors = {1: "#2563eb", 2: "#ea580c", 3: "#15803d"}
        curves = [Series(f"Многочлен P_{d}, степень {d}; MSE = {metrics[d][1]:.6g}",
                         sample(lambda t, c=fits[d]: polynomial_value(c, t), x[0], x[-1]), colors[d]) for d in (1, 2, 3)]
        save_plot(plot_path, "3.3. Метод наименьших квадратов", curves + [
            Series("Исходные данные", list(zip(x, y)), "#18212f", markers=True)
        ])


if __name__ == "__main__":
    parser = parser_for("3.3. Метод наименьших квадратов", 3, "least_squares.svg")
    args = parser.parse_args()
    try:
        run(args.input, None if args.no_plot else args.plot)
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
        parser.error(str(error))
