"""3.1. Многочлены Лагранжа и Ньютона, вариант 14, наборы а и б."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from common import (factor, load_json, make_function, multiply_by_linear, number,
                    parser_for, polynomial_text, print_nodes, print_table, sample, validate_table)
from plotting import Series, save_plot


def lagrange(x: Sequence[float], y: Sequence[float], point: float) -> float:
    """Вычислить сумму y_i * произведение (point - x_j)/(x_i - x_j)."""
    validate_table(x, y)
    return math.fsum(
        yi * math.prod((point - xj) / (xi - xj) for j, xj in enumerate(x) if j != i)
        for i, (xi, yi) in enumerate(zip(x, y))
    )


def divided_differences(x: Sequence[float], y: Sequence[float]) -> list[list[float]]:
    validate_table(x, y)
    size = len(x)
    table = [[0.0] * size for _ in x]
    for i, value in enumerate(y):
        table[i][0] = value
    for order in range(1, size):
        for i in range(size - order):
            table[i][order] = (table[i + 1][order - 1] - table[i][order - 1]) / (x[i + order] - x[i])
    return table


def newton(x: Sequence[float], coefficients: Sequence[float], point: float) -> float:
    """Вложенная форма Ньютона; coefficients — первая строка таблицы разностей."""
    value = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        value = coefficients[i] + (point - x[i]) * value
    return value


def lagrange_coefficients(x: Sequence[float], y: Sequence[float]) -> list[float]:
    result = [0.0] * len(x)
    for i, (xi, yi) in enumerate(zip(x, y)):
        basis = [1.0]
        denominator = 1.0
        for j, xj in enumerate(x):
            if i != j:
                basis = multiply_by_linear(basis, xj)
                denominator *= xi - xj
        for degree, value in enumerate(basis):
            result[degree] += yi * value / denominator
    return result


def newton_coefficients(x: Sequence[float], differences: Sequence[float]) -> list[float]:
    result = [0.0] * len(x)
    basis = [1.0]
    for i, value in enumerate(differences):
        for degree, coefficient in enumerate(basis):
            result[degree] += value * coefficient
        basis = multiply_by_linear(basis, x[i])
    return result


def formula_terms(terms: Sequence[tuple[float, str]]) -> str:
    return " ".join(f"{coefficient:+.12g}" + ("*" + product if product else "")
                    for coefficient, product in terms if coefficient != 0.0).lstrip("+") or "0"


def run(path: str | Path, plot_path: Path | None = None) -> None:
    data = load_json(path, ["function", "node_sets", "x_star"])
    function = make_function(data["function"])
    point = number(data["x_star"])
    if not data["node_sets"]:
        raise ValueError("Нужен хотя бы один набор узлов")
    for set_index, node_set in enumerate(data["node_sets"]):
        print(f"\n3.1. Набор {node_set['name']}; f(x) = {data['function']}")
        x = [number(value) for value in node_set["x"]]
        y = [function(value) for value in x]
        validate_table(x, y)
        print_nodes(x, y)
        terms = [(y[i] / math.prod(x[i] - xj for j, xj in enumerate(x) if j != i),
                  "*".join(factor(xj) for j, xj in enumerate(x) if j != i)) for i in range(len(x))]
        print("\nМногочлен Лагранжа в базисной форме:")
        print("L(x) = " + formula_terms(terms))
        print("L(x) = " + polynomial_text(lagrange_coefficients(x, y)))

        table = divided_differences(x, y)
        print("\nТаблица разделённых разностей (пустые ячейки не используются):")
        print_table(["i", "x_i", "f[x_i]"] + [f"порядок {j}" for j in range(1, len(x))], [
            [i, f"{xi:.10f}"] + [f"{table[i][j]:.12f}" if i + j < len(x) else "" for j in range(len(x))]
            for i, xi in enumerate(x)
        ])
        differences = table[0]
        terms = [(value, "*".join(factor(xj) for xj in x[:i])) for i, value in enumerate(differences)]
        print("\nМногочлен Ньютона:")
        print("N(x) = " + formula_terms(terms))
        print("N(x) = " + polynomial_text(newton_coefficients(x, differences)))

        left_value = lagrange(x, y, point)
        right_value = newton(x, differences, point)
        exact = function(point)
        print(f"\nx* = {point:.12f}")
        print(f"L(x*) = {left_value:.12f}")
        print(f"N(x*) = {right_value:.12f}")
        print(f"Точное значение f(x*) = {exact:.12f}")
        print(f"Абсолютная погрешность Лагранжа |f(x*) - L(x*)| = {abs(exact - left_value):.12e}")
        print(f"Абсолютная погрешность Ньютона  |f(x*) - N(x*)| = {abs(exact - right_value):.12e}")
        print(f"Разность L(x*) - N(x*) = {left_value - right_value:.12e}")
        print(f"Модуль разности |L(x*) - N(x*)| = {abs(left_value - right_value):.12e}")

        if plot_path is not None:
            suffix = chr(ord("a") + set_index)
            target = plot_path.with_name(f"{plot_path.stem}_{suffix}{plot_path.suffix}")
            start, end = min(x[0], point), max(x[-1], point)
            save_plot(target, f"3.1. Интерполяция, набор {node_set['name']}, вариант 14", [
                Series(f"Исходная функция f(x) = {data['function']}", sample(function, start, end), "#2563eb"),
                Series("Многочлен Лагранжа L(x)", sample(lambda t: lagrange(x, y, t), start, end), "#ea580c"),
                Series("Многочлен Ньютона N(x), совпадает с L(x)", sample(lambda t: newton(x, differences, t), start, end), "#15803d", dashed=True),
                Series("Узлы интерполяции", list(zip(x, y)), "#18212f", markers=True),
                Series(f"Точная точка (x*, f(x*)) = ({point:.5f}, {exact:.5f})", [(point, exact)], "#be123c", markers=True),
                Series(f"Интерполяция (x*, L(x*)) = ({point:.5f}, {left_value:.5f})", [(point, left_value)], "#7c3aed", markers=True),
            ], point)


if __name__ == "__main__":
    parser = parser_for("3.1. Интерполяция Лагранжа и Ньютона", 1, "interpolation.svg")
    args = parser.parse_args()
    try:
        run(args.input, None if args.no_plot else args.plot)
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
        parser.error(str(error))
