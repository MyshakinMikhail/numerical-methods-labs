"""3.4. Первая и вторая производные локального квадратичного интерполянта."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from common import load_json, number, parser_for, print_nodes, read_table, validate_table


def select_nodes(x: Sequence[float], point: float) -> int:
    """Начало тройки соседних узлов с минимальным максимальным удалением от x*."""
    if len(x) < 3 or not math.isfinite(point) or not x[0] <= point <= x[-1]:
        raise ValueError("Для дифференцирования нужны не менее трёх узлов и точка внутри таблицы")
    return min(range(len(x) - 2), key=lambda i: (
        max(abs(xj - point) for xj in x[i:i + 3]),
        sum(abs(xj - point) for xj in x[i:i + 3]),
        i,
    ))


def differentiate(x: Sequence[float], y: Sequence[float], point: float) -> tuple[float, float, int]:
    """Формулы работают и для неравномерной сетки. Возвращают f', f'', индекс тройки."""
    validate_table(x, y, minimum=3)
    i = select_nodes(x, point)
    slope_left = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    slope_right = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1])
    second_difference = (slope_right - slope_left) / (x[i + 2] - x[i])
    first = slope_left + second_difference * (2 * point - x[i] - x[i + 1])
    second = 2 * second_difference
    return first, second, i


def run(path: str | Path) -> None:
    data = load_json(path, ["x", "y", "x_star"])
    x, y = read_table(data, minimum=3)
    point = number(data["x_star"])
    first, second, index = differentiate(x, y, point)
    print("3.4. Численное дифференцирование")
    if data.get("note"):
        print(data["note"])
    print_nodes(x, y)
    print(f"\nТочка вычисления x* = {point:.12f}")
    print("Метод: дифференцирование квадратичного многочлена по трём соседним узлам.")
    print("Выбор тройки: минимальное максимальное расстояние до x*; при равенстве — минимальная сумма расстояний.")
    print("Использованные узлы (индексы исходной таблицы):")
    for i in range(index, index + 3):
        print(f"i = {i}: x_i = {x[i]:.12f}, y_i = {y[i]:.12f}")
    print("\nf'(x*) ≈ f[x_i,x_(i+1)] + f[x_i,x_(i+1),x_(i+2)]*(2*x* - x_i - x_(i+1))")
    print("f''(x*) ≈ 2*f[x_i,x_(i+1),x_(i+2)]")
    print(f"Первая производная f'(x*) ≈ {first:.12f}")
    print(f"Вторая производная f''(x*) ≈ {second:.12f}")


if __name__ == "__main__":
    parser = parser_for("3.4. Численное дифференцирование", 4)
    args = parser.parse_args()
    try:
        run(args.input)
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
        parser.error(str(error))
