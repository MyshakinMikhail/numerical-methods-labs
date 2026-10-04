"""Лабораторная 2.1, вариант 14: x³ - 2x² - 10x + 15 = 0."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def f(x: float) -> float:
    return x**3 - 2 * x**2 - 10 * x + 15


def derivative(x: float) -> float:
    return 3 * x**2 - 4 * x - 10


def simple_iteration(x: float, epsilon: float, max_iterations: int) -> tuple[float, list[float]]:
    """Итерационная форма x = (x³ - 2x² + 15) / 10 для меньшего положительного корня."""
    history = [x]
    for _ in range(max_iterations):
        next_x = (x**3 - 2 * x**2 + 15) / 10
        if not math.isfinite(next_x):
            raise RuntimeError("Метод простой итерации расходится")
        history.append(next_x)
        if abs(next_x - x) <= epsilon and abs(f(next_x)) <= epsilon:
            return next_x, history
        x = next_x
    raise RuntimeError("Метод простой итерации не сошёлся")


def newton(x: float, epsilon: float, max_iterations: int) -> tuple[float, list[float]]:
    history = [x]
    for _ in range(max_iterations):
        slope = derivative(x)
        if abs(slope) < 1e-14:
            raise ValueError("Производная близка к нулю")
        next_x = x - f(x) / slope
        if not math.isfinite(next_x):
            raise RuntimeError("Метод Ньютона расходится")
        history.append(next_x)
        if abs(next_x - x) <= epsilon and abs(f(next_x)) <= epsilon:
            return next_x, history
        x = next_x
    raise RuntimeError("Метод Ньютона не сошёлся")


def run(path: str | Path) -> None:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    epsilon = float(data["epsilon"])
    max_iterations = int(data["max_iterations"])
    x0 = float(data["initial_guess"])
    if not 0 < epsilon < 1 or max_iterations < 1:
        raise ValueError("Точность должна быть между 0 и 1, число итераций — положительным")

    # Более точное решение служит только для анализа погрешности по итерациям.
    reference, _ = newton(x0, 1e-14, max_iterations)
    print(f"Начальное приближение по графику: x₀ = {x0:g}")
    for title, method in (("Простая итерация", simple_iteration), ("Метод Ньютона", newton)):
        root, history = method(x0, epsilon, max_iterations)
        print(f"\n{title}: корень = {root:.12f}, итераций = {len(history) - 1}, |f(x)| = {abs(f(root)):.3e}")
        print("  k           x_k        |x_k - x*|")
        for index, value in enumerate(history):
            print(f"{index:3d}  {value:14.10f}  {abs(value - reference):12.3e}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "inputs/1.json")
