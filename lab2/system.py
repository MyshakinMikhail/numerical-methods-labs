"""Лабораторная 2.2, вариант 14 (a = 3)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Point = tuple[float, float]
A = 3.0


def equations(point: Point) -> Point:
    x1, x2 = point
    return x1**2 / A**2 + x2**2 / (A / 2)**2 - 1, A * x2 - math.exp(x1) - x1


def residual(point: Point) -> float:
    return max(abs(value) for value in equations(point))


def simple_iteration(point: Point, epsilon: float, max_iterations: int) -> tuple[Point, list[Point]]:
    """Обновлять x₁ по первому уравнению, затем x₂ по второму."""
    history = [point]
    for _ in range(max_iterations):
        x1, x2 = point
        first = x1**2 / A**2 + x2**2 / (A / 2)**2 - 1
        next_x1 = x1 - 0.5 * first
        next_point = (next_x1, (math.exp(next_x1) + next_x1) / A)
        if not all(math.isfinite(value) for value in next_point):
            raise RuntimeError("Метод простой итерации расходится")
        history.append(next_point)
        if max(abs(new - old) for new, old in zip(next_point, point)) <= epsilon and residual(next_point) <= epsilon:
            return next_point, history
        point = next_point
    raise RuntimeError("Метод простой итерации не сошёлся")


def newton(point: Point, epsilon: float, max_iterations: int) -> tuple[Point, list[Point]]:
    history = [point]
    for _ in range(max_iterations):
        x1, x2 = point
        f1, f2 = equations(point)
        j11, j12 = 2 * x1 / A**2, 2 * x2 / (A / 2)**2
        j21, j22 = -math.exp(x1) - 1, A
        determinant = j11 * j22 - j12 * j21
        if abs(determinant) < 1e-14:
            raise ValueError("Якобиан близок к вырожденному")
        dx1 = (-f1 * j22 + j12 * f2) / determinant
        dx2 = (j21 * f1 - j11 * f2) / determinant
        next_point = (x1 + dx1, x2 + dx2)
        if not all(math.isfinite(value) for value in next_point):
            raise RuntimeError("Метод Ньютона расходится")
        history.append(next_point)
        if max(abs(dx1), abs(dx2)) <= epsilon and residual(next_point) <= epsilon:
            return next_point, history
        point = next_point
    raise RuntimeError("Метод Ньютона не сошёлся")


def run(path: str | Path) -> None:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    epsilon = float(data["epsilon"])
    max_iterations = int(data["max_iterations"])
    initial = data["initial_guess"]
    point = (float(initial[0]), float(initial[1]))
    if not 0 < epsilon < 1 or max_iterations < 1:
        raise ValueError("Точность должна быть между 0 и 1, число итераций — положительным")

    reference, _ = newton(point, 1e-14, max_iterations)
    print(f"Начальное приближение по графику: (x₁, x₂) = {point}")
    for title, method in (("Простая итерация", simple_iteration), ("Метод Ньютона", newton)):
        root, history = method(point, epsilon, max_iterations)
        print(f"\n{title}: решение = ({root[0]:.12f}, {root[1]:.12f}), итераций = {len(history) - 1}, невязка = {residual(root):.3e}")
        print("  k           x₁             x₂       ||x_k - x*||∞")
        for index, value in enumerate(history):
            error = max(abs(a - b) for a, b in zip(value, reference))
            print(f"{index:3d}  {value[0]:14.10f} {value[1]:14.10f}    {error:12.3e}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "inputs/2.json")
