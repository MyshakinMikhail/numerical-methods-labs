"""3.5. Составные квадратурные формулы и уточнение Рунге–Ромберга."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from common import load_json, make_function, number, parser_for, print_table

# Порядки относятся к составным формулам на равномерной сетке.
METHODS = {
    "left": ("Левые прямоугольники", 1),
    "right": ("Правые прямоугольники", 1),
    "midpoint": ("Средние прямоугольники", 2),
    "trapezoid": ("Трапеции", 2),
    "simpson": ("Симпсон", 4),
}


def grid_size(a: float, b: float, h: float) -> int:
    if not all(math.isfinite(value) for value in (a, b, h)) or not a < b or h <= 0:
        raise ValueError("Требуется a < b, h > 0 и конечные параметры")
    ratio = (b - a) / h
    if not math.isfinite(ratio):
        raise ValueError("Слишком маленький шаг")
    count = round(ratio)
    if count < 1 or not math.isclose(ratio, count, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Шаг h={h:g} должен делить длину отрезка [{a:g}, {b:g}] нацело")
    return count


def integrate(function: Callable[[float], float], a: float, b: float, h: float, method: str) -> float:
    count = grid_size(a, b, h)
    step = (b - a) / count
    if method == "left":
        return step * math.fsum(function(a + i * step) for i in range(count))
    if method == "right":
        return step * math.fsum(function(b if i == count else a + i * step) for i in range(1, count + 1))
    if method == "midpoint":
        return step * math.fsum(function(a + (i + 0.5) * step) for i in range(count))
    if method == "trapezoid":
        return step * ((function(a) + function(b)) / 2 + math.fsum(function(a + i * step) for i in range(1, count)))
    if method == "simpson":
        if count % 2:
            raise ValueError(f"Метод Симпсона требует чётное число отрезков; h={h:g} даёт {count}")
        odd = math.fsum(function(a + i * step) for i in range(1, count, 2))
        even = math.fsum(function(a + i * step) for i in range(2, count, 2))
        return step / 3 * (function(a) + function(b) + 4 * odd + 2 * even)
    raise ValueError(f"Неизвестный метод: {method}")


def runge_romberg(coarse: float, fine: float, h1: float, h2: float, order: int) -> tuple[float, float]:
    if not all(math.isfinite(value) for value in (coarse, fine, h1, h2)) or not h1 > h2 > 0 or order < 1:
        raise ValueError("Для Рунге–Ромберга требуется h1 > h2 > 0, p >= 1 и конечные значения")
    correction = (fine - coarse) / ((h1 / h2)**order - 1)
    return fine + correction, abs(correction)


@dataclass
class Result:
    method: str
    order: int
    coarse: float
    fine: float
    refined: float
    estimate: float


def compare_methods(function: Callable[[float], float], a: float, b: float, h1: float, h2: float) -> list[Result]:
    if not h1 > h2 > 0:
        raise ValueError("Нужно задать h1 > h2 > 0: h2 — меньший шаг")
    results = []
    for method, (_, order) in METHODS.items():
        coarse = integrate(function, a, b, h1, method)
        fine = integrate(function, a, b, h2, method)
        refined, estimate = runge_romberg(coarse, fine, h1, h2, order)
        results.append(Result(method, order, coarse, fine, refined, estimate))
    return results


def run(path: str | Path) -> None:
    data = load_json(path, ["function", "a", "b", "h1", "h2"])
    function = make_function(data["function"])
    a, b, h1, h2 = (number(data[key]) for key in ("a", "b", "h1", "h2"))
    results = compare_methods(function, a, b, h1, h2)
    exact = None
    if data.get("antiderivative") is not None:
        antiderivative = make_function(data["antiderivative"])
        exact = antiderivative(b) - antiderivative(a)
    if data.get("exact_value") is not None:
        provided = number(data["exact_value"])
        if exact is not None and not math.isclose(exact, provided, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Заданное exact_value не совпадает с F(b) - F(a)")
        exact = provided
    print("3.5. Численное интегрирование")
    if data.get("note"):
        print(data["note"])
    print(f"f(x) = {data['function']}; [{a:.12g}, {b:.12g}]")
    print(f"h1 = {h1:.12g}, n1 = {grid_size(a, b, h1)}; h2 = {h2:.12g}, n2 = {grid_size(a, b, h2)}")
    print("I_RR = I(h2) + (I(h2) - I(h1))/((h1/h2)^p - 1)")
    print("Оценка ошибки I(h2): |I(h2) - I(h1)|/((h1/h2)^p - 1). Это асимптотическая оценка.")
    print()
    print_table(["метод", "I(h1)", "I(h2)", "p", "I_RR", "оценка ошибки h2"], [
        [METHODS[r.method][0], f"{r.coarse:.12f}", f"{r.fine:.12f}", r.order, f"{r.refined:.12f}", f"{r.estimate:.6e}"]
        for r in results
    ])
    if exact is None:
        print("\nТочное значение не задано. Для аналитического сравнения укажите antiderivative или exact_value в JSON.")
        return
    if data.get("antiderivative") is not None:
        print(f"\nПервообразная F(x) = {data['antiderivative']}")
    print(f"Точное значение интеграла = {exact:.12f}")
    print_table(["метод", "|I(h1) - I|", "|I(h2) - I|", "|I_RR - I|"], [
        [METHODS[r.method][0], f"{abs(r.coarse - exact):.6e}", f"{abs(r.fine - exact):.6e}", f"{abs(r.refined - exact):.6e}"]
        for r in results
    ])
    for title, attribute in (("с шагом h2", "fine"), ("после уточнения Рунге–Ромберга", "refined")):
        minimum = min(abs(getattr(r, attribute) - exact) for r in results)
        winners = [r for r in results if math.isclose(abs(getattr(r, attribute) - exact), minimum, rel_tol=1e-12, abs_tol=1e-14)]
        names = ", ".join(METHODS[r.method][0] for r in winners)
        print(f"Наиболее точный результат {title}: {names}; минимальная абсолютная погрешность = {minimum:.6e}.")


if __name__ == "__main__":
    parser = parser_for("3.5. Численное интегрирование", 5)
    args = parser.parse_args()
    try:
        run(args.input)
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
        parser.error(str(error))
