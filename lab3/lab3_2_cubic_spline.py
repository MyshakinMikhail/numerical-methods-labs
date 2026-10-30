"""3.2. Натуральный кубический сплайн; трёхдиагональная система решается прогонкой."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from common import (load_json, number, parser_for, polynomial_text, print_nodes,
                    print_table, read_table, sample, validate_table)
from plotting import Series, save_plot


@dataclass
class Segment:
    left: float
    right: float
    a: float
    b: float
    c: float
    d: float

    def value(self, x: float) -> float:
        t = x - self.left
        return self.a + t * (self.b + t * (self.c + t * self.d))

    def derivative(self, x: float) -> float:
        t = x - self.left
        return self.b + t * (2 * self.c + 3 * self.d * t)

    def second_derivative(self, x: float) -> float:
        return 2 * self.c + 6 * self.d * (x - self.left)


def natural_spline(x: Sequence[float], y: Sequence[float]) -> list[Segment]:
    validate_table(x, y)
    size = len(x)
    h = [right - left for left, right in zip(x, x[1:])]
    slopes = [(right - left) / step for left, right, step in zip(y, y[1:], h)]
    # c_i = S''(x_i)/2; c_0 = c_{n-1} = 0.
    # h_{i-1}*c_{i-1} + 2*(h_{i-1}+h_i)*c_i + h_i*c_{i+1}
    #     = 3*(slopes_i - slopes_{i-1}).
    alpha, beta, c = [0.0] * size, [0.0] * size, [0.0] * size
    for i in range(1, size - 1):
        denominator = 2 * (h[i - 1] + h[i]) + h[i - 1] * alpha[i - 1]
        alpha[i] = -h[i] / denominator
        beta[i] = (3 * (slopes[i] - slopes[i - 1]) - h[i - 1] * beta[i - 1]) / denominator
    for i in range(size - 2, 0, -1):
        c[i] = alpha[i] * c[i + 1] + beta[i]
    return [Segment(
        x[i], x[i + 1], y[i],
        slopes[i] - h[i] * (2 * c[i] + c[i + 1]) / 3,
        c[i], (c[i + 1] - c[i]) / (3 * h[i]),
    ) for i in range(size - 1)]


def interval_index(segments: Sequence[Segment], point: float) -> int:
    if not segments or not math.isfinite(point) or not segments[0].left <= point <= segments[-1].right:
        raise ValueError("Точка x* должна принадлежать отрезку интерполяции")
    return min(bisect.bisect_right([segment.left for segment in segments], point) - 1, len(segments) - 1)


def spline_value(segments: Sequence[Segment], point: float) -> float:
    return segments[interval_index(segments, point)].value(point)


def run(path: str | Path, plot_path: Path | None = None) -> None:
    data = load_json(path, ["x", "y", "x_star"])
    x, y = read_table(data)
    point = number(data["x_star"])
    segments = natural_spline(x, y)
    index = interval_index(segments, point)
    print("3.2. Натуральный кубический сплайн, вариант 14")
    print_nodes(x, y)
    print("\nS_i(x) = a_i + b_i*t + c_i*t² + d_i*t³, t = x - левая граница участка.")
    print("Интервалы нумеруются с 1; внутренний узел относится к участку справа.")
    print_table(["№", "интервал", "a_i", "b_i", "c_i", "d_i"], [
        [i + 1, f"[{s.left:g}, {s.right:g}{']' if i == len(segments) - 1 else ')'}",
         f"{s.a:.12f}", f"{s.b:.12f}", f"{s.c:.12f}", f"{s.d:.12f}"]
        for i, s in enumerate(segments)
    ])
    for i, segment in enumerate(segments, start=1):
        print(f"S_{i}(x) = {polynomial_text([segment.a, segment.b, segment.c, segment.d], 't')}, t = x - {segment.left:g}")
    value = spline_value(segments, point)
    print(f"\nx* = {point:.12f}; интервал № {index + 1}: [{segments[index].left:g}, {segments[index].right:g}]")
    print(f"S(x*) = {value:.12f}")
    print("\nПроверка прохождения через узлы:")
    print_table(["i", "x_i", "y_i", "S(x_i)", "|S(x_i) - y_i|"], [
        [i, f"{xi:.6f}", f"{yi:.12f}", f"{spline_value(segments, xi):.12f}", f"{abs(spline_value(segments, xi) - yi):.3e}"]
        for i, (xi, yi) in enumerate(zip(x, y))
    ])
    print("\nГраничные условия:")
    print(f"S''({x[0]:g}) = {segments[0].second_derivative(x[0]):.12e}")
    print(f"S''({x[-1]:g}) = {segments[-1].second_derivative(x[-1]):.12e}")
    print("\nПроверка гладкости в местах стыковки (разности слева и справа):")
    print_table(["x_i", "ΔS", "ΔS'", "ΔS''"], [
        [f"{left.right:.6f}", f"{left.value(left.right) - right.value(right.left):.3e}",
         f"{left.derivative(left.right) - right.derivative(right.left):.3e}",
         f"{left.second_derivative(left.right) - right.second_derivative(right.left):.3e}"]
        for left, right in zip(segments, segments[1:])
    ])
    if plot_path is not None:
        colors = ["#2563eb", "#ea580c", "#15803d", "#7c3aed"]
        curves = [Series(f"Участок {i + 1}: [{s.left:g}, {s.right:g}]", sample(s.value, s.left, s.right), colors[i % len(colors)])
                  for i, s in enumerate(segments)]
        save_plot(plot_path, "3.2. Натуральный кубический сплайн, вариант 14", curves + [
            Series("Исходные узлы", list(zip(x, y)), "#18212f", markers=True),
            Series(f"(x*, S(x*)) = ({point:g}, {value:.6f})", [(point, value)], "#be123c", markers=True),
        ], point)


if __name__ == "__main__":
    parser = parser_for("3.2. Кубический сплайн", 2, "spline.svg")
    args = parser.parse_args()
    try:
        run(args.input, None if args.no_plot else args.plot)
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError) as error:
        parser.error(str(error))
