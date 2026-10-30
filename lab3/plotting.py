"""Сохранение самостоятельных SVG-графиков средствами стандартной библиотеки."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Sequence


@dataclass
class Series:
    label: str
    points: Sequence[tuple[float, float]]
    color: str
    dashed: bool = False
    markers: bool = False


def save_plot(path: Path, title: str, series: Sequence[Series], x_star: float | None = None) -> None:
    width, height = 1080, 610 + 27 * len(series)
    left, top, right, bottom = 96, 78, 1036, 536
    all_points = [point for item in series for point in item.points]
    x_min, x_max = min(x for x, _ in all_points), max(x for x, _ in all_points)
    y_min, y_max = min(y for _, y in all_points), max(y for _, y in all_points)
    dx, dy = x_max - x_min or 1.0, y_max - y_min or 1.0
    x_min, x_max = x_min - 0.035 * dx, x_max + 0.035 * dx
    y_min, y_max = y_min - 0.08 * dy, y_max + 0.08 * dy

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<title>{escape(title)}</title>',
           '<style>text { font-family: Arial, sans-serif; fill: #18212f; font-size: 15px; }</style>',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left}" y="34" style="font-size:23px;font-weight:bold">{escape(title)}</text>']
    for i in range(7):
        x = x_min + (x_max - x_min) * i / 6
        y = y_min + (y_max - y_min) * i / 6
        svg.extend([
            f'<path d="M {px(x):.3f} {top} V {bottom}" stroke="#e2e8f0"/>',
            f'<path d="M {left} {py(y):.3f} H {right}" stroke="#e2e8f0"/>',
            f'<text x="{px(x):.3f}" y="{bottom + 27}" text-anchor="middle">{x:.4g}</text>',
            f'<text x="{left - 14}" y="{py(y) + 5:.3f}" text-anchor="end">{y:.4g}</text>',
        ])
    if x_min <= 0 <= x_max:
        svg.append(f'<path d="M {px(0):.3f} {top} V {bottom}" stroke="#a1aab8"/>')
    if y_min <= 0 <= y_max:
        svg.append(f'<path d="M {left} {py(0):.3f} H {right}" stroke="#a1aab8"/>')
    svg.extend([
        f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="none" stroke="#64748b"/>',
        f'<text x="{right + 16}" y="{bottom + 5}">x</text>',
        f'<text x="{left - 8}" y="{top - 15}">y</text>',
    ])
    if x_star is not None:
        svg.extend([
            f'<path d="M {px(x_star):.3f} {top} V {bottom}" stroke="#9f1239" stroke-dasharray="4 5" opacity="0.6"/>',
            f'<text x="{px(x_star) + 7:.3f}" y="{top + 22}">x* = {x_star:.6g}</text>',
        ])
    for i, item in enumerate(series):
        dash = ' stroke-dasharray="9 6"' if item.dashed else ""
        if item.markers:
            for x, y in item.points:
                svg.append(f'<circle cx="{px(x):.3f}" cy="{py(y):.3f}" r="5.5" fill="white" stroke="{item.color}" stroke-width="2.5"/>')
        else:
            points = " ".join(f"{px(x):.3f},{py(y):.3f}" for x, y in item.points)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{item.color}" stroke-width="2.7"{dash}/>')
        legend_y = bottom + 68 + i * 27
        if item.markers:
            svg.append(f'<circle cx="{left + 18}" cy="{legend_y - 5}" r="5" fill="white" stroke="{item.color}" stroke-width="2.5"/>')
        else:
            svg.append(f'<path d="M {left} {legend_y - 5} h 36" stroke="{item.color}" stroke-width="2.7"{dash}/>')
        svg.append(f'<text x="{left + 49}" y="{legend_y}">{escape(item.label)}</text>')
    svg.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"График: {path}")
