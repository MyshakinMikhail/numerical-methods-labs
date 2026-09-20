"""Форматирование численных результатов лабораторных работ."""

from __future__ import annotations

from typing import Sequence

from common.matrix_utils import Number


def _clean(value: float, precision: int) -> float:
    return 0.0 if abs(value) < 0.5 * 10 ** (-precision) else value


def format_number(value: Number, precision: int = 6) -> str:
    if isinstance(value, complex):
        real = _clean(value.real, precision)
        imaginary = _clean(value.imag, precision)
        if imaginary == 0.0:
            return f"{real:.{precision}f}"
        sign = "+" if imaginary >= 0.0 else "-"
        return f"{real:.{precision}f} {sign} {abs(imaginary):.{precision}f}i"
    return f"{_clean(float(value), precision):.{precision}f}"


def format_vector(vector: Sequence[Number], precision: int = 6) -> str:
    return "[" + ", ".join(format_number(value, precision) for value in vector) + "]"


def format_matrix(matrix: Sequence[Sequence[Number]], precision: int = 6) -> str:
    return "\n".join(
        "[" + "  ".join(format_number(value, precision) for value in row) + "]" for row in matrix
    )

