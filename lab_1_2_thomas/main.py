"""Запуск лабораторной 1.2."""

from __future__ import annotations

import sys
from pathlib import Path

from common.formatting import format_number, format_vector
from common.io_utils import load_json, require_vector
from lab_1_2_thomas.thomas import thomas_solve, tridiagonal_residual_inf_norm


def run(path: str | Path) -> None:
    data = load_json(path)
    raw_diagonal = data.get("diagonal")
    if not isinstance(raw_diagonal, list) or not raw_diagonal:
        raise ValueError("diagonal не должна быть пустой")
    size = len(raw_diagonal)
    diagonal = require_vector(raw_diagonal, size, "diagonal")
    lower = require_vector(data.get("lower"), size - 1, "lower")
    upper = require_vector(data.get("upper"), size - 1, "upper")
    rhs = require_vector(data.get("rhs"), size, "rhs")

    result = thomas_solve(lower, diagonal, upper, rhs)
    print("Прогоночные коэффициенты:")
    for index, (alpha, beta) in enumerate(zip(result.alpha, result.beta), start=1):
        print(
            f"i={index}: α={format_number(alpha)}, β={format_number(beta)}"
        )
    print(f"\nРешение x = {format_vector(result.solution)}")
    residual = tridiagonal_residual_inf_norm(
        lower, diagonal, upper, result.solution, rhs
    )
    print(f"Норма невязки ||Ax-b||∞ = {format_number(residual)}")


def main() -> None:
    default_path = Path(__file__).with_name("input.json")
    try:
        run(sys.argv[1] if len(sys.argv) > 1 else default_path)
    except (ValueError, OSError) as error:
        raise SystemExit(f"Ошибка: {error}") from error


if __name__ == "__main__":
    main()

