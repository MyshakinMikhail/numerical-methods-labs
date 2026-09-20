"""Собственный пример с комплексно-сопряжённой парой."""

from __future__ import annotations

import sys
from pathlib import Path

from common.formatting import format_vector
from lab_1_5_qr.main import read_input
from lab_1_5_qr.qr import qr_eigenvalues


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("complex_input.json")
    try:
        matrix, epsilon, max_iterations = read_input(path)
        own_values = qr_eigenvalues(matrix, epsilon, max_iterations).eigenvalues
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка: {error}") from error

    print(f"Собственная реализация: {format_vector(own_values)}")
    try:
        import numpy as np
    except ImportError:
        print("NumPy не установлен. Для сравнения: python -m pip install -r requirements.txt")
        return
    numpy_values = np.linalg.eigvals(np.array(matrix, dtype=float))
    print(f"Проверка NumPy: {format_vector(numpy_values.tolist())}")


if __name__ == "__main__":
    main()

