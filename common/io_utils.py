"""Чтение и проверка входных данных лабораторных работ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.matrix_utils import Matrix, Vector


def load_json(path: str | Path) -> dict[str, Any]:
    """Загрузить JSON-объект из файла."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Файл не найден: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Некорректный JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Корень JSON должен быть объектом")
    return data


def _to_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} должен содержать только числа")
    return float(value)


def require_square_matrix(value: object, name: str = "matrix") -> Matrix:
    """Проверить и преобразовать непустую квадратную матрицу."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} не должна быть пустой")
    if any(not isinstance(row, list) for row in value):
        raise ValueError(f"{name} должна быть массивом строк")
    row_lengths = [len(row) for row in value]
    if len(set(row_lengths)) != 1:
        raise ValueError(f"Строки {name} должны иметь одинаковую длину")
    if row_lengths[0] != len(value):
        raise ValueError(f"{name} должна быть квадратной")
    return [
        [_to_float(item, f"{name}[{row_index}]") for item in row]
        for row_index, row in enumerate(value)
    ]


def require_vector(value: object, expected_length: int, name: str) -> Vector:
    """Проверить числовой вектор ожидаемой длины."""
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{name} должен иметь длину {expected_length}")
    return [_to_float(item, name) for item in value]


def require_positive_float(value: object, name: str) -> float:
    """Получить положительное вещественное число."""
    result = _to_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} должен быть положительным")
    return result


def require_positive_int(value: object, name: str) -> int:
    """Получить положительное целое число."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} должен быть положительным целым числом")
    return value

