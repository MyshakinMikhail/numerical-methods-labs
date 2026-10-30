"""Ввод, проверка данных и небольшие операции для лабораторной 3."""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent
FUNCTIONS = {name: getattr(math, name) for name in (
    "sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "log10", "sqrt", "sinh", "cosh"
)}
FUNCTIONS.update({"abs": abs, "tg": math.tan, "ln": math.log})
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow}


def make_function(source: str, allow_x: bool = True) -> Callable[[float], float]:
    """Разобрать математическую запись без eval и выполнения произвольного кода."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Формула должна быть непустой строкой")
    try:
        tree = ast.parse(source, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"Неверная формула: {source}") from error

    def check(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return
        if isinstance(node, ast.Name) and node.id in (("pi", "e", "x") if allow_x else ("pi", "e")):
            return
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            check(node.left)
            check(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            check(node.operand)
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
            check(node.args[0])
            return
        raise ValueError(f"Недопустимый элемент формулы: {ast.dump(node)}")

    def calculate(node: ast.AST, x: float) -> float:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return {"pi": math.pi, "e": math.e, "x": x}[node.id]
        if isinstance(node, ast.BinOp):
            return BINARY[type(node.op)](calculate(node.left, x), calculate(node.right, x))
        if isinstance(node, ast.UnaryOp):
            value = calculate(node.operand, x)
            return -value if isinstance(node.op, ast.USub) else value
        return FUNCTIONS[node.func.id](calculate(node.args[0], x))

    check(tree)

    def function(x: float) -> float:
        value = float(calculate(tree, x))
        if not math.isfinite(value):
            raise ValueError(f"Формула {source!r} дала не конечное значение при x={x}")
        return value

    return function


def number(value: float | str) -> float:
    result = make_function(value, allow_x=False)(0.0) if isinstance(value, str) else float(value)
    if not math.isfinite(result):
        raise ValueError("Числа должны быть конечными")
    return result


def load_json(path: str | Path, required: Sequence[str]) -> dict:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Корень JSON должен быть объектом")
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise ValueError(f"Заполните данные своего варианта в {path}: {', '.join(missing)}")
    return data


def validate_table(x: Sequence[float], y: Sequence[float], minimum: int = 2) -> None:
    if len(x) != len(y) or len(x) < minimum:
        raise ValueError(f"Нужны одинаковые размеры x, y и не менее {minimum} узлов")
    if not all(math.isfinite(value) for value in (*x, *y)):
        raise ValueError("Таблица должна содержать конечные числа")
    if any(left >= right for left, right in zip(x, x[1:])):
        raise ValueError("Узлы x должны строго возрастать")


def read_table(data: dict, minimum: int = 2) -> tuple[list[float], list[float]]:
    x = [number(value) for value in data["x"]]
    y = [number(value) for value in data["y"]]
    validate_table(x, y, minimum)
    return x, y


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    text = [[str(value) for value in row] for row in rows]
    widths = [max([len(title)] + [len(row[i]) for row in text]) for i, title in enumerate(headers)]
    print("  ".join(title.rjust(width) for title, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in text:
        print("  ".join(value.rjust(width) for value, width in zip(row, widths)))


def print_nodes(x: Sequence[float], y: Sequence[float]) -> None:
    print("Исходная таблица:")
    print_table(["i", "x_i", "y_i"], [
        [i, f"{xi:.12f}", f"{yi:.12f}"] for i, (xi, yi) in enumerate(zip(x, y))
    ])


def polynomial_value(coefficients: Sequence[float], x: float) -> float:
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def multiply_by_linear(coefficients: Sequence[float], root: float) -> list[float]:
    """Умножить многочлен с коэффициентами от a₀ к aₙ на (x - root)."""
    result = [0.0] * (len(coefficients) + 1)
    for i, value in enumerate(coefficients):
        result[i] -= root * value
        result[i + 1] += value
    return result


def polynomial_text(coefficients: Sequence[float], variable: str = "x") -> str:
    parts = []
    for degree in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[degree]
        if coefficient == 0.0:
            continue
        term = f"{abs(coefficient):.12g}"
        if degree:
            term += f"*{variable}" + (f"^{degree}" if degree > 1 else "")
        sign = "-" if coefficient < 0 else "+"
        parts.append((sign + " " if parts else "-" if coefficient < 0 else "") + term)
    return " ".join(parts) or "0"


def factor(root: float) -> str:
    return f"(x {'+' if root < 0 else '-'} {abs(root):.12g})"


def sample(function: Callable[[float], float], left: float, right: float, count: int = 501) -> list[tuple[float, float]]:
    return [(x, function(x)) for x in (left + (right - left) * i / (count - 1) for i in range(count))]


def parser_for(title: str, task: int, graph: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / f"inputs/{task}.json",
                        help="Путь к JSON с исходными данными")
    if graph:
        parser.add_argument("--plot", type=Path, default=ROOT / "graphs" / graph, help="Путь к SVG-графику")
        parser.add_argument("--no-plot", action="store_true", help="Выполнить только расчёт")
    return parser
