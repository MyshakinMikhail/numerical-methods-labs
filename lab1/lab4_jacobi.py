"""Лабораторная 1.4: собственные значения и векторы методом вращений Якоби."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from common import (
    Matrix,
    Vector,
    format_number,
    identity,
    load_json,
    matmul,
    max_matrix_difference,
    print_matrix,
    print_vector,
    square_matrix,
)


def off_diagonal_norm(matrix: Matrix) -> float:
    return math.sqrt(
        sum(matrix[row][column] ** 2 for row in range(len(matrix)) for column in range(row + 1, len(matrix)))
    )


def jacobi_eigen(
    matrix: Matrix, epsilon: float, max_iterations: int
) -> tuple[Vector, Matrix, int, float]:
    current = square_matrix(matrix)
    size = len(current)
    if any(abs(current[row][column] - current[column][row]) > 1e-12 for row in range(size) for column in range(row + 1, size)):
        raise ValueError("Метод Якоби требует симметричную матрицу")
    eigenvectors = identity(size)

    for iteration in range(max_iterations + 1):
        norm = off_diagonal_norm(current)
        if norm <= epsilon:
            pairs = sorted((current[index][index], index) for index in range(size))
            values = [value for value, _ in pairs]
            vectors = [[eigenvectors[row][column] for _, column in pairs] for row in range(size)]
            return values, vectors, iteration, norm
        if iteration == max_iterations:
            break

        p, q = max(
            ((row, column) for row in range(size) for column in range(row + 1, size)),
            key=lambda pair: abs(current[pair[0]][pair[1]]),
        )
        angle = 0.5 * math.atan2(2.0 * current[p][q], current[p][p] - current[q][q])
        cosine, sine = math.cos(angle), math.sin(angle)
        a_pp, a_qq, a_pq = current[p][p], current[q][q], current[p][q]

        for index in range(size):
            if index in (p, q):
                continue
            a_ip, a_iq = current[index][p], current[index][q]
            current[index][p] = current[p][index] = cosine * a_ip + sine * a_iq
            current[index][q] = current[q][index] = -sine * a_ip + cosine * a_iq

        current[p][p] = cosine**2 * a_pp + 2.0 * cosine * sine * a_pq + sine**2 * a_qq
        current[q][q] = sine**2 * a_pp - 2.0 * cosine * sine * a_pq + cosine**2 * a_qq
        current[p][q] = current[q][p] = 0.0

        for row in range(size):
            old_p, old_q = eigenvectors[row][p], eigenvectors[row][q]
            eigenvectors[row][p] = cosine * old_p + sine * old_q
            eigenvectors[row][q] = -sine * old_p + cosine * old_q

    raise RuntimeError(f"Метод Якоби не сошёлся за {max_iterations} итераций")


def run(path: str | Path) -> None:
    data = load_json(path)
    matrix = square_matrix(data["matrix"])
    values, vectors, iterations, norm = jacobi_eigen(
        matrix, float(data["epsilon"]), int(data["max_iterations"])
    )
    diagonal = [[values[row] if row == column else 0.0 for column in range(len(values))] for row in range(len(values))]
    left, right = matmul(matrix, vectors), matmul(vectors, diagonal)

    print(f"Количество вращений: {iterations}")
    print(f"Норма внедиагональной части: {format_number(norm)}")
    print_vector("Собственные значения:", values)
    for column in range(len(values)):
        print_vector(f"Собственный вектор h{column + 1}:", [vectors[row][column] for row in range(len(values))])
    print_matrix("Матрица собственных векторов V:", vectors)
    print_matrix("A · V:", left)
    print_matrix("V · Λ:", right)
    print(f"max|A·V-V·Λ| = {format_number(max_matrix_difference(left, right))}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "inputs/4.json")
