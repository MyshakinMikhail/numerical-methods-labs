"""Собственные значения и векторы симметричной матрицы методом Якоби."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

from common.io_utils import require_positive_float, require_positive_int, require_square_matrix
from common.matrix_utils import Matrix, Vector, identity, is_symmetric, off_diagonal_norm


@dataclass(frozen=True)
class JacobiEigenResult:
    eigenvalues: Vector
    eigenvectors: Matrix
    iterations: int
    off_diagonal_norm: float


def jacobi_eigen(
    matrix: Matrix, epsilon: float, max_iterations: int
) -> JacobiEigenResult:
    """Найти собственные пары вещественной симметричной матрицы."""
    current = require_square_matrix(matrix)
    epsilon = require_positive_float(epsilon, "epsilon")
    max_iterations = require_positive_int(max_iterations, "max_iterations")
    if not is_symmetric(current):
        raise ValueError("Матрица должна быть симметричной")

    size = len(current)
    eigenvectors = identity(size)
    current_norm = off_diagonal_norm(current)
    if current_norm <= epsilon:
        pairs = sorted(
            ((current[index][index], index) for index in range(size)),
            key=lambda pair: pair[0],
        )
        values = [value for value, _ in pairs]
        sorted_vectors = [
            [eigenvectors[row][index] for _, index in pairs] for row in range(size)
        ]
        return JacobiEigenResult(values, sorted_vectors, 0, current_norm)

    for iteration in range(1, max_iterations + 1):
        p, q = max(
            ((row, column) for row in range(size) for column in range(row + 1, size)),
            key=lambda pair: abs(current[pair[0]][pair[1]]),
        )
        phi = 0.5 * atan2(
            2.0 * current[p][q], current[p][p] - current[q][q]
        )
        cosine = cos(phi)
        sine = sin(phi)
        a_pp = current[p][p]
        a_qq = current[q][q]
        a_pq = current[p][q]

        for index in range(size):
            if index in (p, q):
                continue
            a_ip = current[index][p]
            a_iq = current[index][q]
            new_ip = cosine * a_ip + sine * a_iq
            new_iq = -sine * a_ip + cosine * a_iq
            current[index][p] = current[p][index] = new_ip
            current[index][q] = current[q][index] = new_iq

        current[p][p] = (
            cosine * cosine * a_pp
            + 2.0 * cosine * sine * a_pq
            + sine * sine * a_qq
        )
        current[q][q] = (
            sine * sine * a_pp
            - 2.0 * cosine * sine * a_pq
            + cosine * cosine * a_qq
        )
        current[p][q] = current[q][p] = 0.0

        for row in range(size):
            old_p = eigenvectors[row][p]
            old_q = eigenvectors[row][q]
            eigenvectors[row][p] = cosine * old_p + sine * old_q
            eigenvectors[row][q] = -sine * old_p + cosine * old_q

        current_norm = off_diagonal_norm(current)
        if current_norm <= epsilon:
            pairs = sorted(
                ((current[index][index], index) for index in range(size)),
                key=lambda pair: pair[0],
            )
            values = [value for value, _ in pairs]
            sorted_vectors = [
                [eigenvectors[row][index] for _, index in pairs] for row in range(size)
            ]
            return JacobiEigenResult(values, sorted_vectors, iteration, current_norm)

    raise RuntimeError(f"Метод вращений Якоби не сошёлся за {max_iterations} итераций")
