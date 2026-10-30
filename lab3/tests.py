"""Проверки математических свойств методов и заданных данных варианта 14."""

import json
import math
import unittest

from common import ROOT, make_function, number, polynomial_value
from lab3_1_interpolation import (divided_differences, lagrange, lagrange_coefficients,
                                  newton, newton_coefficients)
from lab3_2_cubic_spline import interval_index, natural_spline, spline_value
from lab3_3_least_squares import least_squares, squared_error
from lab3_4_differentiation import differentiate
from lab3_5_integration import METHODS, compare_methods, integrate, runge_romberg


def input_data(task):
    return json.loads((ROOT / f"inputs/{task}.json").read_text(encoding="utf-8"))


class InterpolationTests(unittest.TestCase):
    def test_reproduce_cubic_on_nonuniform_grid(self):
        x = [-1.0, 0.1, 1.2, 3.0]
        coefficients = [2.0, -1.0, 3.0, -0.5]
        y = [polynomial_value(coefficients, xi) for xi in x]
        differences = divided_differences(x, y)[0]
        for point in [*x, -0.3, 0.8, 2.0]:
            expected = polynomial_value(coefficients, point)
            self.assertAlmostEqual(lagrange(x, y, point), expected, places=11)
            self.assertAlmostEqual(newton(x, differences, point), expected, places=11)
        for actual in (lagrange_coefficients(x, y), newton_coefficients(x, differences)):
            for value, expected in zip(actual, coefficients):
                self.assertAlmostEqual(value, expected, places=11)

    def test_variant_both_node_sets(self):
        data = input_data(1)
        function = make_function(data["function"])
        point = number(data["x_star"])
        for node_set in data["node_sets"]:
            with self.subTest(node_set=node_set["name"]):
                x = [number(value) for value in node_set["x"]]
                y = [function(xi) for xi in x]
                differences = divided_differences(x, y)[0]
                for xi, yi in zip(x, y):
                    self.assertAlmostEqual(lagrange(x, y, xi), yi, places=12)
                    self.assertAlmostEqual(newton(x, differences, xi), yi, places=12)
                self.assertLess(abs(lagrange(x, y, point) - newton(x, differences, point)), 1e-12)

    def test_duplicate_nodes_rejected(self):
        for method in (lagrange, divided_differences):
            with self.assertRaises(ValueError):
                if method is lagrange:
                    method([0, 0, 1], [1, 2, 3], 0.5)
                else:
                    method([0, 0, 1], [1, 2, 3])


class SplineTests(unittest.TestCase):
    def test_variant_nodes_boundaries_and_continuity(self):
        data = input_data(2)
        x, y = data["x"], data["y"]
        segments = natural_spline(x, y)
        for xi, yi in zip(x, y):
            self.assertAlmostEqual(spline_value(segments, xi), yi, places=12)
        self.assertAlmostEqual(segments[0].second_derivative(x[0]), 0.0, places=12)
        self.assertAlmostEqual(segments[-1].second_derivative(x[-1]), 0.0, places=12)
        for left, right in zip(segments, segments[1:]):
            for method in ("value", "derivative", "second_derivative"):
                self.assertAlmostEqual(getattr(left, method)(left.right), getattr(right, method)(right.left), places=11)
        self.assertEqual(interval_index(segments, 1.5), 1)
        self.assertAlmostEqual(spline_value(segments, 1.5), 4965253 / 3780000, places=12)
        self.assertEqual(interval_index(segments, x[-1]), 3)
        self.assertEqual(interval_index(segments, x[1]), 1)

    def test_linear_data_on_nonuniform_grid(self):
        x = [-1.0, -0.2, 1.5, 4.0]
        segments = natural_spline(x, [2 * xi + 3 for xi in x])
        for point in [-1.0, 0.0, 2.0, 4.0]:
            self.assertAlmostEqual(spline_value(segments, point), 2 * point + 3, places=12)

    def test_three_node_reference_solution(self):
        # Симметрия и натуральные условия дают c_1 = -3/2, b_0 = 3/2.
        segments = natural_spline([0, 1, 2], [0, 1, 0])
        self.assertAlmostEqual(spline_value(segments, 0.5), 0.6875, places=12)
        self.assertAlmostEqual(spline_value(segments, 1.5), 0.6875, places=12)

    def test_two_nodes_give_line(self):
        segments = natural_spline([1, 3], [4, 8])
        self.assertAlmostEqual(spline_value(segments, 2), 6)
        self.assertEqual(segments[0].c, 0)
        self.assertEqual(segments[0].d, 0)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            spline_value(natural_spline([0, 1], [0, 1]), 1.1)


class LeastSquaresTests(unittest.TestCase):
    def test_reproduce_known_polynomials(self):
        x = [-2, -1, 0, 1, 2, 3]
        for expected in ([1, -2], [1, -2, 0.5], [1, -2, 0.5, 2]):
            y = [polynomial_value(expected, xi) for xi in x]
            actual = least_squares(x, y, len(expected) - 1)
            for value, reference in zip(actual, expected):
                self.assertAlmostEqual(value, reference, places=11)

    def test_variant_optimality_and_decreasing_errors(self):
        data = input_data(3)
        x, y = data["x"], data["y"]
        previous = math.inf
        # Независимые опорные SSE получены рациональной ортогональной проекцией.
        reference_sse = [8.679034651428571, 2.355706084642857, 0.355741072142857]
        for degree in (1, 2, 3):
            coefficients = least_squares(x, y, degree)
            sse = squared_error(x, y, coefficients)
            self.assertAlmostEqual(sse, reference_sse[degree - 1], places=11)
            self.assertLess(sse, previous)
            previous = sse
            # Условие минимума МНК: невязка ортогональна каждому столбцу A.
            for power in range(degree + 1):
                moment = math.fsum((polynomial_value(coefficients, xi) - yi) * xi**power for xi, yi in zip(x, y))
                self.assertAlmostEqual(moment, 0, delta=1e-9)

    def test_insufficient_nodes_rejected(self):
        with self.assertRaises(ValueError):
            least_squares([0, 1, 2], [1, 2, 3], 3)


class DifferentiationTests(unittest.TestCase):
    def test_exact_quadratic_on_uniform_and_nonuniform_grids(self):
        for x in ([0, 1, 2, 3, 4], [-2, -0.5, 0.2, 2, 5]):
            y = [3 * xi**2 - 2 * xi + 4 for xi in x]
            for point in (x[0], x[1], 0.7, x[-1]):
                first, second, index = differentiate(x, y, point)
                self.assertAlmostEqual(first, 6 * point - 2, places=11)
                self.assertAlmostEqual(second, 6, places=11)
                self.assertGreaterEqual(index, 0)
                self.assertLessEqual(index + 2, len(x) - 1)

    def test_invalid_input_rejected(self):
        for x, y, point in (([0, 1], [0, 1], 0.5), ([0, 1, 2], [0, 1, 4], 3), ([0, 1, 2], [0, 1, 4], math.nan)):
            with self.assertRaises(ValueError):
                differentiate(x, y, point)


class IntegrationTests(unittest.TestCase):
    def test_constant_exact_for_all_methods(self):
        for method in METHODS:
            self.assertAlmostEqual(integrate(lambda x: 3, -1, 2, 0.5, method), 9, places=12)

    def test_linear_exact_for_second_and_fourth_order(self):
        for method in ("midpoint", "trapezoid", "simpson"):
            self.assertAlmostEqual(integrate(lambda x: 2 * x + 1, 0, 2, 0.25, method), 6, places=12)

    def test_rectangle_direction_for_increasing_function(self):
        left = integrate(lambda x: x**2, 0, 1, 0.25, "left")
        right = integrate(lambda x: x**2, 0, 1, 0.25, "right")
        self.assertLess(left, 1 / 3)
        self.assertGreater(right, 1 / 3)

    def test_simpson_exact_on_cubic(self):
        self.assertAlmostEqual(integrate(lambda x: x**3, 0, 2, 0.25, "simpson"), 4, places=12)

    def test_runge_romberg_for_quadratic(self):
        results = compare_methods(lambda x: x**2, 0, 1, 0.5, 0.25)
        for result in results:
            if result.method in ("midpoint", "trapezoid", "simpson"):
                self.assertAlmostEqual(result.refined, 1 / 3, places=12)
                self.assertAlmostEqual(result.estimate, abs(result.fine - 1 / 3), places=12)

    def test_runge_romberg_with_ratio_three(self):
        for result in compare_methods(lambda x: x**4, 0, 1, 0.5, 1 / 6):
            if result.method == "simpson":
                self.assertAlmostEqual(result.refined, 0.2, places=12)
                self.assertAlmostEqual(result.estimate, abs(result.fine - 0.2), places=12)

    def test_stated_convergence_orders(self):
        for method, (_, order) in METHODS.items():
            coarse_error = abs(integrate(math.exp, 0, 1, 1 / 16, method) - math.expm1(1))
            fine_error = abs(integrate(math.exp, 0, 1, 1 / 32, method) - math.expm1(1))
            self.assertAlmostEqual(coarse_error / fine_error, 2**order, delta=0.05)

    def test_invalid_grids_and_runge_steps_rejected(self):
        for a, b, h, method in ((0, 1, 0.3, "left"), (0, 1, 0, "left"),
                                (1, 0, 0.1, "left"), (0, 1, 0.2, "simpson")):
            with self.assertRaises(ValueError):
                integrate(math.exp, a, b, h, method)
        with self.assertRaises(ValueError):
            runge_romberg(1, 2, 0.1, 0.2, 2)


class ExpressionTests(unittest.TestCase):
    def test_formulas_and_pi(self):
        self.assertAlmostEqual(number("3*pi/16"), 3 * math.pi / 16)
        self.assertAlmostEqual(make_function("tg(x) + x")(0.4), math.tan(0.4) + 0.4)
        self.assertAlmostEqual(make_function("ln(x) + sqrt(x)")(4), math.log(4) + 2)

    def test_arbitrary_code_rejected(self):
        for source in ("__import__('os')", "x.real", "[x][0]", "(lambda: 1)()"):
            with self.assertRaises(ValueError):
                make_function(source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
