import unittest

from lab2_1_nonlinear_equation import f, newton as equation_newton, simple_iteration as equation_iteration
from lab2_2_nonlinear_system import newton as system_newton, residual, simple_iteration as system_iteration


class NonlinearMethodsTests(unittest.TestCase):
    def test_equation(self):
        for method in equation_iteration, equation_newton:
            root, _ = method(1.4, 1e-8, 100)
            self.assertGreater(root, 0)
            self.assertLess(abs(f(root)), 1e-8)
            self.assertAlmostEqual(root, 1.38196601125, places=8)

    def test_system(self):
        for method in system_iteration, system_newton:
            root, _ = method((1.0, 1.2), 1e-8, 100)
            self.assertTrue(all(value > 0 for value in root))
            self.assertLess(residual(root), 1e-8)
            self.assertAlmostEqual(root[0], 1.11780971539, places=8)
            self.assertAlmostEqual(root[1], 1.39198612064, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
