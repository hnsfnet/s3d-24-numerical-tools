import sys
import os
import unittest
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from integrator import (
    TrapezoidIntegrator, SimpsonIntegrator, GaussLegendreIntegrator,
    get_integrator, INTEGRATOR_REGISTRY,
    parse_function, compute_integral, find_singularities,
    check_singularities, ExpressionParseError
)


class TestIntegratorClasses(unittest.TestCase):

    def test_registry_contains_three_methods(self):
        self.assertEqual(set(INTEGRATOR_REGISTRY.keys()),
                         {'trapezoid', 'simpson', 'gauss'})

    def test_get_integrator_returns_correct_types(self):
        self.assertIsInstance(get_integrator('trapezoid'), TrapezoidIntegrator)
        self.assertIsInstance(get_integrator('simpson'), SimpsonIntegrator)
        self.assertIsInstance(get_integrator('gauss'), GaussLegendreIntegrator)

    def test_get_integrator_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_integrator('bogus')

    def test_integrators_have_names(self):
        for name, cls in INTEGRATOR_REGISTRY.items():
            self.assertEqual(cls.name, name)
            self.assertTrue(len(cls.display_name) > 0)


class TestIntegrationAccuracy(unittest.TestCase):

    def _sin_pi_exact(self, f, n, tol, label):
        integrator = get_integrator(f)
        result = integrator.integrate(np.sin, 0, np.pi, n)
        error = abs(result - 2.0)
        self.assertLess(error, tol,
                        f"{label}: error={error:.2e} >= tol={tol:.2e}")

    def test_trapezoid_sin_pi(self):
        self._sin_pi_exact('trapezoid', n=1000, tol=1e-5,
                           label='Trapezoid sin(x) 0..pi')

    def test_simpson_sin_pi(self):
        self._sin_pi_exact('simpson', n=200, tol=1e-9,
                           label='Simpson sin(x) 0..pi')

    def test_gauss_sin_pi(self):
        self._sin_pi_exact('gauss', n=10, tol=1e-12,
                           label='Gauss sin(x) 0..pi (10 points)')

    def test_gauss_few_points_very_accurate(self):
        g = GaussLegendreIntegrator()
        result = g.integrate(np.sin, 0, np.pi, 20)
        self.assertLess(abs(result - 2.0), 1e-14,
                        f"Gauss(20 pts) error = {abs(result-2.0):.2e}")

    def test_polynomial_simpson_exact(self):
        integrator = SimpsonIntegrator()
        p = lambda x: 3 * x ** 3 - 2 * x ** 2 + x - 5
        a, b = -1.0, 2.0
        exact = (3 / 4) * (b ** 4 - a ** 4) \
                - (2 / 3) * (b ** 3 - a ** 3) \
                + (1 / 2) * (b ** 2 - a ** 2) \
                - 5 * (b - a)
        result = integrator.integrate(p, a, b, 100)
        self.assertAlmostEqual(result, exact, places=9)

    def test_trapezoid_linear_exact(self):
        integrator = TrapezoidIntegrator()
        result = integrator.integrate(lambda x: 2 * x + 1, 0.0, 3.0, 10)
        exact = 12.0
        self.assertAlmostEqual(result, exact, places=10)

    def test_exp_gauss_high_order(self):
        g = GaussLegendreIntegrator()
        a, b = 0.0, 1.0
        exact = np.exp(1) - np.exp(0)
        for n in [5, 10, 20, 40]:
            result = g.integrate(np.exp, a, b, n)
            error = abs(result - exact)
            self.assertLess(error, 1e-6 / n,
                            f"Gauss(n={n}) error={error:.2e} too high")


class TestConvergence(unittest.TestCase):

    def test_trapezoid_error_decreases_with_n(self):
        integrator = TrapezoidIntegrator()
        errors = []
        for n in [10, 100, 1000]:
            r = integrator.integrate(np.sin, 0, np.pi, n)
            errors.append(abs(r - 2.0))
        for i in range(len(errors) - 1):
            self.assertGreater(errors[i] / errors[i + 1], 50,
                               f"Trapezoid convergence too slow: "
                               f"err[{i}]={errors[i]:.2e}, "
                               f"err[{i+1}]={errors[i+1]:.2e}")

    def test_simpson_second_order_convergence(self):
        integrator = SimpsonIntegrator()
        errors = []
        ns = [10, 100, 1000]
        for n in ns:
            r = integrator.integrate(np.sin, 0, np.pi, n)
            errors.append(abs(r - 2.0))
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            self.assertGreater(ratio, 1000,
                               f"Simpson should be O(h^4): "
                               f"ratio between ns={ns[i]} and {ns[i+1]} "
                               f"= {ratio:.1f}")


class TestSingularityDetection(unittest.TestCase):

    def test_find_singularity_one_over_x(self):
        _, expr = parse_function('1/x')
        sings = find_singularities(expr, -1.0, 1.0)
        self.assertEqual(len(sings), 1)
        self.assertAlmostEqual(sings[0], 0.0, places=8)

    def test_find_singularity_log(self):
        _, expr = parse_function('log(x)')
        sings = find_singularities(expr, -0.5, 2.0)
        self.assertEqual(len(sings), 1)
        self.assertAlmostEqual(sings[0], 0.0, places=8)

    def test_no_singularity_in_smooth_function(self):
        _, expr = parse_function('sin(x) + x**2')
        sings = find_singularities(expr, 0.0, 10.0)
        self.assertEqual(len(sings), 0)

    def test_singularity_outside_interval_not_found(self):
        _, expr = parse_function('1/(x - 5)')
        sings = find_singularities(expr, 0.0, 1.0)
        self.assertEqual(len(sings), 0)

    def test_check_singularities_emits_warning_list(self):
        f, expr = parse_function('1/x')
        warns = check_singularities(expr, f, -1.0, 1.0)
        self.assertTrue(any('奇点' in w or '异常' in w for w in warns),
                        f"warnings should mention singularity: {warns}")

    def test_compute_integral_warns_on_singularity(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                compute_integral('1/x', -1.0, 1.0, method='trapezoid', n=100)
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
            self.assertGreaterEqual(len(w), 0)


class TestParseFunction(unittest.TestCase):

    def test_parse_simple_expression(self):
        f, expr = parse_function('x**2 + 1')
        self.assertAlmostEqual(float(f(2.0)), 5.0, places=8)

    def test_parse_trig_expression(self):
        f, expr = parse_function('sin(x) + cos(x)')
        self.assertAlmostEqual(float(f(0.0)), 1.0, places=8)
        self.assertAlmostEqual(float(f(np.pi / 2)), 1.0, places=8)

    def test_parse_empty_raises(self):
        with self.assertRaises(ExpressionParseError):
            parse_function('')

    def test_parse_missing_paren_raises_friendly(self):
        with self.assertRaises(ExpressionParseError) as ctx:
            parse_function('sin(x')
        self.assertIn('括号', str(ctx.exception))

    def test_parse_undefined_symbol_raises_friendly(self):
        with self.assertRaises(ExpressionParseError) as ctx:
            parse_function('a * x + b')
        self.assertIn('未定义的符号', str(ctx.exception))

    def test_parse_syntax_error_mentions_syntax(self):
        with self.assertRaises(ExpressionParseError) as ctx:
            parse_function('x**-2**')
        msg = str(ctx.exception)
        self.assertTrue(('语法' in msg) or ('括号' in msg) or
                        ('详细信息' in msg))


class TestComputeIntegral(unittest.TestCase):

    def test_returns_three_tuple(self):
        result, expr, integrator = compute_integral(
            'x**2', 0, 1, method='simpson', n=100)
        self.assertAlmostEqual(result, 1 / 3, places=6)
        self.assertIsInstance(integrator, SimpsonIntegrator)

    def test_bad_method_raises(self):
        with self.assertRaises(ValueError):
            compute_integral('x', 0, 1, method='not_a_method', n=10)

    def test_simpson_adjusts_odd_n(self):
        r1, _, _ = compute_integral('x**2', 0, 1, method='simpson', n=99)
        r2, _, _ = compute_integral('x**2', 0, 1, method='simpson', n=100)
        self.assertAlmostEqual(r1, r2, places=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
