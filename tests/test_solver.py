import sys
import os
import unittest
import io
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from solver import (
    EulerSolver, HeunSolver, RK4Solver, RK45AdaptiveSolver,
    get_solver, SOLVER_REGISTRY,
    parse_ode_function, solve_ode, check_divergence,
    ExpressionParseError
)


class TestSolverClasses(unittest.TestCase):

    def test_registry_three_methods(self):
        self.assertEqual(set(SOLVER_REGISTRY.keys()),
                         {'euler', 'heun', 'rk4'})

    def test_get_solver_types(self):
        self.assertIsInstance(get_solver('euler'), EulerSolver)
        self.assertIsInstance(get_solver('heun'), HeunSolver)
        self.assertIsInstance(get_solver('rk4'), RK4Solver)

    def test_get_solver_invalid(self):
        with self.assertRaises(ValueError):
            get_solver('invalid')

    def test_solver_names_match_keys(self):
        for name, cls in SOLVER_REGISTRY.items():
            self.assertEqual(cls.name, name)
            self.assertTrue(len(cls.display_name) > 0)


class TestAnalyticSolutionExponential(unittest.TestCase):
    """ dy/dx = y, y(0)=1 => y = e^x """

    def setUp(self):
        self.f = lambda x, y: y
        self.x0, self.y0, self.x_end = 0.0, 1.0, 1.0
        self.exact_end = np.exp(1.0)

    def test_euler_accuracy(self):
        s = EulerSolver()
        x, y = s.solve(self.f, self.x0, self.y0, self.x_end, n=1000)
        err = abs(y[-1] - self.exact_end)
        self.assertLess(err, 0.01, f"Euler error={err:.2e} too large")

    def test_heun_better_than_euler(self):
        euler = EulerSolver()
        heun = HeunSolver()
        n = 50
        _, y_e = euler.solve(self.f, self.x0, self.y0, self.x_end, n)
        _, y_h = heun.solve(self.f, self.x0, self.y0, self.x_end, n)
        err_e = abs(y_e[-1] - self.exact_end)
        err_h = abs(y_h[-1] - self.exact_end)
        self.assertGreater(err_e / err_h, 10,
                           f"Heun should be much better: "
                           f"Euler err={err_e:.2e}, Heun err={err_h:.2e}")

    def test_rk4_high_accuracy(self):
        s = RK4Solver()
        x, y = s.solve(self.f, self.x0, self.y0, self.x_end, n=50)
        err = abs(y[-1] - self.exact_end)
        self.assertLess(err, 1e-8, f"RK4 error={err:.2e} too large")

    def test_rk4_linear_exact(self):
        """ dy/dx = 2x, y(0)=0 => y=x^2, RK4 should be very accurate """
        s = RK4Solver()
        x, y = s.solve(lambda x, y: 2 * x, 0.0, 0.0, 1.0, n=10)
        for xi, yi in zip(x, y):
            self.assertAlmostEqual(yi, xi ** 2, places=8,
                                   msg=f"x={xi}: y={yi} vs x^2={xi**2}")


class TestAnalyticSolutionCoupled(unittest.TestCase):
    """ dy/dx = -2y + e^(-x), y(0)=1 => y = e^(-x) """

    def setUp(self):
        self.f = lambda x, y: -2 * y + np.exp(-x)
        self.x0, self.y0, self.x_end = 0.0, 1.0, 1.0
        self.y_exact_end = np.exp(-1.0)

    def test_rk4_accuracy(self):
        s = RK4Solver()
        x, y = s.solve(self.f, self.x0, self.y0, self.x_end, n=100)
        err = abs(y[-1] - self.y_exact_end)
        self.assertLess(err, 1e-6, f"RK4 error={err:.2e}")

    def test_heun_accuracy(self):
        s = HeunSolver()
        x, y = s.solve(self.f, self.x0, self.y0, self.x_end, n=200)
        err = abs(y[-1] - self.y_exact_end)
        self.assertLess(err, 1e-3, f"Heun error={err:.2e}")


class TestRK45Adaptive(unittest.TestCase):

    def test_exp_ode_high_accuracy(self):
        f = lambda x, y: y
        rk45 = RK45AdaptiveSolver(tol=1e-8, h_max=0.1, h_min=1e-12)
        x, y, steps = rk45.solve(f, 0.0, 1.0, 1.0)
        err = abs(y[-1] - np.exp(1.0))
        self.assertLess(err, 1e-6, f"RK45 adaptive error={err:.2e}")
        self.assertGreater(steps, 0)

    def test_adaptive_better_than_fixed_rk4_on_stiff(self):
        f_stiff = lambda x, y: -1000 * y + 3000 - 2000 * np.exp(-x)
        x0, y0, x_end = 0.0, 0.0, 0.1
        exact_at_end = 3.0 - 2.0 * np.exp(-0.1) - (3.0 - 2.0 - 0.0) * np.exp(-1000 * 0.1)
        exact_at_end = 3.0 - 2.001 * np.exp(-0.1)

        rk4 = RK4Solver()
        _, y_rk4 = rk4.solve(f_stiff, x0, y0, x_end, n=10)
        err_rk4 = abs(y_rk4[-1] - exact_at_end)

        rk45 = RK45AdaptiveSolver(tol=1e-6, h_max=0.05, h_min=1e-12)
        _, y_a, _ = rk45.solve(f_stiff, x0, y0, x_end)
        err_adaptive = abs(y_a[-1] - exact_at_end)

        self.assertLess(err_adaptive, err_rk4,
                        f"Adaptive should be more accurate than fixed "
                        f"(stiff problem): adaptive err={err_adaptive:.2e}, "
                        f"fixed RK4 err={err_rk4:.2e}")

    def test_adaptive_reduces_steps_for_smooth(self):
        f = lambda x, y: y
        rk45 = RK45AdaptiveSolver(tol=1e-6, h_max=0.2, h_min=1e-10)
        _, _, steps = rk45.solve(f, 0.0, 1.0, 1.0)
        self.assertLess(steps, 50, f"Adaptive steps={steps} too many for smooth ODE")


class TestDivergenceDetection(unittest.TestCase):

    def test_check_divergence_stable_series(self):
        ys = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        self.assertFalse(check_divergence(ys))

    def test_check_divergence_rapid_growth(self):
        ys = [1.0, 2.0, 4.0, 1e4, 1e12, 1e20]
        self.assertTrue(check_divergence(ys))

    def test_check_divergence_short_list(self):
        self.assertFalse(check_divergence([1.0, 2.0, 3.0]))

    def test_check_divergence_nan_or_inf(self):
        self.assertTrue(check_divergence([1.0, 2.0, 3.0, 4.0, np.inf, np.nan]))

    def test_check_divergence_huge_value(self):
        self.assertTrue(check_divergence([1.0, 2.0, 3.0, 4.0, 5.0, 1e15]))

    def test_solver_emits_divergence_warning_on_stiff_large_h(self):
        f_stiff = lambda x, y: -1000 * y + 3000 - 2000 * np.exp(-x)
        s = RK4Solver()

        old_stderr = sys.stderr
        buf = io.StringIO()
        sys.stderr = buf
        try:
            s.solve(f_stiff, 0.0, 0.0, 0.1, n=5)
        except Exception:
            pass
        sys.stderr = old_stderr
        output = buf.getvalue()
        self.assertIn('发散', output,
                      "Solver should print divergence warning to stderr")


class TestParseOdeFunction(unittest.TestCase):

    def test_parse_simple(self):
        f, expr = parse_ode_function('x + y')
        self.assertAlmostEqual(float(f(1.0, 2.0)), 3.0, places=8)

    def test_parse_math_functions(self):
        f, expr = parse_ode_function('-2*y + exp(-x)')
        val = float(f(0.0, 1.0))
        self.assertAlmostEqual(val, -1.0, places=6)

    def test_parse_empty(self):
        with self.assertRaises(ExpressionParseError):
            parse_ode_function('')

    def test_parse_missing_paren(self):
        with self.assertRaises(ExpressionParseError) as ctx:
            parse_ode_function('sin(x + y')
        self.assertIn('括号', str(ctx.exception))

    def test_parse_undefined_symbol(self):
        with self.assertRaises(ExpressionParseError) as ctx:
            parse_ode_function('a * x + b * y')
        self.assertIn('未定义的符号', str(ctx.exception))


class TestSolveOdeHighLevel(unittest.TestCase):

    def test_solve_ode_rk4_returns_tuple(self):
        import tempfile
        import csv
        with tempfile.TemporaryDirectory() as td:
            out_csv = os.path.join(td, 'result.csv')
            x, y, expr, out_file, method_used, steps = solve_ode(
                'y', 0, 1, 1, method='rk4', n=100, output_file=out_csv)
            self.assertEqual(method_used, 'rk4')
            self.assertEqual(steps, 100)
            self.assertEqual(out_file, out_csv)
            self.assertTrue(os.path.exists(out_csv))
            with open(out_csv, encoding='utf-8') as f:
                r = csv.reader(f)
                header = next(r)
                self.assertEqual(header, ['x', 'y'])
                rows = list(r)
                self.assertEqual(len(rows), 101)

    def test_solve_ode_adaptive_returns_rk45(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out_csv = os.path.join(td, 'ada.csv')
            _, _, _, _, method_used, _ = solve_ode(
                'y', 0, 1, 1, adaptive=True, tolerance=1e-6, output_file=out_csv)
            self.assertEqual(method_used, 'rk45')


if __name__ == '__main__':
    unittest.main(verbosity=2)
