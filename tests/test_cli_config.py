import sys
import os
import unittest
import tempfile
import shutil
import io
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cli import build_parser
from config import Config, DEFAULT_CONFIG
from integrator import ExpressionParseError
from solver import ExpressionParseError as _spare
import main as main_module


class TestCLIParser(unittest.TestCase):

    def setUp(self):
        self.parser = build_parser()

    def test_integrate_basic_args(self):
        args = self.parser.parse_args([
            'integrate',
            '--from', '0',
            '--to', '3.14',
            '--func', 'sin(x)',
            '--method', 'simpson',
            '--steps', '100'
        ])
        self.assertEqual(args.command, 'integrate')
        self.assertAlmostEqual(args.a, 0.0)
        self.assertAlmostEqual(args.b, 3.14)
        self.assertEqual(args.func, 'sin(x)')
        self.assertEqual(args.method, 'simpson')
        self.assertEqual(args.n, 100)

    def test_integrate_defaults(self):
        args = self.parser.parse_args([
            'integrate',
            '--from', '0',
            '--to', '1',
            '--func', 'x**2',
        ])
        self.assertIsNone(args.method)
        self.assertIsNone(args.n)
        self.assertFalse(args.plot)

    def test_integrate_plot_args(self):
        args = self.parser.parse_args([
            'integrate',
            '--from', '0',
            '--to', '1',
            '--func', 'x',
            '--plot',
            '--output', 'out.png',
            '--dpi', '300',
        ])
        self.assertTrue(args.plot)
        self.assertEqual(args.output, 'out.png')
        self.assertEqual(args.dpi, 300)

    def test_ode_basic_args(self):
        args = self.parser.parse_args([
            'ode',
            '--x0', '0',
            '--y0', '1',
            '--to', '1',
            '--func', 'y',
            '--method', 'rk4',
        ])
        self.assertEqual(args.command, 'ode')
        self.assertAlmostEqual(args.x0, 0.0)
        self.assertAlmostEqual(args.y0, 1.0)
        self.assertAlmostEqual(args.x_end, 1.0)
        self.assertEqual(args.func, 'y')
        self.assertEqual(args.method, 'rk4')

    def test_ode_adaptive(self):
        args = self.parser.parse_args([
            'ode',
            '--x0', '0', '--y0', '1', '--to', '1',
            '--func', 'y',
            '--adaptive',
            '--tolerance', '1e-8',
        ])
        self.assertTrue(args.adaptive)
        self.assertAlmostEqual(args.tolerance, 1e-8)

    def test_plot_args(self):
        args = self.parser.parse_args([
            'plot',
            '--func', 'x**2',
            '--from', '-2',
            '--to', '2',
            '--integrate',
            '--style', 'ggplot',
            '--image-format', 'pdf',
        ])
        self.assertEqual(args.command, 'plot')
        self.assertEqual(args.func, 'x**2')
        self.assertTrue(args.integrate)
        self.assertEqual(args.style, 'ggplot')
        self.assertEqual(args.image_format, 'pdf')

    def test_missing_command_errors(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_integrate_missing_required_errors(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(['integrate', '--func', 'x'])


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_config_when_no_file(self):
        cfg = Config('nonexistent.yaml')
        self.assertEqual(cfg.get('integrate', 'method'), 'simpson')
        self.assertEqual(cfg.get('integrate', 'steps'), 100)
        self.assertEqual(cfg.get('ode', 'method'), 'rk4')
        self.assertEqual(cfg.get('ode', 'adaptive'), False)
        self.assertEqual(cfg.get('plot', 'style'), 'seaborn-v0_8-whitegrid')

    def test_load_yaml_overrides_defaults(self):
        yaml_path = os.path.join(self.tmpdir, 'cfg.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("""
integrate:
  method: gauss
  steps: 20
ode:
  adaptive: true
  tolerance: 1e-8
plot:
  dpi: 300
""")
        cfg = Config(yaml_path)
        self.assertEqual(cfg.get('integrate', 'method'), 'gauss')
        self.assertEqual(cfg.get('integrate', 'steps'), 20)
        self.assertTrue(cfg.get('ode', 'adaptive'))
        self.assertAlmostEqual(cfg.get('ode', 'tolerance'), 1e-8)
        self.assertEqual(cfg.get('plot', 'dpi'), 300)
        self.assertEqual(cfg.get('plot', 'image_format'), 'png')  # still default

    def test_get_missing_returns_default(self):
        cfg = Config()
        self.assertEqual(cfg.get('integrate', 'does_not_exist', 'abc'), 'abc')
        self.assertIsNone(cfg.get('integrate', 'nope'))
        self.assertEqual(cfg.get('nonesections', 'key', 42), 42)

    def test_set_and_get_section(self):
        cfg = Config()
        cfg.set('integrate', 'steps', 500)
        self.assertEqual(cfg.get('integrate', 'steps'), 500)
        self.assertEqual(cfg.get_section('integrate')['steps'], 500)

    def test_invalid_yaml_warns(self):
        yaml_path = os.path.join(self.tmpdir, 'bad.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("integrate:\n  : : :bad yaml::  \n  ")
        old_err = sys.stderr
        sys.stderr = io.StringIO()
        try:
            cfg = Config(yaml_path)
        finally:
            sys.stderr = old_err
        self.assertEqual(cfg.get('integrate', 'method'),
                         DEFAULT_CONFIG['integrate']['method'])

    def test_apply_args_integrate_overrides(self):
        cfg = Config()
        cfg.set('integrate', 'method', 'simpson')
        cfg.set('integrate', 'steps', 100)

        class A:
            method = 'gauss'
            n = 50
            adaptive = None
            tolerance = None
            style = None
            image_format = None
            dpi = None
            figsize = None
            integrate = False

        cfg.apply_args(A(), 'integrate')
        self.assertEqual(cfg.get('integrate', 'method'), 'gauss')
        self.assertEqual(cfg.get('integrate', 'steps'), 50)

    def test_apply_args_none_does_not_override(self):
        cfg = Config()
        cfg.set('integrate', 'method', 'gauss')

        class A:
            method = None
            n = None
            adaptive = None
            tolerance = None
            style = None
            image_format = None
            dpi = None
            figsize = None
            integrate = False

        cfg.apply_args(A(), 'integrate')
        self.assertEqual(cfg.get('integrate', 'method'), 'gauss')

    def test_apply_args_ode_overrides(self):
        cfg = Config()

        class A:
            method = 'euler'
            n = 500
            adaptive = True
            tolerance = 1e-7
            style = None
            image_format = None
            dpi = None
            figsize = None
            integrate = False

        cfg.apply_args(A(), 'ode')
        self.assertEqual(cfg.get('ode', 'method'), 'euler')
        self.assertEqual(cfg.get('ode', 'steps'), 500)
        self.assertTrue(cfg.get('ode', 'adaptive'))
        self.assertAlmostEqual(cfg.get('ode', 'tolerance'), 1e-7)


class TestMainCommandDispatch(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_main_with_args(self, argv):
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        exit_code = 0
        try:
            sys.argv = ['main.py'] + argv
            main_module.main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 1
        finally:
            out = sys.stdout.getvalue()
            err = sys.stderr.getvalue()
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return exit_code, out, err

    def test_integrate_sin_dispatches(self):
        code, out, err = self._run_main_with_args([
            'integrate',
            '--from', '0',
            '--to', '3.141592653589793',
            '--func', 'sin(x)',
            '--steps', '100',
            '--method', 'simpson'
        ])
        self.assertEqual(code, 0, f"stderr={err}")
        self.assertIn('辛普森', out)
        self.assertIn('2.0000', out)

    def test_integrate_bad_expr_friendly_error(self):
        code, out, err = self._run_main_with_args([
            'integrate',
            '--from', '0', '--to', '1',
            '--func', 'sin(x'
        ])
        self.assertNotEqual(code, 0)
        self.assertIn('括号', err)
        self.assertNotIn('SympifyError', err.split('\n')[0] if err else '')

    def test_integrate_undefined_symbol_error(self):
        code, out, err = self._run_main_with_args([
            'integrate',
            '--from', '0', '--to', '1',
            '--func', 'a * x'
        ])
        self.assertNotEqual(code, 0)
        self.assertIn('未定义的符号', err)

    def test_ode_solve_dispatches(self):
        code, out, err = self._run_main_with_args([
            'ode',
            '--x0', '0', '--y0', '1', '--to', '1',
            '--func', 'y',
            '--method', 'rk4',
            '--steps', '50',
        ])
        self.assertEqual(code, 0, f"stderr={err}")
        self.assertIn('RK4', out)
        self.assertIn('龙格', out)
        self.assertTrue(os.path.exists('ode_result.csv'))

    def test_ode_adaptive_dispatches(self):
        code, out, err = self._run_main_with_args([
            'ode',
            '--x0', '0', '--y0', '1', '--to', '1',
            '--func', 'y',
            '--adaptive',
            '--tolerance', '1e-6',
        ])
        self.assertEqual(code, 0, f"stderr={err}")
        self.assertIn('RK45', out)
        self.assertIn('自适应', out)

    def test_ode_bad_expr_error(self):
        code, out, err = self._run_main_with_args([
            'ode',
            '--x0', '0', '--y0', '1', '--to', '1',
            '--func', 'y + exp(x'
        ])
        self.assertNotEqual(code, 0)
        self.assertIn('括号', err)

    def test_plot_dispatches(self):
        code, out, err = self._run_main_with_args([
            'plot',
            '--func', 'x**2',
            '--from', '-2',
            '--to', '2',
        ])
        self.assertEqual(code, 0, f"stderr={err}")
        self.assertIn('x**2', out)
        self.assertTrue(os.path.exists('function_plot.png'))

    def test_plot_integrate_mode(self):
        code, out, err = self._run_main_with_args([
            'plot',
            '--func', 'sin(x)',
            '--from', '0', '--to', '3.14159',
            '--integrate',
        ])
        self.assertEqual(code, 0, f"stderr={err}")
        self.assertIn('积分区域填充', out)


class TestConfigPriorityEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        with open('config.yaml', 'w', encoding='utf-8') as f:
            f.write("""
integrate:
  method: gauss
  steps: 10
""")

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_main_with_args(self, argv):
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        exit_code = 0
        try:
            sys.argv = ['main.py'] + argv
            main_module.main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 1
        finally:
            out = sys.stdout.getvalue()
            err = sys.stderr.getvalue()
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return exit_code, out, err

    def test_config_file_overridden_by_cli(self):
        code, out, err = self._run_main_with_args([
            'integrate',
            '--from', '0', '--to', '3.141592653589793',
            '--func', 'sin(x)',
            '--method', 'simpson',
            '--steps', '100'
        ])
        self.assertEqual(code, 0, f"stderr={err}")
        self.assertIn('辛普森', out)
        self.assertIn('100', out)

    def test_config_file_used_when_no_cli(self):
        code, out, err = self._run_main_with_args([
            'integrate',
            '--from', '0', '--to', '3.141592653589793',
            '--func', 'sin(x)',
        ])
        self.assertEqual(code, 0, f"stderr={err}")
        self.assertIn('高斯', out)
        self.assertIn('10', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
