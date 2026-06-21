import sympy as sp
import numpy as np
import sys
from abc import ABC, abstractmethod


class ExpressionParseError(Exception):
    pass


class Integrator(ABC):
    name = 'base'
    display_name = '基类'

    @abstractmethod
    def integrate(self, f, a, b, n):
        pass


class TrapezoidIntegrator(Integrator):
    name = 'trapezoid'
    display_name = '梯形法则'

    def integrate(self, f, a, b, n):
        if n <= 0:
            raise ValueError("n 必须是正整数")
        x = np.linspace(a, b, n + 1)
        y = np.array([float(f(xi)) for xi in x])
        h = (b - a) / n
        result = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))
        return float(result)


class SimpsonIntegrator(Integrator):
    name = 'simpson'
    display_name = '辛普森法则'

    def integrate(self, f, a, b, n):
        if n <= 0:
            raise ValueError("n 必须是正整数")
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n + 1)
        y = np.array([float(f(xi)) for xi in x])
        h = (b - a) / n
        result = (h / 3) * (
            y[0] + y[-1]
            + 4 * np.sum(y[1:-1:2])
            + 2 * np.sum(y[2:-2:2])
        )
        return float(result)


class GaussLegendreIntegrator(Integrator):
    name = 'gauss'
    display_name = '高斯-勒让德求积'

    def integrate(self, f, a, b, n):
        if n <= 0:
            raise ValueError("n 必须是正整数")
        nodes, weights = np.polynomial.legendre.leggauss(n)
        x_transformed = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        y = np.array([float(f(xi)) for xi in x_transformed])
        result = 0.5 * (b - a) * np.sum(weights * y)
        return float(result)


INTEGRATOR_REGISTRY = {
    'trapezoid': TrapezoidIntegrator,
    'simpson': SimpsonIntegrator,
    'gauss': GaussLegendreIntegrator,
}


def get_integrator(method):
    if method not in INTEGRATOR_REGISTRY:
        raise ValueError(f"未知的积分方法: {method}")
    return INTEGRATOR_REGISTRY[method]()


def parse_function(expr_str, variable='x'):
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise ExpressionParseError("函数表达式为空")

    x = sp.Symbol(variable)
    local_dict = {
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
        'sqrt': sp.sqrt, 'abs': sp.Abs,
        'pi': sp.pi, 'e': sp.E,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'sec': sp.sec, 'csc': sp.csc, 'cot': sp.cot,
    }

    try:
        expr = sp.sympify(expr_str, locals=local_dict)
    except sp.SympifyError as e:
        msg = str(e)
        friendly = f"表达式解析错误: \"{expr_str}\"\n"
        if expr_str.count("(") != expr_str.count(")"):
            friendly += "  提示: 检查括号是否匹配"
        elif "SyntaxError" in msg or "expected" in msg.lower():
            friendly += "  提示: 检查语法是否正确"
        friendly += f"\n  详细信息: {msg}"
        raise ExpressionParseError(friendly)
    except Exception as e:
        raise ExpressionParseError(
            f"表达式解析错误: \"{expr_str}\"\n"
            f"  提示: 请检查表达式是否正确，支持 sin, cos, exp, log, sqrt 等常见函数\n"
            f"  详细信息: {e}"
        )

    free = expr.free_symbols
    allowed = {x}
    extra_symbols = free - allowed
    if extra_symbols:
        names = ", ".join(str(s) for s in extra_symbols)
        raise ExpressionParseError(
            f"表达式中包含未定义的符号: {names}\n"
            f"  提示: 只允许使用变量 '{variable}' 和支持的数学函数/常量"
        )

    try:
        f = sp.lambdify(x, expr, modules=['numpy', 'sympy'])
    except Exception as e:
        raise ExpressionParseError(
            f"表达式转换为可计算函数失败: {e}"
        )

    return f, expr


def find_singularities(expr, a, b, variable='x'):
    x = sp.Symbol(variable)
    candidates = []
    try:
        for func in sp.preorder_traversal(expr):
            if isinstance(func, sp.Pow) and func.exp.is_negative:
                base = func.base
                try:
                    sols = sp.solve(base, x)
                    for s in sols:
                        if s.is_real:
                            candidates.append(float(s))
                except Exception:
                    pass
            elif isinstance(func, sp.log):
                try:
                    sols = sp.solve(func.args[0], x)
                    for s in sols:
                        if s.is_real:
                            candidates.append(float(s))
                except Exception:
                    pass
            elif isinstance(func, sp.tan):
                try:
                    sols = sp.solve(sp.cos(func.args[0]), x)
                    for s in sols:
                        if s.is_real:
                            candidates.append(float(s))
                except Exception:
                    pass
            elif isinstance(func, sp.cot):
                try:
                    sols = sp.solve(sp.sin(func.args[0]), x)
                    for s in sols:
                        if s.is_real:
                            candidates.append(float(s))
                except Exception:
                    pass
            elif isinstance(func, sp.sec):
                try:
                    sols = sp.solve(sp.cos(func.args[0]), x)
                    for s in sols:
                        if s.is_real:
                            candidates.append(float(s))
                except Exception:
                    pass
            elif isinstance(func, sp.csc):
                try:
                    sols = sp.solve(sp.sin(func.args[0]), x)
                    for s in sols:
                        if s.is_real:
                            candidates.append(float(s))
                except Exception:
                    pass
    except Exception:
        pass

    singularities = []
    for s in sorted(set(candidates)):
        if a < s < b:
            singularities.append(s)
    return singularities


def evaluate_sample_points(f, a, b, n=100):
    x_vals = np.linspace(a, b, n)
    bad_points = []
    for xi in x_vals:
        try:
            yi = float(f(xi))
            if not np.isfinite(yi):
                bad_points.append(float(xi))
        except Exception:
            bad_points.append(float(xi))
    return bad_points


def check_singularities(expr, f, a, b):
    warnings = []
    sym_sing = find_singularities(expr, a, b)
    for s in sym_sing:
        warnings.append(f"检测到可能的奇点 x = {s:.6f}")
    num_sing = evaluate_sample_points(f, a, b)
    for s in num_sing:
        if not any(abs(s - ss) < 1e-4 for ss in sym_sing):
            warnings.append(f"数值检测到异常点 x ≈ {s:.6f}")
    return warnings


def compute_integral(expr_str, a, b, method='simpson', n=100):
    integrator = get_integrator(method)
    f, expr = parse_function(expr_str, 'x')

    warnings = check_singularities(expr, f, a, b)
    if warnings:
        print("=" * 50, file=sys.stderr)
        print("警告: 积分区间内可能存在奇点或不连续点", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        print("  积分结果可能不准确或无意义", file=sys.stderr)
        print("=" * 50, file=sys.stderr)

    try:
        result = integrator.integrate(f, a, b, n)
        if not np.isfinite(result):
            raise ValueError("积分结果为无穷大或 NaN，可能区间内存在奇点")
    except ZeroDivisionError:
        raise ValueError("计算过程中出现除零错误，区间内可能存在奇点")
    except (ValueError, OverflowError) as e:
        raise ValueError(f"积分计算失败: {e}")
    except Exception as e:
        raise ValueError(f"积分计算失败: {e}")

    return result, expr, integrator
