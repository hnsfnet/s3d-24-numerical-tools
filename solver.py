import sympy as sp
import numpy as np
import csv
import sys
import os
from abc import ABC, abstractmethod
from integrator import ExpressionParseError


def parse_ode_function(expr_str, variable='x', dep_variable='y'):
    if not isinstance(expr_str, str) or not expr_str.strip():
        raise ExpressionParseError("函数表达式为空")

    x = sp.Symbol(variable)
    y = sp.Symbol(dep_variable)
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
    allowed = {x, y}
    extra_symbols = free - allowed
    if extra_symbols:
        names = ", ".join(str(s) for s in extra_symbols)
        raise ExpressionParseError(
            f"表达式中包含未定义的符号: {names}\n"
            f"  提示: 只允许使用变量 '{variable}'、'{dep_variable}' 和支持的数学函数/常量"
        )

    try:
        f = sp.lambdify((x, y), expr, modules=['numpy', 'sympy'])
    except Exception as e:
        raise ExpressionParseError(
            f"表达式转换为可计算函数失败: {e}"
        )

    return f, expr


def check_divergence(y_list, threshold=1e10, window=5):
    if len(y_list) < window + 1:
        return False
    recent = y_list[-window:]
    if any(not np.isfinite(v) for v in recent):
        return True
    max_growth = 0
    for i in range(1, len(recent)):
        if abs(recent[i - 1]) > 1e-10:
            ratio = abs(recent[i] / recent[i - 1])
            if ratio > max_growth:
                max_growth = ratio
    if max_growth > 100:
        return True
    if max(abs(v) for v in recent) > threshold:
        return True
    return False


def _warn_divergence(x_val, y_val, reason="方程为刚性方程，或步长过大",
                     suggestion="减小步长 (--steps)，或使用自适应方法 (--adaptive)"):
    print("=" * 50, file=sys.stderr)
    print("警告: 检测到数值可能发散或剧烈增长", file=sys.stderr)
    print(f"  当前位置: x = {x_val:.6f}, y = {y_val:.6e}", file=sys.stderr)
    print(f"  可能原因: {reason}", file=sys.stderr)
    print(f"  建议: {suggestion}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


class Solver(ABC):
    name = 'base'
    display_name = '基类'

    @abstractmethod
    def solve(self, f, x0, y0, x_end, n, warn_divergence=True):
        pass


class EulerSolver(Solver):
    name = 'euler'
    display_name = '欧拉法'

    def solve(self, f, x0, y0, x_end, n, warn_divergence=True):
        if n <= 0:
            raise ValueError("n 必须是正整数")
        h = (x_end - x0) / n
        x = np.linspace(x0, x_end, n + 1)
        y = np.zeros(n + 1)
        y[0] = y0
        diverged = False
        for i in range(n):
            try:
                k = float(f(x[i], y[i]))
                if not np.isfinite(k):
                    raise ValueError(f"f({x[i]:.6f}, {y[i]:.6f}) 计算结果无效")
            except Exception as e:
                raise ValueError(f"在 x = {x[i]:.6f}, y = {y[i]:.6f} 处计算 f(x, y) 失败: {e}")
            y[i + 1] = y[i] + h * k
            if not np.isfinite(y[i + 1]):
                raise ValueError(
                    f"数值在 x = {x[i + 1]:.6f} 处发散为 {y[i + 1]}，步长可能太大"
                )
            if warn_divergence and not diverged and check_divergence(y[:i + 2]):
                diverged = True
                _warn_divergence(x[i + 1], y[i + 1])
        return x, y


class HeunSolver(Solver):
    name = 'heun'
    display_name = '改进欧拉法 (Heun)'

    def solve(self, f, x0, y0, x_end, n, warn_divergence=True):
        if n <= 0:
            raise ValueError("n 必须是正整数")
        h = (x_end - x0) / n
        x = np.linspace(x0, x_end, n + 1)
        y = np.zeros(n + 1)
        y[0] = y0
        diverged = False
        for i in range(n):
            try:
                k1 = float(f(x[i], y[i]))
                if not np.isfinite(k1):
                    raise ValueError(f"f({x[i]:.6f}, {y[i]:.6f}) 计算结果无效")
            except Exception as e:
                raise ValueError(f"在 x = {x[i]:.6f}, y = {y[i]:.6f} 处计算 f(x, y) 失败: {e}")
            y_predict = y[i] + h * k1
            try:
                k2 = float(f(x[i + 1], y_predict))
                if not np.isfinite(k2):
                    raise ValueError(f"f({x[i + 1]:.6f}, {y_predict:.6f}) 计算结果无效")
            except Exception as e:
                raise ValueError(f"在校正步计算 f(x, y) 失败 (x = {x[i + 1]:.6f}, y ≈ {y_predict:.6f}): {e}")
            y[i + 1] = y[i] + (h / 2) * (k1 + k2)
            if not np.isfinite(y[i + 1]):
                raise ValueError(
                    f"数值在 x = {x[i + 1]:.6f} 处发散为 {y[i + 1]}，步长可能太大"
                )
            if warn_divergence and not diverged and check_divergence(y[:i + 2]):
                diverged = True
                _warn_divergence(x[i + 1], y[i + 1])
        return x, y


class RK4Solver(Solver):
    name = 'rk4'
    display_name = '四阶龙格-库塔法 (RK4)'

    def solve(self, f, x0, y0, x_end, n, warn_divergence=True):
        if n <= 0:
            raise ValueError("n 必须是正整数")
        h = (x_end - x0) / n
        x = np.linspace(x0, x_end, n + 1)
        y = np.zeros(n + 1)
        y[0] = y0
        diverged = False
        for i in range(n):
            try:
                k1 = float(f(x[i], y[i]))
                if not np.isfinite(k1):
                    raise ValueError("k1 无效")
                k2 = float(f(x[i] + h / 2, y[i] + h / 2 * k1))
                if not np.isfinite(k2):
                    raise ValueError("k2 无效")
                k3 = float(f(x[i] + h / 2, y[i] + h / 2 * k2))
                if not np.isfinite(k3):
                    raise ValueError("k3 无效")
                k4 = float(f(x[i] + h, y[i] + h * k3))
                if not np.isfinite(k4):
                    raise ValueError("k4 无效")
            except Exception as e:
                raise ValueError(f"在 x = {x[i]:.6f}, y = {y[i]:.6f} 处计算 RK4 系数失败: {e}")
            y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.isfinite(y[i + 1]):
                raise ValueError(
                    f"数值在 x = {x[i + 1]:.6f} 处发散为 {y[i + 1]}，步长可能太大"
                )
            if warn_divergence and not diverged and check_divergence(y[:i + 2]):
                diverged = True
                _warn_divergence(x[i + 1], y[i + 1])
        return x, y


class RK45AdaptiveSolver:
    name = 'rk45'
    display_name = '自适应 RK45 (Dormand-Prince)'

    def __init__(self, tol=1e-6, h_max=0.1, h_min=1e-10, max_steps=1000000):
        self.tol = tol
        self.h_max = h_max
        self.h_min = h_min
        self.max_steps = max_steps

    def solve(self, f, x0, y0, x_end, warn_divergence=True):
        c2 = 1/5
        c3 = 3/10
        c4 = 4/5
        c5 = 8/9
        c6 = 1.0
        c7 = 1.0

        a21 = 1/5
        a31 = 3/40
        a32 = 9/40
        a41 = 44/45
        a42 = -56/15
        a43 = 32/9
        a51 = 19372/6561
        a52 = -25360/2187
        a53 = 64448/6561
        a54 = -212/729
        a61 = 9017/3168
        a62 = -355/33
        a63 = 46732/5247
        a64 = 49/176
        a65 = -5103/18656
        a71 = 35/384
        a72 = 0
        a73 = 500/1113
        a74 = 125/192
        a75 = -2187/6784
        a76 = 11/84

        b1 = 35/384
        b3 = 500/1113
        b4 = 125/192
        b5 = -2187/6784
        b6 = 11/84

        b1s = 5179/57600
        b3s = 7571/16695
        b4s = 393/640
        b5s = -92097/339200
        b6s = 187/2100
        b7s = 1/40

        x = [x0]
        y = [y0]
        h = self.h_max
        step_count = 0
        diverged = False
        h_warned = False

        while x[-1] < x_end and step_count < self.max_steps:
            step_count += 1
            h = min(h, x_end - x[-1])
            if h < self.h_min:
                if not h_warned:
                    h_warned = True
                    print("=" * 50, file=sys.stderr)
                    print("警告: 步长已达到最小值", file=sys.stderr)
                    print(f"  当前位置: x = {x[-1]:.6f}", file=sys.stderr)
                    print("  可能原因: 方程刚性强，或精度要求过高", file=sys.stderr)
                    print("=" * 50, file=sys.stderr)
                h = self.h_min

            xi = x[-1]
            yi = y[-1]

            try:
                k1 = float(f(xi, yi))
                if not np.isfinite(k1):
                    raise ValueError("k1 无效")
                k2 = float(f(xi + c2 * h, yi + h * a21 * k1))
                if not np.isfinite(k2):
                    raise ValueError("k2 无效")
                k3 = float(f(xi + c3 * h, yi + h * (a31 * k1 + a32 * k2)))
                if not np.isfinite(k3):
                    raise ValueError("k3 无效")
                k4 = float(f(xi + c4 * h, yi + h * (a41 * k1 + a42 * k2 + a43 * k3)))
                if not np.isfinite(k4):
                    raise ValueError("k4 无效")
                k5 = float(f(xi + c5 * h, yi + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)))
                if not np.isfinite(k5):
                    raise ValueError("k5 无效")
                k6 = float(f(xi + c6 * h, yi + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)))
                if not np.isfinite(k6):
                    raise ValueError("k6 无效")
                k7 = float(f(xi + c7 * h, yi + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)))
                if not np.isfinite(k7):
                    raise ValueError("k7 无效")
            except Exception as e:
                raise ValueError(f"在 x = {xi:.6f}, y = {yi:.6f} 处计算 RK45 系数失败: {e}")

            y5 = yi + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
            y4 = yi + h * (b1s * k1 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)

            error = abs(y5 - y4)

            if error <= self.tol or h <= self.h_min:
                x.append(xi + h)
                y.append(y5)
                if not np.isfinite(y5):
                    raise ValueError(
                        f"数值在 x = {xi + h:.6f} 处发散为 {y5}"
                    )
                if warn_divergence and not diverged and check_divergence(y):
                    diverged = True
                    _warn_divergence(xi + h, y5,
                                     reason="方程为刚性方程，或容差设置不当",
                                     suggestion="减小容差 (--tolerance)")

            if error == 0:
                factor = 5.0
            else:
                factor = 0.9 * (self.tol / error) ** 0.2

            factor = max(0.2, min(factor, 5.0))
            h = h * factor
            h = max(self.h_min, min(self.h_max, h))

        if step_count >= self.max_steps:
            print(f'警告: 达到最大步数 {self.max_steps}，可能未完全收敛', file=sys.stderr)

        return np.array(x), np.array(y), step_count


SOLVER_REGISTRY = {
    'euler': EulerSolver,
    'heun': HeunSolver,
    'rk4': RK4Solver,
}


def get_solver(method):
    if method not in SOLVER_REGISTRY:
        raise ValueError(f"未知的 ODE 求解方法: {method}")
    return SOLVER_REGISTRY[method]()


def solve_ode(expr_str, x0, y0, x_end, method='rk4', n=100,
              adaptive=False, tolerance=1e-6, output_file='ode_result.csv'):
    f, expr = parse_ode_function(expr_str, 'x', 'y')

    if adaptive:
        h_init = (x_end - x0) / n if n > 0 else 0.1
        solver = RK45AdaptiveSolver(
            tol=tolerance,
            h_max=h_init * 5,
            h_min=1e-12
        )
        x, y, steps_used = solver.solve(f, x0, y0, x_end)
        method_used = 'rk45'
    else:
        solver = get_solver(method)
        x, y = solver.solve(f, x0, y0, x_end, n)
        steps_used = n
        method_used = method

    out_dir = os.path.dirname(output_file)
    if out_dir and out_dir != '.':
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for xi, yi in zip(x, y):
            writer.writerow([f'{xi:.10f}', f'{yi:.10f}'])

    return x, y, expr, output_file, method_used, steps_used
