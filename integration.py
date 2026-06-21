import sympy as sp
import numpy as np


def parse_function(expr_str, variable='x'):
    x = sp.Symbol(variable)
    local_dict = {
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
        'sqrt': sp.sqrt, 'abs': sp.Abs,
        'pi': sp.pi, 'e': sp.E,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sec': sp.sec, 'csc': sp.csc, 'cot': sp.cot,
    }
    expr = sp.sympify(expr_str, locals=local_dict)
    f = sp.lambdify(x, expr, modules=['numpy', 'sympy'])
    return f, expr


def trapezoidal_rule(f, a, b, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")
    x = np.linspace(a, b, n + 1)
    y = np.array([float(f(xi)) for xi in x])
    h = (b - a) / n
    result = h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))
    return float(result)


def simpsons_rule(f, a, b, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")
    if n % 2 != 0:
        raise ValueError("辛普森法则要求 n 为偶数")
    x = np.linspace(a, b, n + 1)
    y = np.array([float(f(xi)) for xi in x])
    h = (b - a) / n
    result = (h / 3) * (
        y[0] + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-2:2])
    )
    return float(result)


def gauss_legendre_quadrature(f, a, b, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    x_transformed = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    y = np.array([float(f(xi)) for xi in x_transformed])
    result = 0.5 * (b - a) * np.sum(weights * y)
    return float(result)


INTEGRATE_METHODS = {
    'trapezoid': trapezoidal_rule,
    'simpson': simpsons_rule,
    'gauss': gauss_legendre_quadrature,
}


def integrate(expr_str, a, b, method='simpson', n=100):
    if method not in INTEGRATE_METHODS:
        raise ValueError(f"未知的积分方法: {method}")
    f, expr = parse_function(expr_str, 'x')
    func = INTEGRATE_METHODS[method]

    if method == 'simpson' and n % 2 != 0:
        n += 1

    result = func(f, a, b, n)
    return result, expr
