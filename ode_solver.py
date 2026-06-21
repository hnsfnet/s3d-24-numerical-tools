import sympy as sp
import numpy as np
import csv


def parse_ode_function(expr_str, variable='x', dep_variable='y'):
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
    expr = sp.sympify(expr_str, locals=local_dict)
    f = sp.lambdify((x, y), expr, modules=['numpy', 'sympy'])
    return f, expr


def euler_method(f, x0, y0, x_end, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")
    h = (x_end - x0) / n
    x = np.linspace(x0, x_end, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * float(f(x[i], y[i]))
    return x, y


def heun_method(f, x0, y0, x_end, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")
    h = (x_end - x0) / n
    x = np.linspace(x0, x_end, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = float(f(x[i], y[i]))
        y_predict = y[i] + h * k1
        k2 = float(f(x[i + 1], y_predict))
        y[i + 1] = y[i] + (h / 2) * (k1 + k2)
    return x, y


def rk4_method(f, x0, y0, x_end, n):
    if n <= 0:
        raise ValueError("n 必须是正整数")
    h = (x_end - x0) / n
    x = np.linspace(x0, x_end, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = float(f(x[i], y[i]))
        k2 = float(f(x[i] + h / 2, y[i] + h / 2 * k1))
        k3 = float(f(x[i] + h / 2, y[i] + h / 2 * k2))
        k4 = float(f(x[i] + h, y[i] + h * k3))
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, y


ODE_METHODS = {
    'euler': euler_method,
    'heun': heun_method,
    'rk4': rk4_method,
}


def solve_ode(expr_str, x0, y0, x_end, method='rk4', n=100, output_file='ode_result.csv'):
    if method not in ODE_METHODS:
        raise ValueError(f"未知的 ODE 求解方法: {method}")
    f, expr = parse_ode_function(expr_str, 'x', 'y')
    func = ODE_METHODS[method]
    x, y = func(f, x0, y0, x_end, n)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for xi, yi in zip(x, y):
            writer.writerow([f'{xi:.10f}', f'{yi:.10f}'])

    return x, y, expr, output_file
