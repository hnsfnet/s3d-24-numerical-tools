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


def rk45_adaptive(f, x0, y0, x_end, tol=1e-6, h_max=0.1, h_min=1e-10, max_steps=1000000):
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
    h = h_max
    step_count = 0

    while x[-1] < x_end and step_count < max_steps:
        step_count += 1
        h = min(h, x_end - x[-1])
        if h < h_min:
            h = h_min

        xi = x[-1]
        yi = y[-1]

        k1 = float(f(xi, yi))
        k2 = float(f(xi + c2 * h, yi + h * a21 * k1))
        k3 = float(f(xi + c3 * h, yi + h * (a31 * k1 + a32 * k2)))
        k4 = float(f(xi + c4 * h, yi + h * (a41 * k1 + a42 * k2 + a43 * k3)))
        k5 = float(f(xi + c5 * h, yi + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)))
        k6 = float(f(xi + c6 * h, yi + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)))
        k7 = float(f(xi + c7 * h, yi + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)))

        y5 = yi + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        y4 = yi + h * (b1s * k1 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)

        error = abs(y5 - y4)

        if error <= tol or h <= h_min:
            x.append(xi + h)
            y.append(y5)

        if error == 0:
            factor = 5.0
        else:
            factor = 0.9 * (tol / error) ** 0.2

        factor = max(0.2, min(factor, 5.0))
        h = h * factor
        h = max(h_min, min(h_max, h))

    if step_count >= max_steps:
        print(f'警告: 达到最大步数 {max_steps}，可能未完全收敛')

    return np.array(x), np.array(y), step_count


ODE_METHODS = {
    'euler': euler_method,
    'heun': heun_method,
    'rk4': rk4_method,
}


def solve_ode(expr_str, x0, y0, x_end, method='rk4', n=100,
              adaptive=False, tolerance=1e-6, output_file='ode_result.csv'):
    f, expr = parse_ode_function(expr_str, 'x', 'y')

    if adaptive:
        h_init = (x_end - x0) / n if n > 0 else 0.1
        x, y, steps_used = rk45_adaptive(
            f, x0, y0, x_end,
            tol=tolerance,
            h_max=h_init * 5,
            h_min=1e-12
        )
        method_used = 'rk45'
    else:
        if method not in ODE_METHODS:
            raise ValueError(f"未知的 ODE 求解方法: {method}")
        func = ODE_METHODS[method]
        x, y = func(f, x0, y0, x_end, n)
        steps_used = n
        method_used = method

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for xi, yi in zip(x, y):
            writer.writerow([f'{xi:.10f}', f'{yi:.10f}'])

    return x, y, expr, output_file, method_used, steps_used
