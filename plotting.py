import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from integration import parse_function


def plot_function(expr_str, a, b, show_integral=False, num_points=500,
                  title=None, xlabel='x', ylabel='f(x)',
                  figsize=(10, 6), style='seaborn-v0_8-whitegrid',
                  output_file=None, image_format='png', dpi=150):
    f, expr = parse_function(expr_str, 'x')
    x = np.linspace(a, b, num_points)
    y = np.array([float(f(xi)) for xi in x])

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, color='#2563eb', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')

    if show_integral:
        ax.fill_between(x, y, 0, where=(x >= a) & (x <= b),
                        alpha=0.3, color='#3b82f6', edgecolor='#1d4ed8',
                        linewidth=1, linestyle='--',
                        label=f'积分区域 $[{a}, {b}]$')
        ax.axhline(y=0, color='#64748b', linewidth=0.8, linestyle='-')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or f'$f(x) = {sp.latex(expr)}$', fontsize=14, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()

    if output_file:
        if not output_file.lower().endswith(f'.{image_format.lower()}'):
            output_file = f'{output_file}.{image_format}'
        plt.savefig(output_file, format=image_format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_file, expr
    else:
        plt.savefig(f'function_plot.{image_format}', format=image_format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return f'function_plot.{image_format}', expr


def plot_ode_solution(x, y, x0, y0, expr=None,
                      title=None, xlabel='x', ylabel='y(x)',
                      figsize=(10, 6), style='seaborn-v0_8-whitegrid',
                      output_file=None, image_format='png', dpi=150):
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, color='#dc2626', linewidth=2, label='数值解 $y(x)$')

    ax.scatter([x0], [y0], color='#16a34a', s=100, zorder=5,
               edgecolor='black', linewidth=1.5,
               label=f'初始条件 $y({x0}) = {y0}$')
    ax.annotate(f'$({x0}, {y0})$',
                xy=(x0, y0),
                xytext=(x0 + 0.05 * (max(x) - min(x)), y0 + 0.05 * (max(y) - min(y))),
                fontsize=11, color='#166534', fontweight='bold')

    if expr is not None:
        import sympy as sp
        ax.set_title(title or f'ODE 解: $dy/dx = {sp.latex(expr)}$', fontsize=14, pad=15)
    else:
        ax.set_title(title or 'ODE 数值解', fontsize=14, pad=15)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()

    if output_file:
        if not output_file.lower().endswith(f'.{image_format.lower()}'):
            output_file = f'{output_file}.{image_format}'
        plt.savefig(output_file, format=image_format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_file
    else:
        plt.savefig(f'ode_plot.{image_format}', format=image_format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return f'ode_plot.{image_format}'


import sympy as sp
