import sys
import os
import numpy as np

from cli import build_parser
from config import Config
from integrator import compute_integral, ExpressionParseError, INTEGRATOR_REGISTRY
from solver import solve_ode, SOLVER_REGISTRY
from plotting import plot_function, plot_ode_solution


def _get_plot_output(args, config, default_name):
    if args.output:
        return args.output
    out_dir = config.get('output', 'plot_dir', '.')
    if out_dir != '.':
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, default_name)
    return default_name


def cmd_integrate(args, config):
    method = config.get('integrate', 'method', 'simpson')
    n = config.get('integrate', 'steps', 100)

    try:
        result, expr, integrator = compute_integral(
            expr_str=args.func,
            a=args.a,
            b=args.b,
            method=method,
            n=n
        )

        print('=' * 50)
        print(f'方法: {integrator.display_name}')
        print(f'被积函数: f(x) = {expr}')
        print(f'积分区间: [{args.a}, {args.b}]')
        if integrator.name == 'gauss':
            print(f'高斯点数: n = {n}')
        else:
            print(f'分段数: n = {n}')
        print('-' * 50)
        print(f'积分结果 ≈ {result:.12f}')
        print('=' * 50)

        if args.plot:
            output_file = _get_plot_output(args, config, 'integral_plot')
            img_file, _ = plot_function(
                args.func, args.a, args.b,
                show_integral=True,
                style=config.get('plot', 'style', 'seaborn-v0_8-whitegrid'),
                image_format=config.get('plot', 'image_format', 'png'),
                dpi=config.get('plot', 'dpi', 150),
                figsize=tuple(config.get('plot', 'figsize', [10, 6])),
                output_file=output_file
            )
            print(f'积分区域图像已保存: {img_file}')
    except ExpressionParseError as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)


def cmd_ode(args, config):
    method = config.get('ode', 'method', 'rk4')
    n = config.get('ode', 'steps', 100)
    adaptive = config.get('ode', 'adaptive', False)
    tolerance = config.get('ode', 'tolerance', 1e-6)

    try:
        x, y, expr, output_file, method_used, steps_used = solve_ode(
            expr_str=args.func,
            x0=args.x0,
            y0=args.y0,
            x_end=args.x_end,
            method=method,
            n=n,
            adaptive=adaptive,
            tolerance=tolerance,
            output_file=args.output
        )

        if method_used == 'rk45':
            method_name = '自适应 RK45 (Dormand-Prince)'
            h_avg = (args.x_end - args.x0) / steps_used
            h_min_val = float(min(np.diff(x)))
            h_max_val = float(max(np.diff(x)))
        else:
            solver_cls = SOLVER_REGISTRY.get(method_used)
            method_name = solver_cls.display_name if solver_cls else method_used

        print('=' * 50)
        print(f'方法: {method_name}')
        print(f'方程: dy/dx = {expr}')
        print(f'初始条件: y({args.x0}) = {args.y0}')
        print(f'求解区间: [{args.x0}, {args.x_end}]')
        print(f'总步数: {steps_used}')
        if method_used == 'rk45':
            print(f'容差: {tolerance}')
            print(f'平均步长: {h_avg:.6f}')
            print(f'最小步长: {h_min_val:.6f}')
            print(f'最大步长: {h_max_val:.6f}')
        else:
            print(f'固定步长: h = {(args.x_end - args.x0) / n:.8f}')
        print('-' * 50)
        print('前 5 个点:')
        print(f'  {"x":>15}  {"y":>15}')
        show_count = min(5, len(x))
        for i in range(show_count):
            print(f'  {x[i]:15.10f}  {y[i]:15.10f}')
        if len(x) > 5:
            print('  ...')
            last_i = len(x) - 1
            print(f'  {x[last_i]:15.10f}  {y[last_i]:15.10f}')
        print('-' * 50)
        print(f'结果已写入: {output_file}')
        print('=' * 50)

        if args.plot:
            plot_output = args.plot_output
            if not plot_output:
                out_dir = config.get('output', 'plot_dir', '.')
                if out_dir != '.':
                    os.makedirs(out_dir, exist_ok=True)
                    plot_output = os.path.join(out_dir, 'ode_plot')
                else:
                    plot_output = 'ode_plot'

            img_file = plot_ode_solution(
                x, y, args.x0, args.y0, expr=expr,
                style=config.get('plot', 'style', 'seaborn-v0_8-whitegrid'),
                image_format=config.get('plot', 'image_format', 'png'),
                dpi=config.get('plot', 'dpi', 150),
                figsize=tuple(config.get('plot', 'figsize', [10, 6])),
                output_file=plot_output
            )
            print(f'ODE 解曲线图像已保存: {img_file}')
    except ExpressionParseError as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)


def cmd_plot(args, config):
    try:
        output_file = _get_plot_output(args, config, 'function_plot')
        img_file, expr = plot_function(
            args.func, args.a, args.b,
            show_integral=args.integrate,
            style=config.get('plot', 'style', 'seaborn-v0_8-whitegrid'),
            image_format=config.get('plot', 'image_format', 'png'),
            dpi=config.get('plot', 'dpi', 150),
            figsize=tuple(config.get('plot', 'figsize', [10, 6])),
            output_file=output_file
        )

        print('=' * 50)
        print(f'函数: f(x) = {expr}')
        print(f'区间: [{args.a}, {args.b}]')
        if args.integrate:
            print('模式: 显示积分区域填充')
        print(f'图像格式: {config.get("plot", "image_format", "png")}')
        print(f'分辨率: {config.get("plot", "dpi", 150)} DPI')
        print('-' * 50)
        print(f'图像已保存: {img_file}')
        print('=' * 50)
    except ExpressionParseError as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = Config(args.config)
    config.apply_args(args, args.command)

    if args.command == 'integrate':
        cmd_integrate(args, config)
    elif args.command == 'ode':
        cmd_ode(args, config)
    elif args.command == 'plot':
        cmd_plot(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
