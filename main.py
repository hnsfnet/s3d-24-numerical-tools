import argparse
import sys
import os
from integration import integrate, INTEGRATE_METHODS, ExpressionParseError
from ode_solver import solve_ode, ODE_METHODS
from plotting import plot_function, plot_ode_solution
from config import load_config, merge_args


def build_parser():
    parser = argparse.ArgumentParser(
        description='数值计算工具：数值积分与常微分方程求解'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    integrate_parser = subparsers.add_parser(
        'integrate',
        help='数值积分：计算定积分近似值'
    )
    integrate_parser.add_argument(
        '--method',
        choices=list(INTEGRATE_METHODS.keys()),
        default=None,
        help='积分方法: trapezoid(梯形), simpson(辛普森), gauss(高斯-勒让德)'
    )
    integrate_parser.add_argument(
        '--from', dest='a', type=float, required=True,
        help='积分下限 a'
    )
    integrate_parser.add_argument(
        '--to', dest='b', type=float, required=True,
        help='积分上限 b'
    )
    integrate_parser.add_argument(
        '--func', type=str, required=True,
        help='被积函数表达式 f(x)，例如 "sin(x)", "x**2 + 2*x"'
    )
    integrate_parser.add_argument(
        '--steps', dest='n', type=int, default=None,
        help='分段数 n (高斯法表示高斯点数)'
    )
    integrate_parser.add_argument(
        '--plot', action='store_true',
        help='同时绘制积分区域图像'
    )
    integrate_parser.add_argument(
        '--output', type=str, default=None,
        help='图像输出文件路径'
    )
    integrate_parser.add_argument(
        '--style', type=str, default=None,
        help='matplotlib 样式，如 seaborn-v0_8-whitegrid, ggplot 等'
    )
    integrate_parser.add_argument(
        '--image-format', type=str, default=None,
        help='图像格式: png, pdf, svg, jpg 等'
    )
    integrate_parser.add_argument(
        '--dpi', type=int, default=None,
        help='图像分辨率 DPI'
    )
    integrate_parser.add_argument(
        '--figsize', type=float, nargs=2, default=None,
        help='图像尺寸 (宽 高)'
    )

    ode_parser = subparsers.add_parser(
        'ode',
        help='常微分方程求解：dy/dx = f(x, y)'
    )
    ode_parser.add_argument(
        '--method',
        choices=list(ODE_METHODS.keys()),
        default=None,
        help='ODE 方法: euler(欧拉), heun(改进欧拉), rk4(四阶龙格-库塔)'
    )
    ode_parser.add_argument(
        '--x0', type=float, required=True,
        help='初始点 x0'
    )
    ode_parser.add_argument(
        '--y0', type=float, required=True,
        help='初始值 y(x0) = y0'
    )
    ode_parser.add_argument(
        '--to', dest='x_end', type=float, required=True,
        help='求解区间终点 x_end'
    )
    ode_parser.add_argument(
        '--func', type=str, required=True,
        help='f(x, y) 表达式，例如 "x + y", "-2*y + exp(-x)"'
    )
    ode_parser.add_argument(
        '--steps', dest='n', type=int, default=None,
        help='步数 n (自适应模式下作为初始步长参考)'
    )
    ode_parser.add_argument(
        '--adaptive', action='store_true', default=None,
        help='使用自适应步长 RK45 方法'
    )
    ode_parser.add_argument(
        '--tolerance', type=float, default=None,
        help='自适应方法的容差 (默认: 1e-6)'
    )
    ode_parser.add_argument(
        '--output', type=str, default='ode_result.csv',
        help='输出 CSV 文件路径'
    )
    ode_parser.add_argument(
        '--plot', action='store_true',
        help='同时绘制 ODE 解曲线'
    )
    ode_parser.add_argument(
        '--plot-output', type=str, default=None,
        help='ODE 图像输出文件路径'
    )

    plot_parser = subparsers.add_parser(
        'plot',
        help='函数图像绘制'
    )
    plot_parser.add_argument(
        '--func', type=str, required=True,
        help='函数表达式 f(x)，例如 "x**2", "sin(x)"'
    )
    plot_parser.add_argument(
        '--from', dest='a', type=float, required=True,
        help='区间起点'
    )
    plot_parser.add_argument(
        '--to', dest='b', type=float, required=True,
        help='区间终点'
    )
    plot_parser.add_argument(
        '--integrate', action='store_true',
        help='填充积分区域显示面积'
    )
    plot_parser.add_argument(
        '--style', type=str, default=None,
        help='matplotlib 样式，如 seaborn-v0_8-whitegrid, ggplot 等'
    )
    plot_parser.add_argument(
        '--image-format', type=str, default=None,
        help='图像格式: png, pdf, svg, jpg 等'
    )
    plot_parser.add_argument(
        '--dpi', type=int, default=None,
        help='图像分辨率 DPI'
    )
    plot_parser.add_argument(
        '--figsize', type=float, nargs=2, default=None,
        help='图像尺寸 (宽 高)'
    )
    plot_parser.add_argument(
        '--output', type=str, default=None,
        help='图像输出文件路径'
    )

    return parser


def cmd_integrate(args, config):
    try:
        result, expr = integrate(
            expr_str=args.func,
            a=args.a,
            b=args.b,
            method=args.method,
            n=args.n
        )
        method_name = {
            'trapezoid': '梯形法则',
            'simpson': '辛普森法则',
            'gauss': '高斯-勒让德求积',
        }[args.method]

        print('=' * 50)
        print(f'方法: {method_name}')
        print(f'被积函数: f(x) = {expr}')
        print(f'积分区间: [{args.a}, {args.b}]')
        if args.method == 'gauss':
            print(f'高斯点数: n = {args.n}')
        else:
            print(f'分段数: n = {args.n}')
        print('-' * 50)
        print(f'积分结果 ≈ {result:.12f}')
        print('=' * 50)

        if args.plot:
            plot_cfg = config.get('plot', {})
            output_file = args.output
            if not output_file:
                out_dir = config.get('output', {}).get('plot_dir', '.')
                if out_dir != '.':
                    os.makedirs(out_dir, exist_ok=True)
                    output_file = os.path.join(out_dir, 'integral_plot')
                else:
                    output_file = 'integral_plot'

            img_file, _ = plot_function(
                args.func, args.a, args.b,
                show_integral=True,
                style=args.style or plot_cfg.get('style', 'seaborn-v0_8-whitegrid'),
                image_format=args.image_format or plot_cfg.get('image_format', 'png'),
                dpi=args.dpi or plot_cfg.get('dpi', 150),
                figsize=tuple(args.figsize) if args.figsize else tuple(plot_cfg.get('figsize', [10, 6])),
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
    try:
        x, y, expr, output_file, method_used, steps_used = solve_ode(
            expr_str=args.func,
            x0=args.x0,
            y0=args.y0,
            x_end=args.x_end,
            method=args.method,
            n=args.n,
            adaptive=args.adaptive,
            tolerance=args.tolerance,
            output_file=args.output
        )

        if method_used == 'rk45':
            method_name = f'自适应 RK45 (Dormand-Prince)'
            h_avg = (args.x_end - args.x0) / steps_used
            h_min_val = min(np.diff(x))
            h_max_val = max(np.diff(x))
        else:
            method_name = {
                'euler': '欧拉法',
                'heun': '改进欧拉法 (Heun)',
                'rk4': '四阶龙格-库塔法 (RK4)',
            }.get(method_used, method_used)

        print('=' * 50)
        print(f'方法: {method_name}')
        print(f'方程: dy/dx = {expr}')
        print(f'初始条件: y({args.x0}) = {args.y0}')
        print(f'求解区间: [{args.x0}, {args.x_end}]')
        print(f'总步数: {steps_used}')
        if method_used == 'rk45':
            print(f'容差: {args.tolerance}')
            print(f'平均步长: {h_avg:.6f}')
            print(f'最小步长: {h_min_val:.6f}')
            print(f'最大步长: {h_max_val:.6f}')
        else:
            print(f'固定步长: h = {(args.x_end - args.x0) / args.n:.8f}')
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
            plot_cfg = config.get('plot', {})
            plot_output = args.plot_output
            if not plot_output:
                out_dir = config.get('output', {}).get('plot_dir', '.')
                if out_dir != '.':
                    os.makedirs(out_dir, exist_ok=True)
                    plot_output = os.path.join(out_dir, 'ode_plot')
                else:
                    plot_output = 'ode_plot'

            img_file = plot_ode_solution(
                x, y, args.x0, args.y0, expr=expr,
                style=plot_cfg.get('style', 'seaborn-v0_8-whitegrid'),
                image_format=plot_cfg.get('image_format', 'png'),
                dpi=plot_cfg.get('dpi', 150),
                figsize=tuple(plot_cfg.get('figsize', [10, 6])),
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
        output_file, expr = plot_function(
            args.func, args.a, args.b,
            show_integral=args.integrate,
            style=args.style or config.get('plot', {}).get('style', 'seaborn-v0_8-whitegrid'),
            image_format=args.image_format or config.get('plot', {}).get('image_format', 'png'),
            dpi=args.dpi or config.get('plot', {}).get('dpi', 150),
            figsize=tuple(args.figsize) if args.figsize else tuple(config.get('plot', {}).get('figsize', [10, 6])),
            output_file=args.output
        )

        print('=' * 50)
        print(f'函数: f(x) = {expr}')
        print(f'区间: [{args.a}, {args.b}]')
        if args.integrate:
            print('模式: 显示积分区域填充')
        print(f'图像格式: {args.image_format or config.get("plot", {}).get("image_format", "png")}')
        print(f'分辨率: {args.dpi or config.get("plot", {}).get("dpi", 150)} DPI')
        print('-' * 50)
        print(f'图像已保存: {output_file}')
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

    config = load_config(args.config)
    args = merge_args(args, config, args.command)

    if args.command == 'integrate':
        cmd_integrate(args, config)
    elif args.command == 'ode':
        cmd_ode(args, config)
    elif args.command == 'plot':
        cmd_plot(args, config)
    else:
        parser.print_help()
        sys.exit(1)


import numpy as np


if __name__ == '__main__':
    main()
