import argparse
import sys
from integration import integrate, INTEGRATE_METHODS
from ode_solver import solve_ode, ODE_METHODS


def build_parser():
    parser = argparse.ArgumentParser(
        description='数值计算工具：数值积分与常微分方程求解'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    integrate_parser = subparsers.add_parser(
        'integrate',
        help='数值积分：计算定积分近似值'
    )
    integrate_parser.add_argument(
        '--method',
        choices=list(INTEGRATE_METHODS.keys()),
        default='simpson',
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
        '--steps', dest='n', type=int, default=100,
        help='分段数 n (高斯法表示高斯点数)'
    )

    ode_parser = subparsers.add_parser(
        'ode',
        help='常微分方程求解：dy/dx = f(x, y)'
    )
    ode_parser.add_argument(
        '--method',
        choices=list(ODE_METHODS.keys()),
        default='rk4',
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
        '--steps', dest='n', type=int, default=100,
        help='步数 n'
    )
    ode_parser.add_argument(
        '--output', type=str, default='ode_result.csv',
        help='输出 CSV 文件路径'
    )

    return parser


def cmd_integrate(args):
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
    except Exception as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)


def cmd_ode(args):
    try:
        x, y, expr, output_file = solve_ode(
            expr_str=args.func,
            x0=args.x0,
            y0=args.y0,
            x_end=args.x_end,
            method=args.method,
            n=args.n,
            output_file=args.output
        )
        method_name = {
            'euler': '欧拉法',
            'heun': '改进欧拉法 (Heun)',
            'rk4': '四阶龙格-库塔法 (RK4)',
        }[args.method]

        print('=' * 50)
        print(f'方法: {method_name}')
        print(f'方程: dy/dx = {expr}')
        print(f'初始条件: y({args.x0}) = {args.y0}')
        print(f'求解区间: [{args.x0}, {args.x_end}]')
        print(f'步数: n = {args.n}')
        print(f'步长: h = {(args.x_end - args.x0) / args.n:.8f}')
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
    except Exception as e:
        print(f'错误: {e}', file=sys.stderr)
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'integrate':
        cmd_integrate(args)
    elif args.command == 'ode':
        cmd_ode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
