import argparse
from integrator import INTEGRATOR_REGISTRY
from solver import SOLVER_REGISTRY


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
        choices=list(INTEGRATOR_REGISTRY.keys()),
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
        choices=list(SOLVER_REGISTRY.keys()),
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
