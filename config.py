import os
import yaml


DEFAULT_CONFIG = {
    'integrate': {
        'method': 'simpson',
        'steps': 100,
    },
    'ode': {
        'method': 'rk4',
        'steps': 100,
        'adaptive': False,
        'tolerance': 1e-6,
    },
    'plot': {
        'style': 'seaborn-v0_8-whitegrid',
        'image_format': 'png',
        'dpi': 150,
        'figsize': [10, 6],
    },
    'output': {
        'ode_csv_dir': '.',
        'plot_dir': '.',
    },
}


def load_config(config_path='config.yaml'):
    config = DEFAULT_CONFIG.copy()

    if not os.path.exists(config_path):
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)

        if user_config is None:
            return config

        _deep_update(config, user_config)
    except yaml.YAMLError as e:
        print(f'警告: 配置文件解析错误: {e}', file=sys.stderr)
    except Exception as e:
        print(f'警告: 读取配置文件时出错: {e}', file=sys.stderr)

    return config


def _deep_update(base, updates):
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def merge_args(args, config, command):
    if command == 'integrate':
        cfg = config.get('integrate', {})
        if args.method is None and 'method' in cfg:
            args.method = cfg['method']
        if args.n is None and 'steps' in cfg:
            args.n = cfg['steps']
    elif command == 'ode':
        cfg = config.get('ode', {})
        if args.method is None and 'method' in cfg:
            args.method = cfg['method']
        if args.n is None and 'steps' in cfg:
            args.n = cfg['steps']
        if args.adaptive is None and 'adaptive' in cfg:
            args.adaptive = cfg['adaptive']
        if args.tolerance is None and 'tolerance' in cfg:
            args.tolerance = cfg['tolerance']

        out_cfg = config.get('output', {})
        if args.output == 'ode_result.csv' and 'ode_csv_dir' in out_cfg:
            out_dir = out_cfg['ode_csv_dir']
            if out_dir and out_dir != '.':
                os.makedirs(out_dir, exist_ok=True)
                args.output = os.path.join(out_dir, os.path.basename(args.output))
    elif command == 'plot':
        cfg = config.get('plot', {})
        if args.style is None and 'style' in cfg:
            args.style = cfg['style']
        if args.image_format is None and 'image_format' in cfg:
            args.image_format = cfg['image_format']
        if args.dpi is None and 'dpi' in cfg:
            args.dpi = cfg['dpi']
        if args.figsize is None and 'figsize' in cfg:
            args.figsize = tuple(cfg['figsize'])
        if args.output is None:
            out_cfg = config.get('output', {})
            if 'plot_dir' in out_cfg and out_cfg['plot_dir'] != '.':
                out_dir = out_cfg['plot_dir']
                os.makedirs(out_dir, exist_ok=True)
                base_name = 'integral_plot' if args.integrate else 'function_plot'
                args.output = os.path.join(out_dir, base_name)

    return args


import sys
