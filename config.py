import os
import sys
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


class Config:
    def __init__(self, config_path='config.yaml'):
        self._config = self._deep_copy(DEFAULT_CONFIG)
        self.load(config_path)

    def _deep_copy(self, obj):
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        return obj

    def _coerce_value(self, key, value, default_val):
        if value is None:
            return value
        if isinstance(default_val, bool) and not isinstance(value, bool):
            try:
                return bool(value)
            except (ValueError, TypeError):
                return value
        if isinstance(default_val, float):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if isinstance(default_val, int) and not isinstance(value, bool):
            try:
                if isinstance(value, float):
                    return int(value)
                return int(value)
            except (ValueError, TypeError):
                return value
        return value

    def _deep_update(self, base, updates):
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                if key in base:
                    value = self._coerce_value(key, value, base[key])
                base[key] = value

    def load(self, config_path='config.yaml'):
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)

            if user_config is None:
                return

            self._deep_update(self._config, user_config)
        except yaml.YAMLError as e:
            print(f'警告: 配置文件解析错误: {e}', file=sys.stderr)
        except Exception as e:
            print(f'警告: 读取配置文件时出错: {e}', file=sys.stderr)

    def get(self, section, key, default=None):
        section_data = self._config.get(section, {})
        if not isinstance(section_data, dict):
            return default
        value = section_data.get(key, default)
        if default is not None and value is not None and not isinstance(value, type(default)):
            try:
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int):
                    value = int(float(value))
                elif isinstance(default, float):
                    value = float(value)
            except (ValueError, TypeError):
                pass
        return value

    def get_section(self, section):
        return self._config.get(section, {})

    def set(self, section, key, value):
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def apply_args(self, args, command):
        if command == 'integrate':
            if args.method is not None:
                self.set('integrate', 'method', args.method)
            if args.n is not None:
                self.set('integrate', 'steps', args.n)
        elif command == 'ode':
            if args.method is not None:
                self.set('ode', 'method', args.method)
            if args.n is not None:
                self.set('ode', 'steps', args.n)
            if args.adaptive is not None:
                self.set('ode', 'adaptive', args.adaptive)
            if args.tolerance is not None:
                self.set('ode', 'tolerance', args.tolerance)
        elif command == 'plot':
            if args.style is not None:
                self.set('plot', 'style', args.style)
            if args.image_format is not None:
                self.set('plot', 'image_format', args.image_format)
            if args.dpi is not None:
                self.set('plot', 'dpi', args.dpi)
            if args.figsize is not None:
                self.set('plot', 'figsize', list(args.figsize))

    @property
    def all(self):
        return self._config
