from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / 'configs'
def read_yaml(path):
    with open(path, encoding='utf8') as f:
        return yaml.safe_load(f)
def load_config(path=None):
    return read_yaml(path or CONFIGS / 'default.yaml')
def platform_spec(kind):
    return read_yaml(CONFIGS / 'platforms' / (kind + '.yaml'))
def world_spec(name):
    p = Path(name)
    if not p.is_file():
        p = CONFIGS / 'worlds' / (name + '.yaml')
    return read_yaml(p)
