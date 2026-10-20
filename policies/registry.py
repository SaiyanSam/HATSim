"""Policy discovery. A policy lives in policies/<name>/main.py."""
from importlib import import_module
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def available_policies():
    return sorted(p.name for p in ROOT.iterdir()
                  if p.is_dir() and (p / "main.py").is_file()
                  and _NAME.fullmatch(p.name))


def select_actions(sim, policy="highest_reward"):
    if not isinstance(policy, str) or not _NAME.fullmatch(policy):
        raise ValueError(f"Invalid policy name: {policy!r}")
    if policy not in available_policies():
        raise ValueError(f"Unknown policy {policy!r}. Available: {', '.join(available_policies())}")
    method = import_module(f"policies.{policy}.main")
    if not callable(getattr(method, "select_actions", None)):
        raise TypeError(f"Policy {policy!r} must define select_actions(sim)")
    actions = method.select_actions(sim)
    if not isinstance(actions, dict):
        raise TypeError(f"Policy {policy!r} returned {type(actions).__name__}; expected dict")
    return actions
