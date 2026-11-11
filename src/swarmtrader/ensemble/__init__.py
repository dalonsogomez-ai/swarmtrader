"""Combinacion de agentes sin fuga de informacion."""
from .stacking import StackConfig, StackResult, per_date_ic, stack_agents

__all__ = ["StackConfig", "StackResult", "stack_agents", "per_date_ic"]
