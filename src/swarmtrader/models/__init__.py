"""Modelos profundos (opcional: requiere el extra [deep])."""
from .deep import TORCH_AVAILABLE, DeepConfig, best_device

__all__ = ["DeepConfig", "TORCH_AVAILABLE", "best_device"]
