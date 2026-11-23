"""Frozen base forecasts; adapters load model dependencies only when requested."""
from .registry import BASE_MODELS, MAIN_BASES, get_base, check_base_dataset

__all__ = ['BASE_MODELS', 'MAIN_BASES', 'get_base', 'check_base_dataset']
