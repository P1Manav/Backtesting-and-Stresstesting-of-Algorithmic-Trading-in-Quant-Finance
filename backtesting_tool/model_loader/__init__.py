"""
Model Loader Module
Handles loading and analyzing ANY ML/DL/RL model for backtesting.
"""

from .model_analyzer import ModelAnalyzer
from .model_loader import ModelLoader
from .universal_models import (
    UniversalModelLoader,
    UniversalLSTM,
    UniversalGRU,
    UniversalRNN,
    UniversalCNN,
    UniversalTransformer,
    UniversalMLP,
    UniversalBiLSTM,
    PYTORCH_ARCHITECTURES,
    auto_detect_architecture,
    get_default_config,
    list_available_architectures
)

__all__ = [
    'ModelAnalyzer', 
    'ModelLoader',
    'UniversalModelLoader',
    'UniversalLSTM',
    'UniversalGRU',
    'UniversalRNN',
    'UniversalCNN',
    'UniversalTransformer',
    'UniversalMLP',
    'UniversalBiLSTM',
    'PYTORCH_ARCHITECTURES',
    'auto_detect_architecture',
    'get_default_config',
    'list_available_architectures'
]
