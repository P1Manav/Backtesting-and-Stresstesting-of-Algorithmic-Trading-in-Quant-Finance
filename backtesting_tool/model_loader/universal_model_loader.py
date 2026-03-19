"""Load TorchScript models"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from .model_analyzer import ModelAnalyzer

class UniversalModelLoader:
    """Load TorchScript models with automatic architecture detection"""

    def __init__(self, device: Optional[str] = None):
        """Initialize model loader"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def load(self, model_path: str) -> tuple:
        """Load TorchScript model and return (model, info)"""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if p.suffix != '.pt':
            raise ValueError(
                f"Expected a .pt (TorchScript) file, got '{p.suffix}'.\n"
                f"Only TorchScript (.pt) models are supported.\n"
                f"Convert your model using:\n"
                f"  traced = torch.jit.trace(model, sample_input)\n"
                f"  torch.jit.save(traced, 'model.pt')"
            )

        print(f"\n  Loading TorchScript model: {model_path}")

        analyzer = ModelAnalyzer(model_path)
        info = analyzer.analyze()
        analyzer.summary()

        model = analyzer.model
        model.to(self.device)
        model.eval()

        print(f"  [OK] Model loaded successfully on {self.device}")
        return model, info

