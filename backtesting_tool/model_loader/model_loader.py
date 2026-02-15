"""
Model Loader Module
Loads PyTorch models from .pth files.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union


class ModelLoader:
    """
    Loads PyTorch models from .pth files for backtesting.
    
    Supports:
    - Full model saves (torch.save(model, path))
    - State dict saves (torch.save(model.state_dict(), path))
    - JIT/TorchScript models
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Model Loader.
        
        Args:
            device: Device to load model on ('cpu', 'cuda', 'cuda:0', etc.)
                    If None, automatically selects based on availability
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._loaded_model: Optional[nn.Module] = None
        self._model_path: Optional[str] = None
        self._model_metadata: Dict[str, Any] = {}
    
    def load_model(self, 
                   model_path: str,
                   model_class: Optional[type] = None,
                   strict: bool = True) -> nn.Module:
        """
        Load a PyTorch model from a .pth file.
        
        Args:
            model_path: Path to the .pth model file
            model_class: Optional model class for state_dict loading
            strict: Whether to strictly enforce state_dict key matching
            
        Returns:
            Loaded PyTorch model in eval mode
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not model_path.endswith(('.pth', '.pt', '.pkl')):
            raise ValueError(f"Unsupported model format. Expected .pth, .pt, or .pkl")
        
        self._model_path = model_path
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        try:
            # Try loading as TorchScript model first
            model = self._try_load_torchscript(model_path)
            if model is not None:
                self._model_metadata['load_type'] = 'torchscript'
                self._loaded_model = model
                return model
            
            # Try loading as full model
            model = self._try_load_full_model(model_path)
            if model is not None:
                self._model_metadata['load_type'] = 'full_model'
                self._loaded_model = model
                return model
            
            # Try loading as state dict with provided class
            if model_class is not None:
                model = self._try_load_state_dict(model_path, model_class, strict)
                if model is not None:
                    self._model_metadata['load_type'] = 'state_dict'
                    self._loaded_model = model
                    return model
            
            raise RuntimeError(
                "Could not load model. If the file contains only state_dict, "
                "please provide the model_class parameter."
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _try_load_torchscript(self, path: str) -> Optional[nn.Module]:
        """Try loading as TorchScript model."""
        try:
            model = torch.jit.load(path, map_location=self.device)
            model.eval()
            print("Loaded as TorchScript model")
            return model
        except:
            return None
    
    def _try_load_full_model(self, path: str) -> Optional[nn.Module]:
        """Try loading as full model (nn.Module)."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Check if it's a full model
            if isinstance(checkpoint, nn.Module):
                checkpoint.to(self.device)
                checkpoint.eval()
                print("Loaded as full model")
                return checkpoint
            
            # Check if it's a checkpoint dict with 'model' key
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
                if isinstance(model, nn.Module):
                    model.to(self.device)
                    model.eval()
                    print("Loaded model from checkpoint dict")
                    return model
            
            return None
        except:
            return None
    
    def _try_load_state_dict(self, 
                              path: str, 
                              model_class: type,
                              strict: bool = True) -> Optional[nn.Module]:
        """Load model from state dict using provided class."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Get the state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                return None
            
            # Instantiate model and load state dict
            model = model_class()
            model.load_state_dict(state_dict, strict=strict)
            model.to(self.device)
            model.eval()
            print("Loaded from state dict with provided class")
            return model
            
        except Exception as e:
            print(f"Failed to load state dict: {e}")
            return None
    
    def get_loaded_model(self) -> Optional[nn.Module]:
        """Get the currently loaded model."""
        return self._loaded_model
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded model."""
        return self._model_metadata
    
    def get_device(self) -> torch.device:
        """Get the device being used."""
        return self.device
    
    def save_model(self, 
                   model: nn.Module,
                   save_path: str,
                   save_type: str = 'full') -> str:
        """
        Save a PyTorch model.
        
        Args:
            model: Model to save
            save_path: Path to save the model
            save_type: 'full', 'state_dict', or 'torchscript'
            
        Returns:
            Path to saved model
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        if save_type == 'full':
            torch.save(model, save_path)
        elif save_type == 'state_dict':
            torch.save(model.state_dict(), save_path)
        elif save_type == 'torchscript':
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, save_path)
        else:
            raise ValueError(f"Unknown save_type: {save_type}")
        
        print(f"Model saved to: {save_path}")
        return save_path
