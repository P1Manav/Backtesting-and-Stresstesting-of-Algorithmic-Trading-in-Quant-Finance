"""
Model Analyzer Module
Analyzes PyTorch model structure, input/output dimensions, and model type.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import Enum


class ModelType(Enum):
    """Enumeration of model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    POLICY = "policy"  # RL-style policy network
    UNKNOWN = "unknown"


class OutputType(Enum):
    """Enumeration of model output types."""
    SIGNALS = "signals"  # Buy/Sell/Hold signals
    POSITIONS = "positions"  # Position sizes
    SCORES = "scores"  # Continuous scores
    PROBABILITIES = "probabilities"  # Class probabilities
    UNKNOWN = "unknown"


class ModelAnalyzer:
    """
    Analyzes PyTorch models to understand their structure and behavior.
    
    Inspects:
    - Input tensor dimensions
    - Output shape (classification / regression / action)
    - Model architecture
    - Whether model outputs signals, positions, or continuous scores
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize the Model Analyzer.
        
        Args:
            model: PyTorch model to analyze
            device: Device to use for inference tests
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        
        self._analysis_results: Dict[str, Any] = {}
    
    def analyze(self, 
                sample_input_shape: Optional[Tuple[int, ...]] = None,
                num_features: int = 10,
                sequence_length: int = 30) -> Dict[str, Any]:
        """
        Perform comprehensive model analysis.
        
        Args:
            sample_input_shape: Optional tuple specifying input shape
            num_features: Number of features to test with if shape not provided
            sequence_length: Sequence length to test with if shape not provided
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'model_class': self.model.__class__.__name__,
            'device': str(self.device),
            'parameters': self._count_parameters(),
            'architecture': self._get_architecture_summary(),
            'layers': self._get_layer_info(),
            'input_analysis': {},
            'output_analysis': {},
            'model_type': ModelType.UNKNOWN.value,
            'output_type': OutputType.UNKNOWN.value
        }
        
        # Infer input shape if not provided
        if sample_input_shape is None:
            sample_input_shape = self._infer_input_shape(num_features, sequence_length)
        
        results['input_analysis'] = self._analyze_input(sample_input_shape)
        
        # Perform inference to analyze output
        try:
            output_analysis = self._analyze_output(sample_input_shape)
            results['output_analysis'] = output_analysis
            
            # Infer model and output types
            results['model_type'] = self._infer_model_type(output_analysis).value
            results['output_type'] = self._infer_output_type(output_analysis).value
            
        except Exception as e:
            results['output_analysis'] = {'error': str(e)}
        
        self._analysis_results = results
        return results
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def _get_architecture_summary(self) -> str:
        """Get a string summary of model architecture."""
        return str(self.model)
    
    def _get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer."""
        layers = []
        
        for name, module in self.model.named_modules():
            if name == '':
                continue
            
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters(recurse=False))
            }
            
            # Add layer-specific info
            if isinstance(module, nn.Linear):
                layer_info['in_features'] = module.in_features
                layer_info['out_features'] = module.out_features
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                layer_info['input_size'] = module.input_size
                layer_info['hidden_size'] = module.hidden_size
                layer_info['num_layers'] = module.num_layers
                layer_info['bidirectional'] = module.bidirectional
            elif isinstance(module, nn.Conv1d):
                layer_info['in_channels'] = module.in_channels
                layer_info['out_channels'] = module.out_channels
                layer_info['kernel_size'] = module.kernel_size
            
            layers.append(layer_info)
        
        return layers
    
    def _infer_input_shape(self, 
                           num_features: int,
                           sequence_length: int) -> Tuple[int, ...]:
        """
        Infer the expected input shape from the model architecture.
        
        Returns:
            Tuple representing input shape (batch, ...)
        """
        # Check first layer to infer input shape
        first_layer = None
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d)):
                first_layer = module
                break
        
        if first_layer is None:
            return (1, sequence_length, num_features)
        
        if isinstance(first_layer, nn.Linear):
            # Linear layer: (batch, in_features)
            return (1, first_layer.in_features)
        elif isinstance(first_layer, (nn.LSTM, nn.GRU, nn.RNN)):
            # RNN: (batch, seq_len, input_size)
            return (1, sequence_length, first_layer.input_size)
        elif isinstance(first_layer, nn.Conv1d):
            # Conv1d: (batch, in_channels, seq_len)
            return (1, first_layer.in_channels, sequence_length)
        
        return (1, sequence_length, num_features)
    
    def _analyze_input(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Analyze model input requirements."""
        return {
            'shape': input_shape,
            'dimensions': len(input_shape),
            'batch_size': input_shape[0],
            'feature_dims': input_shape[1:] if len(input_shape) > 1 else None
        }
    
    def _analyze_output(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Analyze model output by performing dummy inference.
        
        Args:
            input_shape: Input shape to use for inference
            
        Returns:
            Dictionary with output analysis
        """
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        with torch.no_grad():
            output = self.model(dummy_input)
        
        # Handle different output types
        if isinstance(output, tuple):
            # Some models return (output, hidden_state)
            main_output = output[0]
            has_hidden = True
        else:
            main_output = output
            has_hidden = False
        
        output_shape = tuple(main_output.shape)
        
        analysis = {
            'shape': output_shape,
            'dimensions': len(output_shape),
            'has_hidden_state': has_hidden,
            'output_size': output_shape[-1] if len(output_shape) > 0 else 0,
            'value_range': {
                'min': float(main_output.min()),
                'max': float(main_output.max()),
                'mean': float(main_output.mean())
            }
        }
        
        # Check for activation patterns
        analysis['likely_activation'] = self._detect_activation(main_output)
        
        return analysis
    
    def _detect_activation(self, output: torch.Tensor) -> str:
        """Detect the likely output activation function."""
        min_val = output.min().item()
        max_val = output.max().item()
        
        if 0 <= min_val and max_val <= 1:
            # Check if sums to 1 (softmax)
            if output.dim() > 1:
                sums = output.sum(dim=-1)
                if torch.allclose(sums, torch.ones_like(sums), atol=0.01):
                    return 'softmax'
            return 'sigmoid'
        elif -1 <= min_val and max_val <= 1:
            return 'tanh'
        else:
            return 'linear/none'
    
    def _infer_model_type(self, output_analysis: Dict[str, Any]) -> ModelType:
        """Infer the type of model based on output analysis."""
        output_size = output_analysis.get('output_size', 0)
        activation = output_analysis.get('likely_activation', '')
        
        # Classification: softmax output or small discrete output
        if activation == 'softmax':
            return ModelType.CLASSIFICATION
        
        # RL-style policy: outputs action probabilities or Q-values
        if output_size in [2, 3, 4]:  # Common action spaces
            if activation in ['softmax', 'sigmoid']:
                return ModelType.POLICY
        
        # Regression: single continuous output
        if output_size == 1:
            return ModelType.REGRESSION
        
        # Default to regression for continuous outputs
        if activation in ['linear/none', 'tanh']:
            return ModelType.REGRESSION
        
        return ModelType.UNKNOWN
    
    def _infer_output_type(self, output_analysis: Dict[str, Any]) -> OutputType:
        """Infer the type of output the model produces."""
        output_size = output_analysis.get('output_size', 0)
        activation = output_analysis.get('likely_activation', '')
        value_range = output_analysis.get('value_range', {})
        
        # Trading signals (Buy/Sell/Hold = 3 classes)
        if output_size == 3 and activation == 'softmax':
            return OutputType.SIGNALS
        
        # Binary signals (Long/Short or Buy/Sell)
        if output_size == 2 and activation in ['softmax', 'sigmoid']:
            return OutputType.SIGNALS
        
        # Probabilities
        if activation == 'softmax':
            return OutputType.PROBABILITIES
        
        # Position sizes (typically -1 to 1 range)
        if activation == 'tanh':
            return OutputType.POSITIONS
        
        # Continuous scores
        if activation in ['linear/none', 'sigmoid']:
            return OutputType.SCORES
        
        return OutputType.UNKNOWN
    
    def get_input_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the analyzed input shape."""
        if 'input_analysis' in self._analysis_results:
            return self._analysis_results['input_analysis'].get('shape')
        return None
    
    def get_output_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the analyzed output shape."""
        if 'output_analysis' in self._analysis_results:
            return self._analysis_results['output_analysis'].get('shape')
        return None
    
    def get_model_type(self) -> str:
        """Get the inferred model type."""
        return self._analysis_results.get('model_type', ModelType.UNKNOWN.value)
    
    def get_output_type(self) -> str:
        """Get the inferred output type."""
        return self._analysis_results.get('output_type', OutputType.UNKNOWN.value)
    
    def log_analysis(self) -> None:
        """Print analysis results to console."""
        if not self._analysis_results:
            print("No analysis performed yet. Call analyze() first.")
            return
        
        print("\n" + "="*60)
        print("MODEL ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nModel Class: {self._analysis_results['model_class']}")
        print(f"Device: {self._analysis_results['device']}")
        
        params = self._analysis_results['parameters']
        print(f"\nParameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Non-trainable: {params['non_trainable']:,}")
        
        input_info = self._analysis_results['input_analysis']
        print(f"\nInput Analysis:")
        print(f"  Shape: {input_info.get('shape')}")
        print(f"  Dimensions: {input_info.get('dimensions')}")
        
        output_info = self._analysis_results['output_analysis']
        if 'error' not in output_info:
            print(f"\nOutput Analysis:")
            print(f"  Shape: {output_info.get('shape')}")
            print(f"  Output Size: {output_info.get('output_size')}")
            print(f"  Likely Activation: {output_info.get('likely_activation')}")
            print(f"  Value Range: {output_info.get('value_range')}")
        
        print(f"\nInferred Model Type: {self._analysis_results['model_type']}")
        print(f"Inferred Output Type: {self._analysis_results['output_type']}")
        print("="*60 + "\n")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        return {
            'model_class': self._analysis_results.get('model_class'),
            'parameters': self._analysis_results.get('parameters', {}).get('total', 0),
            'input_shape': self.get_input_shape(),
            'output_shape': self.get_output_shape(),
            'model_type': self.get_model_type(),
            'output_type': self.get_output_type()
        }
