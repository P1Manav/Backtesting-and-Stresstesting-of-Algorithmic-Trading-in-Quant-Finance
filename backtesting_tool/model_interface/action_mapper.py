"""
Action Mapper Module
Maps model outputs to standardized trading actions.
"""

import torch
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from enum import Enum


class TradingAction(Enum):
    """Standardized trading actions."""
    LONG = 1
    FLAT = 0
    SHORT = -1


class ActionMapper:
    """
    Maps model outputs to standardized trading actions.
    
    Converts various model output formats to:
    - Long / Short / Flat positions
    - Position sizes (-1 to 1)
    """
    
    def __init__(self,
                 output_type: str = 'auto',
                 threshold: float = 0.5,
                 signal_mapping: Optional[Dict[int, TradingAction]] = None):
        """
        Initialize the Action Mapper.
        
        Args:
            output_type: Type of model output:
                - 'auto': Automatically detect
                - 'signals': Discrete buy/sell/hold signals
                - 'scores': Continuous scores
                - 'positions': Position sizes
                - 'probabilities': Class probabilities
            threshold: Threshold for converting continuous outputs to actions
            signal_mapping: Custom mapping from class indices to actions
        """
        self.output_type = output_type
        self.threshold = threshold
        
        # Default signal mapping for 3-class classification
        self.signal_mapping = signal_mapping or {
            0: TradingAction.SHORT,
            1: TradingAction.FLAT,
            2: TradingAction.LONG
        }
        
        # For binary classification
        self.binary_mapping = {
            0: TradingAction.SHORT,
            1: TradingAction.LONG
        }
    
    def map_output(self, 
                   output: Union[torch.Tensor, np.ndarray],
                   output_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map model output to trading actions and position sizes.
        
        Args:
            output: Model output tensor or array
            output_type: Override the configured output type
            
        Returns:
            Tuple of (actions, positions):
            - actions: Array of TradingAction values (-1, 0, 1)
            - positions: Array of position sizes (-1 to 1)
        """
        # Convert to numpy
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        
        output_type = output_type or self.output_type
        
        # Auto-detect output type
        if output_type == 'auto':
            output_type = self._detect_output_type(output)
        
        if output_type == 'signals':
            return self._map_signals(output)
        elif output_type == 'scores':
            return self._map_scores(output)
        elif output_type == 'positions':
            return self._map_positions(output)
        elif output_type == 'probabilities':
            return self._map_probabilities(output)
        else:
            # Default: treat as scores
            return self._map_scores(output)
    
    def _detect_output_type(self, output: np.ndarray) -> str:
        """Auto-detect the type of model output."""
        if output.ndim == 1:
            output = output.reshape(-1, 1)
        
        num_outputs = output.shape[-1]
        
        # Single output: continuous scores or positions
        if num_outputs == 1:
            values = output.flatten()
            if np.all((values >= -1) & (values <= 1)):
                if np.mean(np.abs(values)) > 0.3:
                    return 'positions'
            return 'scores'
        
        # Two outputs: binary classification
        if num_outputs == 2:
            if np.allclose(output.sum(axis=-1), 1.0, atol=0.01):
                return 'probabilities'
            return 'scores'
        
        # Three outputs: Buy/Sell/Hold classification
        if num_outputs == 3:
            if np.allclose(output.sum(axis=-1), 1.0, atol=0.01):
                return 'probabilities'
            return 'signals'
        
        # More outputs: probably probabilities or signals
        if np.allclose(output.sum(axis=-1), 1.0, atol=0.01):
            return 'probabilities'
        
        return 'signals'
    
    def _map_signals(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map discrete signal outputs to actions."""
        if output.ndim == 1:
            signals = output
        else:
            # Take argmax for multi-class
            signals = np.argmax(output, axis=-1)
        
        num_classes = len(np.unique(signals))
        
        if num_classes <= 2:
            mapping = self.binary_mapping
        else:
            mapping = self.signal_mapping
        
        # Map to actions
        actions = np.array([mapping.get(int(s), TradingAction.FLAT).value for s in signals])
        
        # Positions are same as actions for discrete signals
        positions = actions.astype(np.float32)
        
        return actions, positions
    
    def _map_scores(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map continuous score outputs to actions."""
        if output.ndim > 1:
            scores = output[:, 0]  # Take first column
        else:
            scores = output
        
        # Normalize scores to position sizes
        positions = np.clip(scores, -1, 1).astype(np.float32)
        
        # Convert to discrete actions using thresholds
        actions = np.zeros_like(positions, dtype=np.int32)
        actions[positions > self.threshold] = TradingAction.LONG.value
        actions[positions < -self.threshold] = TradingAction.SHORT.value
        
        return actions, positions
    
    def _map_positions(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map position size outputs to actions."""
        if output.ndim > 1:
            positions = output[:, 0]
        else:
            positions = output
        
        # Clip to valid range
        positions = np.clip(positions, -1, 1).astype(np.float32)
        
        # Convert to discrete actions
        actions = np.zeros_like(positions, dtype=np.int32)
        actions[positions > 0.1] = TradingAction.LONG.value
        actions[positions < -0.1] = TradingAction.SHORT.value
        
        return actions, positions
    
    def _map_probabilities(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map probability outputs to actions."""
        if output.ndim == 1:
            output = output.reshape(1, -1)
        
        num_classes = output.shape[-1]
        
        # Get predicted class
        predicted_class = np.argmax(output, axis=-1)
        
        if num_classes == 2:
            # Binary: class 0 = short, class 1 = long
            actions = np.where(predicted_class == 1, 
                              TradingAction.LONG.value, 
                              TradingAction.SHORT.value)
            # Position size based on confidence
            positions = output[:, 1] * 2 - 1  # Map [0,1] to [-1,1]
            
        elif num_classes == 3:
            # Three classes: short, flat, long
            actions = np.array([
                self.signal_mapping.get(int(c), TradingAction.FLAT).value 
                for c in predicted_class
            ])
            # Position size from probability difference
            long_prob = output[:, 2] if num_classes > 2 else output[:, 1]
            short_prob = output[:, 0]
            positions = (long_prob - short_prob).astype(np.float32)
            
        else:
            # Generic handling for more classes
            actions = np.where(predicted_class > num_classes // 2,
                              TradingAction.LONG.value,
                              TradingAction.SHORT.value)
            positions = (predicted_class / num_classes * 2 - 1).astype(np.float32)
        
        return actions.astype(np.int32), positions.astype(np.float32)
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable name for an action."""
        if action == TradingAction.LONG.value:
            return "LONG"
        elif action == TradingAction.SHORT.value:
            return "SHORT"
        else:
            return "FLAT"
    
    def actions_to_names(self, actions: np.ndarray) -> List[str]:
        """Convert array of actions to names."""
        return [self.get_action_name(a) for a in actions]
