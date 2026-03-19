"""Analyze TorchScript model architecture and parameters"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

INPUT_SIZE_TO_FEATURES = {
    1: ['Close'],
    2: ['Close', 'Volume'],
    3: ['Open', 'High', 'Close'],
    4: ['Open', 'High', 'Low', 'Close'],
    5: ['Open', 'High', 'Low', 'Close', 'Volume'],
}

class ModelAnalyzer:
    """Analyze model architecture from TorchScript file"""

    def __init__(self, model_path: str):
        """Initialize analyzer"""
        self.model_path = model_path
        self.model: Optional[nn.Module] = None
        self.state_dict: Dict[str, torch.Tensor] = {}
        self.info: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """Analyze model and return architecture info"""
        self._load_model()
        arch = self._detect_architecture()
        params = self._detect_params(arch)

        input_size = params.get('input_size', 1)
        features = INPUT_SIZE_TO_FEATURES.get(
            input_size, [f'feature_{i}' for i in range(input_size)]
        )

        self.info = {
            'architecture': arch,
            'input_size': input_size,
            'hidden_size': params.get('hidden_size'),
            'num_layers': params.get('num_layers', 1),
            'output_size': params.get('output_size', 1),
            'bidirectional': params.get('bidirectional', False),
            'feature_columns': features,
            'total_parameters': sum(t.numel() for t in self.state_dict.values()),
        }
        return self.info

    def summary(self) -> None:
        """Print model summary"""
        if not self.info:
            self.analyze()
        i = self.info
        print(f"\n  Architecture : {i['architecture'].upper()}")
        print(f"  Input size   : {i['input_size']}  ->  features = {i['feature_columns']}")
        if i.get('hidden_size'):
            print(f"  Hidden size  : {i['hidden_size']}")
        print(f"  Layers       : {i['num_layers']}")
        print(f"  Output size  : {i['output_size']}")
        if i['bidirectional']:
            print(f"  Bidirectional: Yes")
        print(f"  Parameters   : {i['total_parameters']:,}")

    def _load_model(self):
        """Load TorchScript model from path"""
        p = Path(self.model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if p.suffix != '.pt':
            raise ValueError(
                f"Expected a .pt (TorchScript) file, got '{p.suffix}'.\n"
                f"Only TorchScript (.pt) models are supported."
            )

        try:
            self.model = torch.jit.load(self.model_path, map_location='cpu')
            self.model.eval()
            self.state_dict = self.model.state_dict()
            print(f"  [OK] Loaded TorchScript model successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TorchScript model from '{self.model_path}'.\n"
                f"Ensure the file was saved with torch.jit.save().\n"
                f"Error: {e}"
            )

    def _detect_architecture(self) -> str:
        """Detect neural network architecture from weights"""
        keys = set(self.state_dict.keys())

        if 'lstm.weight_ih_l0' in keys or 'lstm.weight_hh_l0' in keys:
            if 'lstm.weight_ih_l0_reverse' in keys:
                return 'bilstm'
            return 'lstm'

        if 'gru.weight_ih_l0' in keys or 'gru.weight_hh_l0' in keys:
            return 'gru'

        if 'rnn.weight_ih_l0' in keys or 'rnn.weight_hh_l0' in keys:
            return 'rnn'

        if any('transformer' in k for k in keys) or any('encoder' in k for k in keys):
            return 'transformer'

        if any('conv' in k for k in keys):
            return 'cnn'

        if any('network' in k or 'fc1' in k or 'linear' in k for k in keys):
            return 'mlp'

        if 'fc.weight' in keys:
            return 'lstm'

        raise RuntimeError(
            f"Could not auto-detect architecture from keys: {sorted(keys)[:20]}"
        )

    def _detect_params(self, arch: str) -> Dict[str, Any]:
        """Extract model parameters from weights"""
        params: Dict[str, Any] = {}
        sd = self.state_dict

        if arch in ('lstm', 'gru', 'rnn', 'bilstm'):
            rnn_name = 'lstm' if arch == 'bilstm' else arch
            ih_key = f'{rnn_name}.weight_ih_l0'

            if ih_key in sd:
                gate_mult = {'lstm': 4, 'gru': 3, 'rnn': 1, 'bilstm': 4}[arch]
                ih_shape = sd[ih_key].shape
                params['input_size'] = ih_shape[1]
                params['hidden_size'] = ih_shape[0] // gate_mult

            layer_keys = [k for k in sd if f'{rnn_name}.weight_hh_l' in k and 'reverse' not in k]
            params['num_layers'] = len(layer_keys) if layer_keys else 1

            params['bidirectional'] = arch == 'bilstm'

            if 'fc.weight' in sd:
                params['output_size'] = sd['fc.weight'].shape[0]

        elif arch == 'cnn':
            if 'conv1.weight' in sd:
                params['input_size'] = sd['conv1.weight'].shape[1]
            fc_keys = sorted([k for k in sd if k.startswith('fc') and 'weight' in k])
            if fc_keys:
                params['output_size'] = sd[fc_keys[-1]].shape[0]

        elif arch == 'transformer':
            proj_key = None
            for k in sd:
                if 'input_projection' in k and 'weight' in k:
                    proj_key = k
                    break
            if proj_key:
                params['input_size'] = sd[proj_key].shape[1]
            if 'fc.weight' in sd:
                params['output_size'] = sd['fc.weight'].shape[0]

        elif arch == 'mlp':
            first_linear = None
            for k in sorted(sd.keys()):
                if 'weight' in k:
                    first_linear = k
                    break
            if first_linear:
                params['input_size'] = sd[first_linear].shape[1]
            last_linear = None
            for k in sorted(sd.keys(), reverse=True):
                if 'weight' in k:
                    last_linear = k
                    break
            if last_linear:
                params['output_size'] = sd[last_linear].shape[0]

        return params

