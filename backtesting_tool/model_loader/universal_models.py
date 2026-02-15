"""Universal model architectures for loading notebook-trained models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
import pickle
import joblib


class UniversalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 dropout=0.2, output_size=1):
        super(UniversalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class UniversalGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1):
        super(UniversalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


class UniversalRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1):
        super(UniversalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out


class UniversalCNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=60, output_size=1):
        super(UniversalCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        flat_size = 256 * (sequence_length // 8)
        
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class UniversalTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, output_size=1):
        super(UniversalTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out


class UniversalMLP(nn.Module):
    def __init__(self, input_size=60, hidden_sizes=[256, 128, 64], 
                 dropout=0.2, output_size=1):
        super(UniversalMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.flatten(1)
        return self.network(x)


class UniversalBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1):
        super(UniversalBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


PYTORCH_ARCHITECTURES = {
    'lstm': UniversalLSTM,
    'gru': UniversalGRU,
    'rnn': UniversalRNN,
    'cnn': UniversalCNN,
    'transformer': UniversalTransformer,
    'mlp': UniversalMLP,
    'bilstm': UniversalBiLSTM,
}


class UniversalModelLoader:
    """Universal loader for ML/DL/RL models trained on OHLCV data."""
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.model_type = None
        self.framework = None
    
    def load_model(self, model_path: str, model_config: Optional[Dict[str, Any]] = None) -> Any:
        if model_path.endswith(('.pth', '.pt')):
            return self._load_pytorch(model_path, model_config)
        elif model_path.endswith(('.pkl', '.pickle')):
            return self._load_sklearn(model_path)
        elif model_path.endswith('.joblib'):
            return self._load_joblib(model_path)
        else:
            raise ValueError(f"Unsupported format: {model_path}")
    
    def _load_pytorch(self, path: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        try:
            # Try loading as full model first
            model = torch.load(path, map_location=self.device)
            if isinstance(model, nn.Module):
                model.eval()
                self.model = model
                self.framework = 'pytorch'
                self.model_type = 'full_model'
                return model
        except Exception as e:
            print(f"Could not load as full model: {e}")
        
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        if config and 'architecture' in config:
            arch_name = config['architecture'].lower()
            if arch_name in PYTORCH_ARCHITECTURES:
                detected_params = {}
                if isinstance(checkpoint, dict):
                    param_mappings = {
                        'input_size': ['input_dim', 'input_size', 'n_features'],
                        'hidden_size': ['hidden_dim', 'hidden_size', 'hidden_units'],
                        'num_layers': ['num_layers', 'n_layers'],
                        'output_size': ['output_dim', 'output_size']
                    }
                    
                    for target_key, possible_keys in param_mappings.items():
                        for key in possible_keys:
                            if key in checkpoint:
                                detected_params[target_key] = checkpoint[key]
                                break
                
                if not detected_params:
                    detected_params = self._detect_model_params(state_dict, arch_name)
                
                final_config = {**detected_params, **config}
                final_config.pop('architecture', None)
                
                if detected_params:
                    print(f"  Detected model configuration:")
                    for key, value in detected_params.items():
                        print(f"    {key}: {value}")
                
                ModelClass = PYTORCH_ARCHITECTURES[arch_name]
                model = ModelClass(**final_config)
                
                try:
                    model.load_state_dict(state_dict, strict=False)
                    model.to(self.device)
                    model.eval()
                    
                    self.model = model
                    self.framework = 'pytorch'
                    self.model_type = 'state_dict'
                    return model
                except Exception as e:
                    raise RuntimeError(f"Failed to load state_dict: {e}")
        
        raise RuntimeError(
            "Could not load PyTorch model. Please provide model_config with architecture details.\n"
            f"Available architectures: {list(PYTORCH_ARCHITECTURES.keys())}\n"
            "Example config: {'architecture': 'lstm', 'input_size': 1, 'hidden_size': 64, 'num_layers': 2}"
        )
    
    def _detect_model_params(self, state_dict: dict, architecture: str) -> Dict[str, Any]:
        params = {}
        
        if architecture in ['lstm', 'gru', 'rnn', 'bilstm']:
            key_prefix = architecture if architecture != 'bilstm' else 'lstm'
            weight_key = f'{key_prefix}.weight_hh_l0'
            
            if weight_key in state_dict:
                weight_shape = state_dict[weight_key].shape
                hidden_size = weight_shape[1]
                params['hidden_size'] = hidden_size
            
            layer_keys = [k for k in state_dict.keys() if f'{key_prefix}.weight_hh_l' in k]
            if layer_keys:
                max_layer = max([int(k.split('_l')[1].split('.')[0].split('[')[0]) for k in layer_keys])
                params['num_layers'] = max_layer + 1
            
            input_key = f'{key_prefix}.weight_ih_l0'
            if input_key in state_dict:
                weight_shape = state_dict[input_key].shape
                # For LSTM: weight_ih shape is (4*hidden_size, input_size)
                input_size = weight_shape[1]
                params['input_size'] = input_size
            
            # Detect output_size from fc layer
            if 'fc.weight' in state_dict:
                output_size = state_dict['fc.weight'].shape[0]
                params['output_size'] = output_size
        
        elif architecture == 'mlp':
            # Detect input_size from first layer
            if 'network.0.weight' in state_dict:
                params['input_size'] = state_dict['network.0.weight'].shape[1]
        
        elif architecture == 'cnn':
            # Detect input_channels from first conv layer
            if 'conv1.weight' in state_dict:
                params['input_channels'] = state_dict['conv1.weight'].shape[1]
        
        return params
    
    def _load_sklearn(self, path: str) -> Any:
        """Load scikit-learn model from pickle."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        self.model = model
        self.framework = 'sklearn'
        self.model_type = 'sklearn_model'
        return model
    
    def _load_joblib(self, path: str) -> Any:
        """Load model from joblib."""
        model = joblib.load(path)
        
        self.model = model
        self.framework = 'sklearn'
        self.model_type = 'joblib_model'
        return model
    
    def predict(self, x):
        """Universal predict method."""
        if self.framework == 'pytorch':
            if isinstance(x, torch.Tensor):
                with torch.no_grad():
                    return self.model(x)
            else:
                x_tensor = torch.FloatTensor(x).to(self.device)
                if len(x_tensor.shape) == 2:
                    x_tensor = x_tensor.unsqueeze(0)
                with torch.no_grad():
                    return self.model(x_tensor).cpu().numpy()
        
        elif self.framework == 'sklearn':
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], -1)
            return self.model.predict(x)
        
        else:
            raise ValueError(f"Unknown framework: {self.framework}")


def auto_detect_architecture(model_path: str) -> Optional[str]:
    filename = model_path.lower()
    
    for arch_name in PYTORCH_ARCHITECTURES.keys():
        if arch_name in filename:
            return arch_name
    
    return None


def get_default_config(architecture: str, input_size: int = 1, 
                       sequence_length: int = 60) -> Dict[str, Any]:
    base_config = {
        'architecture': architecture,
        'input_size': input_size,
        'output_size': 1
    }
    
    if architecture in ['lstm', 'gru', 'rnn', 'bilstm']:
        base_config.update({
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        })
    elif architecture == 'cnn':
        base_config.update({
            'input_channels': input_size,
            'sequence_length': sequence_length
        })
    elif architecture == 'transformer':
        base_config.update({
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1
        })
    elif architecture == 'mlp':
        base_config.update({
            'input_size': sequence_length * input_size,
            'hidden_sizes': [256, 128, 64],
            'dropout': 0.2
        })
    
    return base_config


def list_available_architectures() -> list:
    return list(PYTORCH_ARCHITECTURES.keys())
