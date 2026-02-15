from typing import List, Tuple


class ModelPreset:
    
    def __init__(self,
                 name: str,
                 sequence_length: int,
                 feature_columns: List[str],
                 scaling_method: str,
                 scaling_range: Tuple[float, float],
                 description: str):
        self.name = name
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.scaling_method = scaling_method
        self.scaling_range = scaling_range
        self.description = description
    
    def __repr__(self):
        return f"ModelPreset('{self.name}', seq={self.sequence_length}, features={len(self.feature_columns)})"


PRESETS = {
    '1': ModelPreset(
        name="Single Feature (Close price only)",
        sequence_length=60,
        feature_columns=['Close'],
        scaling_method='minmax',
        scaling_range=(-1, 1),
        description="LSTM/GRU/RNN trained on Close price sequences"
    ),
    '2': ModelPreset(
        name="OHLC Features",
        sequence_length=60,
        feature_columns=['Open', 'High', 'Low', 'Close'],
        scaling_method='minmax',
        scaling_range=(-1, 1),
        description="Model trained on Open, High, Low, Close prices"
    ),
    '3': ModelPreset(
        name="OHLCV Features (with Volume)",
        sequence_length=60,
        feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
        scaling_method='minmax',
        scaling_range=(-1, 1),
        description="Model trained on all OHLCV features"
    ),
    '4': ModelPreset(
        name="OHLCV with Standard Scaling",
        sequence_length=60,
        feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
        scaling_method='standard',
        scaling_range=(0, 1),  # Not used for standard scaling
        description="StandardScaler normalization (mean=0, std=1)"
    ),
    '5': ModelPreset(
        name="Short Sequence (30 days)",
        sequence_length=30,
        feature_columns=['Close'],
        scaling_method='minmax',
        scaling_range=(0, 1),
        description="Short-term prediction model (30-day window)"
    ),
    '6': ModelPreset(
        name="Long Sequence (100 days)",
        sequence_length=100,
        feature_columns=['Open', 'High', 'Low', 'Close'],
        scaling_method='minmax',
        scaling_range=(-1, 1),
        description="Long-term context model (100-day window)"
    ),
    '7': ModelPreset(
        name="No Scaling",
        sequence_length=60,
        feature_columns=['Close'],
        scaling_method='none',
        scaling_range=(0, 1),
        description="Model trained on raw prices (no normalization)"
    ),
}


def get_preset(preset_id: str) -> ModelPreset:
    if preset_id not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_id}")
    return PRESETS[preset_id]


def list_presets():
    print("\n" + "=" * 70)
    print("AVAILABLE MODEL CONFIGURATION PRESETS")
    print("=" * 70)
    for key, preset in PRESETS.items():
        print(f"\n{key}. {preset.name}")
        print(f"   Sequence Length: {preset.sequence_length}")
        print(f"   Features: {', '.join(preset.feature_columns)}")
        print(f"   Scaling: {preset.scaling_method}")
        print(f"   Description: {preset.description}")
    print("=" * 70)


def create_custom_preset() -> ModelPreset:
    print("\n" + "=" * 70)
    print("CREATE CUSTOM MODEL CONFIGURATION")
    print("=" * 70)
    
    name = input("\nModel name: ").strip() or "Custom Model"
    
    seq_len = input("Sequence length (default: 60): ").strip()
    sequence_length = int(seq_len) if seq_len else 60
    
    print("\nSelect features (comma-separated, or press Enter for Close only):")
    print("Available: Open, High, Low, Close, Volume")
    features_input = input("Features: ").strip()
    
    if features_input:
        feature_columns = [f.strip() for f in features_input.split(',')]
    else:
        feature_columns = ['Close']
    
    print("\nScaling method:")
    print("1. MinMax (-1 to 1)")
    print("2. MinMax (0 to 1)")
    print("3. Standard (z-score)")
    print("4. None (raw values)")
    
    scaling_choice = input("Choice (1-4, default: 1): ").strip() or "1"
    
    if scaling_choice == "1":
        scaling_method = 'minmax'
        scaling_range = (-1, 1)
    elif scaling_choice == "2":
        scaling_method = 'minmax'
        scaling_range = (0, 1)
    elif scaling_choice == "3":
        scaling_method = 'standard'
        scaling_range = (0, 1)
    else:
        scaling_method = 'none'
        scaling_range = (0, 1)
    
    description = input("\nDescription (optional): ").strip() or "Custom configuration"
    
    return ModelPreset(name, sequence_length, feature_columns, scaling_method, scaling_range, description)
