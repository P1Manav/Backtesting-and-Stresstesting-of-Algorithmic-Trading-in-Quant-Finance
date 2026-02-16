import torch
import torch.nn as nn
import numpy as np


class PredictionController:

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> np.ndarray:
        """Run forward pass on a numpy window of shape (1, seq_len, n_features)."""
        x = torch.from_numpy(window).float().to(self.device)
        output = self.model(x)
        return output.cpu().numpy()
