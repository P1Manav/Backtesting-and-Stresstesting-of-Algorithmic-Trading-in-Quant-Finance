import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class InferenceValidator:

    def __init__(self,
                 expected_output_dim: Optional[int] = None,
                 clamp_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
                 strict: bool = False):
        self.expected_output_dim = expected_output_dim
        self.clamp_range = clamp_range
        self.strict = strict

        self._last_valid: Optional[np.ndarray] = None
        self.total_predictions = 0
        self.nan_count = 0
        self.inf_count = 0
        self.clamp_count = 0
        self.shape_errors = 0

    # Validate and sanitise a single prediction output.
    def validate(self, output) -> np.ndarray:
        if not isinstance(output, np.ndarray):
            try:
                output = output.detach().cpu().numpy()
            except AttributeError:
                output = np.asarray(output)

        self.total_predictions += 1

        if self.expected_output_dim is not None:
            if output.shape[-1] != self.expected_output_dim:
                self.shape_errors += 1
                msg = (f"[InferenceValidator] Shape mismatch: expected last "
                       f"dim={self.expected_output_dim}, got {output.shape}")
                if self.strict:
                    raise ValueError(msg)
                if self._last_valid is not None:
                    return self._last_valid.copy()
                return np.zeros_like(output)

        has_nan = bool(np.any(np.isnan(output)))
        has_inf = bool(np.any(np.isinf(output)))

        if has_nan:
            self.nan_count += 1
        if has_inf:
            self.inf_count += 1

        if has_nan or has_inf:
            if self.strict:
                raise ValueError(
                    f"[InferenceValidator] Non-finite output detected: "
                    f"NaN={has_nan}, Inf={has_inf}")
            fallback_val = self._last_valid if self._last_valid is not None else 0.0
            output = np.where(np.isfinite(output), output, fallback_val)

        if self.clamp_range is not None:
            lo, hi = self.clamp_range
            before_clamp = output.copy()
            output = np.clip(output, lo, hi)
            if not np.array_equal(before_clamp, output):
                self.clamp_count += 1

        self._last_valid = output.copy()
        return output

    # Return validation diagnostics summary.
    def summary(self) -> dict:
        return {
            'total_predictions': self.total_predictions,
            'nan_detections': self.nan_count,
            'inf_detections': self.inf_count,
            'clamp_corrections': self.clamp_count,
            'shape_errors': self.shape_errors,
            'clean_rate_pct': round(
                (1 - (self.nan_count + self.inf_count + self.shape_errors)
                 / max(1, self.total_predictions)) * 100, 2),
        }

class PredictionController:

    def __init__(self, model: nn.Module, device: torch.device,
                 validator: Optional[InferenceValidator] = None):
        self.model = model
        self.device = device
        self.model.eval()
        self.validator = validator or InferenceValidator()

    # Run forward pass on a numpy window and validate the output.
    @torch.no_grad()
    def predict(self, window: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(window).float().to(self.device)
        output = self.model(x)
        raw = output.cpu().numpy()

        validated = self.validator.validate(raw)
        return validated

    # Return inference validation diagnostics.
    def get_validation_summary(self) -> dict:
        return self.validator.summary()
