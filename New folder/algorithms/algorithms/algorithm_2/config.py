from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DCTContextualConfig:
    block_size: int = 32
    overlap: float = 0.5
    orientation_sigma_degrees: float = 18.0
    frequency_sigma: float = 0.055
    minimum_frequency: float = 1.0 / 16.0
    maximum_frequency: float = 1.0 / 3.0
    base_strength: float = 0.45
    contrast_clip_percentile: float = 1.0
    proposed_quality_exponent: float = 0.65
    proposed_confidence_exponent: float = 0.65
    proposed_min_strength_ratio: float = 1.00
    proposed_ridge_gain: float = 1.40
    proposed_detail_gain: float = 0.65

    @property
    def stride(self) -> int:
        return max(1, int(round(self.block_size * (1.0 - self.overlap))))

    def validate(self) -> None:
        if self.block_size < 8 or self.block_size % 2:
            raise ValueError("block_size must be an even integer of at least 8")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if not 0 < self.minimum_frequency < self.maximum_frequency:
            raise ValueError("invalid ridge-frequency range")
        if self.proposed_quality_exponent <= 0 or self.proposed_confidence_exponent <= 0:
            raise ValueError("proposed strength exponents must be positive")
        if not 0 <= self.proposed_min_strength_ratio <= 1:
            raise ValueError("proposed_min_strength_ratio must be in [0, 1]")
        if self.proposed_ridge_gain < 0:
            raise ValueError("proposed_ridge_gain must be non-negative")
        if self.proposed_detail_gain < 0:
            raise ValueError("proposed_detail_gain must be non-negative")
