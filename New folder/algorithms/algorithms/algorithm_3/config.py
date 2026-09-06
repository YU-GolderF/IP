"""
Configuration for Gabor Fingerprint Enhancement - GENTLE VERSION
Balanced between enhancement and structure preservation.
"""

from dataclasses import dataclass


@dataclass
class GaborConfig:
    """
    Gentle configuration for fingerprint enhancement.
    """
    
    block_size: int = 16
    blend_ratio: float = 0.1  
    apply_clahe: bool = True
    clip_limit: float = 0.2   
    sharpen_amount: float = 0.0 
    
    def to_dict(self):
        return {
            'block_size': self.block_size,
            'blend_ratio': self.blend_ratio,
            'apply_clahe': self.apply_clahe,
            'clip_limit': self.clip_limit,
            'sharpen_amount': self.sharpen_amount,
        }