from dataclasses import dataclass


@dataclass(frozen=True)
class RHLTConfig:
    """Configuration for legacy RHLT diagnostics and ridge-flow restoration."""

    # Retained legacy RHLT parameters.
    gaussian_sigma: float = 1.0
    psf_size: int = 65
    aperture_ratio: float = 0.90
    topological_charge: int = 1
    apodisation: float = 0.00
    edge_gain: float = 0.75
    stretch_low_percentile: float = 1.0
    stretch_high_percentile: float = 99.0
    segmentation_sigma: float = 9.0
    min_component_area: int = 20
    minutiae_border: int = 12
    minutiae_min_distance: int = 8

    # Ridge-flow segmentation and orientation parameters.
    block_size: int = 16
    segmentation_min_std: float = 6.0
    segmentation_threshold_scale: float = 0.55
    orientation_smoothing_sigma: float = 1.0
    orientation_bins: int = 12

    # Conservative orientation-adaptive Gabor parameters.
    gabor_kernel_size: int = 17
    gabor_sigma: float = 3.0
    gabor_lambda: float = 10.0
    gabor_gamma: float = 0.5
    gabor_psi: float = 0.0
    gabor_strength: float = 0.75
    gabor_blend_strength: float = 48.0

    def validate(self) -> None:
        if self.psf_size < 15 or self.psf_size % 2 == 0:
            raise ValueError("psf_size must be an odd integer >= 15.")
        if not 0.2 <= self.aperture_ratio <= 1.0:
            raise ValueError("aperture_ratio must be between 0.2 and 1.0.")
        if self.topological_charge < 1:
            raise ValueError("topological_charge must be >= 1.")
        if not 0.0 <= self.apodisation <= 0.95:
            raise ValueError("apodisation must be between 0.0 and 0.95.")
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be >= 0.")
        if self.edge_gain < 0:
            raise ValueError("edge_gain must be >= 0.")
        if self.block_size < 4:
            raise ValueError("block_size must be >= 4.")
        if self.segmentation_min_std < 0:
            raise ValueError("segmentation_min_std must be >= 0.")
        if self.segmentation_threshold_scale <= 0:
            raise ValueError("segmentation_threshold_scale must be > 0.")
        if self.orientation_smoothing_sigma < 0:
            raise ValueError("orientation_smoothing_sigma must be >= 0.")
        if not 4 <= self.orientation_bins <= 36:
            raise ValueError("orientation_bins must be between 4 and 36.")
        if self.gabor_kernel_size < 5 or self.gabor_kernel_size % 2 == 0:
            raise ValueError("gabor_kernel_size must be an odd integer >= 5.")
        if self.gabor_sigma <= 0 or self.gabor_lambda <= 0 or self.gabor_gamma <= 0:
            raise ValueError("Gabor sigma, lambda and gamma must be positive.")
        if not 0.0 <= self.gabor_strength <= 5.0:
            raise ValueError("gabor_strength must be between 0 and 5.")
        if self.gabor_blend_strength <= 0:
            raise ValueError("gabor_blend_strength must be positive.")
