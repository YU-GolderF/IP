from dataclasses import dataclass


@dataclass(frozen=True)
class RHLTConfig:
    """Configuration for the RHLT ridge-flow restoration pipeline."""

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

    # Hybrid RHLT+Gabor fusion parameters.
    # Maximum fraction of each pixel that Gabor support may contribute (0..1).
    hybrid_gabor_max_weight: float = 0.45
    # Orientation coherence below this threshold is treated as unreliable.
    minimum_orientation_coherence: float = 0.10
    # Gamma applied to the per-pixel support weight to keep it conservative.
    rhlt_support_gamma: float = 0.60

    # Quality-adaptive fusion. Local standard deviation is measured in uint8
    # intensity units; clear regions are deliberately protected from support.
    local_quality_sigma: float = 3.0
    weak_ridge_target_contrast: float = 65.0
    weak_ridge_strength: float = 1.50
    clear_region_protection: float = 1.10
    rhlt_edge_gamma: float = 0.55
    rhlt_evidence_floor: float = 0.40

    # Local ridge-wavelength estimation and adaptive Gabor bank.
    frequency_block_size: int = 24
    minimum_ridge_wavelength: float = 3.0
    maximum_ridge_wavelength: float = 14.0
    frequency_smoothing_size: int = 3
    frequency_bins: int = 6
    use_local_frequency: bool = True
    use_quality_adaptive_fusion: bool = True

    # Candidate selection. Improved RHLT is the proposed primary output when it
    # is structurally safe and is not materially worse than Traditional RHLT.
    # This tolerance prevents tiny no-reference score noise from defeating the
    # proposed method while retaining a genuine regression guard.
    selector_regression_tolerance: float = 0.010
    selector_ssim_floor: float = 0.55

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
        if not 0.0 <= self.hybrid_gabor_max_weight <= 0.5:
            raise ValueError(
                "hybrid_gabor_max_weight must be between 0.0 and 0.5 so RHLT remains primary."
            )
        if not 0.0 <= self.minimum_orientation_coherence <= 1.0:
            raise ValueError("minimum_orientation_coherence must be between 0.0 and 1.0.")
        if self.rhlt_support_gamma <= 0:
            raise ValueError("rhlt_support_gamma must be positive.")
        if self.local_quality_sigma <= 0:
            raise ValueError("local_quality_sigma must be positive.")
        if self.weak_ridge_target_contrast <= 0:
            raise ValueError("weak_ridge_target_contrast must be positive.")
        if self.weak_ridge_strength < 0:
            raise ValueError("weak_ridge_strength must be non-negative.")
        if self.clear_region_protection <= 0:
            raise ValueError("clear_region_protection must be positive.")
        if self.rhlt_edge_gamma <= 0:
            raise ValueError("rhlt_edge_gamma must be positive.")
        if not 0.0 <= self.rhlt_evidence_floor <= 1.0:
            raise ValueError("rhlt_evidence_floor must be between 0.0 and 1.0.")
        if self.frequency_block_size < 8:
            raise ValueError("frequency_block_size must be >= 8.")
        if self.minimum_ridge_wavelength < 2.0:
            raise ValueError("minimum_ridge_wavelength must be >= 2.0 pixels.")
        if self.maximum_ridge_wavelength <= self.minimum_ridge_wavelength:
            raise ValueError("maximum_ridge_wavelength must exceed the minimum.")
        if self.frequency_smoothing_size < 1 or self.frequency_smoothing_size % 2 == 0:
            raise ValueError("frequency_smoothing_size must be a positive odd integer.")
        if not 2 <= self.frequency_bins <= 16:
            raise ValueError("frequency_bins must be between 2 and 16.")
        if self.selector_regression_tolerance < 0:
            raise ValueError("selector_regression_tolerance must be non-negative.")
        if not 0.0 <= self.selector_ssim_floor <= 1.0:
            raise ValueError("selector_ssim_floor must be between 0.0 and 1.0.")
