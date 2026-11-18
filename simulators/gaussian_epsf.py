import numpy as np


def gaussian_epsf(distance_sq, fwhm_nm):
    """Dimensionless time-gated MINSTED E-PSF, with broadcastable inputs."""
    return np.exp(
        -4.0 * np.log(2.0) * np.asarray(distance_sq)
        / (np.asarray(fwhm_nm) ** 2 + 1e-12)
    )
