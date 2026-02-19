import numpy as np

from optimization_pipeline import (
    biomechanical_model,
    chambolle_tv_denoise_1d,
    pain_from_stress,
    total_variation_1d,
)


def test_biomechanical_model_shape_numpy():
    params = np.array(
        [
            [0.2, 8.0, 0.3, 2.0],
            [0.4, 15.0, 1.2, 3.5],
        ],
        dtype=float,
    )
    stress = biomechanical_model(params, n_steps=64)
    assert stress.shape == (2, 64)


def test_pain_from_stress_minimum_near_target():
    stress = np.array([3.0, 3.8], dtype=float)
    pain = pain_from_stress(stress, stress_target=3.0)
    assert pain[0] < pain[1]


def test_total_variation_constant_signal_is_zero():
    signal = np.ones(200, dtype=float) * 2.5
    tv = float(total_variation_1d(signal))
    assert np.isclose(tv, 0.0)


def test_chambolle_denoising_reduces_total_variation():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0.0, 2.0 * np.pi, 300))
    noisy = x + rng.normal(0.0, 0.25, size=x.shape)
    denoised = chambolle_tv_denoise_1d(noisy, weight=0.2, n_iter=100)

    tv_noisy = float(total_variation_1d(noisy))
    tv_denoised = float(total_variation_1d(denoised))
    assert tv_denoised < tv_noisy
