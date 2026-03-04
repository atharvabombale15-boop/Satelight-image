import numpy as np

from inference import compute_mask, compute_overlay


def test_compute_mask_threshold():
    prob = np.array([[0.1, 0.2], [0.15, 0.9]], dtype=np.float32)
    mask = compute_mask(prob, 0.15)
    expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    assert np.array_equal(mask, expected)


def test_compute_overlay():
    base = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    overlay = compute_overlay(base, mask, alpha=0.5)

    expected = np.array(
        [
            [[132, 10, 15], [40, 50, 60]],
            [[70, 80, 90], [177, 55, 60]],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(overlay, expected)
