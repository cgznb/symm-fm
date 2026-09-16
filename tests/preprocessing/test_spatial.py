from __future__ import annotations

import numpy as np

from ispy2_symmflow.preprocessing.spatial import reorient_volume


def test_reorientation_permutes_flips_and_preserves_physical_coordinates() -> None:
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    affine_lps = np.eye(4, dtype=np.float64)
    oriented, new_affine, spacing, codes = reorient_volume(
        volume, affine_lps, "SAR"
    )
    assert oriented.shape == (1, 4, 3, 2)
    assert codes == ("S", "A", "R")
    assert spacing == (1.0, 1.0, 1.0)

    # New [s, a, r] corresponds to old [1-r, 2-a, s] for this identity LPS affine.
    for new_index in ((0, 0, 0), (3, 2, 1), (1, 1, 0)):
        s, a, r = new_index
        old_index = (1 - r, 2 - a, s)
        assert oriented[(0, *new_index)] == volume[(0, *old_index)]
        old_world_lps = affine_lps @ np.asarray((*old_index, 1.0))
        new_world_lps = new_affine @ np.asarray((*new_index, 1.0))
        np.testing.assert_allclose(new_world_lps, old_world_lps)


def test_typical_axial_dhw_affine_only_needs_in_plane_flips() -> None:
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    affine_lps = np.asarray(
        [
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [2.5, 0.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    oriented, new_affine, spacing, codes = reorient_volume(
        volume, affine_lps, "SAR"
    )
    assert oriented.shape == volume.shape
    np.testing.assert_array_equal(oriented, volume[:, :, ::-1, ::-1])
    assert spacing == (2.5, 1.0, 1.0)
    assert codes == ("S", "A", "R")
    np.testing.assert_allclose(
        new_affine @ np.asarray((0, 0, 0, 1.0)),
        affine_lps @ np.asarray((0, 2, 3, 1.0)),
    )
