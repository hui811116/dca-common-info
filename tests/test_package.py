import numpy as np

from dcacinfo.dataset import synToyNView
from dcacinfo.solver._primal_dual_wynerdca import _cmi_subset
from dcacinfo.utils import calcEnt, calcMI


def test_entropy_and_mutual_information():
    assert np.isclose(calcEnt(np.array([0.5, 0.5])), np.log(2))
    assert np.isclose(calcMI(np.full((2, 2), 0.25)), 0.0)


def test_synthetic_multiview_distribution_is_normalized():
    data = synToyNView(ny=2, nb=2, corr=0.1, nv=3)
    joint = data["p_joint"]

    assert joint.shape == (4, 4, 4)
    assert np.all(joint >= 0.0)
    assert np.isclose(joint.sum(), 1.0)


def test_cmi_is_zero_for_views_independent_given_latent():
    px1_given_z = np.array([[0.8, 0.2], [0.3, 0.7]])
    px2_given_z = np.array([[0.7, 0.3], [0.2, 0.8]])
    pzx1x2 = np.stack(
        [
            0.5 * np.outer(px1_given_z[0], px2_given_z[0]),
            0.5 * np.outer(px1_given_z[1], px2_given_z[1]),
        ]
    )
    set_dict = {"tuple_list": [((0,), (1,))]}

    cmi = _cmi_subset(pzx1x2, set_dict)

    assert np.allclose(cmi, 0.0, atol=1e-12)
