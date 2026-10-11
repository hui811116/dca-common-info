import numpy as np

from dcacinfo.dataset.dataset import synToyNView
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
