import matplotlib
matplotlib.use("Agg")

import numpy as np

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash
from pymash.plots import mash_plot_meta


def test_mash_plot_meta_returns_axes():
    rng = np.random.default_rng(42)
    Bhat = rng.normal(size=(40, 5))
    Shat = np.ones_like(Bhat)
    data = mash_set_data(Bhat, Shat)
    U = cov_canonical(data)
    m = mash(data, Ulist=U, outputlevel=2)

    ax = mash_plot_meta(m, 0)
    assert ax is not None
    assert ax.get_xlabel() == "Effect size"
    assert ax.get_ylabel() == "Condition"
    # 5 conditions = 5 y-tick labels
    assert len(ax.get_yticklabels()) == 5


def test_mash_plot_meta_custom_labels():
    rng = np.random.default_rng(43)
    Bhat = rng.normal(size=(30, 3))
    Shat = np.ones_like(Bhat)
    data = mash_set_data(Bhat, Shat)
    U = cov_canonical(data)
    m = mash(data, Ulist=U, outputlevel=2)

    labels = ["Tissue A", "Tissue B", "Tissue C"]
    ax = mash_plot_meta(m, 0, labels=labels)
    tick_texts = [t.get_text() for t in ax.get_yticklabels()]
    assert tick_texts == labels
