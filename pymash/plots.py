from __future__ import annotations

import numpy as np

from .mash import MashResult
from .results import get_pm, get_psd


def mash_plot_meta(
    m: MashResult,
    i: int,
    xlab: str = "Effect size",
    ylab: str = "Condition",
    labels: list[str] | None = None,
    ci: float = 1.96,
    ax=None,
):
    """Forest plot of posterior mean +/- CI for effect *i* across conditions.

    Parameters
    ----------
    m : MashResult
        Fitted mash result.
    i : int
        Row index of the effect to plot.
    xlab, ylab : str
        Axis labels.
    labels : list of str, optional
        Condition names (length R).  Defaults to ``0, 1, ...``.
    ci : float
        Number of posterior SDs for the confidence interval (default 1.96).
    ax : matplotlib Axes, optional
        Axes to draw on.  Created if *None*.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. "
            "Install it with: pip install pymashrink[plot]"
        ) from None

    pm = get_pm(m)[i, :]
    psd = get_psd(m)[i, :]
    R = pm.shape[0]

    if labels is None:
        labels = [str(j) for j in range(R)]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2, 0.4 * R)))

    y = np.arange(R)
    ax.errorbar(pm, y, xerr=ci * psd, fmt="o", color="black", capsize=3)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.invert_yaxis()

    return ax


__all__ = ["mash_plot_meta"]
