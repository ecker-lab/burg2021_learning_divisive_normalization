import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def compute_confidence_interval(lst, alpha=0.95):
    return stats.t.interval(alpha, len(lst) - 1, loc=np.mean(lst), scale=stats.sem(lst, ddof=1))


def compute_fev_summary_stats(fev_lst, alpha=0.95):
    res = {
        "mean": [],
        "max": [],
        "max_corr": [],
        "sem": [],
        "conf_int": [],
        "shapiro": [],
        "shapiro_reject": [],
    }
    for fev in fev_lst:
        res["mean"].append(np.mean(fev))
        res["max"].append(np.max(fev))
        res["max_corr"].append(fev[0])
        res["sem"].append(stats.sem(fev, ddof=1))
        res["conf_int"].append(compute_confidence_interval(fev, alpha))
        shapiro = stats.shapiro(fev)
        res["shapiro"].append(shapiro)
        res["shapiro_reject"].append(shapiro[1] < (1 - alpha))
        if res["shapiro_reject"] == True:
            print(
                "WARNING: shapiro: null hypothesis that data comes from normal distribution can be rejected. Conficende intervals meaningful?"
            )
    return res


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def gausswin(w, d=2):
    k = w.shape[-1]
    ii = np.linspace(-d, d, k, endpoint=True)
    ii, jj = np.meshgrid(ii, ii)
    return np.exp(-(ii ** 2 + jj ** 2) / 2.0)


def circ_var(p, fmin=0.3, fmax=0.7, plot=False):
    """
    Returns mean resultant vector in relvant frequency band.

    The length of the resultant vector [np.abs(v)] indicates the degree
    of orientation tuning while its angle [np.angle(v)] indicates the
    preferred orientation.
    """
    p = np.fft.fftshift(p)
    k = p.shape[-1]
    x = np.linspace(-1, 1, k, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx ** 2 + yy ** 2)
    phi = np.arctan2(xx, yy)
    p_band = p * ((r > fmin) & (r < fmax))
    p_band = p_band / (p_band ** 2).mean()
    if plot:
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(p, cmap="gray")
        axes[1].imshow(p_band, cmap="gray")
    mean_resultant_vector = (p_band * np.exp(2j * phi)).mean()
    return mean_resultant_vector


def angles_circ_var(features, threshold=0.2):
    """Compute filter orientation with circular variance.

    Args:
        features: np.array of shape (feature, size, size) of a batch of features

    Returns:
        np.array of orientation angles.
    """

    w = features
    w = w / (w ** 2).sum(axis=(1, 2), keepdims=True)

    # Power spectrum (windowed)
    N = 64
    ww = w * gausswin(w)
    ww = ww / (ww ** 2).sum(axis=(1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(ww, s=(N, N)))

    v_tot = np.array([circ_var(p) for p in P])  # Get circular variance
    idx_unori = np.abs(v_tot) <= threshold  # Select oriented features
    v_tot[idx_unori] = np.nan
    angles = np.angle(v_tot) / 2  # Orientation is circular on np.pi (angle on 2 * np.pi)

    # Project all angle values into the interval [0, np.pi]
    for idx, a in enumerate(angles):
        if a < 0:
            a = a + 2 * np.pi
            angles[idx] = a
        if a >= np.pi:
            a = a - np.pi
            angles[idx] = a

    return angles

    """compute difference in angles. Return angle_diff as float[,]"""


def angle_diff(angles):
    """Compute pairwise orientation differences.

    Args:
        angles: np.array of angles, shape (num_angles,)

    Returns:
        np.array of pairwise orientation differences, shape (num_angles, num_angles)
    """

    angle_diff = np.zeros((np.shape(angles)[0], np.shape(angles)[0]))
    for k in range(np.shape(angles)[0]):
        a_k = angles[k]

        for l in range(np.shape(angles)[0]):
            a_l = angles[l]

            if a_k < 0:
                print("WARNING: a_k < 0", a_k)

            if a_l < 0:
                print("WARNING: a_k < 0", a_l)

            delta_a = np.abs(np.abs(a_k) - np.abs(a_l))

            if delta_a > np.pi / 2:
                delta_a = np.pi - delta_a

            if delta_a < 0:
                print("WARNING: delta Angle ", delta_a, "< 0")
                print(a_k)
                print(a_l)
                print(delta_a)

            if delta_a > np.pi / 2:
                print("WARNING: delta Angle ", delta_a, "> PI/2")
                print(a_k)
                print(a_l)
                print(delta_a)

            angle_diff[k, l] = delta_a

    return angle_diff


def orientation_masks(angle_diff, angle_crit=45):
    """Compute mask arrays for unoriented, similarly oriented and dissimilarly oriented angles.

    Args:
        angle_diff (array-like): pairwise angle differences, shape (num_angles, num_angles)
        angle_crit (float): critical angle in degree to split into similarly and disimilarly oriented features

    Returns:
        np.array, np.array, np.array: unoriented mask, similarly oriented mask, dissimilarly oriented mask
    """

    angle_crit = np.deg2rad(angle_crit)
    unor_mask = np.isnan(angle_diff)
    sim_mask = np.logical_and((angle_diff < angle_crit), np.logical_not(unor_mask))
    dissim_mask = np.logical_and((angle_diff >= angle_crit), np.logical_not(unor_mask))
    return unor_mask, sim_mask, dissim_mask


def norm_input(pooled, p):
    """Calculates the normalization input matrix (shape: in-chan, out-chan) from the divisive normalization model's pooled and p variables.

    Args:
        p (array-like): normalization weights, shape (1,1, in chan, out chan)
        pooled (array-like): pooled normalization input, shape (batch, space, space, chan)
    """

    # average over images and space
    pooled_avg = np.average(pooled, axis=(0, 1, 2))
    # expand pooled_avg in out-dim -> shape (in, out) = (32, 1)
    pooled_avg = np.expand_dims(pooled_avg, -1)

    p = np.squeeze(p)
    # weight * activity
    contrib = p * pooled_avg

    return contrib


def plot_contribution_matrix_chan_first(
    contrib,
    features,
    index_permutation_lst,
    angle_difference,
    oriented_bools,
    figsize,
):
    """Plots matrix showing normalization input.

    Colorscale: white = 0, darkest blue = np.max(contrib)

    Args:
        contrib (array): Normalization input with shape (in-chan, out-chan)
        features (array): shape (chan, space, space)
        index_permutation_lst (array): shape (chan)

    Returns:
        fig object
    """

    idc = index_permutation_lst
    num_features = features.shape[0]

    no_rows = 33
    no_columns = 33
    linewidth_rescale = figsize[0] / no_rows

    fig, axes = plt.subplots(no_rows, no_columns, figsize=figsize, dpi=300)

    # plot w column
    for i in range(num_features):
        im = features[idc[i]]
        vmax = np.max(np.abs(im))
        vmin = -vmax

        ridx = i + 1  # row index
        cidx = 0  # column index
        ax = axes[ridx, cidx]
        _ = ax.imshow(im, vmin=vmin, vmax=vmax, cmap="Greys")
        ax.tick_params(which="both", bottom=False, labelbottom=False, left=False, labelleft=False)
        for axis_name in ["top", "bottom", "left", "right"]:
            ax.spines[axis_name].set_linewidth(linewidth_rescale)

    # plot w row
    for i in range(num_features):
        im = features[idc[i]]
        vmax = np.max(np.abs(im))
        vmin = -vmax

        ridx = 0
        cidx = i + 1
        ax = axes[ridx, cidx]
        _ = ax.imshow(im, vmin=vmin, vmax=vmax, cmap="Greys")
        ax.tick_params(which="both", bottom=False, labelbottom=False, left=False, labelleft=False)
        for axis_name in ["top", "bottom", "left", "right"]:
            ax.spines[axis_name].set_linewidth(linewidth_rescale)

    # plot empty
    for cidx in range(33, no_columns):
        ax = axes[0, cidx]
        _ = ax.imshow(np.zeros((1, 1)), vmin=0, vmax=0, cmap="Greys")
        ax.axis("off")

    ax = axes[0, 0]
    _ = ax.imshow(np.zeros((1, 1)), vmin=0, vmax=0, cmap="Greys")
    ax.axis("off")

    vmax = np.max(contrib[oriented_bools][:, oriented_bools])
    vmin = 0

    # sort contrib according to idc
    contrib = contrib[idc, :]
    contrib = contrib[:, idc]

    # insert lines if at 45deg desicion boundary
    a_diff = angle_difference
    a_crit = np.deg2rad(45)

    lw = 5 * linewidth_rescale
    col = "#000000"

    # sort contrib according to idc
    a_diff = a_diff[idc, :]
    a_diff = a_diff[:, idc]

    # iidx: in-chan-idx, oidx: out-chan-idx
    for iidx, oidx in itertools.product(range(contrib.shape[0]), range(contrib.shape[1])):
        im = contrib[iidx, oidx]
        im = np.reshape(im, (1, 1))

        ridx = oidx + 1
        cidx = iidx + 1
        ax = axes[ridx, cidx]
        _ = ax.imshow(im, vmin=vmin, vmax=vmax, cmap="Blues")
        ax.tick_params(which="both", bottom=False, labelbottom=False, left=False, labelleft=False)

        # turn spines off
        for _, spine in ax.spines.items():
            spine.set_visible(False)

        # horizontal lines
        if oidx > 0:
            if (a_diff[iidx, oidx] >= a_crit and a_diff[iidx, oidx - 1] < a_crit) or (
                a_diff[iidx, oidx] < a_crit and a_diff[iidx, oidx - 1] >= a_crit
            ):
                ax.spines["top"].set_visible(True)
                ax.spines["top"].set_linewidth(lw)
                ax.spines["top"].set_color(col)

        # vertical lines
        if iidx > 0:
            if (a_diff[iidx, oidx] >= a_crit and a_diff[iidx - 1, oidx] < a_crit) or (
                a_diff[iidx, oidx] < a_crit and a_diff[iidx - 1, oidx] >= a_crit
            ):
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_linewidth(lw)
                ax.spines["left"].set_color(col)

    return fig


def cohens_d(x1, x2):
    """Compute Cohen's d."""

    x1 = np.array(x1)
    x2 = np.array(x2)

    n1 = x1.shape[0]
    n2 = x2.shape[0]

    s1_sq = np.var(x1, ddof=1)
    s2_sq = np.var(x2, ddof=1)

    s = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
    d = (np.mean(x1) - np.mean(x2)) / s

    return d
