from IPython import get_ipython
from IPython.core.display import display, HTML
import numpy as np
import datetime
import os
import numpy as np
import pickle
import time

import ast
import pandas as pd

from divisivenormalization.models import (
    DivisiveNetOutputNonlin,
    DivisiveNetUnspecificOutputNonlin,
    ConvNet,
    ConvSubunitNetOutputNonlin,
)


def config_ipython():
    if get_ipython() is not None:
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
        get_ipython().run_line_magic("matplotlib", "inline")
        get_ipython().run_line_magic("config", "InlineBackend.print_figure_kwargs = {'bbox_inches':'tight'}")
        # make notebook display width scalable
        display(HTML("<style>.container { width:98% !important; }</style>"))


def load_dn_model(run_no, resulsts_df, data, log_dir, eval_batches=256):
    df = resulsts_df
    df_slice = df.loc[df["run_no"] == run_no]

    log_hash = get_log_hash(run_no)
    filter_sizes = ast.literal_eval(df_slice.iloc[0]["filter_sizes"])
    out_channels = ast.literal_eval(df_slice.iloc[0]["out_channels"])
    strides = ast.literal_eval(df_slice.iloc[0]["strides"])
    paddings = ast.literal_eval(df_slice.iloc[0]["paddings"])
    smooth_weights = ast.literal_eval(df_slice.iloc[0]["smooth_weights"])
    sparse_weights = ast.literal_eval(df_slice.iloc[0]["sparse_weights"])
    readout_sparse_weight = df_slice.iloc[0]["readout_sparse_weight"]
    output_nonlin_smooth_weight = df_slice.iloc[0]["output_nonlin_smooth_weight"]
    pool_kernel_size = ast.literal_eval(df_slice.iloc[0]["pool_kernel_size"])
    pool_type = df_slice.iloc[0]["pool_type"]
    dilation = ast.literal_eval(df_slice.iloc[0]["dilation"])
    M = df_slice.iloc[0]["M"]
    dn_u_size = ast.literal_eval(df_slice.iloc[0]["dn_u_size"])
    abs_v_l1_weight = df_slice.iloc[0]["abs_v_l1_weight"]
    dn_padding = df_slice.iloc[0]["dn_padding"]

    hyperparam_dn = DivisiveNetOutputNonlin(
        filter_sizes,
        out_channels,
        strides,
        paddings,
        smooth_weights,
        sparse_weights,
        readout_sparse_weight,
        output_nonlin_smooth_weight,
        pool_kernel_size,
        pool_type,
        dilation,
        M,
        dn_u_size,
        abs_v_l1_weight,
        dn_padding=dn_padding,
        data=data,
        log_dir=log_dir,
        log_hash=log_hash,
        eval_batches=eval_batches,
    )
    hyperparam_dn.load_best()

    return hyperparam_dn


def load_dn_nonspecific_model(run_no, resulsts_df, data, log_dir, eval_batches=256):
    df = resulsts_df
    df_slice = df.loc[df["run_no"] == run_no]

    log_hash = get_log_hash(run_no)
    filter_sizes = ast.literal_eval(df_slice.iloc[0]["filter_sizes"])
    out_channels = ast.literal_eval(df_slice.iloc[0]["out_channels"])
    strides = ast.literal_eval(df_slice.iloc[0]["strides"])
    paddings = ast.literal_eval(df_slice.iloc[0]["paddings"])
    smooth_weights = ast.literal_eval(df_slice.iloc[0]["smooth_weights"])
    sparse_weights = ast.literal_eval(df_slice.iloc[0]["sparse_weights"])
    readout_sparse_weight = df_slice.iloc[0]["readout_sparse_weight"]
    output_nonlin_smooth_weight = df_slice.iloc[0]["output_nonlin_smooth_weight"]
    pool_kernel_size = ast.literal_eval(df_slice.iloc[0]["pool_kernel_size"])
    pool_type = df_slice.iloc[0]["pool_type"]
    dilation = ast.literal_eval(df_slice.iloc[0]["dilation"])
    M = df_slice.iloc[0]["M"]
    dn_u_size = ast.literal_eval(df_slice.iloc[0]["dn_u_size"])
    abs_v_l1_weight = df_slice.iloc[0]["abs_v_l1_weight"]
    dn_padding = df_slice.iloc[0]["dn_padding"]

    hyperparam_dn = DivisiveNetUnspecificOutputNonlin(
        filter_sizes,
        out_channels,
        strides,
        paddings,
        smooth_weights,
        sparse_weights,
        readout_sparse_weight,
        output_nonlin_smooth_weight,
        pool_kernel_size,
        pool_type,
        dilation,
        M,
        dn_u_size,
        abs_v_l1_weight,
        dn_padding=dn_padding,
        data=data,
        log_dir=log_dir,
        log_hash=log_hash,
        eval_batches=eval_batches,
    )
    hyperparam_dn.load_best()

    return hyperparam_dn


def load_subunit_model(run_no, resulsts_df, data, log_dir, eval_batches=256):
    df = resulsts_df
    df_slice = df.loc[df["run_no"] == run_no]

    log_hash = get_log_hash(run_no)
    filter_sizes = ast.literal_eval(df_slice.iloc[0]["filter_sizes"])
    out_channels = ast.literal_eval(df_slice.iloc[0]["out_channels"])
    strides = ast.literal_eval(df_slice.iloc[0]["strides"])
    paddings = ast.literal_eval(df_slice.iloc[0]["paddings"])
    smooth_weights = ast.literal_eval(df_slice.iloc[0]["smooth_weights"])
    sparse_weights = ast.literal_eval(df_slice.iloc[0]["sparse_weights"])
    readout_sparse_weight = df_slice.iloc[0]["readout_sparse_weight"]
    output_nonlin_smooth_weight = df_slice.iloc[0]["output_nonlin_smooth_weight"]

    hyperparam_dn = ConvSubunitNetOutputNonlin(
        filter_sizes,
        out_channels,
        strides,
        paddings,
        smooth_weights,
        sparse_weights,
        readout_sparse_weight,
        output_nonlin_smooth_weight,
        data=data,
        log_dir=log_dir,
        log_hash=log_hash,
        eval_batches=eval_batches,
    )
    hyperparam_dn.load_best()

    return hyperparam_dn


def load_cnn3_model(run_no, resulsts_df, data, log_dir, eval_batches=256):
    df = resulsts_df
    df_slice = df.loc[df["run_no"] == run_no]

    log_hash = get_log_hash(run_no)
    filter_sizes = [13, 3, 3]
    out_channels = [32, 32, 32]
    strides = [1, 1, 1]
    paddings = ["VALID", "SAME", "SAME"]
    smooth_weights = ast.literal_eval(df_slice.iloc[0]["smooth_weights"])
    sparse_weights = ast.literal_eval(df_slice.iloc[0]["sparse_weights"])
    readout_sparse_weight = df_slice.iloc[0]["readout_sparse_weight"]
    output_nonlin_smooth_weight = df_slice.iloc[0]["output_nonlin_smooth_weight"]

    model = ConvNet(data=data, log_dir=log_dir, log_hash=log_hash, eval_batches=eval_batches)
    model.build(
        filter_sizes=filter_sizes,
        out_channels=out_channels,
        strides=strides,
        paddings=paddings,
        smooth_weights=smooth_weights,
        sparse_weights=sparse_weights,
        readout_sparse_weight=readout_sparse_weight,
        output_nonlin_smooth_weight=output_nonlin_smooth_weight,
    )
    model.load_best()

    return model


def simplify_df(df):
    """Return simplified pandas df of results"""

    df_plain = df.copy()
    df_plain = df_plain.iloc[0:-1]

    # evaluate strings
    for ind, _ in df_plain.iterrows():
        for key in [
            "stim_size",
            "filter_sizes",
            "out_channels",
            "strides",
            "paddings",
            "smooth_weights",
            "dn_u_size",
        ]:
            df_plain.at[ind, key] = ast.literal_eval(df_plain.at[ind, key])[0]
        for key in ["pool_kernel_size", "dilation"]:
            df_plain.at[ind, key] = ast.literal_eval(df_plain.at[ind, key])[1]

    # convert strings to numerics
    to_numeric_key_list = list(df_plain.keys())
    # do not convert strings for following keys:
    rm_key_list = ["paddings", "pool_type", "time_stamp", "dn_padding"]
    for key in rm_key_list:
        if key in to_numeric_key_list:
            to_numeric_key_list.remove(key)

    df_plain[to_numeric_key_list] = df_plain[to_numeric_key_list].apply(pd.to_numeric, errors="coerce")

    df_plain = df_plain.sort_values(by="avg_corr_after_train", ascending=False)
    df_plain["norm_pool_span"] = df_plain["dilation"] * df_plain["dn_u_size"]

    return df_plain


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def inch2cm(x):
    return x * 2.54


def cm2inch(x):
    return x / 2.54


def log_uniform(low=0, high=1, base=10):
    """draw samples from a uniform distribution in logspace"""
    return np.power(base, np.random.uniform(low, high))


def time_stamp():
    """creates a string timestamp YYYY-MM-DD HH:MM:SS"""
    s = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    return s


def get_log_hash(run_no):
    return "run-{0:0>15}".format(run_no)


def get_weights(dn):
    with dn.session.as_default():
        w = dn.W[-1].eval()
        features_chanfirst = w[:, :, 0, :].transpose()

        p = dn.p.eval()
        u = dn.u.eval()
        v = dn.v.eval()
        dn_exponent = dn.dn_exponent.eval()
        readout_feat_w = dn.W_features.eval()

        # get normalization input
        images = dn.data.val()[0]
        responses = dn.data.val()[1]
        real_responses = dn.data.val()[2]

        # evaluate with eval batches
        batches = 2320
        numpts = images.shape[0]
        numbatch = int(np.ceil(numpts / batches))

        pooled = np.zeros((numpts, 28, 28, 32))
        for batch_idx in range(0, numbatch):
            if batches * (batch_idx + 1) > numpts:
                idx = (batch_idx * batches) + np.arange(0, numpts - (batch_idx * batches))
            else:
                idx = (batch_idx * batches) + np.arange(0, batches)

            feed_dict = {
                dn.images: images[
                    idx,
                ],
                dn.responses: responses[
                    idx,
                ],
                dn.real_responses: real_responses[
                    idx,
                ],
                dn.is_training: False,
            }
            pooled[
                idx,
            ] = dn.pooled.eval(feed_dict)

    return features_chanfirst, p, pooled, readout_feat_w, u, v, dn_exponent


def pkl_dump(var, run_no, name, path):
    fname = "run_" + str(run_no) + "_" + str(name)
    fpath = os.path.join(path, fname)
    with open(fpath, "wb") as f:
        pickle.dump(var, f)


def pkl_load(run_no, name, path):
    fname = "run_" + str(run_no) + "_" + str(name)
    fpath = os.path.join(path, fname)
    with open(fpath, "rb") as f:
        var = pickle.load(f)
    return var


from colorsys import rgb_to_hls, hls_to_rgb


def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls_to_rgb(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def lighten_color(r, g, b, factor=0.1):
    c = adjust_color_lightness(r, g, b, 1 + factor)
    return [cc / 255 for cc in c]


def darken_color(r, g, b, factor=0.1):
    c = adjust_color_lightness(r, g, b, 1 - factor)
    return [cc / 255 for cc in c]
