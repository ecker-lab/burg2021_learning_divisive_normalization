import os

import datajoint as dj
import numpy as np

from .stimuli import CenterSurround, GaborSet
from .data import *
from .utils import *

schema = dj.schema("mburg_dnrebuttal_insilico", locals())

BATCH_SIZE = 1024
MAX_CONTRAST = 2.52
MIN_CONTRAST = 0.01 * MAX_CONTRAST

MIN_SIZE = 4
MAX_SIZE = 40


class path_args:
    fname_best_csv = "df_best.csv"
    fname_results_csv = "results.csv"
    logs_dir = "train_logs"
    dn_path = "/projects/burg2021_learning-divisive-normalization/divisive_net"
    nonspecific_path = "/projects/burg2021_learning-divisive-normalization/nonspecific_divisive_net"
    subunit_path = "/projects/burg2021_learning-divisive-normalization/subunit_net"
    cnn3_path = "/projects/burg2021_learning-divisive-normalization/cnn3"


def get_best_run_nums(model_type):
    """Return array of run numbers for a `model_type`, sorted by descending validation correlation."""
    return (Fit() & dict(model_type=model_type)).fetch("run_no", order_by="val_corr DESC")


def get_fit_entries(path, model_type, num_best):
    """Get entries to insert into the Fit() table, describing the fitted models of a certain model type."""
    data = []
    for idx in range(num_best):
        df = pd.read_csv(os.path.join(path, path_args.fname_best_csv))
        run_no = int(df.iloc[idx]['run_no'])
        val_corr = float(df.iloc[idx]['avg_corr_after_train'])
        test_fev = float(df.iloc[idx]['fev'])
        model = Fit().load_model_run_no(run_no, model_type)
        fev_per_neuron = model.evaluate_fev_testset_per_neuron()
        data.append((run_no, model_type, val_corr, test_fev, fev_per_neuron))
    return data


@schema
class Fit(dj.Manual):
    @property
    def definition(self):
        return """
            run_no           : int unsigned  # run number, unique id
            model_type       : varchar(64)   # model type, e.g. dn, cnn3, ...
            ---
            val_corr         : float         # loss on validation set
            test_fev         : float         # correlation on test set
            fev_per_neuron   : blob          # lst with length of number of neurons
        """

    @property
    def key_source(self):
        return Fit()

    def load_model(self, key):
        print("load model key", key)
        key = (self.key_source & key).fetch1(dj.key)
        run_no = key["run_no"]
        model_type = key["model_type"]

        model = self.load_model_run_no(run_no, model_type)

        return model

    def load_model_run_no(self, run_no, model_type):
        data_dict = Dataset.get_clean_data()
        data = MonkeySubDataset(data_dict, seed=1000, train_frac=0.8, subsample=2, crop=30)

        if model_type == "dn":
            results_df = pd.read_csv(os.path.join(path_args.dn_path, path_args.fname_results_csv))
            model = load_dn_model(
                run_no,
                results_df,
                data,
                os.path.join(path_args.dn_path, path_args.logs_dir),
            )

        elif model_type == "dn-nonspecific":
            results_df = pd.read_csv(os.path.join(path_args.nonspecific_path, path_args.fname_results_csv))
            model = load_dn_nonspecific_model(
                run_no,
                results_df,
                data,
                os.path.join(path_args.nonspecific_path, path_args.logs_dir),
            )

        elif model_type == "subunit":
            results_df = pd.read_csv(os.path.join(path_args.subunit_path, path_args.fname_results_csv))
            model = load_subunit_model(
                run_no,
                results_df,
                data,
                os.path.join(path_args.subunit_path, path_args.logs_dir),
            )

        elif model_type == "cnn3":
            results_df = pd.read_csv(os.path.join(path_args.cnn3_path, path_args.fname_results_csv))
            model = load_cnn3_model(
                run_no,
                results_df,
                data,
                os.path.join(path_args.cnn3_path, path_args.logs_dir),
            )

        return model


@schema
class GaborParams(dj.Lookup):
    definition = """
        param_id        : tinyint unsigned  # id for parameter set
        ---
        x_start         : tinyint unsigned  # start location in x
        x_end           : tinyint unsigned  # end location in x
        y_start         : tinyint unsigned  # start location in y
        y_end           : tinyint unsigned  # end location in y
        min_size        : float             # minimum size in pixels
        num_sizes       : tinyint unsigned  # number of different sizes
        size_increment  : float             # relative size increments
        min_sf          : float             # minimum spatial frequency
        num_sf          : tinyint unsigned  # number of spatial frequencies (SF)
        sf_increment    : float             # relative SF increments
        min_contrast    : float             # minimum contrast (Michelson)
        num_contrasts   : tinyint unsigned  # number of contrast levels
        contrast_increment : float          # relative contrast increments
        num_orientations : tinyint unsigned # number of orientations
        num_phases      : tinyint unsigned  # number of phases
        """
    num_contrasts = 6
    contrast_increment = (MAX_CONTRAST / MIN_CONTRAST) ** (1 / (num_contrasts - 1))

    num_sizes = 8
    size_increment = (MAX_SIZE / MIN_SIZE) ** (1 / (num_sizes - 1))
    contents = [
        [
            1,
            0,
            40,
            0,
            40,
            MIN_SIZE,
            num_sizes,
            size_increment,
            (1.3 ** -1),
            10,
            1.3,
            MIN_CONTRAST,
            num_contrasts,
            contrast_increment,
            12,
            8,
        ],
    ]

    def gabor_set(self, key, canvas_size):
        p = (self & key).fetch1()
        center_range = [p["x_start"], p["x_end"], p["y_start"], p["y_end"]]
        sizes = p["min_size"] * p["size_increment"] ** np.arange(p["num_sizes"])
        sfs = p["min_sf"] * p["sf_increment"] ** np.arange(p["num_sf"])
        c = p["min_contrast"] * p["contrast_increment"] ** np.arange(p["num_contrasts"])
        g = GaborSet(
            canvas_size,
            center_range,
            sizes,
            sfs,
            c,
            p["num_orientations"],
            p["num_phases"],
        )
        return g


@schema
class OptimalGabor(dj.Computed):
    definition = """
        -> GaborParams
        -> Fit
        ---
        """

    class Unit(dj.Part):
        definition = """
            -> master
            unit_id      : int unsigned     # output unit id
            ---
            max_response : float            # response to optimal Gabor
            max_index    : int              # index of optimal Gabor
            """

        def params(self, key):
            gabor_set = GaborParams().gabor_set(key, None)
            idx = (self & key).fetch1("max_index")
            return gabor_set.params_from_idx(idx)

        def params_dict(self, key):
            gabor_set = GaborParams().gabor_set(key, None)
            idx = (self & key).fetch1("max_index")
            return gabor_set.params_dict_from_idx(idx)

        def optimal_gabor(self, key, canvas_size=[40, 40]):
            gabor_set = GaborParams().gabor_set(key, canvas_size)
            idx = (self & key).fetch1("max_index")
            return gabor_set.gabor_from_idx(idx)

    @property
    def key_source(self):
        key = Fit().fetch(dj.key, order_by="val_corr DESC")
        print(key)
        return Fit() * GaborParams() & key

    def _make_tuples(self, key):
        model = Fit().load_model(key)
        canvas_size = [model.data.px_y, model.data.px_x]
        g = GaborParams().gabor_set(key, canvas_size)
        max_response, max_idx = 0, 0
        for batch_idx, images in enumerate(g.image_batches(BATCH_SIZE)):
            dummy_responses = np.zeros((images.shape[0], model.data.num_neurons))
            dummy_real_res = dummy_responses * np.nan
            r = model.eval(
                images=images[..., None],
                responses=dummy_responses,
                real_responses=dummy_real_res,
            )
            r = r[-1]
            max_r = r.max(axis=0)
            max_i = batch_idx * BATCH_SIZE + r.argmax(axis=0)
            new_max = max_response < max_r
            max_response = np.maximum(max_response, max_r)
            max_idx = ~new_max * max_idx + new_max * max_i
            if not (batch_idx % 10):
                print(batch_idx, max_response.mean())
                dj.conn().ping()

        self.insert1(key)
        tuples = [
            dict(key, unit_id=id, max_response=mr, max_index=mi)
            for id, mr, mi in zip(np.arange(len(max_idx)), max_response, max_idx)
        ]
        self.Unit().insert(tuples)


@schema
class OrthPlaidsContrastParams(dj.Lookup):
    definition = """
        opc_param_id     : tinyint unsigned  # id for parameter set
        ---
        min_contrast    : float             # minimum contrast (Michelson)
        num_contrasts   : tinyint unsigned  # number of contrast levels
        contrast_increment : float          # relative contrast increments
        num_phases      : tinyint unsigned  # number of phases, lin spaced
        """

    num_contrasts = 9
    contrast_increment = (0.5 * MAX_CONTRAST / MIN_CONTRAST) ** (1 / (num_contrasts - 1))

    contents = [[1, MIN_CONTRAST, num_contrasts, contrast_increment, 8]]

    def contrasts(self, key):
        min_contrast, contrast_increment, num_contrasts = (self & key).fetch1(
            "min_contrast", "contrast_increment", "num_contrasts"
        )
        c = min_contrast * contrast_increment ** np.arange(num_contrasts)
        c = np.concatenate([np.zeros(1), c], axis=0)
        return c

    def phases(self, key):
        num_phases = (self & key).fetch1("num_phases")
        phases = np.linspace(start=0, stop=2 * np.pi, endpoint=False, num=num_phases)
        return phases

    def gabor_set(self, key, canvas_size, loc, size, spatial_freq, orientation, phase):
        center_range = [loc[0], loc[0] + 1, loc[1], loc[1] + 1]
        contrasts = self.contrasts(key)
        g = GaborSet(
            canvas_size,
            center_range,
            [size],
            [spatial_freq],
            contrasts,
            [orientation],
            [phase],
            relative_sf=False,
        )
        return g

    def plaids(self, key, ph_shift=0, canvas_size=(40, 40)):
        canvas_size = list(canvas_size)
        loc, sz, sf, _, ori, ph = OptimalGabor.Unit().params(key)
        g_pref = OrthPlaidsContrastParams().gabor_set(key, canvas_size, loc, sz, sf, ori, ph)
        g_orth = OrthPlaidsContrastParams().gabor_set(key, canvas_size, loc, sz, sf, ori + np.pi / 2, ph + ph_shift)
        comps_pref = g_pref.images()
        comps_orth = g_orth.images()
        plaids = comps_pref[None, ...] + comps_orth[:, None, ...]
        return plaids


@schema
class OrthPlaidsContrast(dj.Computed):
    definition = """
        -> OptimalGabor
        ---
        contrasts    : blob        # list of contrasts
        orth_phase_shifts : blob   # list of phase shifts
    """

    class Unit(dj.Part):
        definition = """
            -> master
            -> OptimalGabor.Unit
            orth_phase_shift_idx   : tinyint unsigned   # index for phase shift of master table
            ---
            tuning_curve  : blob  # contrast preferred x contrast orthogonal
            nonlin_input  : blob  # contrast preferred x contrast orthogonal, input values to outpout nonlinearity
        """

    def orth_phase_shift(self, key):
        ph_shift_idx = (self.Unit() & key).fetch1("orth_phase_shift_idx")
        ph_shifts = (self & key).fetch1("orth_phase_shifts")
        ph_shift = ph_shifts[ph_shift_idx]
        return ph_shift

    def plaids(self, key):
        ph_shift = self.orth_phase_shift(key)
        return OrthPlaidsContrastParams().plaids(key, ph_shift=ph_shift)

    def _make_tuples(self, key):
        model = Fit().load_model(key)
        canvas_size = [model.data.px_y, model.data.px_x]
        contrasts = OrthPlaidsContrastParams().contrasts(key)
        key.update(contrasts=contrasts)
        orth_phase_shifts = OrthPlaidsContrastParams().phases(key)
        key.update(orth_phase_shifts=orth_phase_shifts)
        self.insert1(key)

        for i, key in enumerate((OptimalGabor.Unit() & key).fetch(dj.key)):
            for ph_i, ph_shift in enumerate(orth_phase_shifts):
                plaids = OrthPlaidsContrastParams().plaids(key, ph_shift=ph_shift)
                plaids = np.reshape(plaids, [-1] + canvas_size + [1])

                feed_dict = {model.images: plaids, model.is_training: False}
                tuning_curve = model.session.run(model.prediction[:, key["unit_id"]], feed_dict=feed_dict)
                with model.session.as_default():
                    h_out = model.h_out.eval(feed_dict)[:, key["unit_id"]]
                    b_out = model.b_out.eval()[key["unit_id"]]
                nonlin_input = h_out + b_out

                tupl = key
                tupl["tuning_curve"] = tuning_curve.reshape([len(contrasts), len(contrasts)])
                tupl["nonlin_input"] = nonlin_input.reshape([len(contrasts), len(contrasts)])
                tupl["orth_phase_shift_idx"] = ph_i
                self.Unit.insert1(tupl)

            if not (i % 10):
                print("Unit {:d}".format(i))
                dj.conn().ping()


@schema
class OrthPlaidsContrastInhibPercentPhaseAvg(dj.Computed):
    definition = """
        -> OrthPlaidsContrast
        ---
    """

    class Unit(dj.Part):
        definition = """
            -> master
            unit_id                 : int unsigned
            ---
            inhib_percent_curve     : blob
            inhib_percent           : float
        """

    def _make_tuples(self, key):
        self.insert1(key)
        _ = key.pop("orth_phase_shift_idx", None)
        for i, uid in enumerate((OrthPlaidsContrast.Unit() & key & dict(orth_phase_shift_idx=0)).fetch("unit_id")):
            key["unit_id"] = uid
            t = (OrthPlaidsContrast().Unit() & key & dict(unit_id=uid)).fetch("tuning_curve")
            t = np.array([np.array(a) for a in t]).mean(axis=0)
            index_curve = 1 - (t / t[:1, :])
            key["inhib_percent_curve"] = index_curve
            key["inhib_percent"] = np.max(index_curve)
            self.Unit.insert1(key)

            if not (i % 10):
                print("Unit {:d}".format(i))
                dj.conn().ping()


@schema
class CenterSurroundParams(dj.Lookup):
    definition = """
        param_id                    : tinyint unsigned
        ---
        total_size                  : float  # radius
        min_size_center             : float
        num_sizes_center            : tinyint unsigned
        size_center_increment       : float
        min_size_surround           : float
        num_sizes_surround          : tinyint unsigned
        size_surround_increment     : float
        min_contrast_center         : float
        num_contrasts_center        : tinyint unsigned
        contrast_center_increment   : float
        min_contrast_surround       : float
        num_contrasts_surround      : tinyint unsigned
        contrast_surround_increment : float
        """

    contents = [
        [
            1,
            60,
            0.05,
            15,
            1.23859,
            0.05,
            15,
            1.23859,
            MAX_CONTRAST,
            1,
            1,
            MAX_CONTRAST,
            1,
            1,
        ]
    ]

    def center_set(self, key, canvas_size, loc, sf, ori, phase):
        p = (self & key).fetch1()
        center_range = [loc[0], loc[0] + 1, loc[1], loc[1] + 1]
        sizes_center = [-0.01] + list(
            p["min_size_center"] * p["size_center_increment"] ** np.arange(p["num_sizes_center"])
        )
        sizes_surround = [1]
        contrasts_center = p["min_contrast_center"] * p["contrast_center_increment"] ** np.arange(
            p["num_contrasts_center"]
        )
        contrasts_surround = p["min_contrast_surround"] * p["contrast_surround_increment"] ** np.arange(
            p["num_contrasts_surround"]
        )

        center_surround = CenterSurround(
            canvas_size=canvas_size,
            center_range=center_range,
            sizes_total=[p["total_size"]],
            sizes_center=sizes_center,
            sizes_surround=sizes_surround,
            contrasts_center=contrasts_center,
            contrasts_surround=contrasts_surround,
            orientations_center=[ori],
            orientations_surround=[ori],
            spatial_frequencies=[sf],
            phases=[phase],
        )

        return center_surround

    def surround_set(self, key, canvas_size, loc, sf, ori, phase):
        p = (self & key).fetch1()
        center_range = [loc[0], loc[0] + 1, loc[1], loc[1] + 1]
        sizes_center = [-0.01]
        sizes_surround = [-0.01] + list(
            p["min_size_surround"] * p["size_surround_increment"] ** np.arange(p["num_sizes_surround"])
        )
        contrasts_center = p["min_contrast_center"] * p["contrast_center_increment"] ** np.arange(
            p["num_contrasts_center"]
        )
        contrasts_surround = p["min_contrast_surround"] * p["contrast_surround_increment"] ** np.arange(
            p["num_contrasts_surround"]
        )

        center_surround = CenterSurround(
            canvas_size=canvas_size,
            center_range=center_range,
            sizes_total=[p["total_size"]],
            sizes_center=sizes_center,
            sizes_surround=sizes_surround,
            contrasts_center=contrasts_center,
            contrasts_surround=contrasts_surround,
            orientations_center=[ori],
            orientations_surround=[ori],
            spatial_frequencies=[sf],
            phases=[phase],
        )

        return center_surround


@schema
class CenterSurroundResponses(dj.Computed):
    definition = """
        -> CenterSurroundParams
        -> OptimalGabor
        ---
        """

    class Unit(dj.Part):
        definition = """
            -> master
            -> OptimalGabor.Unit
            ---
            tuning_curve_center   : blob
            tuning_curve_surround : blob
        """

    def _make_tuples(self, key):
        model = Fit().load_model(key)
        canvas_size = [model.data.px_y, model.data.px_x]
        self.insert1(key)

        for i, key in enumerate((OptimalGabor.Unit() & key).fetch(dj.key)):
            loc, _, sf, _, ori, phase = OptimalGabor.Unit().params(key)
            center_set = CenterSurroundParams().center_set(key, canvas_size, loc, sf, ori, phase)
            surround_set = CenterSurroundParams().surround_set(key, canvas_size, loc, sf, ori, phase)

            feed_dict = {
                model.images: center_set.images()[..., None],
                model.is_training: False,
            }
            pred_center = model.session.run(model.prediction[:, key["unit_id"]], feed_dict=feed_dict)

            feed_dict = {
                model.images: surround_set.images()[..., None],
                model.is_training: False,
            }
            pred_surround = model.session.run(model.prediction[:, key["unit_id"]], feed_dict=feed_dict)

            key["tuning_curve_center"] = pred_center
            key["tuning_curve_surround"] = pred_surround
            self.Unit.insert1(key)

            if not (i % 10):
                print("Unit {:d}".format(i))
                dj.conn().ping()


@schema
class CenterSurroundStats(dj.Computed):
    definition = """
        -> CenterSurroundResponses
        ---
        """

    class Unit(dj.Part):
        definition = """
        -> master
        -> CenterSurroundResponses.Unit
        ---
        gsf_pixel                               : float
        gsf_scale                               : float
        gsf_global_max_pixel                    : float
        surround_diameter_pixel                 : float
        surround_diameter_scale                 : float
        amrf_pixel                              : float
        amrf_scale                              : float
        suppression_index                       : float
        """

    def get_gsf(self, key, canvas_size=(64, 36)):
        """return image of center stimulus at gsf size"""
        p = (CenterSurroundParams() & key).fetch1()
        total_size = p["total_size"]
        loc, _, sf, _, ori, phase = OptimalGabor.Unit().params(key)
        center_range = [loc[0], loc[0] + 1, loc[1], loc[1] + 1]
        gsf = (self & key).fetch1("gsf_scale")

        center_surround = CenterSurround(
            canvas_size=canvas_size,
            center_range=center_range,
            sizes_total=[total_size],
            sizes_center=[gsf],
            sizes_surround=[1],
            contrasts_center=[2],
            contrasts_surround=[2],
            orientations_center=[ori],
            orientations_surround=[ori],
            spatial_frequencies=[sf],
            phases=[phase],
        )
        return center_surround.stimulus_from_idx(0)

    def get_surround(self, key, canvas_size=(64, 36)):
        p = (CenterSurroundParams() & key).fetch1()
        total_size = p["total_size"]
        loc, _, sf, _, ori, phase = OptimalGabor.Unit().params(key)
        center_range = [loc[0], loc[0] + 1, loc[1], loc[1] + 1]
        surround_diameter = (self & key).fetch1("surround_diameter_scale")

        center_surround = CenterSurround(
            canvas_size=canvas_size,
            center_range=center_range,
            sizes_total=[total_size],
            sizes_center=[surround_diameter],
            sizes_surround=[1],
            contrasts_center=[2],
            contrasts_surround=[2],
            orientations_center=[ori],
            orientations_surround=[ori],
            spatial_frequencies=[sf],
            phases=[phase],
        )
        return center_surround.stimulus_from_idx(0)

    def get_amrf(self, key, canvas_size=(64, 36)):
        p = (CenterSurroundParams() & key).fetch1()
        total_size = p["total_size"]
        loc, _, sf, _, ori, phase = OptimalGabor.Unit().params(key)
        center_range = [loc[0], loc[0] + 1, loc[1], loc[1] + 1]
        amrf = (self & key).fetch1("amrf_scale")

        center_surround = CenterSurround(
            canvas_size=canvas_size,
            center_range=center_range,
            sizes_total=[total_size],
            sizes_center=[-0.01],
            sizes_surround=[amrf],
            contrasts_center=[2],
            contrasts_surround=[2],
            orientations_center=[ori],
            orientations_surround=[ori],
            spatial_frequencies=[sf],
            phases=[phase],
        )
        return center_surround.stimulus_from_idx(0)

    def _make_tuples(self, key):
        p = (CenterSurroundParams() & key).fetch1()

        # index 0 corresponds to zero diameter; here negative value is used for stimulus creation
        sizes_center = [-0.01] + list(
            p["min_size_center"] * p["size_center_increment"] ** np.arange(p["num_sizes_center"])
        )
        sizes_surround = [-0.01] + list(
            p["min_size_surround"] * p["size_surround_increment"] ** np.arange(p["num_sizes_surround"])
        )
        total_size = p["total_size"]
        self.insert1(key)

        for i, key in enumerate((CenterSurroundResponses().Unit() & key).fetch(dj.key)):
            tuning_curve_center = (CenterSurroundResponses().Unit() & key).fetch1("tuning_curve_center")

            empty_response = tuning_curve_center[0]
            center_max_global = np.max(tuning_curve_center)
            center_max_global_i = np.argmax(tuning_curve_center)
            key["gsf_global_max_pixel"] = np.max([0, sizes_center[center_max_global_i]]) * total_size
            for center_max_i, center_max in enumerate(tuning_curve_center):
                if center_max >= 0.95 * center_max_global:
                    break

            key["gsf_pixel"] = np.max([0, sizes_center[center_max_i]]) * total_size
            key["gsf_scale"] = np.max([0, sizes_center[center_max_i]])

            center_asymptote = tuning_curve_center[-1]
            surround_mask = tuning_curve_center < 1.05 * center_asymptote
            for surround_i in np.sort(np.where(surround_mask)[0]):
                if surround_i > center_max_i:
                    break

            key["surround_diameter_pixel"] = sizes_center[surround_i] * total_size
            key["surround_diameter_scale"] = sizes_center[surround_i]
            key["suppression_index"] = (center_max - center_asymptote) / center_max

            tuning_curve_surround = (CenterSurroundResponses().Unit() & key).fetch1("tuning_curve_surround")
            for amrf_i, surround_r in enumerate(tuning_curve_surround):
                if surround_r < 0.05 * (center_max - empty_response) + empty_response:
                    break

            key["amrf_pixel"] = sizes_surround[amrf_i] * total_size
            key["amrf_scale"] = sizes_surround[amrf_i]

            self.Unit.insert1(key)
            if not (i % 10):
                print("Unit {:d}".format(i))
                dj.conn().ping()
