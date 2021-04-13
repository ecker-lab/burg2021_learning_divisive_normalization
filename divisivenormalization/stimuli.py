import numpy as np
from numpy import pi


class GaborSet:
    def __init__(
        self,
        canvas_size,  # width x height
        center_range,  # [x_start, x_end, y_start, y_end]
        sizes,  # +/- 2 SD of Gaussian envelope
        spatial_frequencies,  # cycles / 4 SD of envelope (i.e. depends on size)
        contrasts,
        orientations,
        phases,
        relative_sf=True,  # scale spatial frequency by size
    ):
        self.canvas_size = canvas_size
        cr = center_range
        self.locations = np.array([[x, y] for x in range(cr[0], cr[1]) for y in range(cr[2], cr[3])])
        self.sizes = sizes
        self.spatial_frequencies = spatial_frequencies
        self.contrasts = contrasts
        if type(orientations) is not list:
            self.orientations = np.arange(orientations) * pi / orientations
        else:
            self.orientations = orientations
        if type(phases) is not list:
            self.phases = np.arange(phases) * (2 * pi) / phases
        else:
            self.phases = phases
        self.num_params = [
            self.locations.shape[0],
            len(sizes),
            len(spatial_frequencies),
            len(contrasts),
            len(self.orientations),
            len(self.phases),
        ]
        self.relative_sf = relative_sf

    def params_from_idx(self, idx):
        c = np.unravel_index(idx, self.num_params)
        location = self.locations[c[0]]
        size = self.sizes[c[1]]
        spatial_frequency = self.spatial_frequencies[c[2]]
        if self.relative_sf:
            spatial_frequency /= size
        contrast = self.contrasts[c[3]]
        orientation = self.orientations[c[4]]
        phase = self.phases[c[5]]
        return location, size, spatial_frequency, contrast, orientation, phase

    def params_dict_from_idx(self, idx):
        (
            location,
            size,
            spatial_frequency,
            contrast,
            orientation,
            phase,
        ) = self.params_from_idx(idx)
        return {
            "location": location,
            "size": size,
            "spatial_frequency": spatial_frequency,
            "contrast": contrast,
            "orientation": orientation,
            "phase": phase,
        }

    def gabor_from_idx(self, idx):
        return self.gabor(*self.params_from_idx(idx))

    def gabor(self, location, size, spatial_frequency, contrast, orientation, phase):
        x, y = np.meshgrid(
            np.arange(self.canvas_size[0]) - location[0],
            np.arange(self.canvas_size[1]) - location[1],
        )
        R = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )
        coords = np.stack([x.flatten(), y.flatten()])
        x, y = R.dot(coords).reshape((2,) + x.shape)
        envelope = contrast * np.exp(-(x ** 2 + y ** 2) / (2 * (size / 4) ** 2))

        grating = np.cos(spatial_frequency * x * (2 * pi) + phase)
        return envelope * grating

    def image_batches(self, batch_size):
        num_stims = np.prod(self.num_params)
        for batch_start in np.arange(0, num_stims, batch_size):
            batch_end = np.minimum(batch_start + batch_size, num_stims)
            images = [self.gabor_from_idx(i) for i in range(batch_start, batch_end)]
            yield np.array(images)

    def images(self):
        num_stims = np.prod(self.num_params)
        return np.array([self.gabor_from_idx(i) for i in range(num_stims)])


class StimuliSet:
    def __init__(self):
        pass

    def params(self):
        raise NotImplementedError

    def num_params(self):
        return [len(p[0]) for p in self.params()]

    def stimulus(self, *args, **kwargs):
        raise NotImplementedError

    def params_from_idx(self, idx):
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]
        return params

    def params_dict_from_idx(self, idx):
        params = self.params_from_idx(idx)
        return {p[1]: params[i] for i, p in enumerate(self.params())}

    def stimulus_from_idx(self, idx):
        return self.stimulus(**self.params_dict_from_idx(idx))

    def image_batches(self, batch_size):
        num_stims = np.prod(self.num_params())
        for batch_start in np.arange(0, num_stims, batch_size):
            batch_end = np.minimum(batch_start + batch_size, num_stims)
            images = [self.stimulus_from_idx(i) for i in range(batch_start, batch_end)]
            yield np.array(images)

    def images(self):
        num_stims = np.prod(self.num_params())
        return np.array([self.stimulus_from_idx(i) for i in range(num_stims)])


class CenterSurround(StimuliSet):
    def __init__(
        self,
        canvas_size,
        center_range,
        sizes_total,
        sizes_center,  # relative to total size, i.e. between 0 and 1
        sizes_surround,  # relative to total size, i.e. between 0 and 1
        contrasts_center,
        contrasts_surround,
        orientations_center,
        orientations_surround,
        spatial_frequencies,
        phases,
    ):

        self.canvas_size = canvas_size
        self.cr = center_range
        self.locations = np.array(
            [[x, y] for x in range(self.cr[0], self.cr[1]) for y in range(self.cr[2], self.cr[3])]
        )
        self.sizes_total = sizes_total
        self.sizes_center = sizes_center
        self.sizes_surround = sizes_surround
        self.contrasts_center = contrasts_center
        self.contrasts_surround = contrasts_surround

        if type(orientations_center) is not list:
            self.orientations_center = np.arange(orientations_center) * pi / orientations_center
        else:
            self.orientations_center = orientations_center

        if type(orientations_surround) is not list:
            self.orientations_surround = np.arange(orientations_surround) * pi / orientations_surround
        else:
            self.orientations_surround = orientations_surround

        if type(phases) is not list:
            self.phases = np.arange(phases) * (2 * pi) / phases
        else:
            self.phases = phases

        self.spatial_frequencies = spatial_frequencies

    def params(self):
        return [
            (self.locations, "location"),
            (self.sizes_total, "size_total"),
            (self.sizes_center, "size_center"),
            (self.sizes_surround, "size_surround"),
            (self.contrasts_center, "contrast_center"),
            (self.contrasts_surround, "contrast_surround"),
            (self.orientations_center, "orientation_center"),
            (self.orientations_surround, "orientation_surround"),
            (self.spatial_frequencies, "spatial_frequency"),
            (self.phases, "phase"),
        ]

    def stimulus(
        self,
        location,
        size_total,
        size_center,
        size_surround,
        contrast_center,
        contrast_surround,
        orientation_center,
        orientation_surround,
        spatial_frequency,
        phase,
    ):

        x, y = np.meshgrid(
            np.arange(self.canvas_size[0]) - location[0],
            np.arange(self.canvas_size[1]) - location[1],
        )

        R_center = np.array(
            [
                [np.cos(orientation_center), -np.sin(orientation_center)],
                [np.sin(orientation_center), np.cos(orientation_center)],
            ]
        )

        R_surround = np.array(
            [
                [np.cos(orientation_surround), -np.sin(orientation_surround)],
                [np.sin(orientation_surround), np.cos(orientation_surround)],
            ]
        )

        coords = np.stack([x.flatten(), y.flatten()])
        x_center, y_center = R_center.dot(coords).reshape((2,) + x.shape)
        x_surround, y_surround = R_surround.dot(coords).reshape((2,) + x.shape)

        norm_xy_center = np.sqrt(x_center ** 2 + y_center ** 2)
        norm_xy_surround = np.sqrt(x_surround ** 2 + y_surround ** 2)

        envelope_center = contrast_center * (norm_xy_center <= size_center * size_total)
        envelope_surround = (
            contrast_surround * (norm_xy_surround > size_surround * size_total) * (norm_xy_surround <= size_total)
        )

        grating_center = np.cos(spatial_frequency * x_center * (2 * pi) + phase)
        grating_surround = np.cos(spatial_frequency * x_surround * (2 * pi) + phase)
        return envelope_center * grating_center + envelope_surround * grating_surround
