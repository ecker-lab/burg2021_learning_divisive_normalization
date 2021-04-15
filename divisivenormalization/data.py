"""
Data manager script with helper classes for neural system identification
"""

import pickle
import os
import numpy as np


DATA_PATH = os.path.join("/data/data_binned_responses")
FILE = "cadena_ploscb_data.pkl"


class Dataset:
    @staticmethod
    def load_data(file_name=FILE):
        """
        Load dataset pickle and return the contained dict.
        """
        file_path = os.path.join(DATA_PATH, file_name)
        with open(file_path, "rb") as g:
            return pickle.load(g)

    @staticmethod
    def manage_repeats(data_dict):
        """
        This function takes the sessions with only two repeats and duplicates them
        to match those with four repeats replacing the nans in the dict.
        """
        responses = data_dict["responses"].copy()
        idx_two_reps = data_dict["repetitions"] == 2
        responses[2:, :, idx_two_reps] = responses[:2, :, idx_two_reps]
        data_dict["responses"] = responses
        return data_dict

    @staticmethod
    def preprocess_nans(data_dict):
        """
        This function replaces nan responses by zeros and creates
        an array to identify location of original nans.
        """
        responses = data_dict["responses"].copy()
        is_realresponse = ~np.isnan(responses)
        responses[np.isnan(responses)] = 0
        data_dict["responses"] = responses
        data_dict["is_realresponse"] = is_realresponse
        return data_dict

    @staticmethod
    def add_train_test_types(data_dict, types_train="all", types_test="all"):
        """
        This function adds 'train_types' and 'test_types' fields in the
        data dictionary to restrict the stimulus types for training and testing.
        There are five types of stimulus: 'original', 'conv1', 'conv2', 'conv3', 'conv4'
        Specify 'all' to use all of types for training/testing
        """
        main_types = ["original", "conv1", "conv2", "conv3", "conv4"]

        def _process_types(types):
            if types == "all":
                types = main_types
            else:
                if not (isinstance(types, (list,))):
                    types = [types]
            return types

        data_dict["types_train"] = _process_types(types_train)
        data_dict["types_test"] = _process_types(types_test)

        return data_dict

    @staticmethod
    def get_clean_data(file_name=FILE, *args, **kwargs):
        """
        Wrapper of the functions above to get clean data dictionary for preprocessing
        """
        data_dict = Dataset.load_data(file_name)
        data_dict = Dataset.manage_repeats(data_dict)
        data_dict = Dataset.preprocess_nans(data_dict)
        data_dict = Dataset.add_train_test_types(data_dict, *args, **kwargs)
        return data_dict


class MonkeySubDataset:
    def __init__(self, data, seed=None, train_frac=0.8, subsample=1, crop=0):
        """Prepare data to be processed by model.

        Args:
            data (dict): Data loaded by Dataset.get_clean_data()
            seed (int, optional): Seed to split data into train, validation and test set. Defaults to None.
            train_frac (float, optional): Percentage of dataset to be used as train set. Defaults to 0.8.
            subsample (int, optional): Downsampling factor. Defaults to 1.
            crop (int, optional): Crop stimulus by that value from all sides before downsampling. Defaults to 0.
        """
        if crop == 0:
            images = data["images"][:, 0::subsample, 0::subsample, None]
        else:
            images = data["images"][:, crop:-crop:subsample, crop:-crop:subsample, None]
        responses = data["responses"].astype(np.float32)
        real_resps = data["is_realresponse"].astype(np.float32)
        types_train = data["types_train"]
        types_test = data["types_test"]

        # Dimensions
        num_reps, num_images, num_neurons = responses.shape
        num_train_images = int(num_images * train_frac)

        # Normalize images
        image_ids = data["image_ids"].flatten()
        image_types = data["image_types"].flatten()
        imgs_shift = np.mean(images)
        print("Subtracting mean:", imgs_shift)

        imgs_sd = np.std(images)
        images = (images - imgs_shift) / imgs_sd

        self.num_reps = num_reps

        self.images = np.tile(images[:num_train_images, :, :, :], [num_reps, 1, 1, 1])
        self.images_test = images[num_train_images:, :, :, :]

        self.responses = responses[:, :num_train_images, :].reshape([num_train_images * num_reps, num_neurons])
        self.responses_test = responses[:, num_train_images:, :]

        self.real_resps = real_resps[:, :num_train_images, :].reshape([num_train_images * num_reps, num_neurons])
        self.real_resps_test = real_resps[:, num_train_images:, :]

        self.image_ids = np.tile(image_ids[:num_train_images], [num_reps])
        self.image_ids_test = image_ids[num_train_images:]

        self.types = np.tile(image_types[:num_train_images], [num_reps])
        self.types_test = image_types[num_train_images:]

        # Select indices of image types
        idx_trn = np.array([True if x in types_train else False for x in self.types])
        idx_tst = np.array([True if x in types_test else False for x in self.types_test])

        self.images = self.images[idx_trn]
        self.images_test = self.images_test[idx_tst]

        self.responses = self.responses[idx_trn]
        self.responses_test = self.responses_test[:, idx_tst, :]

        self.real_resps = self.real_resps[idx_trn]
        self.real_resps_test = self.real_resps_test[:, idx_tst, :]

        self.image_ids = self.image_ids[idx_trn]
        self.image_ids_test = self.image_ids_test[idx_tst]

        self.types = self.types[idx_trn]
        self.types_test = self.types_test[idx_tst]

        self.num_neurons = num_neurons
        self.num_images = num_images
        self.num_train_images = int(self.images.shape[0] / num_reps)
        self.px_x = images.shape[1]
        self.px_y = images.shape[2]
        if seed:
            np.random.seed(seed)
        perm = np.random.permutation(self.num_train_images)
        train_idx = np.sort(perm[: round(self.num_train_images * train_frac)])
        self.train_idx = np.hstack([train_idx + i * self.num_train_images for i in range(num_reps)])
        val_idx = np.sort(perm[round(self.num_train_images * train_frac) :])
        self.val_idx = np.hstack([val_idx + i * self.num_train_images for i in range(num_reps)])
        self.num_train_samples = self.train_idx.size
        self.next_epoch()

        self.subject_id = data["subject_id"]
        self.repetitions = data["repetitions"]

    def nanarray(self, real_resps, resps):
        """Inserts nan at every position in resps where corresponding positions in real_resps are 0.
        Hence, leading to an array of resps where non-real responses have value nan.
        """

        return np.where(real_resps, resps, np.nan)

    def minibatch(self, batch_size):
        """Returns a minibatch of training images

        Args:
            batch_size (int): Number of samples in minibatch

        Returns:
            np.array, np.array, np.array: Returns test images, responses and real responses
                (if response of a neuron was not recorded for an image, the according real responses value is 0)
        """

        im = self.images[self.train_idx]
        res = self.responses[self.train_idx]
        isreal = self.real_resps[self.train_idx]
        if self.minibatch_idx + batch_size > len(self.train_perm):
            self.next_epoch()
        idx = self.train_perm[self.minibatch_idx + np.arange(0, batch_size)]
        self.minibatch_idx += batch_size
        return im[idx, :, :], res[idx, :], isreal[idx, :]

    def next_epoch(self):
        self.minibatch_idx = 0
        self.train_perm = np.random.permutation(self.num_train_samples)

    def train(self):
        """Get training set.

        Returns:
            np.array, np.array, np.array: Returns training images, responses and real responses
                (if response of a neuron was not recorded for an image, the according real responses value is 0)
        """

        return (
            self.images[self.train_idx],
            self.responses[self.train_idx],
            self.real_resps[self.train_idx],
        )

    def val(self):
        """Get validation set.

        Returns:
            np.array, np.array, np.array: Returns validation images, responses and real responses
                (if response of a neuron was not recorded for an image, the according real responses value is 0)
        """

        return (
            self.images[self.val_idx],
            self.responses[self.val_idx],
            self.real_resps[self.val_idx],
        )

    def test(self):
        """Get test set.

        Returns:
            np.array, np.array, np.array: Returns test images, responses and real responses
                (if response of a neuron was not recorded for an image, the according real responses value is 0)
        """

        return self.images_test, self.responses_test, self.real_resps_test
