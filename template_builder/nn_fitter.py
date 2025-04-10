"""
Generate average image templates from collection of images by fitting a MLP
to the pixel amplitudes.
"""
import gzip
import pickle
from ctapipe.core import Component
from ctapipe.core.traits import List, Int

import astropy.units as u
import numpy as np
import scipy

from scipy.interpolate import LinearNDInterpolator
from keras.models import Sequential
from keras.layers import Dense
import keras

from template_builder.utilities import *
from template_builder.extend_templates import *
from tqdm import tqdm


class NNFitter(Component):

    """
    Class to generate inage templates from collections of images 
    for a given set of shower parameters by fitting  a MLP function.

    """

    bounds = List(
        default_value=[[-5, 1], [-1.5, 1.5]],
        help="X and Y boundaries of template",
    ).tag(config=True)

    bins = List(
        default_value=[601, 301],
        help="X and Y bin numbers of template",
    ).tag(config=True)

    min_fit_pixels = Int(
        default_value=1000,
        help="Minimum number of pixels required to fit image",
    ).tag(config=True)

    def __init__(self, config=None, parent=None):
        super().__init__(config=config, parent=parent)
        return

    def fit_templates(self, amplitude, x_pos, y_pos):
        """
        Perform MLP fit for all shower parameter grid points

        :param amplitude: dict
            Dictionary of pixel amplitudes for each grid point
        :param x_pos: dict
            Dictionary of x position for each grid point
        :param y_pos: dict
            Dictionary of y position for each grid point
        :return: dict
            Dictionary of image templates
        """

        # Create output dictionary
        templates_out = dict()

        # Loop over all templates
        for key in tqdm(list(amplitude.keys())):
            amp = np.array(amplitude[key])
            # Skip if we do not have enough image pixels
            if len(amp) < self.min_fit_pixels:
                continue

            y = y_pos[key]
            x = x_pos[key]

            # Stack up pixel positions
            pixel_pos = np.vstack([x, y])

            # Fit with MLP
            template_output = self.perform_fit(amp, pixel_pos)
            templates_out[key] = template_output.astype(np.float32)



        return templates_out

    def perform_fit(self, amp, pixel_pos, nodes=(64, 64, 64, 64, 64, 64, 64, 64, 64)):
        """
        Fit MLP model to collection of images for singel shower parameter grid point.

        :param amp: ndarray
            Pixel amplitudes
        :param pixel_pos: ndarray
            Pixel XY coordinate format (N, 2)
        :param nodes: tuple
            Node layout of MLP
        :return: MLP
            Fitted MLP model
        """
        pixel_pos = pixel_pos.T

        # Create grid over which to evaluate our fit
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.bins[0])
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.bins[1])
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack((xx.ravel(), yy.ravel()))

        # We expect our images to be symmetric around x=0, so we can duplicate and copy
        # across the values
        pixel_pos_neg = np.array([pixel_pos.T[0], -1 * np.abs(pixel_pos.T[1])]).T
        pixel_pos = np.concatenate((pixel_pos, pixel_pos_neg))
        amp = np.concatenate((amp, amp))
        x, y = pixel_pos.T

        # Fit image pixels using a multi layer perceptron
        model = Sequential()
        model.add(Dense(nodes[0], activation="relu", input_shape=(2,)))
        # We make a very deep network
        for n in nodes[1:]:
            model.add(Dense(n, activation="relu"))

        model.add(Dense(1, activation="linear"))

        def poisson_loss(y_true, y_pred):
            return tensor_poisson_likelihood(y_true, y_pred, 0.5, 1.0)

        # First we have a go at fitting our model with a mean squared loss
        # this gets us most of the way to the answer and is more stable
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0, patience=20, verbose=0, mode="auto"
        )

        model.fit(
            pixel_pos,
            amp,
            epochs=10000,
            batch_size=50000,
            callbacks=[stopping],
            validation_split=0.1,
            verbose=0,
        )
        weights = model.get_weights()

        # Then copy over the weights to a new model but with our poisson loss
        # this should get the final normalisation right
        model.compile(loss=poisson_loss, optimizer="adam", metrics=["accuracy"])
        model.set_weights(weights)

        model.fit(
            pixel_pos,
            amp,
            epochs=10000,
            batch_size=50000,
            callbacks=[stopping],
            validation_split=0.1,
            verbose=0,
        )
        model_pred = model.predict(grid.T, verbose=0)

        # Set everything outside the range of our points to zero
        # This is a bit of a hacky way of doing this, but is fast and works
        lin_range = LinearNDInterpolator(pixel_pos, amp, fill_value=0)
        lin_nan = lin_range(grid.T) == 0
        model_pred[lin_nan] = 0

        return model_pred.reshape((self.bins[1], self.bins[0])).T

    def generate_image_templates(self, x, y, amplitude, output_file="./Template"):
        """
        Generate and save average image templates

        :param x_pos: dict
            Dictionary of x position for each grid point
        :param y_pos: dict
            Dictionary of y position for each grid point
        :param amplitude: dict
            Dictionary of pixel amplitudes for each grid point
        :param output_file: string
            Path to output file and output file base name
        """
        templates = self.fit_templates(x, y, amplitude)
        file_handler = gzip.open(output_file + ".template.gz", "wb")
        pickle.dump(templates, file_handler)
        file_handler.close()

        correction_factor = self.calculate_correction_factors(
            x, y, amplitude, templates
        )
        for t in templates:
            templates[t] = templates[t] * correction_factor
        file_handler = gzip.open(output_file + "_corrected.template.gz", "wb")
        pickle.dump(templates, file_handler)
        file_handler.close()

        return True

    def calculate_correction_factors(self, pixel_x, pixel_y, amplitude, templates):
        """Funtion for performing a simple correction to the template amplitudes to
        match the training images. Only needed if significant fit biases are seen.

        :param file_list: list
            List of sim_telarray input files
        :param template_file: string
            File name of the template file
        :param max_events: int
            Maximum number of events to process

        :return: int
            Correction factor
        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy.optimize import minimize

        # Open up the template file
        keys = amplitude.keys()

        # Define our template binning
        x_bins = np.linspace(self.bounds[0][0], self.bounds[0][1], self.bins[0])
        y_bins = np.linspace(self.bounds[1][0], self.bounds[1][1], self.bins[1])

        amp_vals = None
        pred_vals = None

        # Loop over all templates
        for key in keys:
            try:
                # Get the template matching our data
                template = templates[key]
                # And the event amplitudes and locations
                amp, x, y = amplitude[key], pixel_x[key], pixel_y[key]
                amp = np.array(amp)

                # Create interpolator for our template and predict amplitude
                interpolator = RegularGridInterpolator(
                    (x_bins, y_bins), template, bounds_error=False, fill_value=0
                )
                prediction = interpolator((x, y))
                prediction[prediction < 1e-6] = 1e-6

                # Store the amplitude and prediction
                if amp_vals is None:
                    amp_vals = amp
                    pred_vals = prediction
                else:
                    amp_vals = np.concatenate((amp_vals, amp))
                    pred_vals = np.concatenate((pred_vals, prediction))

            except KeyError:
                True

        # Define the Poissonian fit function used to fit datas
        def scale_like(scale_factor):
            # Reasonable values of single photoelection width and pedestal are used
            # Might need to change for different detector types
            return np.sum(
                poisson_likelihood_gaussian(
                    amp_vals, pred_vals * scale_factor[0], 0.5, 1
                )
            )

        if amp_vals is None:
            return 1

        # Minimise this function to get the scaling factor
        res = minimize(scale_like, [1.0], method="Nelder-Mead")
        # If our fit fails don't scale
        if res.x[0] is None:
            return 1

        # Otherwise return scale factor
        return res.x[0]
