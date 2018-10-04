"""

"""
import gzip
import pickle

import astropy.units as u
import numpy as np
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator
from ctapipe.coordinates import CameraFrame, NominalFrame, GroundFrame, \
    TiltedGroundFrame, HorizonFrame
from ctapipe.io.hessio import hessio_event_source
from ctapipe.reco import ImPACTReconstructor
from sklearn.neural_network import MLPRegressor
from scipy.interpolate import interp1d
from tqdm import tqdm


def find_nearest_bin(array, value):
    """
    Find nearest value in an array

    :param array: ndarray
        Array to search
    :param value: float
        Search value
    :return: float
        Nearest bin value
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class TemplateFitter:

    def __init__(self, eff_fl=1, bounds=((-5, 1), (-1.5, 1.5)), bins=(600, 300),
                 min_fit_pixels=3000, xmax_bins=np.linspace(-150, 250, 17),
                 verbose=False):
        """

        :param eff_fl: float
            Effective focal length scaling of the telescope (to account for distortions)
        :param bounds: tuple
            Boundaries of resultant templates
        :param bins: tuple
            Number of bins in x and y dimensions of template
        :param min_fit_pixels: int
            Minimum number of pixels required in event to perform fit
        """

        self.verbose = verbose
        self.xmax_bins = xmax_bins
        self.eff_fl = eff_fl

        self.bounds = bounds
        self.bins = bins
        self.min_fit_pixels = min_fit_pixels

        self.r1 = HESSIOR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None, None)

    def read_templates(self, filename, max_events=1e9):
        """
        This is a pretty standard ctapipe event loop that calibrates events, rotates
        them into a common frame and then stores the pixel values in a list

        :param filename: str
            Location of input
        :param max_events: int
            Maximum number of events to include in the loop
        :return: tuple
            Return 3 lists of amplitude and rotated x,y positions of all pixels in all
            events
        """

        # Create dictionaries to contain our output
        templates = dict() # Pixel amplitude
        templates_xb = dict() # Rotated X position
        templates_yb = dict() # Rotated Y positions

        source = hessio_event_source(filename.strip())

        grd_tel = None
        num = 0 #Event counter

        for event in tqdm(source):

            point = HorizonFrame(alt=event.mcheader.run_array_direction[1],
                                 az=event.mcheader.run_array_direction[0])

            mc = event.mc
            # Create coordinate objects for source position
            src = HorizonFrame(alt=mc.alt.value * u.rad, az=mc.az.value * u.rad)
            # And transform into nominal system (where we store our templates)
            source = src.transform_to(NominalFrame(array_direction=point))

            # Perform calibration of images
            self.r1.calibrate(event)
            self.dl0.reduce(event)
            self.calibrator.calibrate(event)

            # Store simulated event energy
            energy = mc.energy

            # Store ground position of all telescopes
            # We only want to do this once, but has to be done in event loop
            if grd_tel is None:
                grd_tel = GroundFrame(x=event.inst.subarray.pos_x,
                                      y=event.inst.subarray.pos_y,
                                      z=event.inst.subarray.pos_z)

                # Convert to tilted system
                tilt_tel = grd_tel.transform_to(
                    TiltedGroundFrame(pointing_direction=point))

            # Calculate core position in tilted system
            grd_core_true = GroundFrame(x=np.asarray(mc.core_x) * u.m,
                                        y=np.asarray(mc.core_y) * u.m,
                                        z=np.asarray(0) * u.m)
            tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(
                pointing_direction=point))

            # Loop over triggered telescopes
            for tel_id in event.dl0.tels_with_data:

                #  Get pixel signal (make gain selection if we have 2 channels)
                pmt_signal = event.dl1.tel[tel_id].image[0]
                if len(event.dl1.tel[tel_id].image) > 1:
                    pmt_signal_lg = event.dl1.tel[tel_id].image[1]
                    pmt_signal[pmt_signal_lg > 200] = pmt_signal_lg[pmt_signal_lg > 200]

                # Get pixel coordinates and convert to the nominal system
                geom = event.inst.subarray.tel[tel_id].camera
                fl = event.inst.subarray.tel[tel_id].optics.equivalent_focal_length * \
                     self.eff_fl

                camera_coord = CameraFrame(x=geom.pix_x, y=geom.pix_y, focal_length=fl)
                nom_coord = camera_coord.transform_to(
                    NominalFrame(array_direction=point, pointing_direction=point))

                x = nom_coord.x.to(u.deg)
                y = nom_coord.y.to(u.deg)

                # We only want to keep pixels that fall within the bounds of our
                # final template
                mask = np.logical_and(x > self.bounds[0][0] * u.deg,
                                      x < self.bounds[0][1] * u.deg)
                mask = np.logical_and(mask, y < self.bounds[1][1] * u.deg)
                mask = np.logical_and(mask, y > self.bounds[1][0] * u.deg)

                # Make sure everythin is 32 bit
                x = x[mask].astype(np.float32)
                y = y[mask].astype(np.float32)
                image = pmt_signal[mask].astype(np.float32)

                # Calculate expected rotation angle of the image
                phi = np.arctan2((tilt_tel.y[tel_id - 1] - tilt_core_true.y),
                                 (tilt_tel.x[tel_id - 1] - tilt_core_true.x)) + \
                      180 * u.deg

                # And the impact distance of the shower
                impact = np.sqrt(np.power(tilt_tel.x[tel_id - 1] - tilt_core_true.x, 2) +
                                 np.power(tilt_tel.y[tel_id - 1] - tilt_core_true.y, 2)).\
                    to(u.m).value

                # now rotate and translate our images such that they lie on top of one
                # another
                pix_x_rot, pix_y_rot = \
                    ImPACTReconstructor.rotate_translate(x, y, source.x, source.y, phi)
                pix_x_rot *= -1

                # Store simulated Xmax
                mc_xmax = event.mc.x_max.value / np.cos(20 * u.deg)

                # Calc difference from expected Xmax (for gammas)
                exp_xmax = 300 + 93 * np.log10(energy.value)
                x_diff = mc_xmax - exp_xmax
                x_diff_bin = find_nearest_bin(self.xmax_bins, x_diff)

                zen = 90 - point.alt.to(u.deg).value
                az = point.az.to(u.deg).value

                # Now fill up our output with the X, Y and amplitude of our pixels
                if (zen, az, energy.value, int(impact), x_diff_bin) in templates.keys():
                    # Extend the list if an entry already exists
                    templates[(zen, az, energy.value, int(impact), x_diff_bin)].extend(
                        image)
                    templates_xb[(zen, az, energy.value, int(impact), x_diff_bin)].extend(
                        pix_x_rot.value)
                    templates_yb[(zen, az, energy.value, int(impact), x_diff_bin)].extend(
                        pix_y_rot.value)
                else:
                    templates[(zen, az, energy.value, int(impact), x_diff_bin)] = \
                        image.tolist()
                    templates_xb[(zen, az, energy.value, int(impact), x_diff_bin)] = \
                        pix_x_rot.value.tolist()
                    templates_yb[(zen, az, energy.value, int(impact), x_diff_bin)] = \
                        pix_y_rot.value.tolist()

            if num > max_events:
                return templates, templates_xb, templates_yb

            num += 1

        return templates, templates_xb, templates_yb

    def fit_templates(self, amplitude, x_pos, y_pos):
        """
        Perform MLP fit over a dictionary of pixel lists

        :param amplitude: dict
            Dictionary of pixel amplitudes for each template
        :param x_pos: dict
            Dictionary of x position for each template
        :param y_pos: dict
            Dictionary of y position for each template
        :return: dict
            Dictionary of image templates
        """

        # Create output dictionary
        templates_out = dict()

        # Create grid over which to evaluate our fit
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.bins[0])
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.bins[1])
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack((xx.ravel(), yy.ravel()))

        # Loop over all templates
        for key in amplitude:
            amp = np.array(amplitude[key])

            # Skip if we do not have enough image pixels
            if len(amp) < self.min_fit_pixels:
                continue

            # Stack up pixel positions
            pixel_pos = np.vstack([np.array(x_pos[key]), np.array(y_pos[key])])

            # Fit with MLP
            model = self.perform_fit(amp, pixel_pos)

            # Evaluate MLP fit over our grid
            nn_out = model.predict(grid.T)
            nn_out = nn_out.reshape((self.bins[1], self.bins[0]))
            nn_out[np.isinf(nn_out)] = 0

            templates_out[(key[0], key[1], key[2])] = nn_out

        return templates_out

    @staticmethod
    def perform_fit(amp, pixel_pos):
        """
        Fit MLP model to individual template pixels

        :param amp: ndarray
            Pixel amplitudes
        :param pixel_pos: ndarray
            Pixel XY coordinate format (N, 2)
        :return: MLP
            Fitted MLP model
        """

        # We need a large number of layers to get this fit right

        nodes = (32, 32, 32, 32, 32, 32, 32, 32, 32)
        model = MLPRegressor(hidden_layer_sizes=nodes, activation="relu",
                             max_iter=10000, tol=0,
                             early_stopping=True, verbose=True)

        model.fit(pixel_pos.T, amp)

        return model

    def extend_xmax_range(self, templates):
        """
        Copy templates in empty xmax bins, helps to prevent problems from reaching the
        edge of the interpolation space.

        :param templates: dict
            Image templates
        :return: dict
            Extended image templates
        """

        # Create dictionary for our new templates
        extended_templates = dict()

        # Loop over image templates
        for key in templates:
            min_key_bin = list()
            key_copy = 0

            # For each entry loop forward over possible xmax entries to check if they
            # exist
            for xb in self.xmax_bins:
                key_test = (key[0], key[1], xb)
                # keep looping until we have found the largest xmax value
                if (key_test not in extended_templates.keys()) and \
                        (key_test not in templates.keys()):
                    min_key_bin.append(key_test)
                else:
                    key_copy = key_test
                    break
            # Then copy in the highest xmax valid template into these etries
            for k in min_key_bin:
                if key_copy != 0:
                    extended_templates[k] = templates[key_copy]

            min_key_bin = list()
            key_copy = 0
            # Now we just do the same in reverse
            for xb in reversed(self.xmax_bins):
                key_test = (key[0], key[1], xb)
                if (key_test not in extended_templates.keys()) and \
                        (key_test not in templates.keys()):
                    min_key_bin.append(key_test)
                else:
                    key_copy = key_test
                    break

            for k in min_key_bin:
                if key_copy != 0:
                    extended_templates[k] = templates[key_copy]

        # Copy new template entries into the original
        templates.update(extended_templates)
        return templates

    def extend_distance_range(self, templates, additional_bins=4):
        """
        Copy templates in empty xmax bins, helps to prevent problems from reaching the
        edge of the interpolation space.

        :param templates: dict
            Image templates
        :return: dict
            Extended image templates
        """
        keys = np.array(list(templates.keys()))
        distances = np.unique(keys.T[1])
        energies = np.unique(keys.T[0])

        extended_templates = dict()

        for en in energies:
            for xmax in self.xmax_bins:
                i = 0
                distance_list = list()
                for dist in distances[0:]:
                    key = (en, dist, xmax)
                    if key not in templates.keys():
                        break
                    else:
                        distance_list.append(templates[key])
                        i += 1

                num_dists = len(distance_list)
                if num_dists > 1 and num_dists < len(distances):
                    distance_list = np.array(distance_list)

                    diff = len(distances) - len(distance_list)

                    if diff > additional_bins:
                        diff = additional_bins
                    for j in range(i, i + diff):
                        interp = interp1d(distances[0:i], distance_list, axis=0,
                                          bounds_error=False,
                                          fill_value="extrapolate", kind="linear")

                        int_val = interp(distances[j])
                        int_val[int_val < 0] = 0
                        key = (en, distances[j], xmax)

                        extended_templates[key] = int_val

        templates.update(extended_templates)

        return templates

    def extend_template_coverage(self, templates):

        templates = self.extend_xmax_range(templates)
        templates = self.extend_distance_range(templates, 4)

        return templates

    def generate_templates(self, file_list, output_file, extend_range=True,
                           max_events=1e9):
        """

        :param file_list: list
            List of sim_telarray input files
        :param output_file: string
            Output file name
        :param extend_range: bool
            Extend range of the templates beyond simulations
        :param max_events: int
            Maximum number of events to process
        :return: dict
            Dictionary of image templates

        """
        templates = dict()

        for filename in file_list:
            pix_lists = self.read_templates(filename, max_events)
            file_templates = self.fit_templates(*pix_lists)

            templates.update(file_templates)

            pix_lists = None
            file_templates = None

        # Extend coverage of the templates by extrapolation if requested
        if extend_range:
            templates = self.extend_template_coverage(templates)

        file_handler = gzip.open(output_file, "wb")
        pickle.dump(templates, file_handler)
        file_handler.close()

        return templates
