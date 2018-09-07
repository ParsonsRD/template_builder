"""

"""
import astropy.units as u
import numpy as np
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator
from ctapipe.coordinates import CameraFrame, NominalFrame, GroundFrame, \
    TiltedGroundFrame, HorizonFrame

from ctapipe.io.hessio import hessio_event_source
from ctapipe.reco import ImPACTReconstructor
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor


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
                 min_fit_pixels=3000):
        """

        :param eff_fl:
        :param bounds:
        :param bins:
        :param min_fit_pixels:
        """
        self.xmax_bins = np.linspace(-150, 250, 17)
        self.eff_fl = eff_fl

        self.bounds = bounds
        self.bins = bins
        self.min_fit_pixels = min_fit_pixels

        self.r1 = HESSIOR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None,
                                         None)

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

        point = HorizonFrame(alt=70 * u.deg, az=180 * u.deg)

        for event in tqdm(source):
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
                nom_coord = camera_coord.transform_to(NominalFrame(array_direction=point,
                                                                   pointing_direction=point))

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

                # Now fill up our output with the X, Y and amplitude of our pixels
                if (energy.value, int(impact.value), x_diff_bin) in templates.keys():
                    # Extend the list if an entry already exists
                    templates[(energy.value, int(impact.value), x_diff_bin)].extend(image)
                    templates_xb[(energy.value, int(impact.value), x_diff_bin)].extend(
                        pix_x_rot.value)
                    templates_yb[(energy.value, int(impact.value), x_diff_bin)].extend(
                        pix_y_rot.value)
                else:
                    templates[(energy.value, int(impact.value), x_diff_bin)] = \
                        image.tolist()
                    templates_xb[(energy.value, int(impact.value), x_diff_bin)] = \
                        pix_x_rot.value.tolist()
                    templates_yb[(energy.value, int(impact.value), x_diff_bin)] = \
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

        nodes = (64, 64, 64, 64, 64, 64, 64, 64, 64)
        model = MLPRegressor(hidden_layer_sizes=nodes, activation="relu",
                             max_iter=10000, tol=0,
                             early_stopping=True)

        model.fit(pixel_pos.T, amp)

        return model
