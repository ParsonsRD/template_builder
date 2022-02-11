"""

"""
import gzip
import pickle

import astropy.units as u
import numpy as np
import scipy

from scipy.interpolate import LinearNDInterpolator
from keras.models import Sequential
from keras.layers import Dense
import keras

from template_builder.utilities import *
from template_builder.extend_templates import *

from ctapipe.coordinates import CameraFrame, NominalFrame, GroundFrame, \
    TiltedGroundFrame
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from ctapipe.io import EventSource
from ctapipe.reco import ImPACTReconstructor
from tqdm import tqdm
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import BaselineSubtractedNeighborPeakWindowSum, FixedWindowSum
from ctapipe.calib.camera.gainselection import ThresholdGainSelector
from ctapipe.fitting import lts_linear_regression

class TemplateFitter:

    def __init__(self, eff_fl=1, 
                 bounds=((-5, 1), (-1.5, 1.5)), 
                 bins=(601, 301),
                 min_fit_pixels=3000, 
                 xmax_bins=np.linspace(-150, 200, 15),  
                 offset_bins=np.array([0.0])*u.deg,
                 verbose=False, 
                 rotation_angle=0 * u.deg, 
                 tailcuts=(7, 14), min_amp=30, local_distance_cut=2.*u.deg,
                 gain_threshold=30000):
        """[summary]

        Args:
            eff_fl (int, optional): [description]. Defaults to 1.
            bounds (tuple, optional): [description]. Defaults to ((-5, 1), (-1.5, 1.5)).
            bins (tuple, optional): [description]. Defaults to (601, 301).
            min_fit_pixels (int, optional): [description]. Defaults to 3000.
            xmax_bins ([type], optional): [description]. Defaults to np.linspace(-150, 200, 15).
            maximum_offset ([type], optional): [description]. Defaults to 10*u.deg.
            verbose (bool, optional): [description]. Defaults to False.
            rotation_angle ([type], optional): [description]. Defaults to 0*u.deg.
            tailcuts (tuple, optional): [description]. Defaults to (7, 14).
            min_amp (int, optional): [description]. Defaults to 30.
            local_distance_cut ([type], optional): [description]. Defaults to 2.*u.deg.
            gain_threshold (int, optional): [description]. Defaults to 30000.
        """

        self.verbose = verbose
        self.xmax_bins = xmax_bins
        self.eff_fl = eff_fl

        self.bounds = bounds
        self.bins = bins
        self.min_fit_pixels = min_fit_pixels

        self.rotation_angle = rotation_angle
        self.offset_bins = np.sort(offset_bins)
        self.tailcuts = tailcuts
        self.min_amp = min_amp
        self.local_distance_cut = local_distance_cut

        self.time_slope = dict()  # Pixel amplitude
        self.templates = dict()  # Pixel amplitude

        self.templates_xb = dict()  # Rotated X position
        self.templates_yb = dict()  # Rotated Y positions

        self.count = dict() # Count of events in a given template
        self.count_total = 0 # Total number of events 

        # Switch over point between the two gain channels
        self.gain_threshold = gain_threshold


    def read_templates(self, filename, max_events=1000000):
        """
        This is a pretty standard ctapipe event loop that calibrates events, rotates
        them into a common frame and then stores the pixel values in a list

        :param filename: str
            Location of input
        :param max_events: int
            Maximum number of events to include in the loop
        :param fill_correction: bool
            Fill correction factor table
        :return: tuple
            Return 3 lists of amplitude and rotated x,y positions of all pixels in all
            events
        """

        # Create dictionaries to contain our output
        if max_events > 0:
            print("Warning if limiting event numbers the zero fraction may no longer be correct")
        else:
            max_events = 1e10

        # Create a dummy time for our AltAz objects
        dummy_time = Time('2010-01-01T00:00:00', format='isot', scale='utc')

        if self.verbose:
            print("Reading", filename.strip())
        
        source = EventSource(filename, max_events=max_events, gain_selector_type='ThresholdGainSelector')
        source.gain_selector.threshold = self.gain_threshold # Set our threshodl for gain selection

        window = BaselineSubtractedNeighborPeakWindowSum(source.subarray,
                                                         window_width=7, window_shift=3, lwt=0)
        # This value is currently set for HESS, need to make this more flexible in future
        calib = CameraCalibrator(subarray=source.subarray, image_extractor=window)

        self.count_total += source.simulation_config.num_showers
        grd_tel = None
        num = 0  # Event counter
        scaling_filled = False

        for event in source:
            calib(event)
            alt = event.pointing.array_altitude
            if alt > 90 * u.deg:
                alt = 90*u.deg
            point = SkyCoord(alt=alt, az=event.pointing.array_azimuth,
                            frame=AltAz(obstime=dummy_time))

            if not scaling_filled:
                xmax_scale = create_xmax_scaling(self.xmax_bins, self.offset_bins, point, filename)
                scaling_filled = True

            # Create coordinate objects for source position
            src = SkyCoord(alt=event.simulation.shower.alt.value * u.rad, 
                           az=event.simulation.shower.az.value * u.rad,
                           frame=AltAz(obstime=dummy_time))

            alt_evt = event.simulation.shower.alt
            if alt_evt > 90 * u.deg:
                alt_evt = 90*u.deg

            #print("here1", point.separation(src),  self.maximum_offset)
            #if point.separation(src) > self.maximum_offset:
            #    continue
            offset_bin = find_nearest_bin(self.offset_bins, point.separation(src)).value

            zen = 90 - event.simulation.shower.alt.to(u.deg).value
            # Store simulated Xmax
            mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

            # And transform into nominal system (where we store our templates)
            source_direction = src.transform_to(NominalFrame(origin=point))

            # Store simulated event energy
            energy = event.simulation.shower.energy

            # Store ground position of all telescopes
            # We only want to do this once, but has to be done in event loop
            if grd_tel is None:
                grd_tel = source.subarray.tel_coords


                # Convert to tilted system
                tilt_tel = grd_tel.transform_to(
                    TiltedGroundFrame(pointing_direction=point))

            # Calculate core position in tilted system
            grd_core_true = SkyCoord(x=np.asarray(event.simulation.shower.core_x) * u.m,
                                        y=np.asarray(event.simulation.shower.core_y) * u.m,
                                        z=np.asarray(0) * u.m, frame=GroundFrame())

            tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(
                pointing_direction=point))

            # Loop over triggered telescopes
            for tel_id, dl1 in event.dl1.tel.items():
                #  Get pixel signal

                pmt_signal = dl1.image

                # Get pixel coordinates and convert to the nominal system
                geom = source.subarray.tel[tel_id].camera.geometry
                fl = source.subarray.tel[tel_id].optics.equivalent_focal_length * \
                        self.eff_fl

                camera_coord = SkyCoord(x=geom.pix_x, y=geom.pix_y,
                                        frame=CameraFrame(focal_length=fl,
                                                            telescope_pointing=point))

                nom_coord = camera_coord.transform_to(
                    NominalFrame(origin=point))

                x = nom_coord.fov_lon.to(u.deg)
                y = nom_coord.fov_lat.to(u.deg)

                # Calculate expected rotation angle of the image
                phi = np.arctan2((tilt_tel.y[tel_id - 1] - tilt_core_true.y),
                                    (tilt_tel.x[tel_id - 1] - tilt_core_true.x)) + \
                        90 * u.deg
                phi += self.rotation_angle
                # And the impact distance of the shower
                impact = np.sqrt(np.power(tilt_tel.x[tel_id - 1] - tilt_core_true.x, 2) +
                                    np.power(tilt_tel.y[tel_id - 1] - tilt_core_true.y, 2)). \
                    to(u.m).value
                
                mask510 = tailcuts_clean(geom, pmt_signal,
                                         picture_thresh=self.tailcuts[0],
                                         boundary_thresh=self.tailcuts[1],
                                         min_number_picture_neighbors=1)

                amp_sum = np.sum(pmt_signal[mask510])
                x_cent = np.sum(pmt_signal[mask510] * x[mask510]) / amp_sum
                y_cent = np.sum(pmt_signal[mask510] * y[mask510]) / amp_sum
                
                # now rotate and translate our images such that they lie on top of one
                # another
                x, y = \
                    ImPACTReconstructor.rotate_translate(x, y,
                                                            source_direction.fov_lon,
                                                            source_direction.fov_lat,
                                                            phi)
                x *= -1 # Reverse x axis to fit HESS convention

                mask = mask510
                for i in range(4):
                    mask = dilate(geom, mask)

                # Make our preselection cuts
                if amp_sum < self.min_amp and np.sqrt(x_cent**2 + y_cent**2) < self.local_distance_cut:
                    continue
                
                # Make sure everything is 32 bit
                x = x[mask].astype(np.float32)
                y = y[mask].astype(np.float32)
                image = pmt_signal[mask].astype(np.float32)
                peak_time = dl1.peak_time[mask].astype(np.float32)

                zen = 90 - alt_evt.to(u.deg).value
                # Store simulated Xmax
                mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

                # Calc difference from expected Xmax (for gammas)
                exp_xmax =xmax_expectation(energy.value)
                x_diff = mc_xmax - exp_xmax
                x_diff_bin = find_nearest_bin(self.xmax_bins, x_diff)

                time_mask = image>5
                if np.sum(time_mask)>3:
                    # Linear regression code needs double precision for some reason
                    xv = x.to(u.deg).value[time_mask].astype("float64")
                    pt = peak_time[time_mask].astype("float64")                    
                    time_slope = lts_linear_regression(x=xv, y=pt, samples=3)[0][0]
                else:
                    time_slope = 0.

                az = point.az.to(u.deg).value
                zen = 90. - point.alt.to(u.deg).value

                # Now fill up our output with the X, Y and amplitude of our pixels
                key = zen, az, energy.value, int(impact), x_diff_bin, offset_bin

                if (key) in self.templates.keys():
                    # Extend the list if an entry already exists
                    self.templates[key].extend(image)
                    self.templates_xb[key].extend(x.to(u.deg).value)
                    self.templates_yb[key].extend(y.to(u.deg).value)
                    self.count[key] = self.count[key] + (1  * xmax_scale[(x_diff_bin, offset_bin)])
                    self.time_slope[key].append(time_slope)
                else:
                    self.templates[key] = image.tolist()
                    self.templates_xb[key] = x.value.tolist()
                    self.templates_yb[key] = y.value.tolist()
                    self.count[key] = 1 * xmax_scale[(x_diff_bin, offset_bin)]
                    self.time_slope[key] = [time_slope]

            if num > max_events:
                return True

            num += 1

        return True

    def fit_templates(self, amplitude, x_pos, y_pos):
        """
        Perform MLP fit over a dictionary of pixel lists

        :param amplitude: dict
            Dictionary of pixel amplitudes for each template
        :param x_pos: dict
            Dictionary of x position for each template
        :param y_pos: dict
            Dictionary of y position for each template
        :param make_variance_template: bool
            Should we also make a template of variance
        :param max_fitpoints: int
            Maximum number of points to include in MLP fit
        :return: dict
            Dictionary of image templates
        """

        if self.verbose:
            print("Fitting Templates")
        # Create output dictionary
        templates_out = dict()
        variance_templates_out = dict()

        # Create grid over which to evaluate our fit
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.bins[0])
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.bins[1])

        first = True
        # Loop over all templates
        for key in tqdm(list(amplitude.keys())):
            if self.verbose and first:
                print("Energy", key[2], "TeV")
                first = False

            amp = np.array(amplitude[key])
            if self.verbose:
                print("Template key:", key)
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
            
            #if make_variance_template:
                # need better plan for var templates

        return templates_out

    def perform_fit(self, amp, pixel_pos, 
                    nodes=(64, 64, 64, 64, 64, 64, 64, 64, 64)):
        """
        Fit MLP model to individual template pixels

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

        model.add(Dense(1, activation='linear'))

        def poisson_loss(y_true, y_pred):
            return tensor_poisson_likelihood(y_true, y_pred, 0.5, 1.)

        # First we have a go at fitting our model with a mean squared loss
        # this gets us most of the way to the answer and is more stable
        model.compile(loss="mse",
                        optimizer="adam", metrics=['accuracy'])
        stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0.0,
                                                    patience=20,
                                                    verbose=2, mode='auto')
        
        
        model.fit(pixel_pos, amp, epochs=10000,
                    batch_size=50000,
                    callbacks=[stopping], validation_split=0.1, verbose=0)
        weights = model.get_weights()

        # Then copy over the weights to a new model but with our poisson loss
        # this should get the final normalisation right
        model.compile(loss=poisson_loss,
                        optimizer="adam", metrics=['accuracy'])
        model.set_weights(weights)
        
        hist = model.fit(pixel_pos, amp, epochs=10000,
                    batch_size=50000,
                    callbacks=[stopping], validation_split=0.1, verbose=0)
        model_pred = model.predict(grid.T)

        # Set everything outside the range of our points to zero
        # This is a bit of a hacky way of doing this, but is fast and works
        lin_range = LinearNDInterpolator(pixel_pos, amp, fill_value=0)
        lin_nan = lin_range(grid.T) == 0
        model_pred[lin_nan] = 0

        return model_pred.reshape((self.bins[1], self.bins[0]))

    def save_templates(self, output_file):

        file_templates = self.fit_templates(self.templates, self.templates_xb, self.templates_yb)

        file_handler = gzip.open(output_file, "wb")
        pickle.dump(file_templates, file_handler)
        file_handler.close()

        return file_templates

    def save_time_slope(self, output_file):

        output_dict = {}
        output_dict_unncert = {}

        for key in tqdm(list(self.time_slope.keys())):
            time_slope_list = self.time_slope[key]
            if len(time_slope_list) >9:
                output_dict[key] = scipy.stats.trim_mean(time_slope_list)
                output_dict_unncert[key] = scipy.stats.mstats.trimmed_std(time_slope_list, 0.01)

        file_handler = gzip.open(output_file, "wb")
        pickle.dump(output_dict, file_handler)
        file_handler.close()

        return output_dict

    def save_fraction(self, output_file):

        output_dict = {}
        for key in self.count.keys():
            output_dict[key] = (self.count[key]/self.count_total)

        file_handler = gzip.open(output_file, "wb")
        pickle.dump(output_dict, file_handler)
        file_handler.close()

        return output_dict

    def generate_templates(self, file_list, output_file=None, fraction_output_file=None,
                           time_slope_output_file=None, max_events=1e9):
        """

        :param file_list: list
            List of sim_telarray input files
        :param output_file: string
            Output file name
        :param variance_output_file: string
            Output file name of variance templates
        :param extend_range: bool
            Extend range of the templates beyond simulations
        :param max_events: int
            Maximum number of events to process
        :return: dict
            Dictionary of image templates

        """

        templates = dict()
        variance_templates = dict()

        # Read in the files listed consecutively 
        for filename in file_list:
            self.read_templates(filename, max_events)

        if output_file is not None:
            # Fit them using the method requested
            self.save_templates(output_file)

        if fraction_output_file is not None:
            # Turn our counts into a fraction missed
            self.save_fraction(fraction_output_file)

        if time_slope_output_file is not None:
            # save our time slope output
            self.save_time_slope(time_slope_output_file)

        return templates, variance_templates, self.count

    def calculate_correction_factors(self, file_list, template_file, max_events=1000000):
        """ Funtion for performing a simple correction to the template amplitudes to
        match the training images. Only needed if significant fit biases are seen

        Args:
            file_list (list): List of input file names
            template_file (string): File name of template file
            max_events (int, optional): Maximum number of events to include in fitting. Defaults to 1000000.

        Returns:
            float: Scaling factor to the templates
        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy.optimize import minimize

        # Loop over input file and read in events
        for filename in file_list:
            amplitude, pixel_x, pixel_y = self.read_templates(filename, max_events=max_events)
        
        # Open up the template file
        templates = pickle.load(gzip.open(template_file,"r"))
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
                interpolator = RegularGridInterpolator((y_bins, x_bins), template, bounds_error=False, fill_value=0)
                prediction = interpolator((y, x)) 
                prediction[prediction<1e-6] = 1e-6
                
                # Store the amplitude and prediction
                if amp_vals is None:
                    amp_vals = amp
                    pred_vals = prediction
                else:
                    amp_vals = np.concatenate((amp_vals, amp))
                    pred_vals = np.concatenate((pred_vals, prediction))

            except KeyError:
                print(key, "Key missing")
        
        # Define the Poissonian fit function used to fit datas
        def scale_like(scale_factor):
            # Reasonable values of single photoelection width and pedestal are used
            # Might need to change for different detector types
            return np.sum(poisson_likelihood_gaussian(amp_vals, pred_vals*scale_factor[0], 0.5, 1))

        # Minimise this function to get the scaling factor
        res = minimize(scale_like, [1.], method='Nelder-Mead')
        # If our fit fails don't scale
        if res.x[0] is None:
            return 1
        
        # Otherwise return scale factor
        return res.x[0]
