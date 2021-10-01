"""

"""
import gzip
import pickle

import astropy.units as u
import numpy as np

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
from ctapipe.image.extractor import FullWaveformSum, FixedWindowSum
from ctapipe.calib.camera.gainselection import ThresholdGainSelector


class TemplateFitter:

    def __init__(self, eff_fl=1, 
                 bounds=((-5, 1), (-1.5, 1.5)), 
                 bins=(601, 301),
                 min_fit_pixels=3000, 
                 xmax_bins=np.linspace(-150, 200, 15),  
                 offset_bins=np.array([0.0])*u.deg,
                 verbose=False, 
                 rotation_angle=0 * u.deg, 
                 training_library="keras",
                 tailcuts=(7, 14), min_amp=30, local_distance_cut=2.*u.deg,
                 gain_threshold=30000,
                 amplitude_correction=False):
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
            training_library (str, optional): [description]. Defaults to "keras".
            tailcuts (tuple, optional): [description]. Defaults to (7, 14).
            min_amp (int, optional): [description]. Defaults to 30.
            local_distance_cut ([type], optional): [description]. Defaults to 2.*u.deg.
            gain_threshold (int, optional): [description]. Defaults to 30000.
            amplitude_correction (bool, optional): [description]. Defaults to False.
        """

        self.verbose = verbose
        self.xmax_bins = xmax_bins
        self.eff_fl = eff_fl

        self.bounds = bounds
        self.bins = bins
        self.min_fit_pixels = min_fit_pixels

        self.rotation_angle = rotation_angle
        self.offset_bins = np.sort(offset_bins)
        self.training_library = training_library
        self.tailcuts = tailcuts
        self.min_amp = min_amp
        self.local_distance_cut = local_distance_cut

        self.templates = dict()  # Pixel amplitude
        self.template_fit = dict()  # Pixel amplitude
        self.template_fit_kde = dict()  # Pixel amplitude

        self.templates_xb = dict()  # Rotated X position
        self.templates_yb = dict()  # Rotated Y positions
        self.correction = dict()

        self.count = dict() # Count of events in a given template
        self.count_total = 0 # Total number of events 

        self.amplitude_correction = amplitude_correction
        self.gain_threshold = gain_threshold

        self.amplitude_correction = amplitude_correction

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

        # This value is currently set for HESS, need to make this more flexible in future
        calib = CameraCalibrator(subarray=source.subarray, image_extractor=FixedWindowSum(source.subarray,
                                                                                          window_width=16, window_shift=3, peak_index=3,
                                                                                          apply_integration_correction=False))

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
                
                # now rotate and translate our images such that they lie on top of one
                # another
                x, y = \
                    ImPACTReconstructor.rotate_translate(x, y,
                                                            source_direction.fov_lon,
                                                            source_direction.fov_lat,
                                                            phi)
                x *= -1

                # We only want to keep pixels that fall within the bounds of our
                # final template
                mask = np.logical_and(x > self.bounds[0][0] * u.deg,
                                        x < self.bounds[0][1] * u.deg)
                mask = np.logical_and(mask, y < self.bounds[1][1] * u.deg)
                mask = np.logical_and(mask, y > self.bounds[1][0] * u.deg)

                mask510 = tailcuts_clean(geom, pmt_signal,
                                        picture_thresh=self.tailcuts[0],
                                        boundary_thresh=self.tailcuts[1],
                                        min_number_picture_neighbors=1)

                amp_sum = np.sum(pmt_signal[mask510])
                x_cent = np.sum(pmt_signal[mask510] * x[mask510]) / amp_sum
                y_cent = np.sum(pmt_signal[mask510] * y[mask510]) / amp_sum
                
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

                zen = 90 - alt_evt.to(u.deg).value
                # Store simulated Xmax
                mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

                # Calc difference from expected Xmax (for gammas)
                exp_xmax =xmax_expectation(energy.value)
                x_diff = mc_xmax - exp_xmax
                x_diff_bin = find_nearest_bin(self.xmax_bins, x_diff)
#                print(point.separation(src).value, offset_bin, mc_xmax, exp_xmax,x_diff, x_diff_bin, zen)

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
                else:
                    self.templates[key] = image.tolist()
                    self.templates_xb[key] = x.value.tolist()
                    self.templates_yb[key] = y.value.tolist()
                    self.count[key] = 1 * xmax_scale[(x_diff_bin, offset_bin)]

            if num > max_events:
                return self.templates, self.templates_xb, self.templates_yb

            num += 1

        return self.templates, self.templates_xb, self.templates_yb

    def fit_templates(self, amplitude, x_pos, y_pos,
                      make_variance_template, max_fitpoints):
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
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack((xx.ravel(), yy.ravel()))

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
            template_output = self.perform_fit(amp, pixel_pos, self.training_library,max_fitpoints)
            templates_out[key] = template_output.astype(np.float32)
            
            #if make_variance_template:
                # need better plan for var templates

        return templates_out, variance_templates_out

    def perform_fit(self, amp, pixel_pos,  training_library, max_fitpoints=None,
                    nodes=(64, 64, 64, 64, 64, 64, 64, 64, 64)):
        """
        Fit MLP model to individual template pixels

        :param amp: ndarray
            Pixel amplitudes
        :param pixel_pos: ndarray
            Pixel XY coordinate format (N, 2)
        :param max_fitpoints: int
            Maximum number of points to include in MLP fit
        :param nodes: tuple
            Node layout of MLP
        :return: MLP
            Fitted MLP model
        """
        pixel_pos = pixel_pos.T

        # If we put a limit on this then randomly choose points
        if max_fitpoints is not None and amp.shape[0] > max_fitpoints:
            indices = np.arange(amp.shape[0])
            np.random.shuffle(indices)
            amp = amp[indices[:max_fitpoints]]
            pixel_pos = pixel_pos[indices[:max_fitpoints]]

        if self.verbose:
            print("Fitting template using", training_library, "with", amp.shape[0],
                  "total pixels")

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

        # Fit using kernal density estimates
        if training_library == "kde":
            # We use this fast KDE library rather than the scikit learn or scipy implmentation
            from KDEpy import FFTKDE
            from scipy.interpolate import LinearNDInterpolator

            data = np.vstack((x, y, amp))

            # This bandwidth is a reasonable guess
            bw = 0.02
            kde = FFTKDE(bw=bw, kernel='gaussian').fit(data.T)
            # Estimate pdf on a 3D grid
            z_bins = 400
            points, out = kde.evaluate((self.bins[0], self.bins[1], z_bins))
            points_x, points_y, points_z = points.T
            
            # Then calculate average value on the amplitude axis
            weights = (out*points_z).reshape((self.bins[0], self.bins[1], 400))
            average_value = np.sum(weights, axis=-1) / \
                            np.sum(out.reshape((self.bins[0], self.bins[1], 400)), axis=-1)
            average_value = average_value - 1

            squared_average_value = np.sum(weights**2, axis=-1) / \
                                    np.sum(out.reshape((self.bins[0], self.bins[1], 400)), axis=-1)
            
            points_x = points_x.reshape((self.bins[0], self.bins[1], 400))[:, :, 0].ravel()
            points_y = points_y.reshape((self.bins[0], self.bins[1], 400))[:, :, 0].ravel()

            # Then interpolate on our evaluated grid points to give the template                
            lin = LinearNDInterpolator(np.vstack((points_x, points_y)).T, average_value.ravel(), fill_value=0)
            kde_pred = lin(grid.T)

            return kde_pred.reshape((self.bins[1], self.bins[0]))

        elif training_library == "keras":
            # Fit image pixels using a multi layer perceptron
            from scipy.interpolate import LinearNDInterpolator

            from keras.models import Sequential
            from keras.layers import Dense
            import keras
            
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

    def generate_templates(self, file_list, output_file=None, variance_output_file=None, fraction_output_file=None,
                           extend_range=True, max_events=1e9, max_fitpoints=None):
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
        :param max_fitpoints: int
            Maximum number of points to include in the MLP fit
        :return: dict
            Dictionary of image templates

        """

        make_variance = variance_output_file is not None

        templates = dict()
        variance_templates = dict()

        # Read in the files listed consecutively 
        for filename in file_list:
            pix_lists = self.read_templates(filename, max_events)
        if output_file is not None:
            # Fit them using the method requested
            file_templates, file_variance_templates = self.fit_templates(pix_lists[0],
                                                                        pix_lists[1],
                                                                        pix_lists[2],
                                                                        make_variance,
                                                                        max_fitpoints)
            templates.update(file_templates)
            self.template_fit = templates

            if make_variance:
                variance_templates.update(file_variance_templates)

            # Correct the amplitude by the normalisation from KDE fitting if requested 
            # re-run the template reading, predict the image and take the ratio to the
            # real image
            if self.amplitude_correction:
                self.training_library = "kde"
                file_templates, file_variance_templates = self.fit_templates(pix_lists[0],
                                                                            pix_lists[1],
                                                                            pix_lists[2],
                                                                            make_variance,
                                                                            max_fitpoints)
                self.template_fit_kde = file_templates
                for filename in file_list:
                    _ = self.read_templates(filename, max_events, fill_correction=True)

            pix_lists = None
            file_templates = None
            file_variance_templates = None

            # Extend coverage of the templates by extrapolation if requested
            if extend_range:
                templates = extend_template_coverage(self.xmax_bins, templates)
                if make_variance:
                    variance_templates = extend_template_coverage(self.xmax_bins, variance_templates)

            # Perform correction on the templates
            if self.amplitude_correction:

                for key in self.correction.keys():
                    correction_factor = np.median(self.correction[key])
                    correction_factor_error = np.std(self.correction[key]) / np.sqrt(float(len(self.correction[key])))

                    print(correction_factor_error / correction_factor, correction_factor)
                    if correction_factor > 0 and correction_factor_error / correction_factor < 0.1:
                        self.template_fit[key] = self.template_fit[key] * np.median(self.correction[key])
                    else:
                        self.template_fit.pop(key)

            # Finally write everything out to a gzipped pickle file
            file_handler = gzip.open(output_file, "wb")
            pickle.dump(self.template_fit, file_handler)
            file_handler.close()

            # And variance templates if needed
            if make_variance:
                file_handler = gzip.open(variance_output_file, "wb")
                pickle.dump(variance_templates, file_handler)
                file_handler.close()

        if fraction_output_file is not None:
            # Turn our counts into a fraction missed
            for key in self.count.keys():
                self.count[key] = (self.count[key]/self.count_total)
                
            if extend_range:
                self.count = extend_template_coverage(self.xmax_bins, self.count)

            file_handler = gzip.open(fraction_output_file, "wb")
            pickle.dump(self.count, file_handler)
            file_handler.close()

        return templates, variance_templates, self.count

    def calculate_correction_factors(self, file_list, template_file, max_events=1000000):

        from scipy.interpolate import RegularGridInterpolator
        from ctapipe.image.pixel_likelihood import  neg_log_likelihood_approx
        from scipy.optimize import minimize

        for filename in file_list:
            amplitude, pixel_x, pixel_y = self.read_templates(filename, max_events=max_events)
        
        templates = pickle.load(gzip.open(template_file,"r"))
        keys = amplitude.keys()

        x_bins = np.linspace(self.bounds[0][0], self.bounds[0][1], self.bins[0])
        y_bins = np.linspace(self.bounds[1][0], self.bounds[1][1], self.bins[1])
        scale = []
        
        amp_vals = None
        pred_vals = None

        for key in keys:
            
            key_cut = key[:-1]
            try:
                template = templates[key_cut]
                amp, x, y = amplitude[key], pixel_x[key], pixel_y[key]
                amp = np.array(amp)
                
                interpolator = RegularGridInterpolator((y_bins, x_bins), template, bounds_error=False, fill_value=0)
                prediction = interpolator((y, x)) 
                prediction[prediction<1e-6] = 1e-6
                
                def scale_like(scale_factor):
                    return np.sum(poisson_likelihood_gaussian(amp, prediction*scale_factor[0], 0.5, 1))

                print(key, minimize(scale_like, [1.], method='Nelder-Mead').x)
                if amp_vals is None:
                    amp_vals = amp
                    pred_vals = prediction
                else:
                    amp_vals = np.concatenate((amp_vals, amp))
                    pred_vals = np.concatenate((pred_vals, prediction))

            except KeyError:
                print(key, "Key missing")
        
        print(pred_vals)
        def scale_like(scale_factor):
            return np.sum(poisson_likelihood_gaussian(amp_vals, pred_vals*scale_factor[0], 0.5, 1))

        res = minimize(scale_like, [1.], method='Nelder-Mead')
        print("scale", res.x)
        if res.x[0] is None:
            return 1

        return res.x[0]
            
#                scale.append(scale_factor)
            
#        print("Average scale factor", np.average(scale))
