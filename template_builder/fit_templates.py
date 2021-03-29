"""

"""
import gzip
import pickle

import astropy.units as u
import numpy as np

from ctapipe.coordinates import CameraFrame, NominalFrame, GroundFrame, \
    TiltedGroundFrame
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from ctapipe.io.eventsource import event_source
from ctapipe.reco import ImPACTReconstructor
from scipy.interpolate import interp1d
from tqdm import tqdm
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.calib import CameraCalibrator
from ctapipe.image.extractor import FullWaveformSum


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

    def __init__(self, eff_fl=1, bounds=((-5, 1), (-1.5, 1.5)), bins=(601, 301),
                 min_fit_pixels=3000, xmax_bins=np.linspace(-150, 200, 15),
                 maximum_offset=10*u.deg,
                 verbose=False, rotation_angle=0 * u.deg, training_library="keras",
                 tailcuts=(7, 14), min_amp=30, amplitude_correction=False):
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

        self.rotation_angle = rotation_angle
        self.maximum_offset = maximum_offset
        self.training_library = training_library
        self.tailcuts = tailcuts
        self.min_amp = min_amp

        self.templates = dict()  # Pixel amplitude
        self.template_fit = dict()  # Pixel amplitude
        self.template_fit_kde = dict()  # Pixel amplitude

        self.templates_xb = dict()  # Rotated X position
        self.templates_yb = dict()  # Rotated Y positions
        self.correction = dict()
        self.count = dict()

        self.amplitude_correction = amplitude_correction

    def read_templates(self, filename, max_events=1e9, fill_correction=False):
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

        # Create a dummy time for our AltAz objects
        dummy_time = Time('2010-01-01T00:00:00', format='isot', scale='utc')

        if self.verbose:
            print("Reading", filename.strip())

        calibrator = CameraCalibrator(image_extractor=FullWaveformSum())

        with event_source(input_url=filename.strip()) as source:

            grd_tel = None
            num = 0  # Event counter

            for event in tqdm(source):
                alt = event.mcheader.run_array_direction[1]
                if alt > 90. * u.deg:
                    alt = 90. * u.deg
                point = SkyCoord(alt=alt, az=event.mcheader.run_array_direction[0],
                                 frame=AltAz(obstime=dummy_time))

                mc = event.mc
                # Create coordinate objects for source position
                src = SkyCoord(alt=mc.alt.value * u.rad, az=mc.az.value * u.rad,
                               frame=AltAz(obstime=dummy_time))
                #print("here1", point.separation(src),  self.maximum_offset)
                if point.separation(src) > self.maximum_offset:
                    continue

                # And transform into nominal system (where we store our templates)
                source_direction = src.transform_to(NominalFrame(origin=point))

                # Perform calibration of images
                try:
                    calibrator(event)
                except ZeroDivisionError:
                    print("ZeroDivisionError in calibrator, skipping this event")
                    continue

                # Store simulated event energy
                energy = mc.energy

                # Store ground position of all telescopes
                # We only want to do this once, but has to be done in event loop
                if grd_tel is None:
                    grd_tel = event.inst.subarray.tel_coords


                    # Convert to tilted system
                    tilt_tel = grd_tel.transform_to(
                        TiltedGroundFrame(pointing_direction=point))

                # Calculate core position in tilted system
                grd_core_true = SkyCoord(x=np.asarray(mc.core_x) * u.m,
                                         y=np.asarray(mc.core_y) * u.m,
                                         z=np.asarray(0) * u.m, frame=GroundFrame())

                tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(
                    pointing_direction=point))

                # Loop over triggered telescopes
                for tel_id in event.dl0.tels_with_data:
                   #  Get pixel signal
                    dl1 =  event.dl1.tel[tel_id]
                    try:
#                        print(event.r1.tel[tel_id].waveform.shape)
                        hg, lg =np.sum(event.r1.tel[tel_id].waveform, axis=2)
                        pmt_signal = hg
#                        print(hg[pmt_signal>100],  lg[pmt_signal>100])
                        pmt_signal[pmt_signal>100] = lg[pmt_signal>100]
#                        print(dl1.image[pmt_signal>100], pmt_signal[pmt_signal>100])
                    except ValueError:
                        pmt_signal = np.sum(event.r1.tel[tel_id].waveform, axis=-1)[0]

                    # Get pixel coordinates and convert to the nominal system
                    geom = event.inst.subarray.tel[tel_id].camera
                    fl = event.inst.subarray.tel[tel_id].optics.equivalent_focal_length * \
                         self.eff_fl

                    camera_coord = SkyCoord(x=geom.pix_x, y=geom.pix_y,
                                            frame=CameraFrame(focal_length=fl,
                                                              telescope_pointing=point))

                    nom_coord = camera_coord.transform_to(
                        NominalFrame(origin=point))

                    y = nom_coord.delta_az.to(u.deg)
                    x = nom_coord.delta_alt.to(u.deg)

                    # Calculate expected rotation angle of the image
                    phi = np.arctan2((tilt_tel.y[tel_id - 1] - tilt_core_true.y),
                                     (tilt_tel.x[tel_id - 1] - tilt_core_true.x)) + \
                          180 * u.deg
                    phi += self.rotation_angle
                    # And the impact distance of the shower
                    impact = np.sqrt(np.power(tilt_tel.x[tel_id - 1] - tilt_core_true.x, 2) +
                                     np.power(tilt_tel.y[tel_id - 1] - tilt_core_true.y, 2)). \
                        to(u.m).value
                    
                    # now rotate and translate our images such that they lie on top of one
                    # another
                    x, y = \
                        ImPACTReconstructor.rotate_translate(x, y,
                                                             source_direction.delta_alt,
                                                             source_direction.delta_az,
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
                    
                    if fill_correction:
                        mask = np.logical_and(mask, mask510)
                    
                    mask = mask510
                    for i in range(4):
                        mask = dilate(geom, mask)

                    if amp_sum < self.min_amp and np.sqrt(x_cent**2 + y_cent**2) < 2*u.deg:
                        continue

                    # Make sure everything is 32 bit
                    x = x[mask].astype(np.float32)
                    y = y[mask].astype(np.float32)
                    image = pmt_signal[mask].astype(np.float32)

                    zen = 90 - mc.alt.to(u.deg).value
                    # Store simulated Xmax
                    mc_xmax = event.mc.x_max.value / np.cos(np.deg2rad(zen))

                    # Calc difference from expected Xmax (for gammas)
                    exp_xmax = 300 + 93 * np.log10(energy.value)
                    x_diff = mc_xmax - exp_xmax

                    x_diff_bin = find_nearest_bin(self.xmax_bins, x_diff)
                    az = point.az.to(u.deg).value
                    zen = 90. - point.alt.to(u.deg).value

                    # Now fill up our output with the X, Y and amplitude of our pixels
                    if fill_correction:
                        from scipy.interpolate import RegularGridInterpolator
                        xb = np.linspace(self.bounds[0][0], self.bounds[0][1], self.bins[0])
                        yb = np.linspace(self.bounds[1][0], self.bounds[1][1], self.bins[1])
                        key = zen, az, energy.value, int(impact), x_diff_bin
#                        print(key)
                        if key in self.template_fit.keys():
                            interp = RegularGridInterpolator((yb, xb), self.template_fit[key], bounds_error=False)
                            pred = interp(np.array([y.to(u.deg).value, x.to(u.deg).value]).T)
                            
                            interp_kde = RegularGridInterpolator((yb, xb), self.template_fit_kde[key], bounds_error=False)
                            pred_kde = interp(np.array([y.to(u.deg).value, x.to(u.deg).value]).T)

                            print(np.sum(interp_kde), np.sum(pred_kde))

                            if key in self.correction.keys():
                                self.correction[key] = np.append(self.correction[key], np.sum(pred_kde)/np.sum(pred))
                            else:
                                self.correction[key] = (np.sum(pred_kde)/np.sum(pred))
                    else:
                        if (zen, az, energy.value, int(impact), x_diff_bin) in self.templates.keys():
                            # Extend the list if an entry already exists
                            self.templates[(zen, az, energy.value, int(impact), x_diff_bin)].\
                                extend(image)
                            self.templates_xb[(zen, az, energy.value, int(impact), x_diff_bin)].\
                                extend(x.to(u.deg).value)
                            self.templates_yb[(zen, az, energy.value, int(impact), x_diff_bin)].\
                                extend(y.to(u.deg).value)
                            self.count[(zen, az, energy.value, int(impact), x_diff_bin)] = self.count[(zen, az, energy.value, int(impact), x_diff_bin)] + 1
                        else:
                            self.templates[(zen, az, energy.value, int(impact), x_diff_bin)] = \
                                image.tolist()
                            self.templates_xb[(zen, az, energy.value, int(impact), x_diff_bin)] = \
                                x.value.tolist()
                            self.templates_yb[(zen, az, energy.value, int(impact), x_diff_bin)] = \
                                y.value.tolist()
                            self.count[(zen, az, energy.value, int(impact), x_diff_bin)] = 1

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
        for key in tqdm(list(amplitude.keys())[:10]):
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
            training_library = self.training_library
            model = self.perform_fit(amp, pixel_pos, training_library,max_fitpoints)

            if str(type(model)) == \
                    "<class 'scipy.interpolate.interpnd.LinearNDInterpolator'>":
                nn_out = model(grid.T)
                nn_out = nn_out.reshape((self.bins[1], self.bins[0]))
                nn_out[np.isinf(nn_out)] = 0
            elif self.training_library == "kde":
                points, nn_out = model.evaluate((self.bins[0], self.bins[1]))#grid.T)
                # rint(nn_out)
                nn_out = nn_out.reshape((self.bins[0], self.bins[1]))
                nn_out[np.isinf(nn_out)] = 0
            elif self.training_library == "KNN":
                nn_out = model
            else:
                # Evaluate MLP fit over our grid
#                nn_out = model.predict(grid.T)
#                nn_out = nn_out.reshape((self.bins[1], self.bins[0]))
                nn_out = model
                nn_out[np.isinf(nn_out)] = 0
#            print("sc",scale)
            templates_out[(key[0], key[1], key[2], key[3], key[4])] = \
                nn_out.astype(np.float32)
            
            if make_variance_template:

                if str(type(model)) == \
                        "<class 'scipy.interpolate.interpnd.LinearNDInterpolator'>":
                    predicted_values = model(pixel_pos.T)
                else:
                    predicted_values = model.predict(pixel_pos.T)
                
                predicted_values = predicted_values.ravel()
                predicted_values[np.isinf(predicted_values)] = 0

                # Take absolute and square after as the NN fits the squared deviation
                # This is important due to the 1 sided distribution
                variance = np.abs(amp - predicted_values)
                model_variance = self.perform_fit(variance, pixel_pos,
                                                  self.training_library)

                if str(type(model)) == \
                        "<class 'scipy.interpolate.interpnd.LinearNDInterpolator'>":
                    nn_out_variance = np.power(model_variance(grid.T), 2)
                else:
                    nn_out_variance = np.power(model_variance.predict(grid.T), 2)
                nn_out_variance = nn_out_variance.reshape((self.bins[1], self.bins[0]))
                nn_out_variance[np.isinf(nn_out)] = 0

                variance_templates_out[(key[0], key[1], key[2], key[3], key[4])] = \
                    nn_out_variance.astype(np.float32)

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

        if training_library == "kde":
            from KDEpy import FFTKDE
            from scipy.interpolate import LinearNDInterpolator

            x, y = pixel_pos.T
            amp_fit = np.concatenate((amp, amp))
            amp_fit[amp_fit<1e-3] = 1e-3
            amp_fit = np.log10(amp_fit)

            y_fit = np.concatenate((np.abs(y), -1*np.abs(y)))
            x_fit = np.concatenate((x, x))

            scale = 1
            data = np.vstack((x_fit, y_fit, amp_fit * scale))

            bw = 0.02
            kde = FFTKDE(bw=bw).fit(data.T)
            
            points, out = kde.evaluate((self.bins[0], self.bins[1], 200))
            #points, out = kde.evaluate(grid)
            points_x, points_y, points_z = points.T
            points_z = points_z * scale
            print(points_x, points_y, points_z, np.min(amp))

            av_z = np.average(points_z)
            print(av_z, ((np.max(points_z)-np.min(points_z))/2.) + np.min(points_z))
            weights = (out*points_z).reshape((self.bins[0], self.bins[1], 200))
            average_value = np.sum(weights, axis=-1) / \
                np.sum(out.reshape((self.bins[0], self.bins[1], 200)), axis=-1)
            average_value = np.power(10, average_value)

            squared_average_value = np.sum(weights**2, axis=-1) / \
                np.sum(out.reshape((self.bins[0], self.bins[1], 200)), axis=-1)

        pixel_pos_neg = np.array([pixel_pos.T[0], -1 * np.abs(pixel_pos.T[1])]).T
        pixel_pos = np.concatenate((pixel_pos, pixel_pos_neg))
        amp = np.concatenate((amp, amp))
        x, y = pixel_pos.T

        amp_fit = np.concatenate((amp, amp))
        amp_fit = amp_fit + 1
        amp_fit[amp_fit<1e-3] = 1e-3
        amp_fit= np.log10(amp_fit)
        
        y_fit = np.concatenate((np.abs(y), -1*np.abs(y)))
        x_fit = np.concatenate((x, x))
        
        scale = 1
        data = np.vstack((x_fit, y_fit, amp_fit * scale))
        
        bw = 0.02
        kde = FFTKDE(bw=bw, kernel='gaussian').fit(data.T)
        z_bins = 400
        points, out = kde.evaluate((self.bins[0], self.bins[1], z_bins))
        points_x, points_y, points_z = points.T
        points_z = np.power(10,points_z * scale)
        
        av_z = np.average(points_z)
        
        weights = (out*points_z).reshape((self.bins[0], self.bins[1], 400))
        average_value = np.sum(weights, axis=-1) / \
                        np.sum(out.reshape((self.bins[0], self.bins[1], 400)), axis=-1)
        average_value = average_value - 1

        squared_average_value = np.sum(weights**2, axis=-1) / \
                                np.sum(out.reshape((self.bins[0], self.bins[1], 400)), axis=-1)
        
        variance = squared_average_value - average_value**2
        points_x = points_x.reshape((self.bins[0], self.bins[1], 400))[:, :, 0].ravel()
        points_y = points_y.reshape((self.bins[0], self.bins[1], 400))[:, :, 0].ravel()
            
        lin = LinearNDInterpolator(np.vstack((points_x, points_y)).T, average_value.ravel(), fill_value=0)
        if training_library == "kde":
            return lin
        elif training_library == "keras":
            from keras.models import Sequential
            from keras.layers import Dense
            import keras
            
            model = Sequential()
            model.add(Dense(nodes[0], activation="relu", input_shape=(2,)))

            for n in nodes[1:]:
                model.add(Dense(n, activation="relu"))

            model.add(Dense(1, activation='linear'))
            model.compile(loss='mean_squared_error',
                          optimizer="adam", metrics=['accuracy'])
            stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0.0,
                                                     patience=20,
                                                     verbose=2, mode='auto')
            
            pixel_pos_neg = np.array([pixel_pos.T[0], -1 * np.abs(pixel_pos.T[1])]).T
        
#            pixel_pos = np.concatenate((pixel_pos, pixel_pos_neg))
 #           amp = np.concatenate((amp, amp))
        
            model.fit(pixel_pos, amp, epochs=10000,
                      batch_size=50000,
                      callbacks=[stopping], validation_split=0.1, verbose=0)
            model_pred = model.predict(grid.T)
            kde_pred = lin(grid.T)
            
            sel = kde_pred>4
            print(np.sum(kde_pred[sel])/np.sum(model_pred[sel]), np.min(pixel_pos.T[0]),  np.max(pixel_pos.T[0]), pixel_pos.shape)
            lin_range = LinearNDInterpolator(pixel_pos, amp, fill_value=0)
            lin_nan = lin_range(grid.T) == 0
            model_pred[lin_nan] = 0
            kde_pred[lin_nan] = 0
            print(np.sum(kde_pred[sel])/np.sum(model_pred[sel]), np.sum(lin_nan))
            return model_pred.reshape((self.bins[1], self.bins[0])) * (np.sum(kde_pred[sel])/np.sum(model_pred[sel]))

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
                key_test = (key[0], key[1], key[2], key[3], xb)
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
                key_test = (key[0], key[1], key[2], key[3], xb)
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
        if len(list(templates.keys())) < 1:
            return templates

        distances = np.sort(np.unique(keys.T[3]))
        energies = np.unique(keys.T[2])
        zeniths = np.unique(keys.T[0])
        azimuths = np.unique(keys.T[1])

        extended_templates = dict()
        for zen in zeniths:
            for az in azimuths:
                for en in energies:
                    for xmax in self.xmax_bins:
                        i = 0
                        distance_list = list()

                        # If we have no template at 0 copy the lowest value
                        if distances[0] is not 0.:
                            copied = False
                            for d in distances:
                                key = (zen, az, en, d, xmax)
                                if key in templates.keys() and not copied:
                                    extended_templates[(zen, az, en, 0, xmax)] = \
                                        templates[key]
                                    copied = True

                        for dist in distances[0:]:
                            key = (zen, az, en, dist, xmax)
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
                                key = (zen, az, en, distances[j], xmax)

                                extended_templates[key] = int_val

        templates.update(extended_templates)

        return templates

    def extend_template_coverage(self, templates):

        templates = self.extend_xmax_range(templates)
        templates = self.extend_distance_range(templates, 4)

        return templates

    def generate_templates(self, file_list, output_file, variance_output_file=None,
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

        for filename in file_list:
            pix_lists = self.read_templates(filename, max_events)
        
        file_templates, file_variance_templates = self.fit_templates(pix_lists[0],
                                                                     pix_lists[1],
                                                                     pix_lists[2],
                                                                     make_variance,
                                                                     max_fitpoints)
        templates.update(file_templates)
        self.template_fit = templates

        if make_variance:
            variance_templates.update(file_variance_templates)

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
            templates = self.extend_template_coverage(templates)
            if make_variance:
                variance_templates = self.extend_template_coverage(variance_templates)

        if self.amplitude_correction:

            for key in self.correction.keys():
                correction_factor = np.median(self.correction[key])
                correction_factor_error = np.std(self.correction[key]) / np.sqrt(float(len(self.correction[key])))

                print(correction_factor_error / correction_factor, correction_factor)
                if correction_factor > 0 and correction_factor_error / correction_factor < 0.1:
                    self.template_fit[key] = self.template_fit[key] * np.median(self.correction[key])
                else:
                    self.template_fit.pop(key)

        file_handler = gzip.open(output_file, "wb")
        pickle.dump(self.template_fit, file_handler)
        file_handler.close()

        if make_variance:
            file_handler = gzip.open(variance_output_file, "wb")
            pickle.dump(variance_templates, file_handler)
            file_handler.close()

        return templates, variance_templates
