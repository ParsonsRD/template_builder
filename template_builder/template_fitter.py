"""
Generate ImPACT templates from CORSIKA/sim_telarray simulations implemented as a ctapipe tool
"""
# pylint: disable=W0201
from email.policy import default
import sys

import gzip
import pickle

from tqdm.auto import tqdm
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
import numpy as np
import scipy

from argparse import ArgumentParser
from pathlib import Path

from ctapipe.calib import CameraCalibrator, GainSelector
from ctapipe.core import QualityQuery, Tool, traits
from ctapipe.core.traits import List, classes_with_traits, Unicode, Bool
from ctapipe.image import ImageCleaner, ImageModifier, ImageProcessor
from ctapipe.image.extractor import ImageExtractor
from ctapipe.reco.reconstructor import StereoQualityQuery

from ctapipe.fitting import lts_linear_regression

from ctapipe.io import (
    DataLevel,
    EventSource,
    SimTelEventSource,
    metadata,
)
from ctapipe.coordinates import (
    CameraFrame,
    NominalFrame,
    GroundFrame,
    TiltedGroundFrame,
)

from ctapipe.utils import EventTypeFilter
from ctapipe.image import dilate
from astropy.time import Time

from template_builder.nn_fitter import NNFitter
from template_builder.utilities import *
from template_builder.extend_templates import *

COMPATIBLE_DATALEVELS = [
    DataLevel.R1,
    DataLevel.DL0,
    DataLevel.DL1_IMAGES,
]

from art import tprint

__all__ = ["TemplateFitter"]


class TemplateFitter(Tool):
    """
    ctapipe tool to process CORSIKA/sim_telarray simulations and generate ImPACT templates
    """

    name = "template-fitter"
    examples = """
    To process data with all default values:
    > template-fitter --input events.simtel.gz --output ./Template --progress
    Or use an external configuration file, where you can specify all options:
    > template-fitter --config stage1_config.json --progress
    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main ctapipe code repo.
    """

    input_dir = traits.Path(
        default_value=None,
        help="Input directory",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input sim_telarray simulation files",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*.simtel.zst",
        help="Give a specific file pattern for matching files in ``input_dir``",
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    output_file = Unicode(default_value=".", help="base output file name").tag(
        config=True
    )

    xmax_bins = List(
        default_value=np.linspace(-150, 200, 15).tolist(),
        help="Bin centres for xmax bins to generate templates in",
    ).tag(config=True)

    offset_bins = List(
        default_value=[0.0],
        help="bin centres for offset bins (deg)",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "TemplateFitter.input_files",
        ("o", "output"): "TemplateFitter.output_file",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "EventSource.max_events",
        "image-cleaner-type": "ImageProcessor.image_cleaner_type",
    }

    compute_all = Bool(
        help="Compute all possible templates ?",
        default_value=False,
    ).tag(config=True)

    compute_image = Bool(
        help="Compute image templates ?",
        default_value=False,
    ).tag(config=True)

    compute_time = Bool(
        help="Compute time slope templates ?",
        default_value=False,
    ).tag(config=True)

    compute_fraction = Bool(
        help="Compute trigger fraction templates ?",
        default_value=False,
    ).tag(config=True)

    classes = (
        [
            CameraCalibrator,
            ImageProcessor,
            NNFitter,
            metadata.Instrument,
            metadata.Contact,
        ]
        + classes_with_traits(EventSource)
        + classes_with_traits(ImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
        + classes_with_traits(QualityQuery)
        + classes_with_traits(ImageModifier)
        + classes_with_traits(EventTypeFilter)
        + classes_with_traits(NNFitter)
    )

    def setup(self):
        """
        Parse files and set up class objects
        """
        # setup components:
        args = self.parser.parse_args(self.extra_args)
        self.input_files.extend(args.input_files)
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))

        if not self.input_files:
            self.log.critical(
                "No input files provided, either provide --input-dir "
                "or input files as positional arguments"
            )
            sys.exit(1)

        self.focal_length_choice = "EFFECTIVE"
        try:
            self.event_source = EventSource(
                input_url=self.input_files[0],
                parent=self,
                focal_length_choice=self.focal_length_choice,
            )
        except RuntimeError:
            print("Effective Focal length not availible, defaulting to equivelent")
            self.focal_length_choice = "EQUIVALENT"
            self.event_source = EventSource(
                input_url=self.input_files[0],
                parent=self,
                focal_length_choice=self.focal_length_choice,
            )

        if not self.event_source.has_any_datalevel(COMPATIBLE_DATALEVELS):
            self.log.critical(
                "%s  needs the EventSource to provide either R1 or DL0 or DL1A data"
                ", %s provides only %s",
                self.name,
                self.event_source,
                self.event_source.datalevels,
            )
            sys.exit(1)

        assert len(self.event_source.subarray.telescope_types)==1, "The event source should only contain one telescope type"

        self.telescope_type=str(self.event_source.subarray.telescope_types[0])

        self.calibrate = CameraCalibrator(
            parent=self, subarray=self.event_source.subarray
        )
        self.process_images = ImageProcessor(
            subarray=self.event_source.subarray, parent=self
        )
        self.event_type_filter = EventTypeFilter(parent=self)
        self.check_parameters = StereoQualityQuery(parent=self)

        # warn if max_events prevents writing the histograms
        if (
            isinstance(self.event_source, SimTelEventSource)
            and self.event_source.max_events
            and self.event_source.max_events > 0
        ):
            self.log.warning(
                "No Simulated shower distributions will be written because "
                "EventSource.max_events is set to a non-zero number (and therefore "
                "shower distributions read from the input Simulation file are invalid)."
            )

        if self.compute_all or self.compute_image:
            self.image_computation = True
        else:
            self.image_computation = False
        if self.compute_all or self.compute_time:
            self.time_computation = True
        else:
            self.time_computation = False
        if self.compute_all or self.compute_fraction:
            self.fraction_computation = True
        else:
            self.fraction_computation = False
        if (
            not self.compute_all
            and not self.compute_image
            and not self.compute_time
            and not self.compute_fraction
        ):
            self.fraction_computation = True
            self.time_computation = True
            self.image_computation = True

        # We need this dummy time for coord conversions later
        self.dummy_time = Time("2010-01-01T00:00:00", format="isot", scale="utc")
        # self.point, self.xmax_scale, self.tilt_tel = None, None, None

        if self.time_computation:
            self.time_slope = {}  # Image time gradients
        if self.image_computation:
            self.templates, self.templates_xb, self.templates_yb = (
                {},
                {},
                {},
            )  # Pixel amplitudes and rotated positions
            self.fitter = NNFitter(parent=self)
        if self.fraction_computation:
            self.count = {}  # Count of events in a given template
            self.count_total = 0

        self.key_list = []

    def start(self):
        """
        Process events up to dl1 and fill write out relevant quantities for the templates
        """

        self.event_source.subarray.info(printer=self.log.info)

        for input_file in self.input_files:
            self.event_source = EventSource(
                input_url=input_file,
                parent=self,
                focal_length_choice=self.focal_length_choice,
            )
            self.point, self.xmax_scale, self.tilt_tel = None, None, None

            for event in tqdm(
                self.event_source,
                desc=self.event_source.__class__.__name__,
                total=self.event_source.max_events,
                unit="events",
            ):
                self.log.debug("Processessing event_id=%s", event.index.event_id)
                self.calibrate(event)
                self.process_images(event)

                self.read_template(event)

            # Not sure what else to do here...
            if self.fraction_computation:
                obs_ids = self.event_source.simulation_config.keys()
                for obs_id in obs_ids:
                    self.count_total += self.event_source.simulation_config[
                        obs_id
                    ].n_showers
            self.event_source.close()

    def finish(self):
        """
        Last steps after processing events. Generate and save the templates.
        For the image templates, call the NNFitter.
        """
        if self.image_computation:
            self.fitter.generate_image_templates(
                self.templates_xb,
                self.templates_yb,
                self.templates,
                tel_type=self.telescope_type,
                output_file=self.output_file,
            )
        if self.time_computation:
            self.generate_time_templates()
        if self.fraction_computation:
            self.generate_fraction_templates()

    def read_template(self, event):
        """
        Read a dl1 event, extract the relevant quantities and shower parameters,
        and add to list to generate templates from.

        :param: event
        A simulated sim_telarray event processed to dl1
        """

        # When calculating alt we have to account for the case when it is rounded
        # above 90 deg
        alt_evt = event.simulation.shower.alt
        if alt_evt > 90 * u.deg:
            alt_evt = 90 * u.deg

        # Get the pointing direction and telescope positions of this run
        if self.point is None:
            alt = event.pointing.array_altitude
            if alt > 90 * u.deg:
                alt = 90 * u.deg

            self.point = SkyCoord(
                alt=alt,
                az=event.pointing.array_azimuth,
                frame=AltAz(obstime=self.dummy_time),
            )

            if self.fraction_computation:
                self.xmax_scale = create_xmax_scaling(
                    self.xmax_bins,
                    np.array(self.offset_bins) * u.deg,
                    self.point,
                    self.event_source.input_url,
                )

            grd_tel = self.event_source.subarray.tel_coords
            # Convert to tilted system
            self.tilt_tel = grd_tel.transform_to(
                TiltedGroundFrame(pointing_direction=self.point)
            )

        # These values are keys for the template dict later
        pt_az = self.point.az.to(u.deg).value
        pt_zen = 90.0 - self.point.alt.to(u.deg).value

        # Create coordinate objects for source position
        src = SkyCoord(
            alt=event.simulation.shower.alt.value * u.rad,
            az=event.simulation.shower.az.value * u.rad,
            frame=AltAz(obstime=self.dummy_time),
        )
        # Next key for template dict: Offset
        offset_bin = find_nearest_bin(
            np.array(self.offset_bins) * u.deg, self.point.separation(src)
        ).value

        # Store simulated event energy. A key for the template dict.
        energy = event.simulation.shower.energy

        # Calcualtion of xmax bin as a key for the template dicts later
        zen = 90 - event.simulation.shower.alt.to(u.deg).value
        # Store simulated Xmax
        mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

        # Calc difference from expected Xmax (for gammas).
        exp_xmax = xmax_expectation(energy.value)
        x_diff = mc_xmax - exp_xmax
        x_diff_bin = find_nearest_bin(self.xmax_bins, x_diff)

        # Finally, set up some properties that are needed in the per-telescope calculations

        # Calculate core position in tilted system
        grd_core_true = SkyCoord(
            x=np.asarray(event.simulation.shower.core_x) * u.m,
            y=np.asarray(event.simulation.shower.core_y) * u.m,
            z=np.asarray(0) * u.m,
            frame=GroundFrame(),
        )

        self.tilt_core_true = grd_core_true.transform_to(
            TiltedGroundFrame(pointing_direction=self.point)
        )

        # transform source direction into nominal system (where we store our templates)
        self.source_direction = src.transform_to(NominalFrame(origin=self.point))

        # Loop over triggered telescopes. Everything beyond here is depends on telescope
        for tel_id, dl1 in event.dl1.tel.items():
            #  Get pixel signal
            if np.all(self.check_parameters(parameters=dl1.parameters)) is False:
                continue

            # First set the the last dict key missing, the impact distance
            impact = (
                np.sqrt(
                    np.power(self.tilt_tel.x[tel_id - 1] - self.tilt_core_true.x, 2)
                    + np.power(self.tilt_tel.y[tel_id - 1] - self.tilt_core_true.y, 2)
                )
                .to(u.m)
                .value
            )

            if self.image_computation or self.time_computation:
                x, y = self.get_rotated_translated_pixel_positions(tel_id)

                geom = self.event_source.subarray.tel[tel_id].camera.geometry

                mask = dl1.image_mask

                for _ in range(4):
                    mask = dilate(geom, mask)

                # Apply mask
                x = x[mask].astype(np.float32)
                y = y[mask].astype(np.float32)

                pmt_signal = dl1.image
                image = pmt_signal[mask].astype(np.float32)

            if self.time_computation:
                peak_times = dl1.peak_time[mask]

                time_mask = np.logical_and(peak_times > 0, np.isfinite(peak_times))
                time_mask = np.logical_and(time_mask, image > 5)

                if np.sum(time_mask) > 3:
                    time_slope = lts_linear_regression(
                        x=x[time_mask].to_value(u.deg).astype(np.float64),
                        y=peak_times[time_mask].astype(np.float64),
                        samples=3,
                    )[0][0]
                else:
                    time_slope = None

            # Now fill up our output with the X, Y and amplitude of our pixels
            key = pt_zen, pt_az, energy.value, int(impact), x_diff_bin, offset_bin

            if (key) in self.key_list:
                # Extend the list if an entry already exists
                if self.image_computation:
                    self.templates[key].extend(image)
                    self.templates_xb[key].extend(x.value.tolist())
                    self.templates_yb[key].extend(y.value.tolist())
                if self.fraction_computation:
                    self.count[key] = self.count[key] + (
                        1 * self.xmax_scale[(x_diff_bin, offset_bin)]
                    )
                if self.time_computation and time_slope is not None:
                    self.time_slope[key].append(time_slope)
            else:
                self.key_list.append((key))
                if self.image_computation:
                    self.templates[key] = image.tolist()
                    self.templates_xb[key] = x.value.tolist()  # .value#.tolist()
                    self.templates_yb[key] = y.value.tolist()  # .value#.tolist()
                if self.fraction_computation:
                    self.count[key] = 1 * self.xmax_scale[(x_diff_bin, offset_bin)]
                if self.time_computation:
                    if time_slope is not None:
                        self.time_slope[key] = [time_slope]
                    else:
                        self.time_slope[key] = []

    def get_rotated_translated_pixel_positions(self, tel_id):
        """
        Get pixel coordinates in the nominal frame and rotate and translate
        so the shower source lies in camera center and the shower axis is oriented along the x-axis.

        :param tel_id: int
            ID number of the telescope to apply the operation to

        :return: x,y: list,list
            Rotated and translated x and y pixel positions in the nominal frame
        """

        geom = self.event_source.subarray.tel[tel_id].camera.geometry

        fl = geom.frame.focal_length.to(u.m)

        camera_coord = SkyCoord(
            x=geom.pix_x,
            y=geom.pix_y,
            frame=CameraFrame(focal_length=fl, telescope_pointing=self.point),
        )

        nom_coord = camera_coord.transform_to(NominalFrame(origin=self.point))

        x = nom_coord.fov_lon.to(u.deg)
        y = nom_coord.fov_lat.to(u.deg)

        # Calculate expected rotation angle of the image
        phi = (
            np.arctan2(
                (self.tilt_tel.y[tel_id - 1] - self.tilt_core_true.y),
                (self.tilt_tel.x[tel_id - 1] - self.tilt_core_true.x),
            )
            + 90 * u.deg
        )

        x, y = rotate_translate(
            x, y, self.source_direction.fov_lon, self.source_direction.fov_lat, phi
        )
        x *= -1  # Reverse x axis to fit HESS convention
        x, y = x.ravel(), y.ravel()

        return x, y

    def generate_time_templates(self):
        """Generate and save templates of the expected time gradient and its variance of
            the image along the shower axis for all shower parameter grid points.
        """
        time_slope_template = {}
        for key in tqdm(list(self.time_slope.keys())):
            time_slope_list = np.asarray(self.time_slope[key])
            time_slope_list = time_slope_list[~np.isnan(time_slope_list)]
            if len(time_slope_list) > 5:
                time_slope_template[key] = np.array(
                    (
                        scipy.stats.trim_mean(time_slope_list, 0.01),
                        scipy.stats.mstats.trimmed_std(time_slope_list, 0.01),
                    )
                )

        file_handler = gzip.open(self.output_file + "_time.template.gz", "wb")

        final_out_dict={}
        final_out_dict["data"]=time_slope_template
        final_out_dict["tel_type"]=self.telescope_type
        pickle.dump(final_out_dict, file_handler)
        file_handler.close()

    def generate_fraction_templates(self):
        """Generate and save templates of the expected telescope trigger probability
           for all shower parameter grid points.
        """
        fraction = {}
        for key in self.count.keys():
            fraction[key] = self.count[key] / self.count_total
        file_handler = gzip.open(self.output_file + "_fraction.template.gz", "wb")
        final_out_dict={}
        final_out_dict["data"]=fraction
        final_out_dict["tel_type"]=self.telescope_type
        pickle.dump(final_out_dict, file_handler)
        file_handler.close()


def main():
    """run the tool"""
    print(
        "======================================================================================="
    )
    tprint("Template   Fitter")
    print(
        "======================================================================================="
    )

    tool = TemplateFitter()
    tool.run()


if __name__ == "__main__":
    main()
