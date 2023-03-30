"""
Generate DL1 (a or b) output files in HDF5 format from {R0,R1,DL0} inputs.
"""
# pylint: disable=W0201
from email.policy import default
import sys

from tqdm.auto import tqdm
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
import numpy as np

from ctapipe.calib import CameraCalibrator, GainSelector
from ctapipe.core import QualityQuery, Tool
from ctapipe.core.traits import List, classes_with_traits, Unicode
from ctapipe.image import ImageCleaner, ImageModifier, ImageProcessor
from ctapipe.image.extractor import ImageExtractor
from ctapipe.reco.reconstructor import StereoQualityQuery

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

__all__ = ["ProcessorTool"]


class TemplateFitter(Tool):
    """
    Process data from lower-data levels up to DL1, including both image
    extraction and optinally image parameterization
    """

    name = "template-fitter"
    examples = """
    To process data with all default values:
    > template-fitter --input events.simtel.gz --output events.dl1.h5 --progress
    Or use an external configuration file, where you can specify all options:
    > template-fitter --config stage1_config.json --progress
    The config file should be in JSON or python format (see traitlets docs). For an
    example, see ctapipe/examples/stage1_config.json in the main code repo.
    """

    input_files = Unicode(
        default_value=".", help="list of input files"
    ).tag(config=True)

    output_file = Unicode(
        default_value=".", help="base output file name"
    ).tag(config=True)

    xmax_bins = List(
        default_value = np.linspace(-150, 200, 15).tolist(),
        help = "bin centres for xmax bins",
    ).tag(config=True)

    offset_bins = List(
        default_value = [0.0],
        help = "bin centres for offset bins (deg)",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "TemplateFitter.input_files",
        ("o", "output"): "TemplateFitter.output_file",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "EventSource.max_events",
        "image-cleaner-type": "ImageProcessor.image_cleaner_type",
    }

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

        # setup components:
        self.input_file_list = self.input_files.split(",")

        self.focal_length_choice='EFFECTIVE'
        try:
            self.event_source = EventSource(input_url=self.input_file_list[0], parent=self, 
                    focal_length_choice=self.focal_length_choice)
        except RuntimeError:
            print("Effective Focal length not availible, defaulting to equivelent")
            self.focal_length_choice='EQUIVALENT'
            self.event_source = EventSource(input_url=self.input_file_list[0], parent=self, 
                    focal_length_choice=self.focal_length_choice)

        if not self.event_source.has_any_datalevel(COMPATIBLE_DATALEVELS):
            self.log.critical(
                "%s  needs the EventSource to provide either R1 or DL0 or DL1A data"
                ", %s provides only %s",
                self.name,
                self.event_source,
                self.event_source.datalevels,
            )
            sys.exit(1)

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

        self.fitter = NNFitter()

        # We need this dummy time for coord conversions later
        self.dummy_time = Time('2010-01-01T00:00:00', format='isot', scale='utc')
        self.point, self.xmax_scale, self.tilt_tel = None, None, None

        self.time_slope, self.templates = {}, {}  # Pixel amplitude
        self.templates_xb, self.templates_yb = {}, {} # Rotated Y positions
        self.count = {} # Count of events in a given template
        self.count_total = 0

    def start(self):
        """
        Process events
        """

        self.event_source.subarray.info(printer=self.log.info)

        for input_file in self.input_file_list:
            self.event_source = EventSource(input_url=input_file, parent=self, 
                focal_length_choice=self.focal_length_choice)
            self.point, self.xmax_scale, self.tilt_tel = None, None, None

            for event in tqdm(
                self.event_source,
                desc=self.event_source.__class__.__name__,
                total=self.event_source.max_events,
                unit="events"
                ):

                self.log.debug("Processessing event_id=%s", event.index.event_id)
                self.calibrate(event)
                self.process_images(event)

                self.read_template(event)
            
            # Not sure what else to do here...
            obs_ids = self.event_source.simulation_config.keys()
            for obs_id in obs_ids:
                self.count_total += self.event_source.simulation_config[obs_id].n_showers
            self.event_source.close()

    def finish(self):
        """
        Last steps after processing events.
        """
        self.fitter.generate_templates(self.templates_xb, self.templates_yb, self.templates,
                                    self.time_slope, self.count, self.count_total, 
                                    output_file=self.output_file)

    def read_template(self, event):
        """_summary_

        Args:
            event (_type_): _description_
        """

        # When calculating alt we have to account for the case when it is rounded
        # above 90 deg
        alt_evt = event.simulation.shower.alt
        if alt_evt > 90 * u.deg:
            alt_evt = 90*u.deg

        # Get the pointing direction and telescope positions of this run
        if self.point is None:
            alt = event.pointing.array_altitude
            if alt > 90 * u.deg:
                alt = 90*u.deg

            self.point = SkyCoord(alt=alt, az=event.pointing.array_azimuth,
                    frame=AltAz(obstime=self.dummy_time))
            self.xmax_scale = create_xmax_scaling(self.xmax_bins, np.array(self.offset_bins)*u.deg, \
                    self.point, self.event_source.input_url)

            grd_tel = self.event_source.subarray.tel_coords
            # Convert to tilted system
            self.tilt_tel = grd_tel.transform_to(
                TiltedGroundFrame(pointing_direction=self.point))

        # Create coordinate objects for source position
        src = SkyCoord(alt=event.simulation.shower.alt.value * u.rad, 
                        az=event.simulation.shower.az.value * u.rad,
                        frame=AltAz(obstime=self.dummy_time))


        offset_bin = find_nearest_bin(np.array(self.offset_bins)*u.deg, self.point.separation(src)).value

        zen = 90 - event.simulation.shower.alt.to(u.deg).value
        # Store simulated Xmax
        mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

        # And transform into nominal system (where we store our templates)
        source_direction = src.transform_to(NominalFrame(origin=self.point))

        # Store simulated event energy
        energy = event.simulation.shower.energy

        # Calculate core position in tilted system
        grd_core_true = SkyCoord(x=np.asarray(event.simulation.shower.core_x) * u.m,
                                    y=np.asarray(event.simulation.shower.core_y) * u.m,
                                    z=np.asarray(0) * u.m, frame=GroundFrame())

        tilt_core_true = grd_core_true.transform_to(TiltedGroundFrame(
            pointing_direction=self.point))

        # Loop over triggered telescopes
        for tel_id, dl1 in event.dl1.tel.items():
            #  Get pixel signal
            if np.all(self.check_parameters(parameters=dl1.parameters)) is False:
                continue

            pmt_signal = dl1.image

            # Get pixel coordinates and convert to the nominal system
            geom = self.event_source.subarray.tel[tel_id].camera.geometry
            fl = self.event_source.subarray.tel[tel_id].optics.equivalent_focal_length

            camera_coord = SkyCoord(x=geom.pix_x, y=geom.pix_y,
                                    frame=CameraFrame(focal_length=fl,
                                                        telescope_pointing=self.point))

            nom_coord = camera_coord.transform_to(
                NominalFrame(origin=self.point))

            x = nom_coord.fov_lon.to(u.deg)
            y = nom_coord.fov_lat.to(u.deg)

            # Calculate expected rotation angle of the image
            phi = np.arctan2((self.tilt_tel.y[tel_id - 1] - tilt_core_true.y),
                                (self.tilt_tel.x[tel_id - 1] - tilt_core_true.x)) + \
                    90 * u.deg

            # And the impact distance of the shower
            impact = np.sqrt(np.power(self.tilt_tel.x[tel_id - 1] - tilt_core_true.x, 2) +
                                np.power(self.tilt_tel.y[tel_id - 1] - tilt_core_true.y, 2)). \
                to(u.m).value
            
            mask = event.dl1.tel[tel_id].image_mask
            # now rotate and translate our images such that they lie on top of one
            # another
            x, y = rotate_translate(x, y,
                                    source_direction.fov_lon,
                                    source_direction.fov_lat,
                                    phi)
            x *= -1 # Reverse x axis to fit HESS convention
            x, y = x.ravel(), y.ravel()
    
            for i in range(4):
                mask = dilate(geom, mask)

            # Make sure everything is 32 bit
            x = x[mask].astype(np.float32)
            y = y[mask].astype(np.float32)
            image = pmt_signal[mask].astype(np.float32)
            time_slope = dl1.parameters.timing.slope.value

            # Store simulated Xmax
            mc_xmax = event.simulation.shower.x_max.value / np.cos(np.deg2rad(zen))

            # Calc difference from expected Xmax (for gammas)
            exp_xmax = xmax_expectation(energy.value)
            x_diff = mc_xmax - exp_xmax
            x_diff_bin = find_nearest_bin(self.xmax_bins, x_diff)

            az = self.point.az.to(u.deg).value
            zen = 90. - self.point.alt.to(u.deg).value
            
            # Now fill up our output with the X, Y and amplitude of our pixels
            key = zen, az, energy.value, int(impact), x_diff_bin, offset_bin

            if (key) in self.templates.keys():
                # Extend the list if an entry already exists
                self.templates[key].extend(image)
                self.templates_xb[key].extend(x.value.tolist())
                self.templates_yb[key].extend(y.value.tolist())
                self.count[key] = self.count[key] + (1  * self.xmax_scale[(x_diff_bin, offset_bin)])
                self.time_slope[key].append(time_slope)
            else:
                self.templates[key] = image.tolist()
                self.templates_xb[key] = x.value.tolist()#.value#.tolist()
                self.templates_yb[key] = y.value.tolist()#.value#.tolist()
                self.count[key] = 1 * self.xmax_scale[(x_diff_bin, offset_bin)]
                self.time_slope[key] = [time_slope]


def main():
    """run the tool"""
    print("=======================================================================================")
    tprint("Template   Fitter")
    print("=======================================================================================")

    tool = TemplateFitter()
    tool.run()

if __name__ == "__main__":
    main()