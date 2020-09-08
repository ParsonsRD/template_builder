"""
The purpose of this class is to create the input cards required for launching the CORSIKA
jobs used in ImPACT template generation
"""

import numpy as np
from ctapipe.coordinates import *
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u

particle_lookup = {"gamma": "1", "electron": "2", "proton": "3", "nitrogen": "1407",
                   "silicon": "2814", "iron": "5626"}


class CORSIKAInput:

    def __init__(self, input_parameters, energy_scaling=False,
                 event_scaling_index=-1, min_events=200, primary_particle="gamma"):
        """
        Generates CORSIKA input cards for a

        :param input_parameters: dict
            Dictionary of shared CORSIKA input parameters
        :param energy_scaling: bool
            Perform scaling of energies by cos zenith angle
        :param event_scaling_index: float
            Power law index the scale event numbers by energy
        :param min_events: int
            Minimum number of events to simulate
        """
        self.input_parameters = input_parameters
        self.common_input = self.generate_common_input(self.input_parameters)
        self.energy_scaling = energy_scaling
        self.event_scaling_index = event_scaling_index
        self.min_events = min_events
        self.primary_particle = primary_particle

    @staticmethod
    def generate_common_input(input_parameters):
        """
        A simple function to create the common aspects of the CORSIKA input file for
        all energies zenith angles etc

        :param input_parameters: dict()
            dictionary of common input parameters
        :return: str
            common CORSIKA input card entries
        """
        common_input = str()
        for input_line in input_parameters:
            common_input += "%s %s \n" % (input_line.upper(), input_parameters[input_line])

        return common_input

    def simulation_range(self, altitude, azimuth, energy_orig, core_distance, rotation_angle):
        """

        :param altitude: ndarray
            Simulated altitude
        :param azimuth: ndarray
            Simulated azimuth
        :param energy: ndarray
            Simulated energy
        :param core_distance: ndarray
            Telescope core distances required in simulation
        :param rotation_angle: ndarray
            Rotation angle of telescopes (degrees)
        :return: dict
            Dictiontary of telescope position for each simulation required
        """

        arrang = 0
        if "ARRANG" in self.input_parameters.keys():
            arrang = float(self.input_parameters["ARRANG"])

        # First lets make sure everything is an array
        altitude = np.array(altitude)
        azimuth = np.array(azimuth)
        energy_orig = np.array(energy_orig)
        rotation_angle = np.array(rotation_angle)

        # Create x, y positions of telescopes
        xt = np.array(core_distance)
        yt = np.zeros_like(xt)

        xr, yr = list(), list()
        # Rotate telescope positions by each rotation angle requested
        for phi in np.nditer(rotation_angle):
            xr.append(xt * np.cos(np.deg2rad(phi)) - yt * np.sin(np.deg2rad(phi)))
            yr.append(xt * np.sin(np.deg2rad(phi)) + yt * np.cos(np.deg2rad(phi)))

        # Create 1D array of x, y, z positions of each telescope requested
        xr = np.array(xr).ravel()
        yr = np.array(yr).ravel()
        simulation_dict = {}
        # Now loop over altitude, azimuth and energy
        for alt in np.nditer(altitude):
            for az in np.nditer(azimuth):
                # We will need this later for coordinate conversions
                horizon_system = SkyCoord(alt=alt*u.deg, az=az*u.deg - arrang*u.deg,
                                          frame=AltAz())

                # Scale the simulated energies if requested
                if self.energy_scaling:
                   # print( np.cos(np.deg2rad(90-alt)))
                    energy = energy_orig / np.cos(np.deg2rad(90-alt))
                else:
                    energy = energy_orig
#                print("here", self.energy_scaling, energy)

                for en in np.nditer(energy):
                    # We define our core distance in the tilted system, but when we
                    # simulate we do this in the ground system, so we need to
                    # project these values onto the ground
                    tilted_system = SkyCoord(x=xr*u.m, y=yr*u.m,
                                             frame=TiltedGroundFrame(pointing_direction=
                                                                     horizon_system))
                    ground_system = project_to_ground(tilted_system)

                    simulation_dict[(90-float(alt), float(az), float(en))] = \
                        np.array([ground_system.x.value,
                                  ground_system.y.value,
                                  np.zeros_like(ground_system.y.value)]).T

        return simulation_dict

    def create_corsika_input(self, simuation_dict, num_showers,
                             diameter, cherenkov_output):
        """
        Create CORSIKA input cards for each of the simulation sets

        :param simuation_dict: dict
            Dictionary of telescope positions for each simulation set
        :param num_showers: str
            Number of showers to simulate
        :param diameter: float
            Diameter of telescope region
        :param cherenkov_output: str
            Name of sim_telarray run script
        :return: dict
            Dictionary of input cards for each simulation
        """
        card_dict = {}
        for zen, az, en in simuation_dict:

            particle_input = "PRMPAR %s \n" % \
                             (particle_lookup[self.primary_particle.lower()])

            zenith_input = "THETAP %.1f %.1f \n" % (zen, zen)
            azimuth_input = "PHIP %.1f %.1f \n" % (az, az)
            energy_input = "ERANGE %.1f %.1f \n" % (en * 1000, en * 1000)

            num = float(num_showers) * np.power(float(en), self.event_scaling_index)
            if num < self.min_events:
                num = self.min_events

            number_input = "NSHOW %d \n" % (num)

            input_params = particle_input + zenith_input + azimuth_input + energy_input\
                           + number_input
            tel_input = ""
            for tel in simuation_dict[(zen, az, en)]:
                tel_input += "TELESCOPE %.1f %.1f %.1f %.1f\n" % (tel[0] * 100,
                                                                  tel[1] * 100,
                                                                  tel[2] * 100,
                                                                  diameter * 100)

            cherenkov_output_file = "TELFIL " + "|${SIM_TELARRAY_PATH}/"+cherenkov_output

            card_dict[(zen, az, en)] = input_params + tel_input + self.common_input + \
                                       cherenkov_output_file

        return card_dict

    def get_input_cards(self, num_showers, altitude, azimuth,
                        energy, core_distance, rotation_angle, diameter,
                        cherenkov_output):
        """
        Create CORSIKA input cards for a given range of altitude, azimuth, energy,
        core distance and telescope rotation angle

        :param num_showers: float
            Number of showers to simulate (normalised at 1 TeV)
        :param altitude: ndarray
            Simulated altitudes
        :param azimuth: ndarrray
            Simulated azimuths
        :param energy: ndarray
            Simulated energies
        :param core_distance: ndarray
            Simulated core distance
        :param rotation_angle: ndarray
            simulated rotation angles
        :param diameter: float
            Diameter of telescope region
        :param cherenkov_output: str
            Name of sim_telarray run script
        :return: dict
            Dictionary of CORSIKA input cards
        """
        sim_range = self.simulation_range(altitude, azimuth, energy,
                                          core_distance, rotation_angle)

        return self.create_corsika_input(sim_range, num_showers, diameter,
                                         cherenkov_output)
