"""
In order to perform the telescope simulation we need to prepare the correct directory
structures and create some shell scripts that pass the configuration options through
the simulation

This module prepares the correct structure and returns the command for running the
required simulation set and where to look for the results
"""
import shutil
from pathlib import Path


def get_run_script(config_name):
    """

    :param config_name: str
        Name of the configuration
    :return: str
        Name of sim_telarray run script
    """
    return "run_sim_template_" + config_name


class SimTelArrayConfig:

    def __init__(self, config_name, config_file, altitude, atmospheric_profile,
                 optical_efficiency=1, extra_defines="", offsets=[0.0]):
        """
        :param config_name: str
            Name of the configuration
        :param config_file: str
            Location of the telescope config file (relative to base sim_telarray
            directory)
        :param altitude: float
            Observation level altitude above sea level
        :param atmospheric_profile: str
            Name of atmospheric profile to be used in simulations
        :param offsets: list
            List of source offsets to simulate
        """

        # Just copy everything into the class
        self.config_name = config_name
        self.config_file = config_file
        self.atmospheric_profile = atmospheric_profile

        self.extra_defines = extra_defines
        self.optical_efficiency = optical_efficiency
        self.altitude = altitude

        self.offsets = offsets

    def run_setup(self, simtel_directory, corsika_input):
        """
        Create directory structure and return run commands

        :param simtel_directory: str
            Base directory of sim_telarray package
        :param corsika_input: list
            list of locations of CORSIKA input cards to pass through to simulations
        :return: tuple
            List of command to run sim_telarray and where to look for output
        """

        # First set up configuration
        output_paths = self.make_config(simtel_directory)

        # Then generate command to run stuff
        return self.make_run_command(simtel_directory, corsika_input), output_paths

    def make_config(self, simtel_directory):
        """
        Make required config files and directory structure

        :param simtel_directory: str
            Base directory of sim_telarray package

        :return: list
            list of output directories for sim_telarray files
        """

        output_paths = []

        # First make the appropriate directories for the sim_telarray output
        for off in self.offsets:
            path = Path(simtel_directory + "/Data/sim_telarray/"+self.config_name+"/" +
                         str(off)+"deg/Data/")
            path.mkdir(parents=True, exist_ok=True)
            output_paths.append(simtel_directory + "/Data/sim_telarray/"+
                                self.config_name+"/" + str(off)+"deg/Data/")

            path = Path(simtel_directory + "/Data/sim_telarray/"+self.config_name+"/" +
                         str(off)+"deg/Log/")
            path.mkdir(parents=True, exist_ok=True)

            path = Path(simtel_directory + "/Data/sim_telarray/"+self.config_name+"/" +
                         str(off)+"deg/Histograms/")
            path.mkdir(parents=True, exist_ok=True)

        if "/" in self.config_file:
            base_directory = self.config_file.rsplit('/', 1)[0]
        else:
            base_directory = ""

        # Then copy into sim_telarray the config files
        shutil.copy("configs/run_sim_template", simtel_directory + "/sim_telarray/" +
                    get_run_script(self.config_name))
        shutil.copy("configs/cta-temp_run.sh", simtel_directory + "/sim_telarray/" +
                    "/template_run_" + self.config_name + ".sh")
        shutil.copy("configs/array_trigger_temp.dat", simtel_directory + "/sim_telarray/" +
                    base_directory + "/array_trigger_temp.dat")

        # Finally make telescope and multipipe configs
        config = self._make_telescope_configuration(simtel_directory)
        self._make_multi_configuration(simtel_directory, config)

        return output_paths

    def _make_telescope_configuration(self, simtel_directory):
        """
        Create a new configuration file that wraps around the standard telescope config
        file for each telescope included in the simulation
        :param simtel_directory: str
            Base directory of sim_telarray package
        :return: str
            Locations of the telescope configuration
        """
        if "/" in self.config_file:
            base_directory = self.config_file.rsplit('/', 1)[0] + "/"
        else:
            base_directory = ""

        filename = base_directory + \
                   "Template_Configuration_" + self.config_name + ".cfg"
        incfile = "#include " + self.config_file

        f = open(simtel_directory + "/sim_telarray/" + filename, 'w')

        # Take out boilerplate file
        ft = open("configs/simtel_template.cfg", 'r')
        gen = ft.read()
        # And copy in reference to our telescope config
        f.write(gen)
        f.write(incfile)

        return filename

    def _make_multi_configuration(self, simtel_directory, template_config):
        """

        :param simtel_directory: str
            Base directory of sim_telarray package
        :param template_config: str
            Telescope configuration file
        :return:
        """
        filenm = simtel_directory + "/sim_telarray/multi/multi_template_" + \
                   self.config_name + ".cfg"

        f = open(filenm, 'w')

        # Look over offsets to make commands for running simulations
        for off in self.offsets:
            wstr = 'env offset="' + str(off) + '" nsb="0.0" cfg=' + self.config_name + \
                   " cfgfile='" + template_config + "'" + \
                   " transmission=" + self.atmospheric_profile + \
                   " extra_config='-C altitude=" + str(self.altitude) + \
                   " -C MIRROR_DEGRADED_REFLECTION=" + \
                   str(self.optical_efficiency)

            if self.extra_defines is not "":
                wstr += self.extra_defines
            wstr += "'"
            wstr += " ./template_run_" + self.config_name + ".sh"

            f.write(wstr)
            f.write("\n")

        return

    @staticmethod
    def make_run_command(simtel_directory, corsika_input):
        """
        Make command line input to run simulations

        :param simtel_directory: str
            Base directory of sim_telarray package
        :param corsika_input: list
            list of locations of CORSIKA input cards to pass through to simulations
        :return: list
            List of commands
        """
        commands = list()

        for input in corsika_input:
            run_string = "cd " + simtel_directory + "; "\
                         "source examples_common.sh; " \
                         "cd ${CORSIKA_DATA};" \
                         "${SIM_TELARRAY_PATH}/bin/corsika_autoinputs  " \
                         "--run ${CORSIKA_PATH}/corsika " \
                         "-p ${CORSIKA_DATA} " + input

            commands.append(run_string)

        return commands
