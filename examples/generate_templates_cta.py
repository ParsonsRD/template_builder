from argparse import ArgumentParser
import yaml
from template_builder.corsika_input import CORSIKAInput
from template_builder.simtelarray_config import SimTelArrayConfig, get_run_script
from template_builder.fit_templates import TemplateFitter
import os
from os import listdir


def get_file_list(directories):
    """

    :param directories:
    :return:
    """
    files_before_simulation = []

    for path in directories:
        for file in listdir(path):
            files_before_simulation.append(path+file)

    return files_before_simulation


def parse_config(file_list):
    """

    :param file_list:
    :return:
    """

    corsika_dict = dict()
    simulation_input = dict()
    fit_input = dict()
    telescope_input = dict()

    for f in file_list:
        with open(f[0],"r") as yaml_file:
            yaml_file = yaml.load(yaml_file)

        if "CORSIKA" in yaml_file:
            corsika_dict.update(yaml_file["CORSIKA"])
        if "ShowerSimulation" in yaml_file:
            simulation_input.update(yaml_file["ShowerSimulation"])
        if "TelescopeSimulation" in yaml_file:
            telescope_input.update(yaml_file["TelescopeSimulation"])
        if "Fit" in yaml_file:
            fit_input.update(yaml_file["Fit"])

    return corsika_dict, simulation_input, telescope_input, fit_input


def write_corsika_input_cards(sim_telarray_directory, input_cards):

    input_file_names = list()
    for card in input_cards:
        input_name = "input_altitude%.1f_azimuth%.1f_energy%.3f.input" % card
        file_name = sim_telarray_directory + "/corsika-run/" + input_name
        with open(file_name, "w") as input_file:
            input_file.write(input_cards[card])
            input_file.close()

        input_file_names.append(file_name)

    return input_file_names


def generate_templates():

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='append', nargs=1,
                        metavar="config file",
                        help='Configuration YAML file locations')
    parser.add_argument('-o', '--output', default="test.templates.gz",
                        metavar="output file",
                        help='Name of output file')

    parser.add_argument('--simulate-only', dest='simulate_only', action='store_true')

    args = parser.parse_args()

    corsika_input, simulation_input, telescope_input, fit_input = \
        parse_config(args.config)
    output_file = args.output

    corsika = CORSIKAInput(input_parameters=corsika_input)

    cards = corsika.get_input_cards(simulation_input["event_number"],
                                    simulation_input["altitude"],
                                    simulation_input["azimuth"],
                                    simulation_input["energy_bins"],
                                    simulation_input["core_bins"],
                                    simulation_input["rotation_angle"],
                                    simulation_input["diameter"],
                                    get_run_script(telescope_input["config_name"])
                                    )

    corsika_input_file_names = \
        write_corsika_input_cards(telescope_input["sim_telarray_directory"], cards)

    sim_telarray_config = SimTelArrayConfig(telescope_input["config_name"],
                                            telescope_input["config_file"],
                                            float(corsika_input["OBSLEV"])/100,
                                            telescope_input["atmosphere"]
                                            )

    run_commands, output_paths = \
        sim_telarray_config.run_setup(telescope_input["sim_telarray_directory"],
                                      corsika_input_file_names)

    files_before = get_file_list(output_paths)

    for command in run_commands:
        print("Running", command)
        os.system(command)

    print("Simulations complete")

    files_after = get_file_list(output_paths)
    files_after = list(set(files_after) - set(files_before))

    if len(files_after) == 0:
        print("No new simulation files created! Quitting before fit")
        return

    fitter = TemplateFitter()
    fitter.generate_templates(files_after, output_file, max_events=50000)

    return


if __name__ == "__main__":
    generate_templates()
