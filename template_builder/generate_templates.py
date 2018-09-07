from argparse import ArgumentParser
import yaml
from template_builder.corsika_input import CORSIKAInput

def parse_config(filelist):
    """

    :param filelist:
    :return:
    """

    corsika_dict = dict()
    simulation_input = dict()
    fit_input = dict()

    for f in filelist:
        with open(f[0],"r") as yaml_file:
            yaml_file = yaml.load(yaml_file)

        if "CORSIKA" in yaml_file:
            corsika_dict.update(yaml_file["CORSIKA"])
        if "Simulation" in yaml_file:
            simulation_input.update(yaml_file["Simulation"])
        if "Fit" in yaml_file:
            fit_input.update(yaml_file["Fit"])

    return corsika_dict, simulation_input, fit_input


def generate_templates():

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='append', nargs=1,
                        metavar="config file",
                        help='Configuration YAML file locations')

    args = parser.parse_args()

    corsika_input, simulation_input, fit_input = parse_config(args.config)

    corsika = CORSIKAInput(input_parameters=corsika_input)
    print(corsika.common_input)

    return


if __name__ == "__main__": generate_templates()
