"""
This file is a simple example of how to generate templates through all sets of simulation.
This should really only be taken as and example of how to use the classes provided,
although should probably work without too much hassle on most systems
"""
from argparse import ArgumentParser
import yaml
from template_builder.corsika_input import CORSIKAInput
from template_builder.simtelarray_config import SimTelArrayConfig, get_run_script
from template_builder.nn_fitter import TemplateFitter
from template_builder.extend_templates import *

import astropy.units as u
import os
from os import listdir
import numpy as np
from tqdm import tqdm
import pickle
import gzip

def get_file_list(directories):
    """
    Get the absolute locations of all files in a list of directories

    :param directories: list
        list of directories to check
    :return: list
        List of absolute file names
    """
    files_before_simulation = []

    for path in directories:
        for file in listdir(path):
            files_before_simulation.append(path+file)

    return files_before_simulation


def parse_config(file_list):
    """
    Parse required options from a list of configuration files

    :param file_list: list
        list of config files to parse
    :return: tuple
        Dictionaries of the required simulation inputs
    """

    corsika_dict = dict()
    simulation_input = dict()
    fit_input = dict()
    telescope_input = dict()

    # Loop over input config files
    for f in file_list:
        # Open YAML file
        with open(f[0],"r") as yaml_file:
            yaml_file = yaml.safe_load(yaml_file)

        # Get the config options if they exist
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
    """
    Save CORSIKA input cards from a dictionary in the directory provided

    :param sim_telarray_directory: str
        Base directory of sim_telarray package
    :param input_cards:  dict

    :return:
    """
    input_file_names = list()
    for card in input_cards:
        input_name = "input_altitude%.1f_azimuth%.1f_energy%.3f.input" % card
        file_name = sim_telarray_directory + "/corsika-run/" + input_name
        with open(file_name, "w") as input_file:
            input_file.write(input_cards[card])
            input_file.close()

        input_file_names.append(file_name)

    return input_file_names

def produce_sorted_list(filelist, zenith_list, azimuth_list, energy_list):
    zen_list, en_list, az_list = [], [], []

    for f in filelist:
        line = f.split("/")[-1]
        line = np.array(line.split("_"))
        try:
            zen_num = np.where(np.char.find(line, "deg") > -1)[0][0]
            az_num = np.where(np.char.find(line, "deg") > -1)[0][1]
            try:
                en_num = np.where(np.char.find(line, "energy") > -1)[0][0]
            except:
                en_num = 8

            zen = line[zen_num][:-3]
            az = line[az_num][:-3]

            if "energy" in line[en_num]:
                energy = line[en_num][6:]
            else:
                energy = line[en_num]
            zen_list.append(zen)
            en_list.append(energy)
            az_list.append(az)
        except:
            a=1

    en_list = np.unique(en_list).astype("float32")
    en_list = np.sort(en_list).astype("str")
    zen_list = np.unique(zen_list)
    zen_list = np.sort(zen_list)
    az_list = np.unique(az_list)
    az_list = np.sort(az_list)

    filelist = np.array(filelist)

    output_string = {}

    for zen in zen_list:
        for en in en_list:
            for az in az_list:

                if np.isclose(float(en), energy_list, rtol=0.01).any() and \
                   np.isclose(float(zen), zenith_list, rtol=0.01).any() and \
                   np.isclose(float(az), azimuth_list, rtol=0.01).any():

                    zen_sel = np.char.find(filelist, "_"+zen+"deg_"+az+"deg") > -1
                    en_sel = np.char.find(filelist, en) > -1

                    sel = np.logical_and(zen_sel, en_sel)
                    output_string[zen, az, en] = filelist[sel]#list_string
    return output_string

def merge_files(file_list, base_name):
    """
    Merge the contents of multiple files into a single output file for each zenith angle

    Args:
        file_list ([type]): [description]
    """

    zeniths = np.unique(np.array(file_list.keys().T[0]))

    for zen_file in zeniths:
        output_dict = {}

        for zen, az, en in tqdm(file_list.keys()):
            if zen != zen_file:
                continue
            
            file_dict = pickle.load(gzip.open(file_list[zen, az, en]))
            output_dict.update(file_dict)
        output_dict = extend_template_coverage(np.linspace(-150, 200, 15), output_dict)

        for key in output_dict:
            output_dict[key] =  output_dict[key].T

        pickle.dump(output_dict, gzip.open(base_name + "_" + str(zen_file) + "deg.template.gz","w"))

    return None

def generate_templates():
    """
    main() function to call all steps of template production in series

    :return: None
    """

    # First Lets parse the command line
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', action='append', nargs=1,
                        metavar="config file",
                        help='Configuration YAML file locations')
    parser.add_argument('-o', '--output', default="test.templates.gz",
                        metavar="output file",
                        help='Name of output file')
    parser.add_argument('-s', '--split', default="1",
                        metavar="split",
                        help='Number of simulation points to split each run into')

    parser.add_argument('--simulate-only', dest='simulate_only', action='store_true')
    parser.add_argument('--SGE', dest='SGE', action='store_true')

    parser.add_argument('--analyse-only', dest='analyse_only', action='store_true')
    parser.add_argument('--analyse-all', dest='analyse_all', action='store_true')
    parser.add_argument('--merge_zenith', dest='merge_zenith', action='store_true')
    parser.add_argument('--DST', dest='dst', action='store_true')

    args = parser.parse_args()

    # Followed by any config files
    corsika_input, simulation_input, telescope_input, fit_input = \
        parse_config(args.config)
    output_file = args.output
    split_simulations = int(args.split)


    # Generate our range of CORSIKA input cards
    corsika = CORSIKAInput(input_parameters=corsika_input,
                           min_events=int(simulation_input["min_events"]/split_simulations))

    cards = corsika.get_input_cards(int(simulation_input["event_number"]/split_simulations),
                                    simulation_input["altitude"],
                                    simulation_input["azimuth"],
                                    simulation_input["energy_bins"],
                                    simulation_input["core_bins"],
                                    simulation_input["rotation_angle"],
                                    simulation_input["diameter"],
                                    get_run_script(telescope_input["config_name"])
                                )
    # And write them in the sim_telarray directory
    corsika_input_file_names = \
                               write_corsika_input_cards(telescope_input["sim_telarray_directory"], cards)

    # Then create the required sim_telearray telescope config files
    sim_telarray_config = SimTelArrayConfig(telescope_input["config_name"],
                                            telescope_input["config_file"],
                                            float(corsika_input["OBSLEV"])/100,
                                            telescope_input["atmosphere"],
                                            telescope_input["optical_efficiency"],
                                            telescope_input["extra_options"]
    )

    run_commands, output_paths = \
                                 sim_telarray_config.run_setup(telescope_input["sim_telarray_directory"],
                                                               corsika_input_file_names,
                                                               split_simulations=split_simulations)

    # Annoyingly sim_telarray doesn't let us choose our output file name (at least in
    # this script setup). So we instead look in output directory now and after our
    # simulations are complete and take the new files
    files_before = get_file_list(output_paths)
    if args.SGE:
        try:
            from submit_SGE import SubmitSGE
        except ImportError:
            print("submit_SGE package required for cluster submission")

    if not args.analyse_only:
        # Submit to SGE cluster if we can
        if args.SGE:
            submit = SubmitSGE(queue_name="std.q",  maximum_jobs=300, extra_options="-l h_rt=24:00:00 -l h_rss=16000M")
#            print(run_commands)
            submit.submit_job_list(run_commands,
                                   telescope_input["config_name"]+"_temp")

            # Otherwise run on the command line
        else:
            for command in run_commands:
                print("Running", command)
                os.system(command)

        print("Simulations complete")

    if args.dst:
        print(output_paths)
        for i in range(len(output_paths)):
            output_paths[i] += "/DST0/"
        print(output_paths)

    files_after = get_file_list(output_paths)
    # Create a list of newly created files
    if args.analyse_all:
        files_after = files_before
    else:
        files_after = list(set(files_after) - set(files_before))

    if len(files_after) == 0:
        print("No new simulation files created! Quitting before fit")
        return


    analysis_list = produce_sorted_list(files_after,
                                        90 - np.array(simulation_input["altitude"], dtype="int"),
                                        np.array(simulation_input["azimuth"], dtype="int"),
                                        np.array(simulation_input["energy_bins"], dtype="float"))

    # Then generate our templates from these
    fitter = TemplateFitter(min_fit_pixels=fit_input["min_fit_pixels"],
                            eff_fl=fit_input["eff_fl"],
                            bins=fit_input["bins"],
                            bounds=fit_input["bounds"],
                            offset_bins=fit_input["offset"] * u.deg,
                            tailcuts=fit_input["tailcuts"],
                            min_amp=fit_input["min_amp"],
                            local_distance_cut=fit_input["local_distance_cut"] * u.deg,
                            gain_threshold=3000)

    output_dir = os.getcwd() + "/output_temporary/"
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        print("Output directory exists")

    if args.SGE:
        submit = SubmitSGE(queue_name="std.q",  maximum_jobs=300, extra_options="-l h_rt=24:00:00 -l h_rss=16000M")
        input_dir = os.getcwd() + "/input_temporary/"
        try:
            os.mkdir(input_dir)
        except FileExistsError:
            print("Temporary input directory exists")

    command_list = []
    script_location = os.getcwd() + "/" + __file__

    file_dict = {}
    # Loop over our list of templates
    for zen, az, en  in analysis_list:
        input_files = analysis_list[zen, az, en]
        output_file = output_dir + "output_"+str(zen)+"deg_"+str(az)+"az_"+str(en)+"TeV.template.gz"
        file_dict[zen, az, en] = output_file
        if args.SGE:
            #Do stuff
            input_file = input_dir + "config_"+str(zen)+"deg_"+str(az)+"az_"+str(en)+"TeV.yaml"

            data = {}
            data["TelescopeSimulation"] = telescope_input
            simulation_input["altitude"] = 90 - float(zen)
            simulation_input["azimuth"] = float(az)
            simulation_input["energy_bins"] = float(en)

            data["ShowerSimulation"] = simulation_input
            data["Fit"] = fit_input
            data["CORSIKA"] = corsika_input

            with open(input_file, 'w') as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

            command = "cd " + os.getcwd() + ";"
            command += " python " + script_location + " -c " + input_file  + " --analyse-only --analyse-all"
            command_list.append(command)
        else:
            fitter.generate_templates(input_files, output_file, max_events=100000)


    if args.SGE:
        submit.submit_job_list(command_list, "template_fitter")

    if args.merge_zenith:
        merge_files(file_dict, telescope_input["config_name"])

    return


if __name__ == "__main__":
    generate_templates()