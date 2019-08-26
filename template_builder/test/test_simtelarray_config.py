from template_builder.simtelarray_config import SimTelArrayConfig
from pathlib import Path
import os

def test_config_creation():

    config = SimTelArrayConfig(config_name="test_config", config_file="test.cfg",
                               altitude="0", atmospheric_profile="dummy.dat",
                               extra_defines="-DTEST")

    simtel_directory = "./pytest_out"

    path = Path(simtel_directory)
    path.mkdir(parents=True, exist_ok=True)
    path = Path(simtel_directory+"/sim_telarray")
    path.mkdir(parents=True, exist_ok=True)
    path = Path(simtel_directory+"/sim_telarray/multi")
    path.mkdir(parents=True, exist_ok=True)

    run_command = config.run_setup(simtel_directory,[simtel_directory + "/test.card"])

    # First check that the command to run sim_telarray is correct
    assert run_command[0][0] == "cd ./pytest_out; source examples_common.sh; " \
                    "cd ${CORSIKA_DATA};${SIM_TELARRAY_PATH}/bin/corsika_autoinputs  " \
                    "--run ${CORSIKA_PATH}/corsika -p ${CORSIKA_DATA} " \
                    "./pytest_out/test.card"

    assert run_command[1][0] == "./pytest_out/Data/sim_telarray/test_config/0.0deg/Data/"

    # Check that our output directories were created
    assert os.path.exists(simtel_directory +
                          "/Data/sim_telarray/test_config/0.0deg/Data/")
    assert os.path.exists(simtel_directory + "/Data/sim_telarray/test_config/0.0deg/Log/")
    assert os.path.exists(simtel_directory +
                          "/Data/sim_telarray/test_config/0.0deg/Histograms/")

    # Then check our sim telarray files
    assert os.path.isfile(simtel_directory + "/sim_telarray/array_trigger_temp.dat")
    assert os.path.isfile(simtel_directory +
                          "/sim_telarray/Template_Configuration_test_config.cfg")
    assert os.path.isfile(simtel_directory + "/sim_telarray/run_sim_template_test_config")
    assert os.path.isfile(simtel_directory +
                          "/sim_telarray/template_run_test_config.sh")

    # Finally check our multipipe config
    assert os.path.exists(simtel_directory + "/sim_telarray/multi")
    assert os.path.isfile(simtel_directory +
                          "/sim_telarray/multi/multi_template_test_config.cfg")
    multi_file = open(simtel_directory +
                      "/sim_telarray/multi/multi_template_test_config.cfg", "r")
    first_line = multi_file.readline().rstrip()
    assert first_line == "env offset='0.0' nsb='0.0' cfg=test_config " \
                         "cfgfile='Template_Configuration_test_config.cfg' " \
                         "transmission=dummy.dat " \
                         "extra_config='-C altitude=0 -C MIRROR_DEGRADED_REFLECTION=1 " \
                         "-C ARRAY_TRIGGER=array_trigger_temp.dat -DTEST' " \
                         "./template_run_test_config.sh"

    # Finally remove our test directory
    import shutil
    shutil.rmtree(simtel_directory)
