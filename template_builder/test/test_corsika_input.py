from template_builder.corsika_input import CORSIKAInput
import numpy as np

input = {"OBSLVL": "2150"}
corsika_input = CORSIKAInput(input)


def test_standard_input():

    assert corsika_input.common_input.rstrip() == "OBSLVL 2150"


def test_simulation_range():

    # First test format is correct
    sim = corsika_input.simulation_range(90, 0, 1, [0, 100, 200], 0)
    assert np.allclose(sim[(0, 0, 1)], [[0, 0, 0], [100, 0, 0], [200, 0, 0]])

    # Then check rotation works OK
    sim = corsika_input.simulation_range(90, 0, 1, [0, 100, 200], 90)
    assert np.allclose(sim[(0, 0, 1)], [[0, 0, 0], [0, 100, 0], [0, 200, 0]])


def test_corsika_input():
    # Check our output format is correct
    sim = corsika_input.create_corsika_input({(0, 0, 1):np.array([[0, 0, 0],
                                                                  [100, 0, 0]])},
                                             1000, 10, "test_script")
    assert sim[(0, 0, 1)] == 'PRMPAR 1 \n' \
                             'THETAP 0.0 0.0 \nPHIP 0.0 0.0 \n' \
                              'ERANGE 1000.0 1000.0 \n' \
                              'NSHOW 1000 \n' \
                              'TELESCOPE 0.0 0.0 0.0 1000.0\n' \
                              'TELESCOPE 10000.0 0.0 0.0 1000.0\n' \
                              'OBSLVL 2150 \n' \
                              'TELFIL |${SIM_TELARRAY_PATH}/test_script'


def test_full_process():
    # Finally test everything in one step
    cards = corsika_input.get_input_cards(1000, 90, 0, 1, [0, 100, 200], 0, 10.0,
                                          "test_script")
    assert cards[(0, 0, 1.0)] == 'PRMPAR 1 \n' \
                               'THETAP 0.0 0.0 \n' \
                               'PHIP 0.0 0.0 \n' \
                               'ERANGE 1000.0 1000.0 \n' \
                               'NSHOW 1000 \n' \
                               'TELESCOPE 0.0 0.0 0.0 1000.0\n' \
                               'TELESCOPE 10000.0 0.0 0.0 1000.0\n' \
                               'TELESCOPE 20000.0 0.0 0.0 1000.0\n' \
                               'OBSLVL 2150 \n' \
                               'TELFIL |${SIM_TELARRAY_PATH}/test_script'

