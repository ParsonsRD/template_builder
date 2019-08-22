from template_builder.fit_templates import TemplateFitter, find_nearest_bin
import numpy as np


def test_bin_finder():

    test_array = np.linspace(0, 100, 11)

    # First test the lower edge
    assert find_nearest_bin(test_array, 0) == 0.
    assert find_nearest_bin(test_array, -100) == 0.

    # Then the middle
    assert find_nearest_bin(test_array, 54.9) == 50.
    assert find_nearest_bin(test_array, 55.1) == 60.

    # Then the Upper edge
    assert find_nearest_bin(test_array, 500) == 100.
