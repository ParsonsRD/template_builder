from template_builder.utilities import *
import numpy as np
import astropy.units as u
import pkg_resources
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time

# Simple test of xmax function
def test_xmax_expectation():
    assert xmax_expectation(1) == 300

# Make a simple test that our bin finder is working as we expect
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

def test_angular_scaling():

    scaling = create_angular_area_scaling(np.array([0.0])*u.deg, 1*u.deg)
    assert scaling[0.0] == 1.

    scaling = create_angular_area_scaling(np.array([0.5, 1.5])*u.deg, 2*u.deg)
    assert scaling[0.5] == 4.

# Check on the xmax scaling function
def test_xmax_scaling():

    dummy_time = Time('2010-01-01T00:00:00', format='isot', scale='utc')

    xmax_bins = np.linspace(-150, 200, 15)
    offset_bins = np.array([0.5, 1.]) * u.deg
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    data_dir += "/gamma_HESS_example.simhess.gz"

    point = SkyCoord(alt=90*u.deg, az=0*u.deg,
                    frame=AltAz(obstime=dummy_time))

    scale = create_xmax_scaling(xmax_bins, offset_bins, point, data_dir)
    # Make sure the values in the dictionary are in a reasonable range
    for key in scale:
        assert 1./scale[key] >= 0.
        assert 1./scale[key] <= 1.

def test_likelihood():

    signal = np.array([10,10,10,10,10])
    prediction = np.array([1,5,10,15,20])
    likelihood = poisson_likelihood_gaussian(signal, prediction)

    assert np.argmin(likelihood) == 2
