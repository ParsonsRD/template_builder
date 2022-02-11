from template_builder.fit_templates import TemplateFitter, find_nearest_bin
import numpy as np
import pkg_resources
from template_builder.utilities import *
from template_builder.extend_templates import *
import astropy.units as u

def test_template_read():
    # Now lets check out our reading Chain

    # Create our fitter object
    fitter = TemplateFitter()

    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    assert data_dir is not None
    data_dir += "/gamma_HESS_example.simhess.gz"

    # Read in the file
    fitter.read_templates(data_dir)
    amp, raw_x, raw_y = fitter.templates, fitter.templates_xb, fitter.templates_yb
    # First check our output is empty
    assert amp.keys() is not None

    # Check our template parameters are correct
    assert list(amp.keys())[0][0] == 0.  # Alt
    assert list(amp.keys())[0][1] == 0.  # Az
    assert list(amp.keys())[0][2] == 1.  # Energy

    # Can't be sure of the exact template content, but at least check the values make
    # sense
    test_template = (0., 0., 1., 0., 50., 0.)
    assert np.max(amp[test_template]) > 100.  # Max amplitude is reasonable
    # Average y value is about 0.
    assert np.average(raw_y[test_template], weights=amp[test_template]) < 0.05


def test_template_fitting():
    # Now lets check out our reading Chain

    # Create our fitter object
    fitter = TemplateFitter(min_fit_pixels=0)
    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    data_dir += "/gamma_HESS_example.simhess.gz"

    # Read in the file
    fitter.read_templates(data_dir)
    amp, raw_x, raw_y = fitter.templates, fitter.templates_xb, fitter.templates_yb

    test_template = (0., 0., 1., 0., 50., 0.)

    template = fitter.fit_templates(
        {test_template: amp[test_template]},
        {test_template: raw_x[test_template]},
        {test_template: raw_y[test_template]})

    assert template is not None

    x = np.linspace(fitter.bounds[0][0], fitter.bounds[0][1], fitter.bins[0])
    y = np.linspace(fitter.bounds[1][0], fitter.bounds[1][1], fitter.bins[1])

    # Make sure the template is the expected shape
    assert template[test_template].shape[0] == fitter.bins[1]
    assert template[test_template].shape[1] == fitter.bins[0]

        #assert var_template[test_template].shape[0] == fitter.bins[1]
        #assert var_template[test_template].shape[1] == fitter.bins[0]

        # For now we will assume the fit just works

    # Finally we will check that the range extension functions work
    extended_template = extend_xmax_range(fitter.xmax_bins, template)
    xmax_range = np.array(list(extended_template.keys())).T[4]

    # Check the bins are right
    assert np.sort(xmax_range).all() == fitter.xmax_bins.all()
    # And that all templates are the same
    for key in extended_template:
        assert extended_template[key].all() == template[test_template].all()

    template = {test_template: template[test_template]}
    # Finally check the distance extension works
    template[0., 0., 1., 50., 50., 0.] = template[test_template]
    template[0., 0., 1., 100., 0., 0.] = template[test_template]
    template[0., 0., 1., 200., 0., 0.] = template[test_template]

    extended_template = extend_distance_range(fitter.xmax_bins, template)
    assert (0., 0., 1., 100., 50., 0.) in extended_template
    assert (0., 0., 1., 200., 50., 0.) in extended_template


def test_full_fit():
    # Finally check everything

    # Create our fitter object
    fitter = TemplateFitter(min_fit_pixels=300, verbose=False)
    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    data_dir += "/gamma_HESS_example.simhess.gz"

    # Run full template generation
    fitter.generate_templates([data_dir], "./test.template.gz", "./test_fraction.template.gz", "./test_time_slope.template.gz", max_events=10)

    import os.path
    os.path.isfile("./test.template.gz")
    os.path.isfile("./test_time_slope.template.gz")
    os.path.isfile("./test_fraction.template.gz")

    # Open our output files
    import pickle, gzip
    template_fromfile = pickle.load(gzip.open("./test.template.gz","r"))
    fraction_fromfile = pickle.load(gzip.open("./test_fraction.template.gz","r"))
    time_slope_fromfile = pickle.load(gzip.open("./test_time_slope.template.gz","r"))

    os.remove("./test.template.gz")
    os.remove("./test_time_slope.template.gz")
    os.remove("./test_fraction.template.gz")

#test_template_read()
#test_template_fitting()
#test_full_fit()
