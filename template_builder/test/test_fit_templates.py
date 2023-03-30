from template_builder.nn_fitter import NNFitter, find_nearest_bin
import numpy as np
import pkg_resources
from template_builder.utilities import *
from template_builder.extend_templates import *
import astropy.units as u
from template_builder.template_fitter import TemplateFitter
import itertools

def test_template_read():
    # Now lets check out our reading Chain

    # Create our fitter object
    fitter = TemplateFitter()

    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    assert data_dir is not None
    data_dir += "/gamma_HESS_example.simhess.gz"
    fitter.input_files = data_dir

    fitter.setup()

    # Read in the file
    fitter.start()
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
    fitter = TemplateFitter()
    nnfitter = NNFitter()

    fitter.min_fit_pixels=0
    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    data_dir += "/gamma_HESS_example.simhess.gz"

    # Read in the file
    fitter.input_files = data_dir
    fitter.setup()
    # Read in the file
    fitter.start()
    amp, raw_x, raw_y = fitter.templates, fitter.templates_xb, fitter.templates_yb

    test_template = (0., 0., 1., 0., 50., 0.)

    template = nnfitter.fit_templates(
        {test_template: amp[test_template]},
        {test_template: raw_x[test_template]},
        {test_template: raw_y[test_template]})

    assert template is not None

    # Make sure the template is the expected shape
    assert template[test_template].shape[0] == nnfitter.bins[1]
    assert template[test_template].shape[1] == nnfitter.bins[0]

    # For now we will assume the fit just works

    # Finally we will check that the range extension functions work
    extended_template = extend_xmax_range(fitter.xmax_bins, template)
    xmax_range = np.array(list(extended_template.keys())).T[4]

    # Check the bins are right
    assert np.sort(xmax_range).all() == np.array(fitter.xmax_bins).all()
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
    fitter = TemplateFitter()

    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    assert data_dir is not None
    data_dir += "/gamma_HESS_example.simhess.gz"
    fitter.input_files = data_dir
    fitter.output_file = "./test"

    fitter.setup()
    # Read in the file
    fitter.start()
    keys = fitter.templates_xb.keys()
    txb, tyb, tb, tsb, cb = {}, {}, {}, {}, {}

    # Just grab 5 templates so this doesn't take too long
    for key in list(keys)[:5]:
        txb[key] = fitter.templates_xb[key]
        tyb[key] = fitter.templates_yb[key]
        tb[key] = fitter.templates[key]
        tsb[key] = fitter.time_slope[key]
        cb[key] = fitter.count[key]

    fitter.templates_xb = txb
    fitter.templates_yb = tyb
    fitter.templates = tb
    fitter.time_slope = tsb
    fitter.count = cb

    fitter.finish()

    import os.path
    os.path.isfile("./test.template.gz")
    os.path.isfile("./test_corrected.template.gz")
    os.path.isfile("./test_time.template.gz")
    os.path.isfile("./test_fraction.template.gz")

    # Open our output files
    import pickle, gzip
    template_fromfile = pickle.load(gzip.open("./test.template.gz","r"))
    assert len(template_fromfile) == 5

    template_corr_fromfile = pickle.load(gzip.open("./test_corrected.template.gz","r"))
    assert len(template_corr_fromfile) == 5

    fraction_fromfile = pickle.load(gzip.open("./test_fraction.template.gz","r"))
    assert len(fraction_fromfile) == 5

    os.remove("./test.template.gz")
    os.remove("./test_corrected.template.gz")
    os.remove("./test_time.template.gz")
    os.remove("./test_fraction.template.gz")

#test_template_read()
#test_template_fitting()
test_full_fit()
