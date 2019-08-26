from template_builder.fit_templates import TemplateFitter, find_nearest_bin
import numpy as np
import pkg_resources


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


def test_template_read():
    # Now lets check out our reading Chain

    # Create our fitter object
    fitter = TemplateFitter()

    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    assert data_dir is not None
    data_dir += "gamma_HESS_example.simhess.gz"

    # Read in the file
    amp, raw_x, raw_y = fitter.read_templates(data_dir)

    # First check our output is empty
    assert amp.keys() is not None

    # Check our template parameters are correct
    assert list(amp.keys())[0][0] == 0.  # Alt
    assert list(amp.keys())[0][1] == 0.  # Az
    assert list(amp.keys())[0][2] == 1.  # Energy

    # Can't be sure of the exact template content, but at least check the values make
    # sense
    test_template = (0., 0., 1., 0., 50.)
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
    data_dir += "gamma_HESS_example.simhess.gz"

    # Read in the file
    amp, raw_x, raw_y = fitter.read_templates(data_dir)
    test_template = (0., 0., 1., 0., 50.)

    # Then lets fit our example template using the different options
    fit_options = ["sklearn", "KNN"]
    for option in fit_options:
        fitter.training_library = option

        template, var_template = fitter.fit_templates(
            {test_template: amp[test_template]},
            {test_template: raw_x[test_template]},
            {test_template: raw_y[test_template]}, True, 1000)

        assert template is not None
        assert var_template is not None

        x = np.linspace(fitter.bounds[0][0], fitter.bounds[0][1], fitter.bins[0])
        y = np.linspace(fitter.bounds[1][0], fitter.bounds[1][1], fitter.bins[1])

        # Make sure the template is the expected shape
        assert template[test_template].shape[0] == fitter.bins[1]
        assert template[test_template].shape[1] == fitter.bins[0]

        assert var_template[test_template].shape[0] == fitter.bins[1]
        assert var_template[test_template].shape[1] == fitter.bins[0]

        # For now we will assume the fit just works

    # Finally we will check that the range extension functions work
    extended_template = fitter.extend_xmax_range(template)
    xmax_range = np.array(list(extended_template.keys())).T[4]

    # Check the bins are right
    assert np.sort(xmax_range).all() == fitter.xmax_bins.all()
    # And that all templates are the same
    for key in extended_template:
        assert extended_template[key].all() == template[test_template].all()

    template = {test_template: template[test_template]}
    # Finally check the distance extension works
    template[0., 0., 1., 50., 50.] = template[test_template]
    template[0., 0., 1., 100., 0.] = template[test_template]
    template[0., 0., 1., 200., 0.] = template[test_template]

    extended_template = fitter.extend_distance_range(template)
    assert (0., 0., 1., 100., 50.) in extended_template
    assert (0., 0., 1., 200., 50.) in extended_template


def test_full_fit():
    # Finally check everything

    # Create our fitter object
    fitter = TemplateFitter(min_fit_pixels=0, training_library="KNN")
    # Get our example data file (10 events of 1 TeV at 0 Alt, 0 Az)
    data_dir = pkg_resources.resource_filename('template_builder', 'data/')
    # Which needs to actually be there
    data_dir += "gamma_HESS_example.simhess.gz"

    # Run full template generation
    template, var_template = fitter.generate_templates([data_dir], "./test.template.gz",
                                                       "./test_var.template.gz", True)

    # Make sure we get something out
    assert template is not None
    assert var_template is not None

    import os.path
    os.path.isfile("./test.template.gz")
    os.path.isfile("./test_var.template.gz")

    # Open our output files
    import pickle, gzip
    template_fromfile = pickle.load(gzip.open("./test.template.gz","r"))
    var_template_fromfile = pickle.load(gzip.open("./test_var.template.gz","r"))

    # And check the contents are the same
    for key in template:
        assert template[key].all() == template_fromfile[key].all()
        assert var_template[key].all() == var_template_fromfile[key].all()
