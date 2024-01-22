Usage
=====

Installation
------------

First, clone the *template_builder* repository  

.. code-block:: console

    git clone https://github.com/ParsonsRD/template_builder.git

Then create a conda environment

.. code-block:: console

    conda env create -f environment.yml

and install the *template_builder* package

.. code-block:: console

    pip install -e .

If you want to create the input simulations to generate the templates yourself, be it with or without the scripts
provided with this package, you need to also install CORSIKA and sim_telarray. For instructions on how to do this, 
we refer to the `sim_telarray website <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/>`_.
    

Creating and merging templates
------------------------------

Assuming you have generated simulations according to the recipe described under :doc:`input_simulations <input_simulations>`, you have a number of simulation
files each for simulations at a fixed energy. It is advisable to only merge the simulation files with the same energy and then generate 
the templates (which are just python dictionaries) in parallel and then merge the dictionaries into one large template dictionary in the end.

For the generation of all three templates from a single simulation file, use the *template-fitter* tool to be found in template_fitter.py:

.. code-block:: console

    template-fitter --input events.simtel.gz --output ./Template

There are also options to only generate a subset of the three templates or to adjust the Xmax binning. As a ctapipe tool, template-fitter also permits all configuration
options for the CameraCalibrator and ImageProcessor classes of ctapipe which perform the processing of the simulations to dl1 images.

Having generated the template files for all simulation files, they can be merged with the template-merger tool to be found in merge_templates.py

.. code-block:: console

    template-merger --TemplateMerger.input_dir /path/to/individual/templates/ --TemplateMerger.file_pattern *.template.gz --TemplateMerger.output_file ./Merged_Template