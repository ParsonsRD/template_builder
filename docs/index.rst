.. template_builder documentation master file, created by
   sphinx-quickstart on Tue Aug 20 13:11:35 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the template_builder documentation!
==============================================

**template_builder** is a python library to generate *ImPACT* templates from *CORSIKA/sim_telarray* gamma ray simulations. 

What is ImPACT?
---------------------

ImPACT stands for **Image Pixel-wise fit for Atmospheric Cherenkov Telescopes**, it is
an event reconstruction algorithm originally created for the H.E.S.S. experiment. This
reconstruction algorithm uses the full camera image to perform a likelihood fit of the shower geometry
and energy, thereby extracting the maximum possible performance of the instrument.

There are two extensions to this method currently implemented which provide additional terms to the likelihood: In the first the pixel timing 
gradient across the image is calculated and compared to a likelihood depending on the shower parameters. In the second the trigger probability
for each telescope is calculated as a function of the shower parameters and compared to the actual trigger decisions by the telescopes. 

For each of these, the likelihood needs to be formulated as a function of the shower parameters. This is done using Monte Carlo simulations,
from which templates are generated that contain parameter values that are then used in analytic likelihood functions. 


Image Templates
++++++++++++++++++++++

The image templates are essentially the expected images from a "perfect" and extremely finely pixelated camera,
generated for all possible observing conditions of the shower. They are created by extracting the images from each camera from each event and
sorting them into bins of the shower parameters. Then for each of these bins, a multilayer perceptron function is jointly fit to all images in the bin. 


Time Gradient Templates
+++++++++++++++++++++++

Here, the time gradients of all images of all events are calculated and sorted into bins of the shower parameters.
Then for each bin, the (truncated) mean and standard deviation of the time gradient distribution are calculated.
In the actual fit, these are used to create a gaussian likelihood. 


Trigger Fraction Templates
++++++++++++++++++++++++++

For this,the total number of telescopes triggered for each bin of shower parameters is counted and divided by the number
of telescopes that could have triggered for this set of parameters. Note that the corresponding likelihood is currently not implemented in ctapipe. 


All three template types are in the end python dictionaries that have as keys the values of the shower parameters and as the item the values of the parameters
belonging to this combination of shower parameters that is then used to formulate the likelihoods for the reconstruction. The interpolation between the values of the shower
parameters is handled by the ImPACT reconstruction code.
 
Contents
--------

.. toctree::
   
   input_simulations
   usage
   autoapi/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
