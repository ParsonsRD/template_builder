# template_builder

Set of classes to allow the production of image templates for use with the ImPACT Cherenkov telescope event reconstruction. 
For more information about ImPACT see:

https://arxiv.org/abs/1403.2993

https://cta-observatory.github.io/ctapipe/reco/ImPACT.html

Classes are provided to perform the generation of CORSIKA input cards, sim_telarray configuration files and the final neural network fitting of the sim_telarray output. Templates are currently outputted in the ctapipe format, but functions for converting the the H.E.S.S. template format will be provided soon.

[![Build Status](https://travis-ci.com/ParsonsRD/template_builder.svg?branch=master)](https://travis-ci.com/ParsonsRD/template_builder)
