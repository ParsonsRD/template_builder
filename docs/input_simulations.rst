Input simulations
=================

Generating templates requires a very specific simulation setup in order to 
efficiently generate shower images on a grid in the parameter space of shower energies, impact distances and depth of maximum.

The most efficient setup that is implicitly assumed by this code is the following: For a given simulation set, all showers have a fixed energy and impact position.
The simulated telescope array is a line of telescopes with a spacing of below 50 m extending from the shower impact point beyond the maximum trigger distance
for a telescope from a shower with the simulated energy.
When these simulations are generated for multiple fixed enrgies, this setup nicely populates the grid in shower energy and impact distance.
The binning in depth of shower maximum is handled automatically by the *template_builder* code. 

*template_builder* contains a couple of scripts to help write the corresponding configuration files for CORSIKA and sim_telarray.

It is of course also possible to generate the simulations without the help of these scripts.