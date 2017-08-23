## Synopsis

This code aims at predicting the degradation of a lithium-ion battery based on
its operating profile, and operating conditions.

This project was part of my master thesis for the MSc Sustainable Energy
Futures, Imperial College London


## Description

It models calendar and cycling ageing, account for stresses induces by high and
low temperature, high and low state of charge, of LMO, NMC and LFP batteries.
It has been designed so that model parameters can be calculated with a new set
of data, to match the very chemistry of a battery.

## Motivation

Most of the operational cost of a lithium ion battery stems from its degradation.
Batteries being expensive to replace, it is of outstanding importance to estimate
at best their remaining life time.


## Usage


### Model definition

Simply run the file [model_definition_validation.py]

/!\\
When the model parameters are extracted from [model_definition_validation.py], they must be modified manually in
[degradation_model/degradation_model.py]


### Model utilisation

Add some input files in the folder input_data
and then run the file [degradation_estimation.py], after
defining some parameters in that file.

## Dependencies
- matplotlib
- numpy
- pandas
- lmfit
- sys

## Contributors

Jean-Yves Morille, Imperial College
jean-yves.morille16@imperial.ac.uk


## Acknowledgments

This model was inspired by the work published in the following paper:
https://www.researchgate.net/publication/303890624

