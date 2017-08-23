# -*- coding: UTF-8 -*-

"""
This module extracts the parameters of the non-linear SEI model.

Model:
Let's call L the degradation. L = 1 - SoH. So L = 0 for a new battery.

L = 1 - alpha_sei * exp(-N * beta_sei * deg_per_cyc) - (1 - alpha_sei) * exp(-N * deg_per_cyc)

The two parameters to extract are alpha_sei and beta_sei. They have physical meanings so must
be constrained by bounds.
alpha_sei is generally between 2% and 16%
beta_sei must be superior to a number superior to 1, considering that to SEI film formation
occurs at the beginning of the cell life. We'll chose arbitrarily 10 as lower bound.

"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from pandas import read_csv, DataFrame
from degradation_model.degradation_model import nonlinear_cycle_model, residuals_nonlinear_cycle_model


class SEI_fit:

    def __init__(self, data_file_path):
        # data importation
        data_cyc = read_csv(data_file_path)
        N = data_cyc['N']
        SoC = data_cyc['SoC[%]']
        L = 1 - SoC  # converts SoC into degradation (between 0 and 1)
        
        self.data = DataFrame({'N': N, 'SoC': SoC, 'L': L})        
        self.alpha_sei = ''
        self.beta_sei = ''
        self.deg_per_cycle = ''

        self.least_square_fit()

    def least_square_fit(self):
        # parameters definition
        params = Parameters()
        params.add('alpha_sei', value=0.02, max=0.16, min=0.03)
        params.add('beta_sei', value=40, min=10)
        params.add('deg_per_cyc', value=0.02, min=5e-9, max=7e-5)

        # least square algorithm
        out = minimize(fcn=residuals_nonlinear_cycle_model,
                       params=params,
                       args=(self.data['N'], self.data['L'], 1),
                       nan_policy='omit',
                       method='least_squares')

        # print(fit_report(out))

        self.alpha_sei = out.params['alpha_sei'].value
        self.beta_sei = out.params['beta_sei'].value
        self.deg_per_cycle = out.params['deg_per_cyc'].value
        
        print("---- SEI model ------------------")
        print("alpha_sei = ", "{:0.2e}".format(self.alpha_sei))
        print("beta_sei = ", "{:0.2e}".format(self.beta_sei))
        print("deg_per_cyc = ", "{:0.2e}".format(self.deg_per_cycle))

    def plot_sei_model(self, ax):
        # solution
        cycle_num_linspace = np.linspace(self.data['N'].min(), self.data['N'].max(), 100)
        degradation_model = nonlinear_cycle_model(cycle_num_linspace,
                                                  self.alpha_sei,
                                                  self.beta_sei,
                                                  self.deg_per_cycle)
        capa_model = 100-100*degradation_model

        # plot
        ax.plot(cycle_num_linspace, capa_model, 'r-', label='SEI model')
        ax.plot(self.data['N'], 100 * self.data['SoC'], 'bx', label='experimental data')

        ax.set_xlabel('Cycle number')
        ax.set_ylabel('State of Charge [%]')
        ax.set_ylim([75, 100])
        ax.set_xlim([0, None])

        plt.title('SEI model fit')
        plt.legend()
        plt.tight_layout()
        plt.draw()
