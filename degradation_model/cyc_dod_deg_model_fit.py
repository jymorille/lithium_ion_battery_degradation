# -*- coding: UTF-8 -*-

"""
This module extracts the parameters of the cycling degradation model
under standard conditions, which is a function of the depth of
discharge of cycles, and depends on the chemistry.

For LMO and NMC batteries:
N_cycles_at_80_SoH = 1 / (k_d1 * DoD^k_d2 + k_d3)

For LFP batteries:
N_cycles_at_80_SoH = k_d1 * DoD * e^(k_d2 * DoD)

N-B: if the first equation is used to fit LFP data, the behavior
     at low DoD isn't properly modelled
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from degradation_model.degradation_model import emp_deg_model_after_1_dod, residuals_emp_deg_model_after_1_dod, \
                                                exp_deg_model_after_1_dod, residuals_exp_deg_model_after_1_dod
from lmfit import minimize, Parameters, fit_report
from pandas import read_csv


class CyclingDegModelFit:
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey', 'brown']

    def __init__(self, data_file_paths):
        self.data_file_paths = data_file_paths
        self.opt_params = []
        self.chemistry = []
        self.cyc_nb_v = []
        self.dod_v = []

        self.fit_stress_model_dod()

    def fit_stress_model_dod(self):
        print("---- DoD degradation model ------")
        i = 0  # index of file

        for file in self.data_file_paths:
            # imports data
            data_cyc = read_csv(file)
            dod = data_cyc['DoD']
            cyc_nb = data_cyc.iloc[:, 1]

            self.cyc_nb_v.append(cyc_nb)
            self.dod_v.append(dod)

            chemistry = data_cyc.columns[1].split('_')[2]
            stress_data = 0.2 * np.divide(1, cyc_nb)

            # parameters definition
            params = Parameters()

            # the parameters proposed below come from several tests
            # the initial values need to be not to far from the solution
            # in order the algorithm to converge
            if chemistry == "LFP":
                params.add('k_d1', value=1e5)
                params.add('k_d2', value=-0.5)
                params.add('k_d3', value=0, vary=False)
            elif chemistry == "NMC":
                params.add('k_d1', value=1e+04)
                params.add('k_d2', value=-1)
                params.add('k_d3', value=3e+02)
            elif chemistry == "LMO":
                params.add('k_d1', value=1e+05)
                params.add('k_d2', value=-0.5)
                params.add('k_d3', value=-1e+05)

            if chemistry == "LFP":
                residuals = residuals_exp_deg_model_after_1_dod
            elif chemistry == "LMO" or "NMC":
                residuals = residuals_emp_deg_model_after_1_dod
            else:
                sys.exit("The chemistry is incorrect or not supported by the model")

            # least square minimisation
            opt_param = minimize(fcn=residuals,
                                 params=params,
                                 args=(dod, stress_data, 1),
                                 method='leastsq')
            # leastsq seems to work better than least_squares for those types of equations

            # stores optimisation output
            self.opt_params.append(opt_param.params['k_d1'].value)
            self.opt_params.append(opt_param.params['k_d2'].value)
            self.opt_params.append(opt_param.params['k_d3'].value)

            # stores chemistry
            self.chemistry.append(chemistry)

            # prints results
            list_opt_params = ['{:.2e}'.format(opt_param.params['k_d1'].value),
                               '{:.2e}'.format(opt_param.params['k_d2'].value),
                               '{:.2e}'.format(opt_param.params['k_d3'].value)]

            print("Chemistry:", chemistry)
            if chemistry == "LFP":
                print("k_d1 = ", list_opt_params[0])
                print("k_d2 = ", list_opt_params[1])
            else:
                print("k_d1 = ", list_opt_params[0])
                print("k_d2 = ", list_opt_params[1])
                print("k_d3 = ", list_opt_params[2])

            # index incrementation
            i = i + 1

    def plot_dod_graph(self, ax):
        """ Plots the results of the data fit """
        i = 0

        for file in self.data_file_paths:
            # solution
            dod_linspace = np.linspace(1, 100, 100)

            params = Parameters()

            params.add('k_d1', value=self.opt_params[i * 3])
            params.add('k_d2', value=self.opt_params[i * 3 + 1])
            params.add('k_d3', value=self.opt_params[i * 3 + 2])

            if self.chemistry[i] == "LFP":
                stress_dod_model_emp = exp_deg_model_after_1_dod(params, dod_linspace / 100)
            elif self.chemistry[i] == "LMO" or "NMC":
                stress_dod_model_emp = emp_deg_model_after_1_dod(params, dod_linspace / 100)

            # plot
            ax.plot(dod_linspace,
                    0.2 * np.divide(1, stress_dod_model_emp),
                    'b-',
                    label=self.chemistry[i],
                    color=self.colors[i])
            # ax.plot(self.dod_v[i]*100, self.cyc_nb_v[i], 'kx', label='', alpha=0.4)

            # index incrementation
            i = i + 1

        ax.set_xlabel('Depth of Discharge [%]')
        ax.set_ylabel('Cycle No.')
        ax.set_xlim([0, 100])
        ax.set_title('Cycle count at 80% state of Health')

        plt.yscale('log')
        plt.legend()
        plt.tight_layout()  # without it, the text overlap
        plt.draw()
