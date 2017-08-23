# -*- coding: UTF-8 -*-

"""
This module fits the state of charge stress model of a lithium-ion battery

Model: deg_per_time_unit = time_deg_model(t) * soc_stress_model(SoC) * temp_stress_model(T)

       with: soc_stress_model(SoC) = exp(k_SoC*(SoC-SoC_ref))

        considering 2 points at (SoC_ref, T_B) and (SoC_A, T_B):
        deg_per_time_unit(point_A) / deg_per_time_unit(point_ref) = soc_stress_model(SoC_A)
                                                                  = exp(k_SoC*(SoC_A-SoC_ref))
        because: soc_stress_model(SoC_ref) = 1

        so: k_SoC = ln(deg_per_time_unit(point_A) / deg_per_time_unit(point_ref)) / (SoC_A-SoC_ref)

Input: the input file must contain:
       - a first column with the time in year
       - several columns giving the remaining capacity at 25°C under a given state of charge over time
           - those should be called capa_SoC=<SoC> (where <SoC> is to be replaced by the state of charge in %)
           - one of them must be capa_SoC = S_REF (the reference state of charge)

Comments:  - S_ref is defined in the code, and should be around 40-50% SoC
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from degradation_model.degradation_model import nonlinear_cal_model, residuals_nonlinear_cal_model, soc_stress_model
from lmfit import minimize, Parameters, fit_report


class SoCStressModelFit:
    S_REF = 0.5  # this is the reference state of charge at which the SoH stress model is equal to 1
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey', 'brown']

    def __init__(self, data_file_path, alpha_sei=5.87e-02, beta_sei=1.06e+02):
        df_cal_data = read_csv(data_file_path)

        self.t_data = df_cal_data['time[year]']

        # extracts the SoH from the columns' name
        self.SoH_op_data = [int(x.split('=')[1][:-1]) for x in df_cal_data.columns[1:]]

        self.number_of_SoC = len(df_cal_data.columns) - 1  # count the number of SoC at which we have data

        self.alpha_sei = alpha_sei
        self.beta_sei = beta_sei

        self.SoH_data = []
        self.params = Parameters()

        for i in range(1, self.number_of_SoC + 1):  # the first column is skipped (that's the time in year)
            self.SoH_data.append(df_cal_data.iloc[:, i])

        self.list_deg_per_time_unit = []
        self.k_SoC = []

        self.least_square_fit()
        self.print_results()

    def least_square_fit(self):

        self.params.add('alpha_sei', value=self.alpha_sei, vary=False)
        self.params.add('beta_sei', value=self.beta_sei, vary=False)
        self.params.add('deg_per_time_unit', value=0.001, min=0, max=1)

        # least square algorithm
        for i in range(0, self.number_of_SoC):
            deg_per_cyc_i = minimize(fcn=residuals_nonlinear_cal_model,
                                     params=self.params,
                                     args=(self.t_data, self.SoH_data[i], 1),
                                     method='least_squares')
            self.list_deg_per_time_unit.append(deg_per_cyc_i.params['deg_per_time_unit'].value)

        # retrieves the index at which the reference temperature is stored in T
        index_SoH_REF = self.SoH_op_data.index(50)

        for i in [x for x in range(self.number_of_SoC) if x != index_SoH_REF]:
            k_SoC_i = np.log(self.list_deg_per_time_unit[i] / self.list_deg_per_time_unit[index_SoH_REF]) / (self.SoH_op_data[i]/100 - self.S_REF)
            self.k_SoC.append(k_SoC_i)

        self.k_SoC = np.mean(self.k_SoC)  # average of the k_SoC coefficients to get a more precise value

    def plot_model(self, ax):
        soc_linspace = np.linspace(0, 100, 500)
        stress_model_soc = soc_stress_model(soc=soc_linspace / 100, k_soc=self.k_SoC, s_ref=self.S_REF)

        ax.plot(soc_linspace, stress_model_soc, 'b-')

        ax.set_xlabel('State of Charge [%]')
        ax.set_ylabel('Stress')
        ax.set_title('State of Charge stress model')

        # plt.title('State of charge stress model')
        plt.tight_layout()  # without it, the text overlap
        plt.draw()

    def plot_SEI_fit_at_T(self, ax):
        t_linspace = np.linspace(self.t_data.min(), self.t_data.max(), 100)

        for i in range(0, len(self.list_deg_per_time_unit)):
            model = nonlinear_cal_model(t=t_linspace,
                                        alpha_sei=self.params['alpha_sei'],
                                        beta_sei=self.params['beta_sei'],
                                        deg_per_time_unit=self.list_deg_per_time_unit[i])
            label = 'SoC=' + str(self.SoH_op_data[i]) + '%'

            ax.plot(t_linspace, 100 * (1 - model), 'r-', label=label, color=self.colors[i])
            ax.plot(self.t_data, 100 * self.SoH_data[i], 'kx', label='')

        ax.set_xlabel('t [year]')
        ax.set_ylabel('State of charge [%]')
        ax.set_ylim([None, 100])
        ax.set_xlim([0, None])
        ax.set_title('Calendar degradation at 25°C')

        # plt.title('Calendar degradation fit')
        plt.legend()
        plt.tight_layout()  # without it, the text overlap
        plt.draw()

    def print_results(self):
        print('---- SoC stress model -----------')
        print("k_SoC = " + '{:.2e}'.format(self.k_SoC))

