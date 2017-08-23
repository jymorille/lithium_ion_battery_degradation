# -*- coding: UTF-8 -*-

"""
This module fits the temperature stress model of a lithium-ion battery

Model: deg_per_time_unit = time_deg_model(t) * soc_stress_model(SoC) * temp_stress_model(T)

       with: temp_stress_model(T) = exp(k_T*(T-T_ref))

        considering 2 points at (T_ref, SoC_B) and (T_A, SoC_B):
        deg_per_time_unit(point_A) / deg_per_time_unit(point_ref) = temp_stress_model(T_A)
                                                                  = exp(k_T*(T_A-T_ref))
        because: temp_stress_model(T_ref) = 1

        so: k_T = ln(deg_per_time_unit(point_A) / deg_per_time_unit(point_ref)) / (T_A-T_ref)

Input: the input file must contain:
       - a first column with the time in year
       - several columns named giving the remaining capacity over time at the reference state of charge under a given
        temperature
           - those should be called capa_T=<temp> (where <temp> is to be replaced by the temperature in °C)
           - one of them must be capa_T = T_ REF (the reference temperature)

Comments:  - T_ref is defined in the code, and should be around 20-25°C
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from lmfit import minimize, Parameters, fit_report
from degradation_model.degradation_model import nonlinear_cal_model, residuals_nonlinear_cal_model, temp_stress_model


class TempStressModelFit:

    T_REF = 25  # this is the reference temperature at which the temperature stress model is equal to 1
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey', 'brown']

    def __init__(self, data_file_path, alpha_sei=5.87e-02, beta_sei=1.06e+02):
        self.alpha_sei = alpha_sei
        self.beta_sei = beta_sei

        # data importation
        df_cal_data = read_csv(data_file_path)

        T_op_data = [int(x.split('=')[1]) for x in df_cal_data.columns[1:]]
        # extracts the temperatures from the columns' name

        self.number_of_temp = len(df_cal_data.columns) - 1  # count the number of temperature at which we have data

        SoH_data = []
        for i in range(1, self.number_of_temp + 1):  # the first column is skipped (that's the time in year)
            SoH_data.append(df_cal_data.iloc[:, i])

        self.t_data = df_cal_data['time[year]']
        self.T_op_data = T_op_data
        self.SoH_data = SoH_data
        self.k_T = []
        self.list_deg_per_time_unit = []
        self.params = Parameters()

        self.least_square_fit()
        self.print_results()

    def least_square_fit(self):

        self.params.add('alpha_sei', value=self.alpha_sei, vary=False)
        self.params.add('beta_sei', value=self.beta_sei, vary=False)
        self.params.add('deg_per_time_unit', value=0.001, min=0, max=1)

        # least square algorithm
        for i in range(0, self.number_of_temp):
            deg_per_time_unit_i = minimize(fcn=residuals_nonlinear_cal_model,
                                           params=self.params,
                                           args=(self.t_data, self.SoH_data[i], 1),
                                           # nan_policy='omit',
                                           method='least_squares')
            self.list_deg_per_time_unit.append(deg_per_time_unit_i.params['deg_per_time_unit'].value)

        index_T_REF = self.T_op_data.index(25)  # retrieve the index at which the reference temperature is stored in T

        k_T = []
        for i in [x for x in range(self.number_of_temp) if x != index_T_REF]:
            k_T_i = np.log(self.list_deg_per_time_unit[i] / self.list_deg_per_time_unit[index_T_REF]) / (self.T_op_data[i] - self.T_REF)
            k_T.append(k_T_i)

        self.k_T = np.mean(k_T)  # average of the k_T coefficients to get a more precise value

    def plot_model(self, ax):
        T_linspace = np.linspace(-20, 60, 100)
        stress_model_T = temp_stress_model(T=T_linspace, k_T=self.k_T, T_ref=self.T_REF)

        ax.plot(T_linspace, stress_model_T, 'b-')
        ax.set_xlabel('Temperature [°C]')
        ax.set_ylabel('Stress')
        ax.set_title('Temperature stress model')

        # plt.title('Temperature Stress model')
        plt.tight_layout()  # without it, the text overlap
        plt.draw()

    def plot_SEI_fit_at_T(self, ax):
        t_linspace = np.linspace(self.t_data.min(), self.t_data.max(), 100)

        for i in range(0, len(self.list_deg_per_time_unit)):
            model = nonlinear_cal_model(t=t_linspace,
                                        alpha_sei=self.params['alpha_sei'],
                                        beta_sei=self.params['beta_sei'],
                                        deg_per_time_unit=self.list_deg_per_time_unit[i])
            label = 'T=' + str(self.T_op_data[i]) + '°C'

            ax.plot(t_linspace, 100*(1-model), 'r-', label=label, color=self.colors[i])

            ax.plot(self.t_data, 100*self.SoH_data[i], 'kx', label='')

        ax.set_xlabel('t [year]')
        ax.set_ylabel('State of charge [%]')
        ax.set_ylim([None, 100])
        ax.set_xlim([0, None])
        ax.set_title('Calendar degradation at 50% SoC')

        # plt.title('Calendar degradation fit')
        plt.legend()
        plt.tight_layout()  # without it, the text overlap
        plt.draw()

    def print_results(self):
        print('---- Temperature stress model ---')
        print("k_T = " + '{:.2e}'.format(self.k_T))



