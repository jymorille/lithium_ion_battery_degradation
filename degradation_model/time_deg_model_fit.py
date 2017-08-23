# -*- coding: UTF-8 -*-

"""
This module extracts the parameter of the time degradation model

Model: calendar_deg = time_deg_model(t) * soc_stress_model(SoC) * temp_stress_model(T)

       with: time_deg_model(t) = k_t * t
       => k_t = calendar_deg / (t  * soc_stress_model(SoC) * temp_stress_model(T))
              = deg_per_time_unit / (soc_stress_model(SoC) * temp_stress_model(T))

                where t_e is the duration of calendar experiments
"""

import numpy as np
from degradation_model.degradation_model import soc_stress_model, temp_stress_model, residuals_nonlinear_cal_model
from lmfit import minimize, Parameters, fit_report
from pandas import read_csv

class TimeDegModelFit:
    def __init__(self, data_file_path, alpha_sei=5.87e-02, beta_sei=1.06e+02):
        self.alpha_sei = alpha_sei
        self.beta_sei = beta_sei

        df_cal_data = read_csv(data_file_path)

        self.t_data = df_cal_data['time[year]']

        # extracts the SoH from the columns' name
        self.SoH_op_data = [int(x.split('=')[1][:-1]) for x in df_cal_data.columns[1:]]

        self.number_of_SoH = len(df_cal_data.columns) - 1  # count the number of SoH at which we have data

        self.SoH_data = []
        for i in range(1, self.number_of_SoH + 1):  # the first column is skipped (that's the time in year)
            self.SoH_data.append(df_cal_data.iloc[:, i])

        self.params = Parameters()

        self.deg_per_year = []
        self.list_k_t = []
        self.k_t = ''

        self.least_square_fit()
        self.print_results()

    def least_square_fit(self):

        self.params.add('alpha_sei', value=self.alpha_sei, vary=False)
        self.params.add('beta_sei', value=self.beta_sei, vary=False)
        self.params.add('deg_per_time_unit', value=0.001, min=0, max=1)

        for i in range(0, self.number_of_SoH):

            deg_per_year_i = minimize(fcn=residuals_nonlinear_cal_model,
                                      params=self.params,
                                      args=(self.t_data, self.SoH_data[i], 1),
                                      method='least_squares')

            self.deg_per_year.append(deg_per_year_i.params['deg_per_time_unit'].value)

            k_t_i = self.deg_per_year[i] / temp_stress_model(25) / soc_stress_model(self.SoH_op_data[i] / 100)

            self.list_k_t.append(k_t_i)
            self.k_t = np.mean(self.list_k_t)

    def print_results(self):
        print('---- Time degradation model -----')
        print("k_t = ", '{:.2e}'.format(self.k_t/(365.25*24*3600)), "/s")
