""" This modules enables to calculate and plot the estimated degradation of a lithium ion battery, over its operations
defined a file located in the folder input_data"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import time

from degradation_model.degradation_model import final_degradation_model, nonlinear_general_model


colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey', 'brown']


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def even_selection_array(n, array_in):
    step = (len(array_in) - 1) / (n - 1)
    array_out = []
    for i in range(0, n):
        array_out.append(array_in[round(step * i)])
    return array_out


def convert_vector_time(time_array):
    array_out = []
    for i in range(0, len(time_array)):
        diff = time_array.iloc[i] - time_array.iloc[0]
        array_out.append(diff.total_seconds())
    return array_out

time1 = time.time()

fig = plt.figure(figsize=(5, 4), dpi=100)
ax1 = fig.add_subplot(111)

temperature = 21
chemistry = 'NMC'
alpha_sei = 5.87e-02
beta_sei = 1.06e+02

for m, fn in enumerate(os.listdir('input_data/')):
    if os.path.isfile(os.path.join('input_data/', str(fn))):
        filename, file_extension = os.path.splitext(fn)
        if file_extension == '.csv':
            # file processing
            df = pd.read_csv('input_data/' + fn, parse_dates=[0])
            dtm = df.iloc[:, 0]  # date time
            soc = df.iloc[:, 1]  # state of charge

            # calculation of total duration
            last_index = dtm.index[-1]
            first_time = dtm[1]
            last_time = dtm[last_index]
            diff = last_time - first_time
            total_duration = diff.total_seconds()

            time_linspace = even_selection_array(365, dtm)  # selection 100 time points from dtm

            cal_results = [0]
            cyc_results = [0]
            linearised_deg_v = [0]
            nonlinear_deg_v = [0]
            time_index_v = [0]

            for k in range(1, len(time_linspace)):

                # searches in dtm the index at which the date is equal to time_linspace[i]
                index_dtm_i = dtm[dtm == time_linspace[k]].index.tolist()[0]
                time_index_v.append(index_dtm_i)

                # extracts the relevant time data
                time_v = dtm.iloc[time_index_v[k-1]:time_index_v[k]]
                # and converts them to time difference
                time_v = convert_vector_time(time_v)

                # extracts the relevant soc data
                soc_v = soc.iloc[time_index_v[k-1]:time_index_v[k]].as_matrix()

                # calculate calendar and cycling degradation
                cal_deg, cyc_deg = final_degradation_model(time_v=time_v,
                                                           soc_v=soc_v,
                                                           T=temperature,
                                                           time=time_v[-1],
                                                           chemistry=chemistry,
                                                           delta=0.1,
                                                           title='DST')

                # stores the results
                cal_results.append(cal_results[k-1] + cal_deg)
                cyc_results.append(cyc_results[k-1] + cyc_deg)

                linearised_deg = cal_results[k]+cyc_results[k]
                linearised_deg_v.append(linearised_deg)

                non_linear_deg = nonlinear_general_model(alpha_sei, beta_sei, linearised_deg)

                nonlinear_deg_v.append(non_linear_deg)

            remaining_capa = []

            for i in range(0, len(nonlinear_deg_v)):
                remaining_capa.append(100*(1-nonlinear_deg_v[i]))

            ax1.plot(time_linspace, remaining_capa, 'x-', color=colors[m], label=filename)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Remaining capacity [%]')
            ax1.set_ylim([80, 100])

            plt.legend()
            plt.tight_layout()
            plt.draw()
plt.show()