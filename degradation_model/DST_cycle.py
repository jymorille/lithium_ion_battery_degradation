# -*- coding: UTF-8 -*-

""" This module builds C-rate and state of charge profiles of DST cycles
One DST cycle has a pre-defined pattern which discharge a cell of 10% before
charging it at 1 C-rate until the initial state of charge.
This script support depth of discharge which are multiple of 5%.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import floor
import sys
from degradation_model.degradation_model import final_degradation_model, nonlinear_general_model
from pandas import read_csv

class DSTCycleDeg:
    def __init__(self, soc_min, soc_max, alpha_sei, beta_sei, chemistry, temperature):
        """

        :param soc_min: minimum level of state of charge
        :param soc_max: starting level of state of charge

        """

        # checks if the input are correct
        if soc_min >= soc_max:
            sys.exit("soc_min must be strictly inferior to soc_max")
        elif (soc_max-soc_min)/5 - int((soc_max-soc_min)/5) != 0:
            sys.exit("The difference between soc_max and soc_min must be a multiple of 5")

        self.soc_min = soc_min
        self.soc_max = soc_max

        self.num_DST_cycles_linspace = []
        self.v_degradation = []

        precision = 10 # number of points per second

        # definition of one C-rate pattern
        delta_t = np.array([18, 28, 12, 8, 16, 24, 12, 8, 16, 24, 12, 8, 16, 36, 8, 24, 8, 32, 8, 42])
        c_rate = np.array([0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -8, -5, 2, -2, 4, 0])
        cycle_duration = 360

        # number of full DST cycles to perform
        num_DST = floor((soc_max-soc_min)/10)

        # number of half DST cycle to perform
        if (soc_max-soc_min)/10 - int((soc_max-soc_min)/10) == 0.5:
            num_half_DST = 1
        else:
            num_half_DST = 0

        # percentage of a DST cycle after which 5% state of charge is discharged
        perc_DST = 0.6683

        # definition of the C-rate vector of 1 DST cycle
        c_rate_1_DST = []
        for i in range(0, len(delta_t)):
            c_rate_1_DST = c_rate_1_DST + ([c_rate[i]] * delta_t[i] * precision)

        # definition of the state of charge vector of 1 DST cycle
        soc_1_DST = [0]
        for i in range(1, len(c_rate_1_DST)):
            soc_1_DST.append(soc_1_DST[i - 1] + c_rate_1_DST[i] / precision / 36)

        # first DST cycle
        self.soc_v = soc_1_DST[:]
        self.c_rate_v = c_rate_1_DST[:]

        # potential next DST cycles
        for i in range(0, num_DST-1):
            for k in range(0, len(soc_1_DST)):
                self.soc_v.append((i+1)*soc_1_DST[-1] + soc_1_DST[k])
            for k in range(0, len(c_rate_1_DST)):
                self.c_rate_v.append(c_rate_1_DST[k])

        # potential half DST cycle
        if num_half_DST == 1:
            if num_DST == 0:
                # if the range is equal to 5%
                # in that case the first DST cycle is cropped
                self.soc_v = soc_1_DST[:floor(len(soc_1_DST) * perc_DST)]
                self.c_rate_v = c_rate_1_DST[:floor(len(soc_1_DST) * perc_DST)]
            else:
                for k in range(0, floor(len(soc_1_DST)*perc_DST)):
                    self.soc_v.append((num_DST)*soc_1_DST[-1] + soc_1_DST[k])
                for k in range(0, floor(len(c_rate_1_DST)*perc_DST)):
                    self.c_rate_v.append(c_rate_1_DST[k])

        # charge at 1C-rate until initial state of charge
        num_point_half_way = len(self.soc_v)
        for k in range(0, len(self.soc_v)):
            self.c_rate_v.append(1)
            self.soc_v.append((self.soc_min-self.soc_max) + k*(num_DST+num_half_DST)*precision/num_point_half_way)

        # time vector
        self.time_v = np.linspace(0, 2*cycle_duration * (num_DST + num_half_DST * perc_DST),
                                  (len(self.c_rate_v)))

        # scaling
        self.soc_v = [x + soc_max for x in self.soc_v]

        # x-axis: DST cycles

        last_data_cyc_num = {'65_75': 8393.25, '45_75': 6591.46, '25_75': 5226.39, '25_85': 5251.05,
                             '50_100': 5485.85, '40_100': 4986.55, '25_100': 4383.85}

        index = str(self.soc_min) + '_' + str(self.soc_max)

        self.num_DST_cycles_linspace = np.linspace(0, last_data_cyc_num[index], 22)

        # calculate the y-axis: remaining capacity
        for i in range(0, len(self.num_DST_cycles_linspace)):
            cal_deg, cyc_deg_per_DST = final_degradation_model(time_v=self.time_v,
                                                               soc_v=self.soc_v,
                                                               T=temperature,
                                                               time=self.num_DST_cycles_linspace[i] * self.time_v[-1],
                                                               chemistry=chemistry,
                                                               delta=0.1,
                                                               title='DST')

            # total linearised degradation
            linearised_deg = cal_deg + cyc_deg_per_DST * self.num_DST_cycles_linspace[i]

            # total degradation
            non_linear_deg = nonlinear_general_model(alpha_sei, beta_sei, linearised_deg)

            # stores the results
            self.v_degradation.append(non_linear_deg)

    def plot_DST_deg_model(self, ax, color, label):
        SoH = 100*(1-np.array(self.v_degradation))

        ax.plot(self.num_DST_cycles_linspace, SoH, 'x-', color=color, label=label)
        ax.set_ylim([60, 100])
        ax.set_xlim([0, 9000])
        ax.set_xlabel('DST cycle')
        ax.set_ylabel('State of Health [%]')
        ax.set_title('Degradation model')
        ax.grid(color='k', linestyle=':', linewidth=1, alpha=0.3)
        plt.xticks(np.arange(0, 9000, 1000))

        plt.tight_layout()
        plt.legend()
        plt.draw

    def plot_DST_profile(self, ax1, ax2):
        ax1.plot(self.time_v, self.c_rate_v, '-', label='')
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('C-rate')
        ax1.set_title('C-rate profile')
        plt.tight_layout()

        ax2.plot(self.time_v, self.soc_v, '-', label='')
        ax2.set_xlabel('time [s]')
        ax2.set_ylabel('State of Charge [%]')
        ax2.set_title('State of charge profile')
        plt.tight_layout()

        plt.draw()


def plot_DST_experimental_data(ax, data_folder_path):
    v_soc_min = [25, 40, 25, 50, 25, 45, 65]
    v_soc_max = [100, 100, 85, 100, 75, 75, 75]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey', 'brown']

    for i in range(0, 7):
        path = data_folder_path + "DST_" + str(v_soc_min[i]) + '_' + str(v_soc_max[i]) + '.csv'
        df_data_DST = read_csv(path)
        cycles = df_data_DST.iloc[:, 0]
        soh = df_data_DST.iloc[:, 1]
        label = str(v_soc_min[i]) + '-' + str(v_soc_max[i]) + ' at 20°C'

        ax.plot(cycles, soh, 'x-', label=label, color=colors[i])
        ax.set_xlabel('DST cycle')
        ax.set_ylabel('State of Health [%]')
        ax.set_ylim([60, 100])
        ax.set_xlim([0, 9000])
        plt.xticks(np.arange(0, 9000, 1000))

        ax.set_title('Experimental data')
        ax.grid(color='k', linestyle=':', linewidth=1, alpha=0.3)

        plt.tight_layout()
        plt.legend()
        plt.draw()