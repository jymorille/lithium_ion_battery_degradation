#!/usr/bin/env python
"""
This module uses the peak detection and rainflow libraries.
It enables to convert a random signal into simple cycles,
characterised by their range, count (half or full cycle)
and mean value.
"""

import lib.peak_det.peak_det as pkd
import lib.rainflow.rainflow as rf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


class CycleCounter:
    def __init__(self, data_file_path='', time_v=None, soc_v=None, delta=0.1, title=''):

        if data_file_path == '':
            if (time_v is not None) and (soc_v is not None):
                t = time_v
                series = soc_v
            else:
                sys.exit('Either a path or vectors must be argument of CycleCounter')
        else:
            csv_input = pd.read_csv(data_file_path, parse_dates=True)
            t = csv_input.iloc[:, 0]
            series = csv_input.iloc[:, 1]

        self.mean_soc = 0
        self.arr_dod = []
        self.arr_dod = []
        self.arr_n = []

        self.title = title
        self.delta = delta
        self.data = pd.DataFrame({'t': t, 'series': series})
        self.min_points = []
        self.max_points = []

        self.turning_points_extraction()

    def turning_points_extraction(self):
        max_points, min_points = pkd.peakdet(self.data['series'], delta=self.delta)

        self.min_points = min_points
        self.max_points = max_points

    def soc_profile_plot(self, title, ax):
        ax.plot(self.data['t'], self.data['series'], label="x")
        ax.set_ylabel('State of charge [%]')
        ax.set_xlabel('time [s]')
        ax.set_title('SoC profile - ' + self.title)

        plt.draw()

    def turning_point_plot(self, ax):
        ax.plot(self.data['series'], label='turning points')

        ax.scatter(np.array(self.max_points)[:, 0], np.array(self.max_points)[:, 1], color='red', label="max")
        ax.scatter(np.array(self.min_points)[:, 0], np.array(self.min_points)[:, 1], color='blue', label="min")
        ax.set_ylabel('State of charge [%]')
        ax.set_xlabel('time series')

        plt.xticks([])
        plt.legend()
        plt.title('Series of extracted peaks')
        plt.tight_layout()
        plt.draw()

    def rainflow_process(self):
        # concatenation of the turning points
        array_ext = np.concatenate((self.min_points, self.max_points), axis=0)
        array_ext = array_ext[array_ext[:, 0].argsort(), :]
        array_ext = np.transpose(array_ext)
        array_ext = array_ext[1]

        # calculate cycle counts with rainflow algorithm
        # with default values for lfm (0), l_ult (1e16), and uc_mult (0.5)
        array_out = rf.rainflow(array_ext)

        # sort array_out by cycle range
        array_out = array_out[:, array_out[0, :].argsort()]

        self.mean_soc = np.mean(self.data['series'])/100  # converts percentage into number between 0 and 1

        self.arr_dod = array_out[0, :]/100  # converts percentage into number between 0 and 1
        self.arr_n = array_out[3, :]
        self.arr_soc_mean = array_out[1, :]/100  # converts percentage into number between 0 and 1

        # writing output in a .csv file
        # with open(output_file_path, 'w') as fp:
        #     mean_soc = '{:.2f}'.format(np.mean(self.data['series']))
        #     fp.write('Range,Count,Mean_' + mean_soc + '\n')
        #     for i in range(len(array_out.T)):
        #         fp.write('{:.3f},{:.3f},{:.3f}'.format(*array_out[[0, 3, 1], i]) + '\n')
