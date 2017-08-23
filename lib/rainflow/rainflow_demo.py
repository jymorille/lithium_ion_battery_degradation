#!/usr/bin/env python
"""
Demonstration of rainflow cycle counting
Contact: Jennifer Rinker, Duke University
Email:   jennifer.rinker-at-duke.edu
"""
import matplotlib.pyplot as plt
import numpy as np

from degradation_model import cycle_counting_algorithm as rf

if (__name__ == '__main__'):

    # array of turning points
    array_ext = np.array([3, -2, 4, 2, 5, -1, 1, 0, 3])

    # calculate cycle counts with default values for lfm (0),
    #  l_ult (1e16), and uc_mult (0.5)
    array_out = rf.rainflow(array_ext)

    # sort array_out by cycle range
    array_out = array_out[:, array_out[0, :].argsort()]

    # ---------------------------- printing/plotting ------------------------------

    # print cycle range, cycle count, cycle mean, goodman-adjusted range (GAR), and
    #   goodman-adjusted range with zero fixed-load mean (GAR-ZFLM)
    print('\nCalculated cycle count:')
    print('\n{:>7s}{:>8s}{:>12s}'.format('Range', 'Count', 'Mean'))
    print('----------------------------------------------')
    for i in range(len(array_out.T)):
        print('{:7.1f}{:8.1f}{:12.1f}'.format(*array_out[[0, 3, 1], i]))

    # plot turning points for vizualization
    plt.figure(1, figsize=(6.5, 3.5))
    plt.clf()

    plt.plot(array_ext)
    plt.grid('on')
    plt.xlabel('Time')
    plt.ylabel('Turning Points')
    plt.title('Turning Points for Rainflow Demo - J. Rinker')
    plt.tight_layout()

    plt.show()