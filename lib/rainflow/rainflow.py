#!/usr/bin/env python
"""
-------------------------------------------------------------------------------
Rainflow counting function with Goodman correction
Copyright (C) 2015 Jennifer Rinker
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
Contact: Jennifer Rinker, Duke University
Email:   jennifer.rinker-at-duke.edu
-------------------------------------------------------------------------------
USAGE:
To call the function in a script on array of turning points <array_ext>:
    import rainflow as rf
    array_out = rf.rainflow(array_ext)
To run the demonstration from a Python console:
    >>> execfile('demo_rainflow.py')
From the terminal (Windows or UNIX-based):
    $ python demo_rainflow.py
-------------------------------------------------------------------------------
DEPENDENCIES:
- Numpy >= v1.3
-------------------------------------------------------------------------------
NOTES:
Python code modified from rainflow.c code with mex function for Matlab from
WISDEM project: https://github.com/WISDEM/AeroelasticSE/tree/master/src/AeroelasticSE/rainflow

Original c code notes:
/*  RAINFLOW $ Revision: 1.1 $ */
/*  by Adam Nieslony, 2009     */
/*  The original code has been modified by Gregory Hayman 2012 as follows: */
/*    abs() has been replaced everywhere with fabs()                       */
/*    the function now applies the Goodman correction to the damage cycle  */
/*      load ranges using a user-supplied fixed load mean, or a fixed load */
/*      mean of zero.                                                      */
/*    the user can supply a the value of a partial damage cycle: uc_mult   */
-------------------------------------------------------------------------------
"""
from numpy import fabs as fabs
import numpy as np


def rainflow(array_ext,
             flm=0, l_ult=1e16, uc_mult=0.5):
    """ Rainflow counting of a signal's turning points with Goodman correction

        Args:
            array_ext (numpy.ndarray): array of turning points

        Keyword Args:
            flm (float): fixed-load mean [opt, default=0]
            l_ult (float): ultimate load [opt, default=1e16]
            uc_mult (float): partial-load scaling [opt, default=0.5]

        Returns:
            array_out (numpy.ndarray): (5 x n_cycle) array of rainflow values:
                                        1) load range
                                        2) range mean
                                        3) Goodman-adjusted range
                                        4) cycle count
                                        5) Goodman-adjusted range with flm = 0

    """

    flmargin = l_ult - fabs(flm)  # fixed load margin
    tot_num = array_ext.size  # total size of input array
    array_out = np.zeros((5, tot_num - 1))  # initialize output array

    pr = 0  # index of input array
    po = 0  # index of output array
    j = -1  # index of temporary array "a"
    a = np.empty(array_ext.shape)  # temporary array for algorithm

    # loop through each turning point stored in input array
    for i in range(tot_num):

        j += 1  # increment "a" counter
        a[j] = array_ext[pr]  # put turning point into temporary array
        pr += 1  # increment input array pointer

        while ((j >= 2) & (fabs(a[j - 1] - a[j - 2]) <= \
                                   fabs(a[j] - a[j - 1]))):
            lrange = fabs(a[j - 1] - a[j - 2])

            # partial range
            if j == 2:
                mean = (a[0] + a[1]) / 2.
                adj_range = lrange * flmargin / (l_ult - fabs(mean))
                adj_zero_mean_range = lrange * l_ult / (l_ult - fabs(mean))
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if (lrange > 0):
                    array_out[0, po] = lrange
                    array_out[1, po] = mean
                    array_out[2, po] = adj_range
                    array_out[3, po] = uc_mult
                    array_out[4, po] = adj_zero_mean_range
                    po += 1

            # full range
            else:
                mean = (a[j - 1] + a[j - 2]) / 2.
                adj_range = lrange * flmargin / (l_ult - fabs(mean))
                adj_zero_mean_range = lrange * l_ult / (l_ult - fabs(mean))
                a[j - 2] = a[j]
                j = j - 2
                if (lrange > 0):
                    array_out[0, po] = lrange
                    array_out[1, po] = mean
                    array_out[2, po] = adj_range
                    array_out[3, po] = 1.00
                    array_out[4, po] = adj_zero_mean_range
                    po += 1

    # partial range
    for i in range(j):
        lrange = fabs(a[i] - a[i + 1]);
        mean = (a[i] + a[i + 1]) / 2.
        adj_range = lrange * flmargin / (l_ult - fabs(mean))
        adj_zero_mean_range = lrange * l_ult / (l_ult - fabs(mean))
        if (lrange > 0):
            array_out[0, po] = lrange
            array_out[1, po] = mean
            array_out[2, po] = adj_range
            array_out[3, po] = uc_mult
            array_out[4, po] = adj_zero_mean_range
            po += 1

            # get rid of unused entries
    array_out = array_out[:, :po]

    return array_out