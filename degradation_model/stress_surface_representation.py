# -*- coding: UTF-8 -*-

""" This module enables to plot the calendar and cycle stress function as a function of SoC and temperature """

from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from degradation_model.degradation_model import voltage_stress_model, soc_stress_model, temp_stress_model

import numpy as np


def stress_function(soc, temp, include_voltage_stress):
    if include_voltage_stress:
        stress = voltage_stress_model(soc) * soc_stress_model(soc) * temp_stress_model(temp)
    else:
        stress = voltage_stress_model(0) * soc_stress_model(soc) * temp_stress_model(temp)
    return stress


def function_of_meshgrid(X, Y, include_voltage_stress, K):
    gram_matrix = np.zeros((X.shape[0], Y.shape[1]))

    for i, x in enumerate(X):
        for j, y in enumerate(np.matrix.transpose(Y)):
            gram_matrix[i, j] = K(x[j], y[i], include_voltage_stress)
    return gram_matrix


def surface_stress_plot(ax, include_voltage_stress):

    # axes definition
    soc = np.arange(0, 100, 1)
    temp = np.arange(-5, 50, 1)
    soc_mesh, temp_mesh = np.meshgrid(soc, temp)

    # stress calculation
    stress_surface = function_of_meshgrid(soc_mesh/100, temp_mesh, include_voltage_stress, stress_function)

    # Plot the surface.
    ax.plot_surface(soc_mesh, temp_mesh, stress_surface, cmap=cm.jet, rstride=1, cstride=1)

    # Adjust the viewing angle.
    ax.view_init(elev=5.0, azim=230.0)

    ax.set_xlabel('State of charge [%]')
    ax.set_ylabel('Temperature [°C]')
    if include_voltage_stress:
        ax.set_zlabel('$S_{cyc}(\sigma, T) = S_\sigma(\sigma).S_T(T).S_V(\sigma)$', rotation=90)
    else:
        ax.set_zlabel('$S_{cal}(\sigma, T) = S_\sigma(\sigma).S_T(T)$', rotation=90)

    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    # fig.colorbar(surf, shrink=0.5, aspect=5)

