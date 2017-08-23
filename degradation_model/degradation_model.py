# -*- coding: UTF-8 -*-

""" This modules defines all degradation and stress models used in the project.
"""

import matplotlib.pylab as plt
import collections
import numpy as np

from degradation_model.cycle_counting_algorithm import CycleCounter
from lmfit import Parameters
from math import sqrt


# -------- Nonlinear general model ----------------------------------


def nonlinear_general_model(alpha_sei, beta_sei, deg):
    """

    :param cyc_num: cycle number
    :param alpha_sei: coefficient alpha of the SEI model
    :param beta_sei: coefficient beta of the SEI model
    :param deg_per_cyc: degradation per cycle
    :return: total degradation after N cycles, between 0 and 1, 0 meaning new battery
    """
    return 1 - alpha_sei * np.exp(- beta_sei * deg) - (1 - alpha_sei) * np.exp(-deg)


# -------- Cycling degradation model ----------------------------------

def nonlinear_cycle_model(cyc_num, alpha_sei, beta_sei, deg_per_cyc):
    """

    :param cyc_num: cycle number
    :param alpha_sei: coefficient alpha of the SEI model
    :param beta_sei: coefficient beta of the SEI model
    :param deg_per_cyc: degradation per cycle
    :return: total degradation after N cycles, between 0 and 1, 0 meaning new battery
    """
    return 1 - alpha_sei * np.exp(-cyc_num * beta_sei * deg_per_cyc) - (1 - alpha_sei) * np.exp(-cyc_num * deg_per_cyc)


def residuals_nonlinear_cycle_model(params, cyc_num, deg, eps_data):
    """

    :param params: model parameters (alpha_sei, beta_sei, deg_per_cyc)
    :param cyc_num: cycle number
    :param deg: total degradation after N cycles, between 0 and 1, 0 meaning new battery
    :param eps_data: residuals scaling factor
    :return: scaled residuals
    """
    alpha_sei = params['alpha_sei'].value
    beta_sei = params['beta_sei'].value
    deg_per_cyc = params['deg_per_cyc'].value

    deg_model = nonlinear_cycle_model(cyc_num, alpha_sei, beta_sei, deg_per_cyc)
    return (deg - deg_model) / eps_data
# ---------------------------------------------------------------------


# -------- Calendar degradation model ---------------------------------

def nonlinear_cal_model(t, alpha_sei, beta_sei, deg_per_time_unit):
    """

    :param t: time in [year]
    :param alpha_sei: coefficient for the SEI model
    :param beta_sei: coefficient for the SEI model
    :param deg_per_time_unit: degradation per year
    :return: total degradation after a duration of t, between 0 and 1, 0 meaning new battery
    """
    return 1-alpha_sei*np.exp(-t*beta_sei*deg_per_time_unit)-(1-alpha_sei)*np.exp(-t*deg_per_time_unit)


def residuals_nonlinear_cal_model(params, t, soh, eps_data):
    """

    :param params: model parameters (alpha_sei, beta_sei, deg_per_time_unit)
    :param t: time in [year]
    :param soh: state of health, between 0 and 1, 1 meaning new battery
    :param eps_data: residuals scaling factor
    :return: scaled residuals
    """
    alpha_sei = params['alpha_sei'].value
    beta_sei = params['beta_sei'].value
    deg_per_time_unit = params['deg_per_time_unit'].value

    total_deg = 1 - soh
    cal_model = nonlinear_cal_model(t, alpha_sei, beta_sei, deg_per_time_unit)

    return (cal_model - total_deg) / eps_data
# ---------------------------------------------------------------------


# -------- Depth of discharge degradation model -----------------------

def emp_deg_model_after_1_dod(params, dod):
    """

    :param params: parameters of the empirical degradation model (used for LFP batteries)
    :param dod: depth of discharge
    :return: degradation after on cycle of the given depth of discharge
    """
    k_d1 = params['k_d1'].value
    k_d2 = params['k_d2'].value
    k_d3 = params['k_d3'].value
    cycle_num_at_80_soh = np.divide(1, k_d1 * np.power(dod, k_d2) + k_d3)
    return cycle_num_at_80_soh


def residuals_emp_deg_model_after_1_dod(params, dod, deg_data_after_1_dod, eps_data):
    """

    :param params: parameters of the exponential cycle number model
    :param dod: depth of discharge
    :param deg_data_after_1_dod: data of degradation after 1 cycle at the given depth of discharge
    :param eps_data: residuals scaling factor
    :return: scaled residuals
    """
    # residuals_quad is the difference between the empirical model and the experimental data
    return (emp_deg_model_after_1_dod(params, dod) - deg_data_after_1_dod) / eps_data


def exp_deg_model_after_1_dod(params, dod):
    """

    :param params: parameters of the exponential degradation model (used for LMO and NMC batteries)
    :param dod: depth of discharge
    :return: degradation after on cycle of the given depth of discharge
    """
    k_d1 = params['k_d1'].value
    k_d2 = params['k_d2'].value
    k_d3 = params['k_d3'].value  # not used but prevents from implementing if conditions in cyc_dod_deg_model_fit.py
    stress_dod = k_d1 * dod * np.exp(k_d2*dod)
    return stress_dod


def residuals_exp_deg_model_after_1_dod(params, dod, deg_data_after_1_dod, eps_data):
    """

    :param params: parameters of the exponential cycle number model
    :param dod: depth of discharge
    :param deg_data_after_1_dod: data of degradation after 1 cycle at the given depth of discharge
    :param eps_data: residuals scaling factor
    :return: scaled residuals
    """
    # residuals_quad is the difference between the empirical model and the experimental data
    return (exp_deg_model_after_1_dod(params, dod) - deg_data_after_1_dod) / eps_data


def dod_deg_model(chemistry, dod):
    """

    :param chemistry: either NMC, LMO or LFP
    :param dod: depth of discharge (between 0 and 1)
    :return: degradation; 0.2 means end of life of the battery
    """

    params = Parameters()
    if chemistry == "NMC":
        params.add('k_d1', value=1.47e+04)
        params.add('k_d2', value=-1.65e+00)
        params.add('k_d3', value=3.61e+02)
        deg_per_cyc = emp_deg_model_after_1_dod(params, dod)
    elif chemistry == "LMO":
        params.add('k_d1', value=1.39e+05)
        params.add('k_d2', value=-5.09e-01)
        params.add('k_d3', value=-1.21e+05)
        deg_per_cyc = emp_deg_model_after_1_dod(params, dod)
    elif chemistry == "LFP":
        params.add('k_d1', value=9.05e-06)
        params.add('k_d2', value=1.40e+00)
        params.add('k_d3', value=0)
        deg_per_cyc = exp_deg_model_after_1_dod(params, dod)
    else:
        print('The chemistry input of DoD_stress_model isn''t valid')
        return

    return deg_per_cyc


def time_deg_model(t, k_t=4.11e-10):
    """

    :param t: time in second
    :param k_t:
    :return: 0.2 means end of life of the battery
    """

    degradation = k_t * t
    return degradation


def voltage_stress_model(soc, k_v=10.2):

    soc_threshold = 0.85
    if not isinstance(soc, collections.Iterable):
        if soc > soc_threshold:
            stress = np.exp(k_v * (soc - soc_threshold))
        else:
            stress = 1.0
    else:
        stress = []
        for i in range(0, len(soc)):
            if soc[i] > soc_threshold:
                stress.append(np.exp(k_v * (soc[i] - soc_threshold)))
            else:
                stress.append(1.0)
    return stress


def soc_stress_model(soc, k_soc=1.01e+00, s_ref=0.5):
    """

    :param soc: state of charge (between 0 and 1)
    :param k_soc:
    :param s_ref:
    :return: stress between 0 and 1; stress = 1 under standard conditions
    """

    # model type 1 -------------------------------------------------------------------
    if not isinstance(soc, collections.Iterable):
            stress = np.exp(k_soc * (soc - s_ref))
    else:
        stress = []
        for i in range(0, len(soc)):
                stress.append(np.exp(k_soc * (soc[i] - s_ref)))
    return stress

    # model type 2, which an increased stress factor beyond 90% SoC ------------------

    # k_soc_sup = 6.2 * k_soc
    #
    # if type(soc) is float or soc.dtype is not np.dtype(np.float64) or not isinstance(soc, (list, tuple, np.ndarray)):
    #     if soc < 0.9:
    #         stress = np.exp(k_soc * (soc - s_ref))
    #     else:
    #         stress = np.exp(k_soc_sup * (soc - s_ref)) + np.exp(0.4*k_soc) - np.exp(k_soc_sup*0.4)
    # else:
    #     stress = []
    #     for i in range(0, len(soc)):
    #         if soc[i] < 0.9:
    #             stress.append(np.exp(k_soc * (soc[i] - s_ref)))
    #         else:
    #             stress.append(np.exp(k_soc_sup * (soc[i] - s_ref)) + np.exp(0.4*k_soc) - np.exp(k_soc_sup*0.4))
    # return stress


def temp_stress_model(T, k_T=6.71e-02, T_ref=25):
    """

    :param T: temperature in °C
    :param k_T: temperature stress parameter
    :param T_ref: reference temperature, usually around 25°C
    :return: stress between 0 and 1; stress = 1 under reference conditions
    """
    if not isinstance(T, collections.Iterable):
        if T >= T_ref:
            stress = (np.exp(k_T * (T - T_ref)))
        elif T > T_ref - 10:
            stress = 1
        else:
            stress = np.exp(k_T * (- T + T_ref - 10))
    else:
        stress = []
        for i in range(0, len(T)):
            if T[i] >= T_ref:
                stress.append(np.exp(k_T*(T[i]-T_ref)))
            elif T[i] > T_ref - 10:
                stress.append(1)
            else:
                stress.append(np.exp(k_T * (- T[i] + T_ref - 10)))
    return stress


def final_degradation_model(time_v, soc_v, T, time, chemistry, delta=0.1, title=''):

    # cycles counting
    cycle_count1 = CycleCounter(time_v=time_v, soc_v=soc_v, delta=delta, title=title)

    cycle_count1.rainflow_process()

    soc_mean = cycle_count1.mean_soc
    arr_dod = cycle_count1.arr_dod
    arr_n = cycle_count1.arr_n
    arr_soc_mean = cycle_count1.arr_soc_mean

    counted_cycles = len(arr_dod)

    # calendar degradation ------------------------------------------------------------
    time_stress = time_deg_model(time)
    SoC_stress_cal = soc_stress_model(soc_mean)
    temp_stress_cal = temp_stress_model(T)
    cal_degradation = time_stress*SoC_stress_cal*temp_stress_cal

    # cycling degradation -------------------------------------------------------------
    cyc_degradation = 0
    for i in range(0, counted_cycles):
        DoD_stress_cyc_i = dod_deg_model(chemistry, arr_dod[i])
        SoC_stress_cyc_i = soc_stress_model(arr_soc_mean[i])
        temp_stress_cyc_i = temp_stress_model(T)
        voltage_stress_cyc_i = voltage_stress_model(arr_soc_mean[i])

        cyc_degradation += arr_n[i] * DoD_stress_cyc_i * SoC_stress_cyc_i * temp_stress_cyc_i * voltage_stress_cyc_i

    return cal_degradation, cyc_degradation

