# -*- coding: UTF-8 -*-

"""

Model definition:
Let's call L the degradation. L = 1 - SoH. So L = 0 for a new battery.

a) non-linear model
a-1) calendar
L = 1 - alpha_sei * exp(-t * beta_sei * deg_per_time_unit) - (1 - alpha_sei) * exp(-t * deg_per_time_unit)

a-2) cycling
L = 1 - alpha_sei * exp(-N * beta_sei * deg_per_cyc) - (1 - alpha_sei) * exp(-N * deg_per_cyc)

b) linearised model
a-1) calendar
deg_per_time_unit = time_deg_model(t) * soc_stress_model(SoC) * temp_stress_model(T)

a-2) cycling
deg_per_cyc = cyc_dod_deg_model(DoD) * soc_stress_model(SoC) * temp_stress_model(T) * voltage_stress_model(SoC)
-----------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np

from degradation_model.cyc_dod_deg_model_fit import CyclingDegModelFit
from degradation_model.degradation_model import voltage_stress_model
from degradation_model.DST_cycle import DSTCycleDeg, plot_DST_experimental_data
from degradation_model.SEI_fit import SEI_fit
from degradation_model.soc_stress_model_fit import SoCStressModelFit
from degradation_model.temp_stress_model_fit import TempStressModelFit
from degradation_model.time_deg_model_fit import TimeDegModelFit
from degradation_model.stress_surface_representation import surface_stress_plot
from degradation_model.cycle_counting_algorithm import CycleCounter

# -----------------------------------------------------------------------------------
# ----------------- Part 1: model parameters identification -------------------------

print()
print("******** MODEL PARAMETERS ********")

# ----------------- Part 1a: SEI fit of the non linear degradation model ------

# -- SEI fit --------------------------------------------------------
data_file_path = 'degradation_model/data/cycling_degradation/cyc_test_data.csv'
SEI_fit1 = SEI_fit(data_file_path)

# -- Plot --------------------------
fig = plt.figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)
SEI_fit1.plot_sei_model(ax=ax)

# output of SEI_fit, written in order not to launch SEI_fit each time
alpha_sei = 5.87e-02
beta_sei = 1.06e+02


# ----------------- Part 1b: Stress factor outside of standard conditions ------

# -- Temperature stress model fit -------------------------------------
data_file_path = 'degradation_model/data/calendar_degradation/calend_deg_at_50_SoC.csv'
TempStressModelFit1 = TempStressModelFit(data_file_path,
                                         alpha_sei=alpha_sei,
                                         beta_sei=beta_sei)

# -- Plot -------------------------------
fig = plt.figure(figsize=(8, 3), dpi=100)
ax = fig.add_subplot(121)
TempStressModelFit1.plot_SEI_fit_at_T(ax)
ax = fig.add_subplot(122)
TempStressModelFit1.plot_model(ax)


# -- State of Charge stress model fit ---------------------------------
data_file_path = 'degradation_model/data/calendar_degradation/calend_deg_at_25_deg.csv'
SoCStressModelFit1 = SoCStressModelFit(data_file_path,
                                       alpha_sei=alpha_sei,
                                       beta_sei=beta_sei)

# -- Plot -------------------------------
fig = plt.figure(figsize=(8, 3), dpi=100)
ax = fig.add_subplot(121)
SoCStressModelFit1.plot_SEI_fit_at_T(ax)
ax = fig.add_subplot(122)
SoCStressModelFit1.plot_model(ax)


# -- Voltage stress model ----------------------------------------------
fig = plt.figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)
ax.set_xlabel("State of charge [%]")
ax.set_ylabel("Stress")
ax.set_title("Voltage stress model")
plt.tight_layout()

soc_linspace = np.linspace(0, 100, 100)
voltage_stress_model = voltage_stress_model(soc_linspace/100)
ax.plot(soc_linspace, voltage_stress_model, 'b')


# ----------------- Part 1c: Degradation model under standard conditions -------

# -- Time degradation model fit ---------------------------------------
data_file_path = 'degradation_model/data/calendar_degradation/calend_deg_at_25_deg.csv'
TimeDegModelFit1 = TimeDegModelFit(data_file_path,
                                   alpha_sei=alpha_sei,
                                   beta_sei=beta_sei)


# -- Depth of Discharge degradation model fit -------------------------
data_file_paths = ['degradation_model/data/cycling_degradation/cycle_nb_at_80_SoH_NMC.csv',
                   'degradation_model/data/cycling_degradation/cycle_nb_at_80_SoH_LMO.csv',
                   'degradation_model/data/cycling_degradation/cycle_nb_at_80_SoH_LFP.csv']
CyclingDegModelFit1 = CyclingDegModelFit(data_file_paths)

# -- Plot -------------------------------
fig = plt.figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)
CyclingDegModelFit1.plot_dod_graph(ax)


# ----------------- Part 1d: Stress surface plot -------------------------------

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

surface_stress_plot(ax1, include_voltage_stress=False)
surface_stress_plot(ax2, include_voltage_stress=True)


# -----------------------------------------------------------------------------------
# ----------------- Part 2: Model validation with DST cycling data ------------------


# Plot one example of DST cycle
fig = plt.figure(figsize=(8, 3), dpi=100)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

DSTCycleDeg1 = DSTCycleDeg(45, 75, alpha_sei, beta_sei, "NMC", 21)
DSTCycleDeg1.plot_DST_profile(ax1, ax2)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey', 'brown']

v_soc_min = [25, 40, 25, 50, 25, 45, 65]
v_soc_max = [100, 100, 85, 100, 75, 75, 75]

fig = plt.figure(figsize=(9, 4), dpi=100)
ax1 = fig.add_subplot(121)
plot_DST_experimental_data(ax1, data_folder_path='degradation_model/data/DST_cycles/')

for i in range(0, len(v_soc_max)):
    ax2 = fig.add_subplot(122)
    soc_min = v_soc_min[i]
    soc_max = v_soc_max[i]
    label = str(soc_min) + '-' + str(soc_max) + ' at 20°C'
    DSTCycleDeg1 = DSTCycleDeg(soc_min, soc_max, alpha_sei, beta_sei, "LMO", 21)
    DSTCycleDeg1.plot_DST_deg_model(ax2, color=colors[i], label=label)


# -----------------------------------------------------------------------------------
# ----------------- Part 3: Model usage example -------------------------------------

scenario = "new_Flat_9vdb_10MW_5Wh_2014_sc26"
# scenario = "new_Linear_9vdb_10MW_5Wh_2014_sc19"
# scenario = "new_Tailored_9vdb_10MW_5Wh_2014_sc55"
# scenario = "new_Tailored_9vdb_40Triads_10MW_5Wh_2015_sc224"
# scenario = "new_Steep_15vdb_4SOCM_Triads_10MW_5Wh_2014_sc25"

# cycleCount1 = CycleCounter(data_file_path='./data/Green_Hedge/' + scenario + '.csv', delta=0.1)
# cycleCount1.turning_points_extraction()
# cycleCount1.rainflow_process(output_file_path='output/Green_Hedge/' + scenario + '.csv')
#  -----------------------------------------------------------------------------


plt.show()  # this must be at the end of the code

