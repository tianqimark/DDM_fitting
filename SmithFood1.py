#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smith 2018 two food choice DDM fitting (Trial 1)

This is an attempt to fit a DDM model to the smith 2018 two food choice data,
Using the package PyDDM.

Created on Thu Apr 16 22:20:04 2020

@author: hutianqi
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# np.random.seed(666)

start_time = time.time()

###Load data
foodc_um = pd.read_csv('./twofoodchoicedata.csv')
#twofoodchoicedata.csv / LeftRight: subject response (1 = left, 2 = right)
foode_um = pd.read_csv('./twofoodeyedata.csv')
#twofoodeyedata.csv
#Eye gaze data is not used yet in this program

# =============================================================================
# Preprocessing data
# =============================================================================
foodc = foodc_um.copy()

# Drop trials where a correct choice does not exist
foodc = foodc[foodc['ValueLeft'] != foodc['ValueRight']]
foodc = foodc.reset_index(drop = True)

# add a column to record whether the choice made is correct
foodc['Correct'] = np.nan
for i in range(foodc.shape[0]):
    if foodc['ValueLeft'][i] > foodc['ValueRight'][i] and foodc['LeftRight'][i] == 1:
        foodc['Correct'][i] = 1
    elif foodc['ValueLeft'][i] < foodc['ValueRight'][i] and foodc['LeftRight'][i] == 2:
        foodc['Correct'][i] = 1
    else:
        foodc['Correct'][i] = 0


# Create a smaller dataset for test to save time
foodc = foodc[foodc['SubjectNumber'] <= 5]
# foodc = foodc[foodc.index < 500]

foodc = foodc.reset_index(drop = True)


# =============================================================================
# DDM fitting
# =============================================================================

from ddm import Sample

# Create a sample object from the foodc data, and define RT and Correct/Error columns
foodc_sample = Sample.from_pandas_dataframe(foodc, rt_column_name="RT", correct_column_name="Correct")

### Formulate the drift rate
import ddm.models
class DriftFoodc(ddm.models.Drift):
    name = "Drift depends on the values of the food items"
    required_parameters = ["d"] # <-- Parameters we want to include in the model
    required_conditions = ["ValueLeft", "ValueRight"] # <-- Task parameters ("conditions"). Should be the same name as in the sample.

    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.d * (conditions['ValueLeft'] - conditions['ValueRight'])

class NoiseFoodc(ddm.models.Noise):
    name = "Noise of Gaussian structure"
    required_parameters = ["sigma"]
    required_conditions = []

    def get_noise(self, conditions, **kwargs):
        noise = np.random.normal(0, self.sigma)

        # for the default fitting methods (differential evolution) the input cannot be zero
        # This is just a temorary solution
        if noise != 0:
            return noise
        else:
            return noise + 0.001



from ddm import Model, Fittable
from ddm.functions import fit_adjust_model, display_model
from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture
model_foodc = Model(name='Data from Smith 2018 two food choice. Fit a regular DDM to it',
                 drift=DriftFoodc(d = Fittable(minval = 0, maxval = 0.001)),
                 noise=NoiseFoodc(sigma = Fittable(minval = 0, maxval = 0.05)),
                 bound=BoundConstant(B = 1),
                 overlay=OverlayChain(overlays =
                                      [OverlayNonDecision(nondectime = Fittable(minval=0, maxval=1)),
                                       OverlayPoissonMixture(pmixturecoef=.02, rate=1)]),
                 dx=.001, dt=.01, T_dur=10)


# Check if an analytical solution exists.
Model.has_analytical_solution(model_foodc)
# For the current model: True


### Fit the model to the dataset to find parameters

# Use the "differential_evolution" fitting method (default, recommanded fitting method)
fit_model_foodc = fit_adjust_model(sample = foodc_sample, model = model_foodc, method = 'differential_evolution')

# Use the "simple" fitting method (much faster)
# fit_model_foodc = fit_adjust_model(sample = foodc_sample, model = model_foodc, method = 'simple')

# To note, In documentation the key word "method" is documented as "fitting_method",
# which is actually invalid now.


# Display the fitting outcome (parameters)
display_model(fit_model_foodc)


### Plot
import ddm.plot
# import matplotlib.pyplot as plt
ddm.plot.plot_fit_diagnostics(model = fit_model_foodc, sample = foodc_sample, data_dt = .01)
plt.savefig("./foodc_PyDDM1.png")
# plt.savefig("./foodc_PyDDM1.png", bbox_inches='tight')
plt.show()

# At the moment the plot is incorrect

# The warning information from execution:
# /opt/anaconda3/lib/python3.7/site-packages/ddm/plot.py:232:
# UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
#   fig.tight_layout()
















run_time = time.time() - start_time
print("--- %s minutes %.2s seconds ---" % (int(run_time // 60), run_time % 60))
