#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:06:46 2019

@author: hutianqi
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

np.random.seed(666)

start_time = time.time()

# This is an attempt to replicate the food choice aDDM results
# from the Smith 2018 'Attention and Choice Across Domains'
# Dataset is obtained from Smith 2018
# However some paremeters are taken from the Krajbich 2010

###Load data
foodc_um = pd.read_csv('./twofoodchoicedata.csv')
#twofoodchoicedata.csv
foode_um = pd.read_csv('./twofoodeyedata.csv')
#twofoodeyedata.csv


# =============================================================================
# Preparation
# =============================================================================

# Smith 2018 p. 1813
# trials with very long (more than two standard deviations above the log-transformed mean)
# or very short (􏰃300 ms) RTs were removed from analysis
"""It seems that the ourliers have already been taken out."""


foodc = foodc_um.copy()
foode = foode_um.copy()

### add the sum of dwell time to foodc
foodc['DwellSum'] = np.nan

sum_dwell = foode.pivot_table(index = 'SubjectNumber', columns = 'Trial',
                        values = 'DwellLength', aggfunc = 'sum')

for i in range(foodc.shape[0]):
    foodc['DwellSum'][i] = \
    sum_dwell.loc[foodc['SubjectNumber'][i], foodc['Trial'][i]]

# Smith 2018 p. 1811
# left-eye fixation patterns and pupil diameter were recorded at 1000 Hz
# to calculate drift amount based on 1000 HZ

HZ = 1000

foode['DriftAmount'] = np.nan

for i in range(foode.shape[0]):
    foode['DriftAmount'][i] = int(foode['DwellLength'][i] * HZ)

#foode.index = foode['SubjectNumber']

### create matrixs of food (option) values

values_left = foodc.pivot_table(index = 'SubjectNumber', columns = 'Trial',
                                values = 'ValueLeft')

values_right = foodc.pivot_table(index = 'SubjectNumber', columns = 'Trial',
                                values = 'ValueRight')


### create a list of middle gazes
# Smith 2018 p. 1815
# Whenever a simulation did not reach a barrier by the time the subject had actually decided,
# we continued the simulation using alternating gaze locations (e.g., left, right, left, . . .)
# and randomly selected dwell times,
# which we sampled from the pool of all observed middle (i.e., neither first nor last) gazes.

middle_gaze = []
foode['MiddleGaze'] = np.nan
for i in range(foode.shape[0]):
    if i == 0:
        foode['MiddleGaze'] = False
    elif i != foode.shape[0] - 1:
        # all but the last one recording
        if foode['Trial'][i] == foode['Trial'][i-1] and foode['Trial'][i] == foode['Trial'][i+1]:
            foode['MiddleGaze'][i] = True
            middle_gaze.append(foode['DwellLength'][i])
        else:
            foode['MiddleGaze'][i] = False
    else:
        foode['MiddleGaze'][i] = False


# =============================================================================
# These functions return simulated choice (choice_simu) and
# dwell time summation (dwell_simu)for a specific subject's specific trial.
# =============================================================================

# Simulation for a specific subject's specific trial
def addm(subject_num, trial, d, th, sigma):

    boundary_top = 1
    boundary_bot = -1

    rL = values_left.loc[subject_num, trial]
    rR = values_right.loc[subject_num, trial]

    drifts_record = []

    V = 0

    dum1 = foode[foode['SubjectNumber'] == subject_num]
    focus = dum1[dum1['Trial'] == trial]
    focus = focus.reset_index(drop=True)

#   record all simulated drifts based on option values and fixation
#   without considering accumaltion and cross boundry
    for i in range(focus.shape[0]):
        for repe in range(int(focus['DriftAmount'][i])):
            if focus['ROI'][i] == 1:
                #look left
                drift = d * (rL - th * rR) + np.random.normal(0, sigma)
                drifts_record.append(drift)

            else:
                #look right
                drift = - d * (rR - th * rL) + np.random.normal(0, sigma)
                drifts_record.append(drift)

    # calculate the relative decision value by adding drifts together and
    # check if the boundry is crossed.
    # if the boundry is  crossed within the given period
    # of the actual dwell length, The simulation is
    # defined as a 'natural end',
    # othewise the simulation will operate based on some bold assumptions (see below)

    natural_end = np.nan
    drift_counter = 0
    while drift_counter <= (len(drifts_record) - 1):
        V += drifts_record[drift_counter]
        drift_counter += 1

        if V >= boundary_top:
            # to choose option on the left
            choice_simu = 1
            # RT in second
            dwell_simu = drift_counter / HZ
            # the boundry is crossed 'naturally'
            natural_end = 1
            break

        elif V <= boundary_bot:
            # to choose option on the right
            choice_simu = 2
            # RT in second
            dwell_simu = drift_counter / HZ
            # the boundry is crossed 'naturally'
            natural_end = 1
            break

        else:
            natural_end = 0


    if natural_end == True:
        pass

    else:
        ROI = np.random.randint(1,3)

        choice_simu, dwell_simu = random_gaze(d, th, sigma, rL, rR, V, boundary_top, boundary_bot, drift_counter, middle_gaze, ROI)

    return choice_simu, dwell_simu, natural_end


# This function deals with the situation where
# a barrier is not reached by the time the subject had actually decided
def random_gaze(d, th, sigma, rL, rR, V, boundary_top, boundary_bot, drift_counter, middle_gaze, ROI):

    ROI = ROI
    drift_amount = int(np.random.choice(middle_gaze) * HZ)

    extra_drifts_record = []
    for i in range(drift_amount):
        if ROI == 1:
            #look left
            drift = d * (rL - th * rR) + np.random.normal(0, sigma)
            extra_drifts_record.append(drift)
            ROI += 1
            # alternate gaze location in the next round

        else:
            #look right
            drift = - d * (rR - th * rL) + np.random.normal(0, sigma)
            extra_drifts_record.append(drift)
            ROI -= 1

    end = np.nan
    extra_drift_counter = 0
    while extra_drift_counter <= (len(extra_drifts_record) - 1):
        V += extra_drifts_record[extra_drift_counter]
        extra_drift_counter += 1

        if V >= boundary_top:
            # to choose option on the left
            choice_simu = 1
            # RT in second
            drift_counter += extra_drift_counter
            dwell_simu = drift_counter / HZ
            # the boundry is crossed 'naturally'
            end = 1
            break

        elif V <= boundary_bot:
            # to choose option on the right
            choice_simu = 2
            # RT in second
            drift_counter += extra_drift_counter
            dwell_simu = drift_counter / HZ
            # the boundry is crossed 'naturally'
            end = 1
            break

        else:
            end = 0


    if end == True:
        return choice_simu, dwell_simu

    else:
        return random_gaze(d, th, sigma, rL, rR, V, boundary_top, boundary_bot, drift_counter, middle_gaze, ROI)


# =============================================================================
# This funtion calls the addm function and applies it to
# every subject's every trial, then returning a dataframe that
# builds on foodc with additional columns of
# simulatied choice and RT.
# =============================================================================

# Parameter values are taken from Smith 2018 p.1820
# non decision time (non_time) is estimated to be 425ms
def addm_apply(choice_dataset, eye_dataset, d, th, sigma, non_time):

    foodc = choice_dataset
    foode = eye_dataset

    foodc['DwellSum_simu'] = np.nan
    foodc['RT_simu'] = np.nan
    foodc['LeftRight_simu'] = np.nan
    foodc['Natural_Termination'] = np.nan

    choice_simu_record = pd.DataFrame(index = pd.unique(foodc['SubjectNumber']), columns = pd.unique(foodc['Trial']))
    dwell_simu_record = pd.DataFrame(index = pd.unique(foodc['SubjectNumber']), columns = pd.unique(foodc['Trial']))
    natural_end_record = pd.DataFrame(index = pd.unique(foodc['SubjectNumber']), columns = pd.unique(foodc['Trial']))

    for subject_num in np.unique(foode['SubjectNumber']):
#        print(subject_num)
        for trial in np.unique(foode[foode['SubjectNumber'] == subject_num]['Trial']):
#            print(trial)
            choice_simu, dwell_simu, natural_end = addm(subject_num, trial, d, th, sigma)

            choice_simu_record.loc[subject_num, trial] = choice_simu
            dwell_simu_record.loc[subject_num, trial] = dwell_simu
            natural_end_record.loc[subject_num, trial] = natural_end


    for i in range(foodc.shape[0]):
        foodc['DwellSum_simu'][i] = \
        dwell_simu_record.loc[foodc['SubjectNumber'][i], foodc['Trial'][i]]

        foodc['LeftRight_simu'][i] = \
        choice_simu_record.loc[foodc['SubjectNumber'][i], foodc['Trial'][i]]

        foodc['Natural_Termination'][i] = \
        natural_end_record.loc[foodc['SubjectNumber'][i], foodc['Trial'][i]]

    # The simulated RT equals to the simulated dwell time summation plus
    # the estimated non decision time
    foodc['RT_simu'] = foodc['DwellSum_simu'] + non_time

    return foodc


def addm_assess(simulation_output):

    foodc = simulation_output

    # to calculate the accuracy of choice simulation
    counter_c = 0
    for i in range(foodc.shape[0]):
        if foodc['LeftRight'][i] == foodc['LeftRight_simu'][i]:
            counter_c += 1

    choice_simu_accuracy = counter_c / foodc.shape[0]
    print('The accuracy of choice simulation is %.2f percent' % (choice_simu_accuracy * 100))

    # to calculate the accuracy of dwell length and RT simulation
    please_be_good = abs(foodc['DwellSum_simu'] - foodc['DwellSum'])
    print('The simulation on average deviates from the actual dwell length summation by %.2f seconds' % please_be_good.mean())

    please_please_be_good = abs(foodc['RT_simu'] - foodc['RT'])
    print('The simulation on average deviates from the actual RT by %.2f seconds' % please_please_be_good.mean())

    # to display the number of natural termination
    lets_see = foodc['Natural_Termination'].sum()
    print('%.2f percent of the simulation terminated within the actural dwell length' % (lets_see / foodc.shape[0] * 100))


d = 0.0023
th = 0.44
sigma = 0.029
non_time = 0.425

simulation_dataset = addm_apply(foodc, foode, d, th, sigma, non_time)
addm_assess(simulation_dataset)


# =============================================================================
# Plot patterns
# =============================================================================

foodc = simulation_dataset

foodc['Value_Diff_LmR'] = foodc['ValueLeft'] - foodc['ValueRight']

foodc.insert(8, 'Value_Diff_LmR', foodc.pop('Value_Diff_LmR'))

foodc.to_csv("./foodc_simu.csv")
# just to save the data with the addition of simulation outcome

### RT
RT_mean = foodc.pivot_table(index = 'Value_Diff_LmR',
                        values = 'RT', aggfunc = 'mean')

RT_simu_mean = foodc.pivot_table(index = 'Value_Diff_LmR',
                        values = 'RT_simu', aggfunc = 'mean')

RT_mean['RT_simu'] = np.nan
RT_mean['RT_simu'] = RT_simu_mean['RT_simu']

RT_mean.to_csv('./RT_mean.csv')

plt.figure()
RT_plot = RT_mean.plot(style='.-')
RT_plot.set_title("Two Food Choice: Response Time")
RT_plot.set_xlabel("Utility Difference (L - R)")
RT_plot.set_ylabel("Response Time")

RT_plot = RT_plot.get_figure()
RT_plot.savefig('./RT_plot.png')


### Choice

choice_prob = pd.DataFrame(index = foodc['Value_Diff_LmR'].unique(),
                           columns = ['left_prob', 'left_prob_simu'])

foodcgb = foodc.groupby('Value_Diff_LmR')

for index, content in foodcgb:
    prob = content[content['LeftRight'].isin([1])].shape[0] / content.shape[0]
    prob_simu = content[content['LeftRight_simu'].isin([1])].shape[0] / content.shape[0]

    choice_prob.loc[index, 'left_prob'] = prob
    choice_prob.loc[index, 'left_prob_simu'] = prob_simu

choice_prob = choice_prob.sort_index(ascending=True)

choice_prob.to_csv('./choice_prob.csv')

plt.figure()
choice_plot = choice_prob.plot(style='.-')
choice_plot.set_title("Two Food Choice: Choice Probability")
choice_plot.set_xlabel("Utility Difference (L - R)")
choice_plot.set_ylabel("Probability of Choosing Left")

choice_plot = choice_plot.get_figure()
choice_plot.savefig('./choice_plot.png')






run_time = time.time() - start_time
print("--- %s minutes %.2s seconds ---" % (int(run_time // 60), run_time % 60))
