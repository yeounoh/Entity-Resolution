#!/usr/bin/env python
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy
import scipy
import csv
import math
import random
import simplejson
from estimator import chao92, qaChao92, fChao92, sChao92,nominal, vNominal,  sNominal, wChao92, bChao92, dChao92, goodToulmin, vGoodToulmin, remain_switch, gt_switch, gt_marginal, gt_remaining, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, unseen, twoPhase, vRemainSwitch2, minTasks, minTasks2, nominalF1, nominalF2, nominalCov, positiveVotes, minTasks3, extrapolateFromSample
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, simulatedData2, loadInstitution, loadCrowdFlowerData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle
from simulation import plotMulti, plotY1Y2, holdoutRealWorkers




######################################
###########simulated dataset##########
######################################
logscale=False
dirty = 0.1
n_rep = 100

print 'How many items would be discovered with 20 workers/tasks?'

estimators_sim = [nominal,nominalCov,nominalF1,nominalF2, chao92, lambda x: chao92(x,False), lambda x: sChao92(x,shift=1),positiveVotes]
gt_list_sim = [lambda x:gt,lambda x: gt,lambda x: gt, lambda x: gt,lambda x: gt, lambda x: gt, lambda x: gt, lambda x: gt]
legend_sim = ['Observed','Covered','f1','f2', 'chao92','chao92(unc)','vChao92','n']
legend_gt=["Ground Truth"]
yaxis = 'Estimate (# Total Error)'#'Relative Error %'


rel_err = False
err_skew = False

hir = 0.1 #0.1
lowr = 0.02
hiq = 1
lowq = 0.95
fpr = 0.0
fnr = .1

title = 'Perfect Precision'
print title

error_type = 0
n_worker = 410
scale = 10
init = 50

d,gt,prec,rec = simulatedData(items=1000,workers=n_worker,fpr=fpr,fnr=fnr,dirty=dirty,recall=lowr,precision=hiq,err_skew=err_skew)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
min_tasks = minTasks3(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Workers/Tasks',
                        yaxis=yaxis,
                        ymax=200,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_test_fpr0_fnr10.png',
                        logscale=logscale,
                        rel_err=rel_err,
                        min_tasks2=min_tasks
                        )

