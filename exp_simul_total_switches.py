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
from estimator import chao92, qaChao92, fChao92, sChao92,nominal, vNominal,  sNominal, wChao92, bChao92, dChao92, goodToulmin, vGoodToulmin, remain_switch, gt_switch, gt_marginal, gt_remaining, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, unseen, twoPhase, vRemainSwitch2, minTasks, minTasks2, minTasks3, minTasksToCleanAll,  extrapolateFromSample
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, simulatedData2, loadInstitution, loadCrowdFlowerData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle
from simulation import plotMulti, plotY1Y2, holdoutRealWorkers



######################################
###########simulated dataset##########
######################################
print 'Sensitivity of Total Error Estimation'
#estimators_sim = [vNominal, chao92, lambda x:sChao92(x,shift=1),lambda x:vRemainSwitch2(x)]
#legend_sim = ['VOTING','Chao92','V-CHAO','SWITCH']
#gt_list_sim = [lambda x:gt,lambda x:gt,lambda x:gt,lambda x:gt]
estimators_sim = [lambda x:vNominal(x),chao92,lambda x:vRemainSwitch2(x)]
gt_list_sim = [lambda x:gt,lambda x:gt,lambda x:gt]
legend_sim = ['VOTING','Chao92','SWITCH']
legend_gt=["Ground Truth"]

rel_err = False
err_skew = False
logscale=False
dirty = 0.1
n_items = 1000
n_rep = 3
hir = 0.1 #0.1
lowr = 0.015 #30 items per task
fnr = .1 # 10%? was 1%
fpr = .01 # 1%? was 10%

print 'Sensitivity of Total Error Estimation'
#estimators_sim = [vNominal, chao92, lambda x: sChao92(x,shift=1),lambda x: vRemainSwitch2(x)]
estimators_sim = [vNominal]
#estimators_sim = [vNominal, chao92, lambda x: vRemainSwitch2(x)]
#gt_list_sim=[lambda x: gt, lambda x: gt, lambda x: gt, lambda x: gt]
gt_list_sim=[lambda x: gt]
#gt_list_sim=[lambda x: gt, lambda x: gt, lambda x: gt]
#legend_sim = ['VOTING','Chao92','V-CHAO(s=1)','SWITCH']
legend_sim = ['VOTING']
#legend_sim = ['VOTING', 'Chao92', 'SWITCH']
yaxis='# Total Error'

n_worker = 500
scale = 50
init = 0

ymax=190
err_skew = False
error_type = 1
show_min_tasks = False

title = 'Observed # Errors'
print title
d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=fpr,fnr=fnr, err_skew=err_skew,error_type=error_type)
min_tasks=minTasksToCleanAll(d,30)
print min_tasks, 'needed'
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_b.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_b.p','rb'))
#plotMulti(ax[0][1], (X,Y,GT),
#plotMulti(ax[0], (X,Y,GT),
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        ymax=ymax,
                        vertical_y=50,
                        yaxis='',
                        xaxis='Tasks',
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_00.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks,
                        show_min_tasks=show_min_tasks
                        )
"""
title = '(a) False Negative Errors'
print title
error_type = 2

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=fpr,fnr=fnr,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#min_tasks=minTasks2(d,0.8)
min_tasks=minTasksToCleanAll(d,30)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_a.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_a.p','rb'))
plotMulti(ax[0], (X,Y,GT),
#plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        ymax=ymax,
                        vertical_y=50,
                        yaxis=yaxis,
                        xaxis='Tasks',
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_01.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks,
                        show_min_tasks=show_min_tasks
                        )

title = '(c) All Errorrs'
print title
error_type = 0

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=fpr, fnr=fnr,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
min_tasks=minTasksToCleanAll(d,30)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_c.p','wb'))
#(X,Y,GT) = pickle.load(open('dataset/matrix_c.p','rb'))
#plotMulti(ax[0][2], (X,Y,GT),
plotMulti(ax[2], (X,Y,GT),
#plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        yaxis='',
                        title=title, 
                        vertical_y=50,
                        ymax=ymax,
                        xaxis='Tasks',
                        loc='lower right',
                        filename='plot/sim_02.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks,
                        show_min_tasks=show_min_tasks
                        )
"""


#====switch==== 
estimators2 = [lambda x: remain_switch(x) - sNominal(x)]
estimators2a = [lambda x: remain_switch(x,neg_switch=False) - sNominal(x,neg_switch=False)]
estimators2b = [lambda x: remain_switch(x,pos_switch=False) - sNominal(x,pos_switch=False)]
gt_list2 = [lambda x: gt_switch(x,slist)]
gt_list2a = [lambda x: gt_switch(x,slist,neg_switch=False)]
gt_list2b = [lambda x: gt_switch(x,slist,pos_switch=False)]
legend2 = ["REMAIN-SWITCH"]
legend2a = ["REMAIN-SWITCH (+)"]
legend2b = ["REMAIN-SWITCH (-)"]
legend_gt2 = ["Ground Truth"]

fig, ax = plt.subplots(1,3,figsize=(20,5),sharex=True,sharey=True)
#positive switch estimation
print 'switch estimation'
d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=fpr, fnr=fnr,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#(X,Y,GT) = holdoutRealWorkers(d,gt_list2a,range(init,n_worker,scale),estimators2a,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_switches_a.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_switches_a.p','rb'))
plotY1Y2((X,Y,GT),
#plotMulti(ax[1], (X,Y,GT),
                        legend = legend2a,
                        legend_gt = legend_gt2,
                        xaxis="Tasks",
                        yaxis="# Remaining Switches",
                        ymax=80,
                        title='Remaining Positive Switches',
                        filename="plot/sim_tot_err_pos_switch.png",
                        )
#negative estimation
d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=fpr, fnr=fnr,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#(X,Y,GT) = holdoutRealWorkers(d,gt_list2b,range(init,n_worker,scale),estimators2b,rel_err=rel_err,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/matrix_switches_b.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_switches_b.p','rb'))
plotY1Y2((X,Y,GT),
#plotMulti(ax[2], (X,Y,GT),
                        legend = legend2b,
                        legend_gt = legend_gt2,
                        xaxis="Tasks",
                        yaxis="# Remaining Switches",
                        ymax=80,
                        title='Remaining Negative Switches',
                        filename="plot/sim_tot_err_neg_switch.png",
                        )


