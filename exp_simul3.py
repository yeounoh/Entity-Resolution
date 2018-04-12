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
logscale=False
dirty = 0.5
n_items = 1000
n_rep = 3


print 'Sensitivity of Total Error Estimation'
#estimators_sim = [vNominal, chao92, lambda x:sChao92(x,shift=1),lambda x:vRemainSwitch2(x)]
#legend_sim = ['VOTING','Chao92','V-CHAO','SWITCH']
#gt_list_sim = [lambda x:gt,lambda x:gt,lambda x:gt,lambda x:gt]
estimators_sim = [lambda x:vNominal(x),chao92,lambda x:vRemainSwitch2(x)]
gt_list_sim = [lambda x:gt,lambda x:gt,lambda x:gt]
legend_sim = ['VOTING','Chao92','SWITCH']
legend_gt=["Ground Truth"]
yaxis = 'SRMSE'#'Relative Error %'

rel_err = True
err_skew = False

title = 'Tradeoff: False Positives'
recall = 0.1
n_worker=50
font = 20
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    prec = float(i)/10.
    d,gt,pr,re = simulatedData(items=n_items,recall=recall,precision=prec,error_type=0,err_skew=err_skew)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[n_worker],estimators_sim,rel_err=rel_err,rep=n_rep)
    Xs.append(prec*100)
    Ys.append(Y[0])
    GTs.append(GT[0])

plotY1Y2((Xs,Ys,GTs),
                        legend = legend_sim,
                        xaxis='Precision %',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_5a.png',
                        logscale=logscale,
                        rel_err=rel_err,
                        font=font
                        )
title = 'Tradeoff: False Negatives'
precision = 1
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    rec = float(i)/200.
    d,gt,pr,re = simulatedData(items=n_items,recall=rec,precision=precision,error_type=0,err_skew=err_skew)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[n_worker],estimators_sim,rel_err=rel_err,rep=n_rep)
    Xs.append((rec)*100)
    Ys.append(Y[0])
    GTs.append(GT[0])

plotY1Y2((Xs,Ys,GTs),
                        legend = legend_sim,
                        xaxis='Coverage %',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_5d.png',
                        logscale=logscale,
                        rel_err=rel_err,
                        font=font
                        )


print 'Sensitivity of Total Error Estimation'
estimators_sim = [vNominal, chao92, lambda x: sChao92(x,shift=1),lambda x: vRemainSwitch2(x)]
gt_list_sim=[lambda x: gt, lambda x: gt, lambda x: gt, lambda x: gt]
legend_sim = ['VOTING','Chao92','V-CHAO(s=1)','SWITCH']
estimators_sim = [chao92]
gt_list_sim=[ lambda x: gt]
legend_sim = ['Chao92']
yaxis='Estimate (# Total Error)'

rel_err = False

hir = 0.1 #0.1
lowr = 0.015 #30 items per task
hiq = 0.99 
lowq = 0.9
fnr = 0.1 # 10%? was 1%
fpr = 0.01 # 1%? was 10%

title = 'Perfect Precision'
print title

err_skew = False
error_type = 0
n_worker = 330
scale = 30
init = 30

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=1,err_skew=err_skew)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#min_tasks = minTasks2(d,0.8)
min_tasks = minTasksToCleanAll(d)#minTasks3(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/7.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/7.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        ymax=250,
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_7.png',
                        logscale=logscale,
                        rel_err=rel_err,
                        min_tasks2=min_tasks
                        )

fig, ax = plt.subplots(2,3,figsize=(20,8),sharex=True,sharey=True)

n_worker = 500
scale = 50
init = 0

ymax=350
err_skew = False
error_type = 1

title = 'False Positive Errors\n\n(b)'
print title
d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=hiq, err_skew=err_skew,error_type=error_type)
min_tasks=minTasks2(d,0.8)
min_tasks=minTasksToCleanAll(d)#minTasks3(d)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/matrix_b.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_b.p','rb'))
plotMulti(ax[0][1], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        ymax=ymax,
                        yaxis='',
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_00.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        ymax=ymax,
                        yaxis='',
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_00.png',
                        rel_err=rel_err,
                        #min_tasks=min_tasks
                        )

title = 'False Negative Errors\n\n(a)'
print title
error_type = 2

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#min_tasks=minTasks2(d,0.8)
min_tasks=minTasksToCleanAll(d) #minTasks3(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/matrix_a.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_a.p','rb'))
plotMulti(ax[0][0], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        ymax=ymax,
                        vertical_y=ymax-100,
                        yaxis='Uniform Precision\n\n' + yaxis,
                        title=title, 
                        loc='lower right',
                        filename='plot/sim_01.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

title = 'All Errors\n\n(c)'
print title
error_type = 0

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=1-hiq, fnr=1-lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#min_tasks=minTasks2(d,0.8)
min_tasks=minTasksToCleanAll(d) #minTasks3(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/matrix_c.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_c.p','rb'))
plotMulti(ax[0][2], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        yaxis='',
                        title=title, 
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_02.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )


err_skew = True
error_type = 1

title = 'False Positive Errors'
print title
d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=hiq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#min_tasks=minTasks2(d,0.8)
min_tasks=minTasksToCleanAll(d)#minTasks3(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/matrix_e.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_e.p','rb'))
plotMulti(ax[1][1], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis='',
                        title='(e)',
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_10.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

title = 'False Negative Errors'
print title
error_type = 2

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,precision=lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#min_tasks=minTasks2(d,0.8)
min_tasks=minTasks3(d)
min_tasks=minTasksToCleanAll(d)#minTasks3(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/matrix_d.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_d.p','rb'))
plotMulti(ax[1][0], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis='Skewed Precision\n\n' + yaxis,
                        vertical_y=ymax-100,
                        title='(d)',
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_11.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

title = 'All Errors'
print title
error_type = 0

d,gt,prec,rec = simulatedData(items=n_items,workers=n_worker,dirty=dirty,recall=lowr,fpr=1-hiq,fnr=1-lowq,err_skew=err_skew,error_type=error_type)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
#min_tasks=minTasks2(d,0.8)
min_tasks=minTasks3(d)
min_tasks=minTasksToCleanAll(d)#minTasks3(d)
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=rel_err,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/matrix_f.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/matrix_f.p','rb'))
plotMulti(ax[1][2], (X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='Tasks',
                        yaxis='',
                        title='(f)',
                        ymax=ymax,
                        loc='lower right',
                        filename='plot/sim_12.png',
                        rel_err=rel_err,
                        min_tasks=min_tasks
                        )

plt.legend(loc='lower center',bbox_to_anchor = (0,-0.1,1,1),ncol=5,bbox_transform=plt.gcf().transFigure)
#plt.tight_layout()
plt.savefig('plot/sim_matrix_fnr10_fpr1_1000_100.png',bbox_inches='tight')

'''
print 'Figure 10, Heuristics'
n_rep=1
estimators_sim = [lambda x:vRemainSwitch2(x)]
legend_sim = ['SWITCH']
gt_list_sim = [lambda x:gt]
logscale = False
title = 'Heuristic 10% Error'
print title
recall = 0.1
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    eps = float(i)/10.
    d,gt,pr,re = simulatedData(recall=recall,precision=1.,priotized=True,eps=eps,hdirty=0.1)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[50],estimators_sim,rel_err=True,rep=n_rep)
    Xs.append(eps)
    Ys.append(Y[0])
    GTs.append(GT[0])

plotY1Y2((Xs,Ys,GTs),
                        legend = legend_sim,
                        xaxis='Epsilon ($\epsilon$)',
                        yaxis=yaxis,
                        ymax=1.0,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_10a.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )

title = 'Heuristic 50% Error'
print title
n_worker = 50
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    eps = float(i)/10.
    d,gt,pr,re = simulatedData(recall=recall,precision=1.,priotized=True,eps=eps,hdirty=0.5)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[50],estimators_sim,rel_err=True,rep=n_rep)
    Xs.append(eps)
    Ys.append(Y[0])
    GTs.append(GT[0])

plotY1Y2((Xs,Ys,GTs),
                        legend = legend_sim,
                        xaxis='Epsilon ($\epsilon$)',
                        yaxis=yaxis,
                        ymax=1.0,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_10b.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )
'''

"""
print 'Figure 9, Extrapolation vs. species'
legend_sim = ['OBSERVED','EXTRAPOL','SWITCH']
n_rep=1
title = 'Extrapolation vs. Species'
n_worker = 50
s_size = 25
Xs = []
Ys = []
GTs = []
for rep in range(3):
    d,gt,pr,re = simulatedData(workers=n_worker,recall=1,precision=0.7)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    rows = numpy.random.randint(1000,size=s_size)
    obs = numpy.sum(slist[rows]>0)
    q = float(s_size)/1000.
    estimators_sim = [lambda x: nominal(x),lambda x: obs/q,lambda x: vRemainSwitch2(x)]
    gt_list_sim = [lambda x: gt, lambda x:gt, lambda x:gt]
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(10,n_worker+5,5),estimators_sim,rel_err=False,rep=n_rep)

plotHist((X,Y,GT),
                        legend = legend_sim,
                        legend_gt = legend_gt,
                        xaxis='',
                        yaxis='Estimate (# Total Error)',
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_9b.png'
                        )

print 'Figure 9, # Tasks For Perfect Cleaning'
n_rep = 2
n_worker = 200
title = '# Tasks For Perfect Cleaning'
precisions = [.6, .65, .7, .75, .8, .85, .9, 1.0]
Xs, Ys, GTs = [], [], []
for precision in precisions:
    estimators_sim = [lambda x: vNominal(x)]
    gt_list_sim = [lambda x: gt]
    d,gt,pr,re = simulatedData(items=25,workers=n_worker,recall=1,precision=precision)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    x1s = []
    for r in range(n_rep):
        for n_w in range(1,n_worker,5):
            diff = numpy.sum(numpy.logical_xor(numpy.sum(d[:,:n_w] == 1,axis=1) > numpy.sum(d[:,:n_w] != -1,axis=1)/2, slist)) 
            if diff == 0:
                x1s.append(n_w)
                break
    x1 = numpy.mean(x1s)
    #(X,Y1,GT) = holdoutRealWorkers(d,gt_list_sim,range(1,n_worker,1),estimators_sim,rel_err=False,rep=n_rep)
    #x1 = [idx for idx, val in enumerate(Y1) if val == GT[0][0]][0]
    d,gt,pr,re = simulatedData(items=75,workers=n_worker,recall=1,precision=precision)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    x2s = []
    for r in range(n_rep):
        for n_w in range(1,n_worker,5):
            diff = numpy.sum(numpy.logical_xor(numpy.sum(d[:,:n_w] == 1,axis=1) > numpy.sum(d[:,:n_w] != -1,axis=1)/2, slist)) 
            if diff == 0:
                x2s.append(n_w)
                break
    x2 = numpy.mean(x2s)
    #(X,Y2,GT) = holdoutRealWorkers(d,gt_list_sim,range(1,n_worker,1),estimators_sim,rel_err=False,rep=n_rep)
    #x2 = [idx for idx, val in enumerate(Y2) if val == GT[0][0]][0]
    d,gt,pr,re = simulatedData(items=100,workers=n_worker,recall=1,precision=precision)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    x3s = []
    for r in range(n_rep):
        for n_w in range(1,n_worker,5):
            diff = numpy.sum(numpy.logical_xor(numpy.sum(d[:,:n_w] == 1,axis=1) > numpy.sum(d[:,:n_w] != -1,axis=1)/2, slist)) 
            if diff == 0:
                x3s.append(n_w)
                break
    x3 = numpy.mean(x3s)
    #(X,Y3,GT) = holdoutRealWorkers(d,gt_list_sim,range(1,n_worker,1),estimators_sim,rel_err=False,rep=n_rep)
    #x3 = [idx for idx, val in enumerate(Y3) if val == GT[0][0]][0]
    X = range(1,n_worker,1)
    Xs.append(precision)
    print x1, x2, x3
    Ys.append([X[int(x1)],X[int(x2)],X[int(x3)]])
    GTs.append(0)
plotY1Y2((Xs,Ys,GTs),
                        legend = ['25 records', '75 records', '100 records'],
                        xaxis='Worker Accuracy',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_9a.png',
                        logscale=False
                        )


# Extrapolation experiment in Section 2
print 'Extrapolation experiment in Section 2'
estimators_extp = [lambda x:extrapolation(x,pair_solution,len(pair_solution)/50),
                    lambda x: extrapolation(x,pair_solution,len(pair_solution)/50),
                    lambda x: extrapolation(x,pair_solution,len(pair_solution)/50),
                    lambda x: extrapolation(x,pair_solution,len(pair_solution)/50)]
estimators_extp2 = [lambda x:extrapolation2(x,d,sample1)+obvious_err,
                    lambda x:extrapolation2(x,d,sample2)+obvious_err,
                    lambda x:extrapolation2(x,d,sample3)+obvious_err,
                    lambda x:extrapolation2(x,d,sample4)+obvious_err]
estimators_extp3 = [lambda x:max(0,extrapolation3(x,d,sample1)-sNominal(x)),
                    lambda x:max(0,extrapolation3(x,d,sample2)-sNominal(x)),
                    lambda x:max(0,extrapolation3(x,d,sample3)-sNominal(x)),
                    lambda x:max(0,extrapolation3(x,d,sample4)-sNominal(x))]
legend_extp = ['Sample 1','Sample 2','Sample 3','Sample 4']
legend_extp_ = ['Mean','3-Std']
gt_list_extp = [lambda x: gt+obvious_err]
gt_list_extp_ = [lambda x: gt]
gt_list_extp3 = [lambda x: gt_switch(x,slist)]


sample1 = loadRestaurantExtSample(['dataset/extp/restaurant-1.csv'],priotization=priotization)
sample2 = loadRestaurantExtSample(['dataset/extp/restaurant-2.csv'],priotization=priotization)
sample3 = loadRestaurantExtSample(['dataset/extp/restaurant-3.csv'],priotization=priotization)
sample4 = loadRestaurantExtSample(['dataset/extp/restaurant-4.csv'],priotization=priotization)

n_worker = 14
init = 2
scale = 1
n_rep=1

#(X,Y,GT) = holdoutRealWorkers(d,gt_list_extp,range(init,n_worker,scale),estimators_extp2,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/rest2_results_extp.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_extp.p','rb'))
plotExtp((X,Y,GT),
                        legend=legend_extp_,
                        legend_gt=legend_gt,
                        yaxis="Estimate (Total # Error)",
                        xaxis="Workers",
                        title="Total Error: Real Samples of Size 100",
                        ymax=-2,
                        filename="plot/rest2_extp2.png"
                        )

plotExtp(holdoutRealWorkers(d,gt_list_extp3,range(5,n_worker,scale),estimators_extp3,rel_err=False,rep=n_rep),
                        legend=legend_extp_,
                        legend_gt=legend_gt,
                        yaxis="Remaining Switches",
                        xaxis="Workers",
                        title="Switches: Real Samples of Size 100",
                        filename="plot/rest2_switch_extp2.png"
                        )

priotization = False
d,gt,prec,rec = loadRestaurant2(['dataset/good_worker/restaurant_additional.csv','dataset/good_worker/restaurant_new.csv'],priotization=priotization)
pair_solution = pickle.load( open('dataset/pair_solution.p','rb') )
slist = pair_solution.values()
sample1 = loadRestaurantExtSample(['dataset/extp/restaurant-1.csv'],priotization=priotization)
sample2 = loadRestaurantExtSample(['dataset/extp/restaurant-2.csv'],priotization=priotization)
sample3 = loadRestaurantExtSample(['dataset/extp/restaurant-3.csv'],priotization=priotization)
sample4 = loadRestaurantExtSample(['dataset/extp/restaurant-4.csv'],priotization=priotization)
plotHist(holdoutRealWorkers(d,gt_list_extp_,range(5,n_worker,scale),estimators_extp,rep=n_rep),
                        legend=legend_extp,
                        legend_gt=legend_gt,
                        yaxis="Estimate (Total # Error)",
                        xaxis="",
                        titlt="Total Error: Simulated 2% Samples",
                        filename="plot/rest2_extp_hist.png",
                        )
"""
