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
from estimator import chao92, qaChao92, fChao92, sChao92,nominal, vNominal,  sNominal, wChao92, bChao92, dChao92, goodToulmin, vGoodToulmin, remain_switch, gt_switch, gt_marginal, gt_remaining, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, unseen, twoPhase, vRemainSwitch2, extrapolateFromSample, sampleCoverage, minTasks2, minTasks3, minTasks, minTasksToCleanAll
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, loadInstitution, loadCrowdFlowerData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle
from simulation import plotMulti, plotY1Y2, holdoutRealWorkers





##############################################
########### real-world datasets ##############
##############################################

#====total error====
estimators = [lambda x: extrapolateFromSample(x,slist,0.05)+obvious_err,lambda x:vNominal(x)+obvious_err,  lambda x: sChao92(x,shift=1)+obvious_err, lambda x: vRemainSwitch2(x)+obvious_err]
#estimators = [lambda x: vNominal(x) + obvious_err, lambda x: chao92(x)+obvious_err]
legend = ["EXTRAPOL","VOTING","V-CHAO","SWITCH"]
gt_list = [lambda x: gt+obvious_err,lambda x: gt+obvious_err, lambda x: gt+obvious_err, lambda x: gt+obvious_err]
legend_gt=["Ground Truth"]

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

#====relative====
#estimators_rel = [lambda x:vNominal(x)+obvious_err, lambda x: extrapolateFromSample(x,slist,0.1)+obvious_err,lambda x: sChao92(x,shift=1)+obvious_err, lambda x: vRemainSwitch2(x)+obvious_err, lambda x: remain_switch(x,pos_switch=False)-sNominal(x,pos_switch=False), lambda x: remain_switch(x,neg_switch=False)-sNominal(x,neg_switch=False)]
estimators_rel = [lambda x:vNominal(x)+obvious_err,lambda x: sChao92(x,shift=1)+obvious_err, lambda x: vRemainSwitch2(x)+obvious_err, lambda x: remain_switch(x)-sNominal(x)]
gt_list_rel = [lambda x: gt + obvious_err, lambda x: gt + obvious_err, lambda x: gt+ obvious_err, lambda x: gt_switch(x,slist)]
#legend_rel = ["VOTING","EXTRAPOL(1%)","V-CHAO","SWITCH","REMAIN SWITCH(+)", "REMAIN SWITCH(-)"]
legend_rel = ["VOTING","V-CHAO","SWITCH","REMAIN-SWITCH"]
#address dataset
d,gt,prec,rec = loadAddress()
task_sol = pickle.load( open('dataset/addr_solution.p','rb') )
slist = task_sol.values()
min_tasks = minTasks(d)
obvious_err = 0 # no priotization
n_worker = 1200
scale = 100
init = 100
n_rep = 3
priotization = True
logscale = False

"""
(X,Y,GT) = holdoutRealWorkers(d,gt_list,range(init,n_worker,scale),estimators,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/addr_results_1.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/addr_results_1.p','rb'))
plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Tasks",
                        yaxis="Estimate (# Total Error)",
                        #ymax=200,
                        loc='upper right',
                        title='(a) Total Error', 
                        filename="plot/addr_mostly_hard_all.png",
                        )
(X,Y,GT) = holdoutRealWorkers(d,gt_list2,range(init,n_worker,scale),estimators2,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/addr_results_2.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/addr_results_2.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2,  
                        legend_gt = legend_gt2,    
                        xaxis="Tasks", #ymax=100,
                        yaxis="Estimate (# Remaining Switches)",
                        #ymax=150,
                        title='Remaining Switches', 
                        filename="plot/addr_mostly_hard_all_switch.png",
                        )

#positive switch estimation
(X,Y,GT) = holdoutRealWorkers(d,gt_list2a,range(init,n_worker,scale),estimators2a,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/addr_results_3.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/addr_results_3.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2a,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        yaxis="Estimate (# Remaining Switches)",
                        #ymax=20,
                        title='(b) Remaining Positive Switches', 
                        filename="plot/addr_mostly_hard_pos_switch.png",
                        )
#negative estimation_subset
(X,Y,GT) = holdoutRealWorkers(d,gt_list2b,range(init,n_worker,scale),estimators2b,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/rest2_results_4.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_4.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2b,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        yaxis="Estimate (# Remaining Switches)",
                        #ymax=100,
                        title='(c) Remaining Negative Switches', 
                        filename="plot/addr_mostly_hard_neg_switch.png",
                        )
'''
(X,Y,GT) = holdoutRealWorkers(d,gt_list_rel,range(init,n_worker,scale),estimators_rel,rel_err=True,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/addr_results_5.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/addr_results_5.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend_rel,
                        xaxis='Tasks',
                        yaxis='SRMSE',
                        ymax=1,
                        title='(c) Relative Error', 
                        loc='upper right',
                        filename="plot/addr_mse.png",
                        logscale=logscale,
                        rel_err=True
                        )
'''
#restaurant dataset
n_worker = 1400
scale = 100
init = 100
priotization = True
logscale = False

d,gt,prec,rec = loadRestaurant2(['dataset/good_worker/restaurant_additional.csv','dataset/good_worker/restaurant_new.csv'],priotization=priotization)
#['dataset/restaurant_new.csv','dataset/restaurant_new2.csv'],priotization=True)
pair_solution = pickle.load( open('dataset/pair_solution.p','rb') )
# Pair solution ground truth
slist = pair_solution.values()
print 'restaurant data (hueristic pairs):',len(slist),numpy.sum(numpy.array(slist) == 1)
easy_pair_solution = pickle.load( open('dataset/easy_pair_solution.p','rb') )
easy_slist = easy_pair_solution.values()
print 'restaurant data (easy pairs):',len(easy_slist),numpy.sum(numpy.array(easy_slist) == 1)
obvious_err = numpy.sum(numpy.array(easy_slist) == 1)

(X,Y,GT) = holdoutRealWorkers(d,gt_list,range(init,n_worker,scale),estimators,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/rest2_results_1.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_1.p','rb'))
plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Tasks",
                        yaxis="Estimate (# Total Error)",
                        ymax=200,
                        loc='lower right',
                        title='(a) Total Error', 
                        filename="plot/rest2_mostly_hard_all.png",
                        )
(X,Y,GT) = holdoutRealWorkers(d,gt_list2,range(init,n_worker,scale),estimators2,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/rest2_results_2.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_2.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2,  
                        legend_gt = legend_gt2,    
                        xaxis="Tasks", #ymax=100,
                        yaxis="Estimate (# Remaining Switches)",
                        ymax=220,
                        title='Remaining Switches', 
                        filename="plot/rest2_mostly_hard_all_switch.png",
                        )
#positive switch estimation
print 'switch estimation'
(X,Y,GT) = holdoutRealWorkers(d,gt_list2a,range(init,n_worker,scale),estimators2a,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/rest2_results_3.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_3.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2a,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        yaxis="Estimate (# Remaining Switches)",
                        ymax=20,
                        title='(b) Remaining Positive Switches', 
                        filename="plot/rest2_mostly_hard_pos_switch.png",
                        )
#negative estimation
(X,Y,GT) = holdoutRealWorkers(d,gt_list2b,range(init,n_worker,scale),estimators2b,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/rest2_results_4.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_4.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2b,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        yaxis="Estimate (# Remaining Switches)",
                        ymax=100,
                        title='(c) Remaining Negative Switches', 
                        filename="plot/rest2_mostly_hard_neg_switch.png",
                        )
'''
(X,Y,GT) = holdoutRealWorkers(d,gt_list_rel,range(init,n_worker,scale),estimators_rel,rel_err=True,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/rest2_results_5.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_5.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend_rel,
                        xaxis='Tasks',
                        yaxis='SRMSE',
                        ymax=1,
                        title='(c) Relative Error', 
                        loc='upper right',
                        filename="plot/rest2_mse.png",
                        logscale=logscale,
                        rel_err=True
                        )
'''
"""
#Product dataset
print 'loading product data'
d,gt,prec,rec = loadProduct(['dataset/jn_heur/jn_heur_products.csv'])#(['dataset/good_worker/product_new.csv','dataset/good_worker/product_additional.csv'])
pair_solution = pickle.load( open('dataset/jn_heur/pair_solution.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
min_tasks2 = minTasks2(d,.8)
min_tasks3 = minTasks3(d)
min_tasks5 = minTasksToCleanAll(d)
print 'loaded product data'
slist = pair_solution.values()
print 'product data (hueristic pairs):',len(slist),numpy.sum(numpy.array(slist) == 1)
easy_pair_solution = pickle.load( open('dataset/jn_heur/easy_pair_solution.p','rb'))#( open('dataset/products/easy_pair_solution.p','rb') )
easy_slist = easy_pair_solution.values()
print 'product data (easy pairs):',len(easy_slist),numpy.sum(numpy.array(easy_slist) == 1)
obvious_err = numpy.sum(numpy.array(easy_slist) == 1)
n_worker = 4800
scale = 400
init = 400

#(X,Y,GT) = holdoutRealWorkers(d,gt_list,range(init,n_worker,scale),estimators,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/product_results_1b.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/product_results_1b.p','rb'))
plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Tasks",
                        loc='lower right',
                        title="(a) Total Error",
                        yaxis='# Total Error', 
                        ymax=1600,
                        filename="plot/product_mostly_hard_all.png",
                        min_tasks5=min_tasks5,
                        vertical_y=600
                        )

'''
(X,Y,GT) = holdoutRealWorkers(d,gt_list2,range(init,n_worker,scale),estimators2,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/product_results_2b.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/product_results_2b.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2,  
                        legend_gt = legend_gt2,    
                        xaxis="Tasks", #ymax=100,
                        yaxis="Estimate (# Remaining Switches)",
                        title='Remaining Switches', 
                        filename="plot/product_mostly_hard_all_switch.png",
                        )
'''

#positive switch estimation
#(X,Y,GT) = holdoutRealWorkers(d,gt_list2a,range(init,n_worker,scale),estimators2a,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/product_results_3.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/product_results_3.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2a,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        ymax=1200,
                        yaxis="# Remaining Switches",
                        title='(b) Remaining Positive Switches', 
                        filename="plot/product_mostly_hard_pos_switch.png",
                        )

#negative estimation
#(X,Y,GT) = holdoutRealWorkers(d,gt_list2b,range(init,n_worker,scale),estimators2b,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/product_results_4.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/product_results_4.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend2b,    
                        legend_gt = legend_gt2,    
                        xaxis="Tasks",
                        ymax=1200,
                        yaxis="# Remaining Switches",
                        title='(c) Remaining Negative Switches', 
                        filename="plot/product_mostly_hard_neg_switch.png",
                        )
'''
(X,Y,GT) = holdoutRealWorkers(d,gt_list_rel,range(init,n_worker,scale),estimators_rel,rel_err=True,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/product_results_5b.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/product_results_5b.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend_rel,
                        xaxis='Tasks',
                        yaxis='SRMSE',
                        title='(c) Relative Error', 
                        loc='upper right',
                        ymax=1,
                        filename='plot/product_mse.png',
                        logscale=logscale,
                        rel_err=True
                        )
'''

"""
######################################
###########simulated dataset##########
######################################
legend_gt=["Ground Truth"]
print 'Figure 6, Sensitivity of Total Error Estimation'
estimators_sim = [vNominal, chao92]
estimators_sim2 = [vNominal, chao92, lambda x:sChao92(x,shift=1)]
legend_sim = ['VOTING','Chao92']
legend_sim2 = ['VOTING','Chao92','V-CHAO']

logscale = False
rel_err = True
yaxis = 'SRMSE'#'Relative Error %'
priotized = False
err_skew = False
dirty = 0.2

hir = 0.1 #0.1
lowr = 0.01
hiq = 1.0
lowq = 0.85

title = 'Perfect Precision, High Recall'
print title
n_worker = 200
scale = 10
init = 10
n_rep = 5

d,gt,prec,rec = simulatedData(items=1000,workers=n_worker,dirty=dirty,recall=hir,precision=1.0,err_skew=err_skew,priotized=priotized)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
gt_list_sim = [lambda x:gt, lambda x:gt]
gt_list_sim2 = [lambda x:gt, lambda x:gt, lambda x:gt, lambda x:gt]
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=True,rep=n_rep)
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_6a.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )

title = 'Perfect Precision, Low Recall'
print title
d,gt,prec,rec = simulatedData(items=1000,workers=n_worker,dirty=dirty,recall=lowr,precision=1.0,err_skew=err_skew,priotized=priotized)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
gt_list_sim = [lambda x:gt, lambda x:gt]
gt_list_sim2 = [lambda x:gt, lambda x:gt, lambda x:gt, lambda x:gt]
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=True,rep=n_rep)
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_6b.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )

title = 'Imperfect Precision, Low Recall'
print title
d,gt,prec,rec = simulatedData(items=1000,workers=n_worker,dirty=dirty,recall=lowr,precision=0.75,err_skew=err_skew,priotized=priotized)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
gt_list_sim = [lambda x:gt, lambda x:gt]
gt_list_sim2 = [lambda x:gt, lambda x:gt, lambda x:gt, lambda x:gt]
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim2,range(init,n_worker,scale),estimators_sim2,rel_err=True,rep=n_rep)
plotY1Y2((X,Y,GT),
                        legend = legend_sim2,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_6c.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )

print 'Figure 5, Sensitivity of Total Error Estimation'
title = 'Tradeoff: False Positives'
estimators_sim = [vNominal, chao92, lambda x:sChao92(x,shift=1)]
legend_sim = ['VOTING','Chao92','V-CHAO']
rec(i)/10.
    d,gt,pr,re =all = 0.1
Xs = []
Ys = []
GTs = []
for i in range(11):
    prec = float(i)/10.
    d,gt,pr,re = simulatedData(recall=recall,precision=prec)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    gt_list_sim = [lambda x:gt,lambda x:gt,lambda x:gt]
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[50],estimators_sim,rel_err=True,rep=n_rep)
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
                        rel_err=rel_err
                        )

title = 'Tradeoff: False Negatives'
precision = 1.
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    rec = float(i)/200.
    d,gt,pr,re = simulatedData(recall=rec,precision=precision)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    gt_list_sim = [lambda x:gt,lambda x:gt,lambda x:gt]
    (X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,[50],estimators_sim,rel_err=True,rep=n_rep)
    Xs.append((rec)*100)
    Ys.append(Y[0])
    GTs.append(GT[0])

plotY1Y2((Xs,Ys,GTs),
                        legend = legend_sim,
                        xaxis='Coverage %',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_5b.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )


print 'Figure 7: switch estimation'
title = 'Perfect Precision'
d,gt,pr,re = simulatedData(recall=hir,precision=hiq)
slist = pickle.load( open('dataset/sim_label.p','rb') )

estimators_sw = [lambda x: vNominal(x), lambda x: chao92(x), lambda x: remain_switch(x)-sNominal(x)]
gt_list_sw = [lambda x: gt, lambda x: gt, lambda x: gt_switch(x,slist)]
legend_sw = ["VOTING","Chao92","REMAIN-SWITCH"]

(X,Y,GT) = holdoutRealWorkers(d,gt_list_sw,range(init,n_worker,scale),estimators_sw,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/sim_7a_results_extp.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/sim_7a_results_extp.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend_sw,  
                        legend_gt = legend_gt,    
                        xaxis="Tasks", #ymax=100,
                        yaxis="Estimate (# Remaining Switches)",
                        title=title, 
                        filename="plot/sim_7a.png",
                        rel_err=True
                        )

title = 'Imperfect Precision'
d,gt,pr,re = simulatedData(recall=hir,precision=lowq)
slist = pickle.load( open('dataset/sim_label.p','rb') )
estimators_sw = [lambda x: remain_switch(x)-sNominal(x)]
gt_list_sw = [lambda x: gt_switch(x,slist)]
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sw,range(init,n_worker,scale),estimators_sw,rep=n_rep)
pickle.dump((X,Y,GT),open('dataset/sim_7b_results_extp.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/sim_7b_results_extp.p','rb'))
plotY1Y2((X,Y,GT),
                        legend = legend_sw,  
                        legend_gt = legend_gt,    
                        xaxis="Tasks", #ymax=100,
                        yaxis="Estimate (# Remaining Switches)",
                        title=title, 
                        filename="plot/sim_7b.png",
                        rel_err=True
                        )


print 'Figure 7, Sensitivity of Total Error Estimation'
estimators_sim = [vNominal, chao92, lambda x: sChao92(x,shift=1), lambda x: vRemainSwitch2(x)]
legend_sim = ['VOTING','Chao92','V-CHAO','SWITCH']

title = 'Perfect Precision, High Recall'
print title

d,gt,prec,rec = simulatedData(items=1000,workers=n_worker,dirty=dirty,recall=hir,precision=hiq,err_skew=err_skew,priotized=priotized)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
gt_list_sim = [lambda x:gt, lambda x:gt, lambda x:gt, lambda x: gt]
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=True,rep=n_rep)
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_7a2.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )

title = 'Perfect Precision, Low Recall'
print title
d,gt,prec,rec = simulatedData(items=1000,workers=n_worker,dirty=dirty,recall=lowr,precision=hiq,err_skew=err_skew,priotized=priotized)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
gt_list_sim = [lambda x:gt, lambda x:gt, lambda x:gt, lambda x: gt]
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=True,rep=n_rep)
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_7b2.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )

title = 'Imperfect Precision, Low Recall'
print title
d,gt,prec,rec = simulatedData(items=1000,workers=n_worker,dirty=dirty,recall=lowr,precision=lowq,err_skew=err_skew,priotized=priotized)
slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
gt_list_sim = [lambda x:gt, lambda x:gt, lambda x:gt, lambda x:gt]
(X,Y,GT) = holdoutRealWorkers(d,gt_list_sim,range(init,n_worker,scale),estimators_sim,rel_err=True,rep=n_rep)
plotY1Y2((X,Y,GT),
                        legend = legend_sim,
                        xaxis='Tasks',
                        yaxis=yaxis,
                        title=title, 
                        loc='upper right',
                        filename='plot/sim_7c2.png',
                        logscale=logscale,
                        rel_err=rel_err
                        )

print 'Figure 10, Heuristics'
estimators_sim = [lambda x: vRemainSwitch2(x)]
legend_sim = ['SWITCH']
n_rep = 1
logscale = False
title = 'Heuristic 10% Error'
print title
n_worker = 50
Xs = []
Ys = []
GTs = []
for i in range(0,11):
    eps = float(i)/10.
    d,gt,pr,re = simulatedData(recall=0.2,precision=1,priotized=True,eps=eps)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    gt_list_sim = [lambda x:gt]
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
    d,gt,pr,re = simulatedData(recall=0.2,precision=1,priotized=True,eps=eps,hdirty=0.5)
    slist = pickle.load( open('dataset/sim_label.p','rb'))#( open('dataset/products/pair_solution.p','rb'))
    gt_list_sim = [lambda x:gt]
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
"""
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
