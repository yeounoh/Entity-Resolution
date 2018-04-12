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
from estimator import chao92, qaChao92, fChao92, sChao92,nominal, vNominal,  sNominal, wChao92, bChao92, dChao92, goodToulmin, vGoodToulmin, remain_switch, gt_switch, gt_marginal, gt_remaining, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, unseen, twoPhase, vRemainSwitch2, minTasks, minTasks2, minTasks3,  extrapolateFromSample
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, simulatedData2, loadInstitution, loadCrowdFlowerData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle
from simulation import plotExtp, plotHist, plotMulti, plotY1Y2, holdoutRealWorkers



######################################
###########simulated dataset##########
######################################
logscale=False
dirty = 0.2
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
legend_extp_ = ['Mean','Std']
gt_list_extp = [lambda x: gt+obvious_err,lambda x: gt+obvious_err,lambda x: gt+obvious_err, lambda x: gt+obvious_err]
gt_list_extp_ = [lambda x: gt, lambda x: gt, lambda x: gt, lambda x: gt]
gt_list_extp3 = [lambda x: gt_switch(x,slist)]

priotization=True

n_worker = 2000
init = 200
scale = 200
n_rep=1
'''
sample1 = loadRestaurantExtSample(['dataset/extp/restaurant-1.csv'],priotization=priotization)
sample2 = loadRestaurantExtSample(['dataset/extp/restaurant-2.csv'],priotization=priotization)
sample3 = loadRestaurantExtSample(['dataset/extp/restaurant-3.csv'],priotization=priotization)
sample4 = loadRestaurantExtSample(['dataset/extp/restaurant-4.csv'],priotization=priotization)

d,gt,prec,rec = loadRestaurant2(['dataset/good_worker/restaurant_additional.csv','dataset/good_worker/restaurant_new.csv'],priotization=priotization)
pair_solution = pickle.load( open('dataset/pair_solution.p','rb') )
slist = pair_solution.values()
easy_pair_solution = pickle.load( open('dataset/easy_pair_solution.p','rb') )
easy_slist = easy_pair_solution.values()
obvious_err = numpy.sum(numpy.array(easy_slist) == 1)
print obvious_err
'''
#(X,Y,GT) = holdoutRealWorkers(d,gt_list_extp,range(init,n_worker,scale),estimators_extp2,rep=n_rep)
#pickle.dump((X,Y,GT),open('dataset/rest2_results_extp.p','wb'))
(X,Y,GT) = pickle.load(open('dataset/rest2_results_extp.p','rb'))
plotExtp((X,Y,GT),
                        legend=legend_extp_,
                        legend_gt=legend_gt,
                        yaxis="Estimate (Total # Error)",
                        xaxis="Tasks",
                        title="(b) Real Samples of Size 100",
                        ymax=-2,
                        font=20,
                        filename="plot/rest2_extp2.png"
                        )

priotization = False
d,gt,prec,rec = loadRestaurant2(['dataset/good_worker/restaurant_additional.csv','dataset/good_worker/restaurant_new.csv'],priotization=priotization)
pair_solution = pickle.load( open('dataset/pair_solution.p','rb') )
slist = pair_solution.values()
easy_pair_solution = pickle.load( open('dataset/easy_pair_solution.p','rb') )
easy_slist = easy_pair_solution.values()
obvious_err = numpy.sum(numpy.array(easy_slist) == 1)
print obvious_err
sample1 = loadRestaurantExtSample(['dataset/extp/restaurant-1.csv'],priotization=priotization)
sample2 = loadRestaurantExtSample(['dataset/extp/restaurant-2.csv'],priotization=priotization)
sample3 = loadRestaurantExtSample(['dataset/extp/restaurant-3.csv'],priotization=priotization)
sample4 = loadRestaurantExtSample(['dataset/extp/restaurant-4.csv'],priotization=priotization)
plotHist(holdoutRealWorkers(d,gt_list_extp_,range(5,12,1),estimators_extp,rep=n_rep),
                        legend=legend_extp,
                        legend_gt=legend_gt,
                        yaxis="Estimate (Total # Error)",
                        xaxis="",
                        font=20,
                        title="(a) Simulated 2% Samples",
                        filename="plot/rest2_extp_hist.png",
                        )
