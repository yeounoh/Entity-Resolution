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
from estimator import chao92, qaChao92, fChao92, sChao92,nominal, vNominal,  sNominal, wChao92, bChao92, dChao92, goodToulmin, vGoodToulmin, remain_switch, gt_switch, gt_marginal, gt_remaining, extrapolation, extrapolation2, extrapolation3,vRemainSwitch, unseen, twoPhase, vRemainSwitch2, extrapolateFromSample
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, simulatedData2, loadInstitution, loadCrowdFlowerData, loadRestaurant2, loadProduct, loadRestaurantExtSample, loadAddress
import pickle


'''
    @param gt_list: a list of ground truth functions
    @param est_list: a list of estimator functions
    @param rep: repeat the experiments with different worker permutations?
    @param rel_err: output estimates in relative errors? -- note, use mean squared error measure?
'''
def holdoutRealWorkers(bdataset, gt_list, worker_range, est_list,rel_err=False, rep=1):
    X = []
    Y = []
    GT = []

    for w in worker_range:
        #dataset = bdataset[:,0:w]
        random_trial = numpy.zeros((len(est_list)+len(gt_list),rep))
        for t in range(0,rep):
            dataset = bdataset[:,numpy.random.choice(range(len(bdataset[0])),min(w,len(bdataset[0])),replace=False)]
            for e in est_list:
                # ground truth ( len(est_list) == len(gt_list) )
                A = float(gt_list[est_list.index(e)](dataset))
                random_trial[len(est_list) + est_list.index(e),t] = A

                if rel_err:
                    # SRMSE
                    random_trial[est_list.index(e),t] = (e(dataset)-A)**2 / A**2
                else:
                    random_trial[est_list.index(e),t] = e(dataset)
        
        result_array = []
        var_array = []
        if rel_err:
            result_array = numpy.sqrt(numpy.mean(random_trial[0:len(est_list),:],axis=1)) #take median, not mean
            var_array = numpy.std(random_trial[0:len(est_list),:],axis=1)
        else:
            result_array = numpy.mean(random_trial[0:len(est_list),:],axis=1)
            var_array = numpy.std(random_trial[0:len(est_list),:],axis=1)
            #var_array = numpy.ptp(random_trial[0:len(est_list),:],axis=1)

        gt_array = numpy.mean(random_trial[len(est_list):len(est_list)+len(gt_list),:],axis=1)
        if rel_err:
            gt_array = []
         
        Y.append( [list(result_array),list(var_array)] )
        X.append(w)
        GT.append(list(gt_array))#GT.append(ground_truth)

    return (X,Y,GT)

def holdoutRealResponse(bdataset, gt_list, response_range, estimator_list):
    X = []
    Y = []
    GT = []

    for r in response_range:
        result_array = []
        for e in estimator_list:
            random_trial = []
            dataset = bdataset[:,0:r]
            random_trial.append(e(dataset))
            est = numpy.mean(random_tiral)
            result_array.append(est)
        gt_array = []
        for g in gt_list:
            dataset = bdataset[:,0:r]
            gt_array.append(g(dataset))

        Y.append(result_array)
        X.append(r)
        GT.append(gt_array)

    return (X,Y,GT)


def plotExtp(points,
             title="",
             xaxis="# Workers",
             yaxis="Estimate",
             ymax=-1,
             legend=[],
             font=20,
             legend_gt=[],
             loc = 'upper right',
             filename="output.png",
             ):
    #import matplotlib as mpl
    #mpl.use('Agg')
    #import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout': True})

    rcParams.update({'font.size': font})
    fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

    num_estimators = len(points[1][0])
    num_gt = len(legend_gt)#len(points[2][0])

    plt.figure(figsize=(8,5)) 
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    shapes = ['--','-*']
    for i in range(0,num_gt):
        res = [j[i] for j in points[2]]
        plt.plot([x*100 for x in points[0]], res, shapes[i],linewidth=2.5,color="#333333")

    mean = numpy.array([numpy.mean(j) for j in points[1]])
    std = numpy.array([numpy.std(j) for j in points[1]])
    plt.plot([x*100 for x in points[0]], mean, 's-', linewidth=2.5,markersize=7,color=colors[0])
    plt.fill_between([x*100 for x in points[0]], mean-2*std,mean+2*std, alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')

    #plt.plot(points[0], points[2], 'o-', linewidth=2.5,markersize=5,color='#FF6666')
    #plt.plot(points[0], points[2], '--', linewidth=2.5,color='#333333')
    plt.title(title)
    plt.xlabel(xaxis)#,fontproperties=fprop)
    plt.ylabel(yaxis)#,fontproperties=fprop)
    if ymax == -1:
        plt.ylim(ymin=0)
    elif ymax == -2:
        plt.ylim(ymin=90, ymax=110)
    else:
        plt.ylim(ymin=0,ymax=ymax) 
    plt.xlim(xmin=points[0][0]*100, xmax=points[0][len(points[0])-1]*100) 
    plt.legend(legend_gt+legend,loc=loc)
    plt.grid(True)
    plt.savefig(filename,bbox_inches='tight')#,format='pdf')

def plotY1Y2(points,
             title="",
             xaxis="",
             yaxis="Estimate",
             ymax=-1,
             xmin=-1,
             legend=[],
             legend_gt=[],
             loc = 'upper right',
             filename="output.png",
             logscale=False,
             rel_err=False,
             font=20,
             vertical_y=-1,
             min_tasks=-1,
             min_tasks2=-1,
             min_tasks3=-1,
             min_tasks4=-1,
             min_tasks5=-1,
             show_min_tasks=True
             ):
    #import matplotlib as mpl
    #mpl.use('Agg')
    #import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': font})
    fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

    num_estimators = len(legend)#len(points[1][0][0])
    num_gt = len(legend_gt)#len(points[2][0])
    if rel_err:
        num_gt=0

    fig, ax = plt.subplots(1,figsize=(8,5)) 
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    colors = ['#0099ff','#00ff99','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    markers = ['o-','v-','^-','s-','*-','x-','+-','D-']
    markers = ['v-','o-','^-','s-','*-','x-','+-','D-']
    shapes = ['--','-*']
    for i in range(num_gt):
        res = [j[i] for j in points[2]]
        if logscale:
            ax.semilogy(points[0], res, shapes[i],linewidth=2.5,color="#333333",label=legend_gt[i])
        else:
            ax.plot(points[0], res, shapes[i],linewidth=2.5,color="#333333",label=legend_gt[i])

    for i in range(num_estimators):
        #if i is not 0: continue # to plot EXTRAPOL only
        res = numpy.array([j[0][i] for j in points[1]])
        if not rel_err:
            std = numpy.array([j[1][i] for j in points[1]])

        if logscale:
            plt.semilogy(points[0], res, 's-', linewidth=2.5, markersize=7, color=colors[i], label=legend_gt[i])
        else:
            if not rel_err and 'EXTRAPOL' in legend[i]:
                #ax.plot(points[0], numpy.zeros(len(points[0]))+res[0], '--',linewidth=1.5, color='g',label=legend[i]) 
                ax.fill_between(points[0], res[0]-1*std[0], res[0]+1*std[0], alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')#,label=legend[i])
                #ax.errorbar(points[0],numpy.zeros(len(points[0]))+res[0],yerr=std[0],linewidth=2.5,color='g',label=legend[i])
            else:
                if rel_err:
                    ax.plot(points[0], res, markers[i], linewidth=2.5,markersize=7,color=colors[i],label=legend[i])
                else:
                    ax.errorbar(points[0], res, yerr=std, fmt=markers[i], linewidth=2,markersize=7,color=colors[i],label=legend[i])
                #for z in range(len(points[0])):
                #    print legend[i], points[0][z], res[z]
    if show_min_tasks:
        if min_tasks != -1 and min_tasks < points[0][len(points[0])-1] and min_tasks > points[0][0]:
            ax.plot((min_tasks, min_tasks),(0,ymax),'m--',linewidth=1.5)
            ax.text(min_tasks,ymax/3,'at least one vote',rotation='vertical',color='m',fontsize=15)
        if min_tasks2 != -1 and min_tasks2 < points[0][len(points[0])-1] and min_tasks2 > points[0][0]:
            ax.plot((min_tasks2, min_tasks2),(0,ymax),'m--',linewidth=1.5)
            ax.fill_between(range(points[0][0],min_tasks2), 0,ymax, alpha=0.1, facecolor='gray')
            #ax.text(min_tasks2,ymax/3,'Covered 80%',rotation='vertical',color='m',fontsize=15)
            if vertical_y == -1:
                ax.text(min_tasks2,ymax/3,'> 0.5',rotation='vertical',color='m',fontsize=17)
            else:
                ax.text(min_tasks2,vertical_y,'> 0.5',rotation='vertical',color='m',fontsize=17)
        if min_tasks3 != -1 and min_tasks3 < points[0][len(points[0])-1] and min_tasks3 > points[0][0]:
            ax.plot((min_tasks3, min_tasks3),(0,ymax),'m--',linewidth=1.5)
            ax.text(min_tasks3,ymax/3,'25% votes',rotation='vertical',color='m',fontsize=15)
        if min_tasks4 != -1 and min_tasks4 < points[0][len(points[0])-1] and min_tasks4 > points[0][0]:
            ax.plot((min_tasks4, min_tasks4),(0,ymax),'m--',linewidth=1.5)
            ax.text(min_tasks4,ymax/3,'10% votes',rotation='vertical',color='m',fontsize=15)
        if min_tasks5 != -1 and min_tasks5 > points[0][0] and min_tasks5  < points[0][-1]:
            ax.plot((min_tasks5, min_tasks5),(0,ymax),'m--',linewidth=1.5)
            ax.fill_between(range(points[0][0],min_tasks5), 0,ymax, alpha=0.1, facecolor='gray')
            if vertical_y == -1:
                ax.text(min_tasks5,ymax/3,'SCM',rotation='vertical',color='m',fontsize=17)
            else:
                ax.text(min_tasks5,vertical_y,'SCM',rotation='vertical',color='m',fontsize=17)
        #plt.plot(points[[lookup_tbl[(pair_resp[0][1],pair_resp[0][0])],ilist_workers.index(k)] = pair_resp[1]

    #plt.plot(points[0], points[2], '--', linewidth=2.5,color='#333333')
    ax.set_title(title,fontsize=font)
    ax.set_xlabel(xaxis,fontsize=font)#,fontproperties=fprop)
    ax.set_ylabel(yaxis,fontsize=font)#,fontproperties=fprop)
    if not logscale and ymax == -1:
        ax.set_ylim(ymin=0)
    elif not logscale:
        ax.set_ylim(ymin=0,ymax=ymax) 

    if xmin == -1:
        ax.set_xlim(xmin=points[0][0], xmax=points[0][len(points[0])-1]) 
    else:
        ax.set_xlim(xmin=xmin, xmax=points[0][len(points[0])-1]) 
        
    #plt.legend(legend_gt+legend,loc=loc).get_frame().set_alpha(0.5)
    ax.legend(loc=loc,prop={'size':15}).get_frame().set_alpha(0.5)
    ax.grid()
    #plt.grid(True)
    fig.savefig(filename,bbox_inches='tight')#,format='pdf')


def plotHist(points,font=20,title="",xaxis="",yaxis="",ymax=-1,legend=[],legend_gt=[],loc='upper right',filename="output.png",yerr = []):
    #import matplotlib as mpl
    #mpl.use('Agg')
    #import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': font})
    #fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

    num_estimators = len(points[1][0][0])
    num_gt = len(legend_gt)#len(points[2][0])

    plt.figure(figsize=(8,5)) 
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    shapes = ['--','-*']

    width = 0.5     # gives histogram aspect to the bar diagram
    pos = numpy.arange(len(legend))
    pos_gt = [0+width/2, pos[-1]+2*width+width/2]
    for i in range(0,num_gt):
        res = [j[i] for j in points[2]]
        plt.plot(pos_gt, res[0:len(pos_gt)], shapes[i],linewidth=2.5,color="#333333")

    ax = plt.axes()
    ax.set_xticks(pos + 3*(width / 2))
    ax.set_xticklabels(legend)
    frequencies = []
    for i in range(0,num_estimators):
        res = [j[0][i] for j in points[1]]
        res = res[0]
        frequencies.append(res)
    plt.bar(pos+width, frequencies, width, color=colors)

    plt.title(title)
    plt.xlabel(xaxis)#,fontproperties=fprop)
    plt.ylabel(yaxis)#,fontproperties=fprop)
    if ymax == -1:
        plt.ylim(ymin=0)
    elif ymax == -2:
        plt.ylim(ymin=90,ymax=110)
    else:
        plt.ylim(ymin=0,ymax=ymax) 
    print num_gt + num_estimators, legend_gt + legend
    #plt.xlim(xmin=points[0][0], xmax=points[0][len(points[0])-1]) 
    plt.legend(legend_gt,loc=loc)
    plt.grid(True)
    plt.savefig(filename,bbox_inches='tight')#,format='pdf')


def plotMulti(ax,points,title="",xaxis="",yaxis="",
                legend=[],legend_gt=[],loc='upper right',vertical_y=-1,
                filename="output.png",rel_err=False,font=20,ymax=-1,min_tasks=-1,
                show_min_tasks=True):
    #import matplotlib as mpl
    #mpl.use('Agg')
    #import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': font})
    fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    colors = ['#0099ff','#00ff99','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    markers = ['o-','v-','^-','s-','*-','x-','+-','D-']
    markers = ['v-','o-','^-','s-','*-','x-','+-','D-']
    shapes = ['--','-*']

    num_estimators = len(legend) #len(points[1][0][0])
    num_gt = len(legend_gt) #len(points[2][0])
    if rel_err:
        num_gt=0
     
    for i in range(num_gt):
        res = [j[i] for j in points[2]]
        ax.plot(points[0], res, shapes[i],linewidth=2.5,color="#333333",label=legend_gt[i])

    for i in range(num_estimators):
        #if i == 2: continue
        res = numpy.array([j[0][i] for j in points[1]])
        if not rel_err:
            std = numpy.array([j[1][i] for j in points[1]])

        if not rel_err and 'EXTRAPOL' in legend[i]:
            ax.plot(points[0], numpy.zeros(len(points[0]))+res[0], linewidth=1, color='gray',label=legend[i]) 
            ax.fill_between(points[0], res[0]-3*std[0], res[0]+3*std[0], alpha=0.2, edgecolor='#CC4f1B', facecolor='#FF9848')
        else:
            if rel_err:
                ax.plot(points[0], res, markers[i], linewidth=2.5,markersize=7,color=colors[i],label=legend[i])
            else:
                std[res > ymax] =0
                ax.errorbar(points[0], res, yerr=std, fmt=markers[i], label=legend[i],linewidth=2,markersize=7,color=colors[i])

    #plt.plot(points[[lookup_tbl[(pair_resp[0][1],pair_resp[0][0])],ilist_workers.index(k)] = pair_resp[1]

    #plt.plot(points[0], points[2], '--', linewidth=2.5,color='#333333')
    ax.set_title(title,fontsize=font)
    ax.set_xlabel(xaxis,fontsize=font)#,fontproperties=fprop)
    ax.set_ylabel(yaxis,fontsize=font)
    if ymax == -1:
        ax.set_ylim(ymin=0)
    else:
        ax.set_ylim(ymin=0,ymax=ymax) 

    if show_min_tasks:
        if min_tasks != -1 and min_tasks < points[0][len(points[0])-1] and min_tasks > points[0][0]:
            ax.plot((min_tasks, min_tasks),(0,ymax),'m--',linewidth=1.5)
            if vertical_y == -1:
                ax.text(min_tasks,ymax/3,'SCM',rotation='vertical',color='m',fontsize=17)
            else:
                ax.text(min_tasks,vertical_y,'SCM',rotation='vertical',color='m',fontsize=17)
            ax.fill_between(range(points[0][0],min_tasks), 0,ymax, alpha=0.1, facecolor='gray')

    ax.set_xlim(xmin=points[0][0], xmax=points[0][len(points[0])-1]) 
    ax.xaxis.set_ticks(numpy.arange(points[0][0],points[0][len(points[0])-1],100))
    ax.tick_params(labelsize=font)
    #ax.legend(loc=loc,ncol=2).get_frame().set_alpha(0.5)
    ax.grid(True)


