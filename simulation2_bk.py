#!/usr/bin/env python
import numpy
import scipy
import csv
import random
import simplejson
from estimator import chao92, qaChao92, sChao92,nominal, vNominal,  sNominal, wChao92, bChao92, dChao92, goodToulmin, vGoodToulmin, remain_switch, gt_switch,gt_marginal, gt_remaining
from datagen import generateDist, generateDataset, generateWeightedDataset, shuffleList
from dataload import simulatedData, loadInstitution, loadCrowdFlowerData, loadRestaurant2, loadProduct,  loadRestaurant
import pickle

def holdoutWorkers(dist, worker_range, estimator_list, trials=50):
    ground_truth = len([i for i in dist if i > 0]) #number of errors
    
    X = []
    Y = []
    GT = []

    for w in worker_range:
        result_array = []
        for e in estimator_list:
            random_trial = []
            for t in range(0,trials):
                dataset = generateDataset(dist, w)
                random_trial.append(e(dataset))

            est = numpy.mean(random_trial)
            result_array.append(est)

        Y.append(result_array)
        X.append(w)
        GT.append(ground_truth)

    return (X,Y,GT)

def holdoutRealWorkers(bdataset, gt_list, worker_range, estimator_list, relative=False):
    #ground_truth = len([r for r in numpy.sum(bdataset,axis=1) if r > 0]) #number of errors
    
    X = []
    Y = []
    GT = []

    for w in worker_range:
        result_array = []
        for e in estimator_list:
            random_trial = []
            dataset = bdataset[:,0:w]

            if relative:
                random_trial.append(100*float(abs(e(dataset)-gt_list[0](dataset)))/gt_list[0](dataset))
            else:
                random_trial.append(e(dataset))

            est = numpy.mean(random_trial)
            result_array.append(est)
        gt_array = []
        for g in gt_list:
            dataset = bdataset[:,0:w]
            gt_array.append(g(dataset))

        Y.append(result_array)
        X.append(w)
        GT.append(gt_array)#GT.append(ground_truth)

    return (X,Y,GT)

def holdoutRealSwitchWorkers(bdataset, gt_list, worker_range, estimator_list, relative=False):
    #ground_truth = len([r for r in numpy.sum(bdataset,axis=1) if r > 0]) #number of errors
    
    X = []
    Y = []
    GT = []

    for w in worker_range:
        result_array = []
        for e in estimator_list:
            random_trial = []
            dataset = bdataset[:,0:w]

            if relative:
                random_trial.append(100*float(abs(e(dataset)-gt_list[0](dataset)))/gt_list[0](dataset))
            else:
                random_trial.append(e(dataset))

            est = numpy.mean(random_trial)
            result_array.append(est)
        gt_array = []
        for g in gt_list:
            dataset = bdataset
            gt_array.append(g(dataset, w))

        Y.append(result_array)
        X.append(w)
        GT.append(gt_array)#GT.append(ground_truth)

    return (X,Y,GT)

def holdoutRealResponse(bdataset, ground_truth, response_range, estimator_list):
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

        Y.append(result_array)
        X.append(r)
        GT.append(ground_truth)

    return (X,Y,GT)

def holdoutData(dist, sample_range, workers, shuffle, estimator_list, trials=10):
    ground_truth = len([i for i in dist if i > 0]) #number of errors
    N = len(dist)

    X = []
    Y = []
    GT = []

    for s in sample_range:
        result_array = []
    
        N_p = int(round(s/100.0*N))

        for e in estimator_list:
            
            random_trial = []
            for t in range(0,trials):
                dataset = generateWeightedDataset(dist, workers = workers, shuffle=shuffle)
                data_subset = dataset[0][dataset[1][N-N_p:N-1],:]

                random_trial.append(e(data_subset))

            est = numpy.mean(random_trial)
            result_array.append(est)

        Y.append(result_array)
        X.append(s)
        GT.append(ground_truth)

    return (X,Y,GT)


def plotY1Y2(points,
             title="Estimate vs. Workers",
             xaxis="# Workers",
             yaxis="Estimate",
             legend=[],
             legend_gt=[],
             loc = 'upper right',
             filename="output.png",
             ylimm=0,
             gt=True
             ):

    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': 18})
    #fprop = font_manager.FontProperties(fname= 
    #    '/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

    num_estimators = len(points[1][0])
    num_gt = len(points[2][0])

    plt.figure() 
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    shapes = ['--','-*']

    if gt:
        for i in range(0,num_gt):
            res = [j[i] for j in points[2]]
            plt.plot(points[0], res, shapes[i],linewidth=2.5,color="#333333")

    for i in range(0,num_estimators):
        res = [j[i] for j in points[1]]
        plt.plot(points[0], res, 's-', linewidth=2.5,markersize=7,color=colors[i])
        

    #plt.plot(points[0], points[2], 'o-', linewidth=2.5,markersize=5,color='#FF6666')
    #plt.plot(points[0], points[2], '--', linewidth=2.5,color='#333333')
    plt.title(title)
    plt.xlabel(xaxis)#,fontproperties=fprop)
    plt.ylabel(yaxis)#,fontproperties=fprop)
    plt.ylim(ymin=ylimm) 
    plt.xlim(xmin=points[0][0], xmax=points[0][len(points[0])-1]) 

    if gt:
        plt.legend(legend_gt+legend,loc=loc)
    else:
        plt.legend(legend,loc=loc)

    plt.grid(True)
    plt.savefig(filename,bbox_inches='tight')#,format='pdf')

def plotHist(points,
             title="Estimate vs. Workers",
             xaxis="# Workers",
             yaxis="Estimate",
             legend=[],
             legend_gt=[],
             loc = 'upper right',
             filename="output.png",
             ):

    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': 16})
    #fprop = font_manager.FontProperties(fname= 
    #    '/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

    num_estimators = len(points[1][0])
    num_gt = len(points[2][0])
    
    plt.figure() 
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    shapes = ['--','-*']
    fig, ax = plt.subplots()
    N = len(points[0])
    width = 0.80 * (points[0][1]-points[0][0])
    for i in range(0,num_gt):
        res = [j[i] for j in points[2]]
        if i == num_gt-1:
            plt.plot(points[0], res, shapes[i],linewidth=2,color="#333333")
        else:
            ax.bar(points[0], res, width, color='#ffffff',hatch='/')

    for i in range(0,num_estimators):
        res = [j[i] for j in points[1]]
        ax.bar(points[0], res, width, color=colors[i])
        break 
        #plt.plot(points[0], res, 's-', linewidth=2.5,markersize=7,color=colors[i])
    
    #histogram
    ax.set_ylabel('mariginal number of errors')
    ax.set_title(title)
    ax.set_xticks(points[0])
    ax.set_xticklabels( points[0] ) 
    plt.legend(legend_gt+legend,loc=loc)
    plt.savefig(filename,bbox_inches='tight')#,format='pdf')

def plotHowMuchMore(points,
                    title="Estimate vs. Workers",
                    xaxis="# Workers",
                    yaxis="Estimate",
                    ymax=10000,
                    legend=[],
                    filename="output.png",
                    ):
            

    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout' : True})
    rcParams.update({'font.size': 16})
    #fprop = font_manager.FontProperties(fname= 
    #    '/Library/Fonts/Microsoft/Gill Sans MT.ttf') 


    num_estimators = len(points[1][0])

    plt.figure() 
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff']

    for i in range(num_estimators):#range(0,3):
        res = [j[i] for j in points[1]]
        plt.plot(points[0], res, 's-', linewidth=2.5,markersize=7,color=colors[i])
    #for i in range(2,4):
        #res = [j[i] for j in points[1]]
        #plt.plot(points[0], res, 'o-',ominal:  1454.0 total #workers:  360
    #for i in range(4,6):
        #res = [j[i] for j in points[1]]
        #plt.plot(points[0], res, '--', linewidth=2.5,color=colors[i-4])
    plt.plot(points[0], points[2], '--', linewidth=2.5,color='#333333')
    plt.title(title)#,fontproperties=fprop)
    plt.xlabel(xaxis)#,fontproperties=fprop)
    plt.ylabel(yaxis)#,fontproperties=fprop)
    plt.ylim(ymin=0)     
    plt.xlim(xmin=points[0][0], xmax=points[0][len(points[0])-1]) 
    #plt.legend(legend_gt+legend,loc='upper right')
    plt.grid(True)
    plt.savefig(filename,bbox_inches='tight')#,format='pdf')


"""

# Total Error 1
estimators = [lambda x: sChao92(x,shift=0),vNominal]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 5
n_worker = 30

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 1, recall=0.05, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label



plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators, relative=True),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Perfect Precision, High Recall",
                        filename="plot/simulation_results/total_error_1a.png",#"_fn_only.png",
                        gt=False
                        )

# Total Error2
estimators = [lambda x: sChao92(x,shift=0),vNominal]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=1, recall=0.01,err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label



plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators, relative=True),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Perfect Precision, Low Recall",
                        filename="plot/simulation_results/total_error_1b.png",
                        gt=False
                        )




# Total Error 3
estimators = [lambda x: sChao92(x,shift=0),vNominal, lambda x: sChao92(x,shift=1),lambda x: sChao92(x,shift=2)]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)","Chao92(s=1)","Chao92(s=2)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=0.85, recall=0.01,err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators, relative=True),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper left',
                        xaxis="Workers",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Imperfect Precision, Low Recall",
                        filename="plot/simulation_results/total_error_1c.png",
                        gt=False
                        )


#Total Error 4
estimators = [lambda x: sChao92(x,shift=0),vNominal, lambda x: sChao92(x,shift=1)]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)","Chao92(s=1)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

X = []
Y = []
GT = []

for i in range(0,32,2):
    p = (100.0-i)/100
    

    a = numpy.zeros((10,3))
    for t in range(0,10):
        d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=p, recall=0.01,err_skew=False,priotized=False)
        label = pickle.load(open('dataset/sim_label.p','rb'))
        slist = label
        a[t,:]=(holdoutRealWorkers(d,gt_list,[50],estimators, relative=True)[1][0])

    Y.append(numpy.median(a,axis=0))
    GT.append([1.0])
    X.append(i)
    print p

plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper left',
                        xaxis="FPR %",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Tradeoff False Positives",
                        filename="plot/simulation_results/total_error_1d.png",
                        gt=False
                        )


#Total Error 5
estimators = [lambda x: sChao92(x,shift=0),vNominal, lambda x: sChao92(x,shift=1)]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)","Chao92(s=1)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

X = []
Y = []
GT = []

for i in range(0,100,5):
    p = (100.0-i)/100
    

    a = numpy.zeros((10,3))
    for t in range(0,10):
        d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=1.0, recall=0.05*p,err_skew=False,priotized=False)
        label = pickle.load(open('dataset/sim_label.p','rb'))
        slist = label
        a[t,:]=(holdoutRealWorkers(d,gt_list,[50],estimators, relative=True)[1][0])

    Y.append(numpy.median(a,axis=0))
    GT.append([1.0])
    X.append(i)
    print p

plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper left',
                        xaxis="FNR %",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Tradeoff False Negatives",
                        filename="plot/simulation_results/total_error_1e.png",
                        gt=False
                        )


"""

# Switch Error 1
dirty = 0.2
scale = 10
n_worker = 100

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 0.95, recall=0.20, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label

estimators = [lambda x: remain_switch(x, False)-sNominal(x)]
gt_list = [lambda x: gt_switch(x,slist)]
legend=["Chao92+Switch","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Estimate (# Remaining Switches)",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Switches: High Precision, High Coverage",
                        filename="plot/simulation_results/switch_error_1a.png"
                        )

# Switch Error 2
dirty = 0.2
scale = 30
n_worker = 300

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 0.80, recall=0.10, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label

estimators = [lambda x: remain_switch(x, True)-sNominal(x)]
gt_list = [lambda x: gt_switch(x,slist)]
legend=["Chao92+Switch","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Estimate (# Remaining Switches)",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Switches: Low Precision, High Coverage",
                        filename="plot/simulation_results/switch_error_1b.png"
                        )

# Switch Error 3
dirty = 0.2
scale = 30
n_worker = 300

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 0.80, recall=0.10, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label

estimators = [lambda x: remain_switch(x, True)-sNominal(x)]
gt_list = [lambda x: gt_switch(x,slist)]
legend=["Chao92+Switch","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Estimate (# Remaining Switches)",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Switches: Low Precision, Low Coverage",
                        filename="plot/simulation_results/switch_error_1c.png"
                        )



"""

# Switch Error 1
dirty = 0.1
scale = 25
n_worker = 150

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 0.98, recall=0.05, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label

estimators = [lambda x: remain_switch(x)-sNominal(x)]
gt_list = [lambda x: gt_switch(x,slist)]
legend=["Chao92+Switch","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Error (# Remaining Switches)",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Switches: Very High Precision, High Recall",
                        filename="plot/simulation_results/switch_error_1a.png"#"_fn_only.png",
                        )


# Switch Error 1
dirty = 0.1
scale = 25
n_worker = 150

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 0.98, recall=0.01, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label

estimators = [lambda x: remain_switch(x)-sNominal(x)]
gt_list = [lambda x: gt_switch(x,slist)]
legend=["Chao92+Switch","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'center right',
                        xaxis="Workers",
                        yaxis="Error (# Remaining Switches)",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Switches: Very High Precision, Low Recall",
                        filename="plot/simulation_results/switch_error_1b.png"#"_fn_only.png",
                        )

# Switch Error 1
dirty = 0.1
scale = 25
n_worker = 150

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 0.65, recall=0.05, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label

estimators = [lambda x: remain_switch(x)-sNominal(x)]
gt_list = [lambda x: gt_switch(x,slist)]
legend=["Chao92+Switch","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'center right',
                        xaxis="Workers",
                        yaxis="Error (# Remaining Switches)",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Switches: Low Precision, High Recall",
                        filename="plot/simulation_results/switch_error_1c.png"#"_fn_only.png",
                        )


# Switch Error 1
dirty = 0.1
scale = 25
n_worker = 150

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision= 0.98, recall=0.05, err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label

estimators = [lambda x: remain_switch2(x)]
gt_list = [gt_switch2]
legend=["Chao92+Switch","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]


plotY1Y2(holdoutRealSwitchWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Error (# Remaining Switches)",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Switches: Very High Precision, High Recall",
                        filename="plot/simulation_results/switch_error_1a.png"#"_fn_only.png",
                        )

# Total Error2
estimators = [lambda x: sChao92(x,shift=0),vNominal]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=1, recall=0.01,err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label



plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators, relative=True),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Perfect Precision, Low Recall",
                        filename="plot/simulation_results/total_error_1b.png",
                        gt=False
                        )




# Total Error 3
estimators = [lambda x: sChao92(x,shift=0),vNominal, lambda x: sChao92(x,shift=1),lambda x: sChao92(x,shift=2)]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)","Chao92(s=1)","Chao92(s=2)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=0.85, recall=0.01,err_skew=False,priotized=False)
label = pickle.load(open('dataset/sim_label.p','rb'))
slist = label


plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators, relative=True),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper left',
                        xaxis="Workers",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Imperfect Precision, Low Recall",
                        filename="plot/simulation_results/total_error_1c.png",
                        gt=False
                        )


#Total Error 4
estimators = [lambda x: sChao92(x,shift=0),vNominal, lambda x: sChao92(x,shift=1)]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)","Chao92(s=1)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

X = []
Y = []
GT = []

for i in range(0,32,2):
    p = (100.0-i)/100
    

    a = numpy.zeros((10,3))
    for t in range(0,10):
        d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=p, recall=0.01,err_skew=False,priotized=False)
        label = pickle.load(open('dataset/sim_label.p','rb'))
        slist = label
        a[t,:]=(holdoutRealWorkers(d,gt_list,[50],estimators, relative=True)[1][0])

    Y.append(numpy.median(a,axis=0))
    GT.append([1.0])
    X.append(i)
    print p

plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper left',
                        xaxis="FPR %",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Tradeoff False Positives",
                        filename="plot/simulation_results/total_error_1d.png",
                        gt=False
                        )


#Total Error 5
estimators = [lambda x: sChao92(x,shift=0),vNominal, lambda x: sChao92(x,shift=1)]
gt_list = [lambda x: gt]
legend=["Chao92","Nominal (Voting)","Chao92(s=1)"]
legend_gt=["Ground Truth (total)"]
dirty = 0.1
scale = 10
n_worker = 120

X = []
Y = []
GT = []

for i in range(0,100,5):
    p = (100.0-i)/100
    

    a = numpy.zeros((10,3))
    for t in range(0,10):
        d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=1.0, recall=0.05*p,err_skew=False,priotized=False)
        label = pickle.load(open('dataset/sim_label.p','rb'))
        slist = label
        a[t,:]=(holdoutRealWorkers(d,gt_list,[50],estimators, relative=True)[1][0])

    Y.append(numpy.median(a,axis=0))
    GT.append([1.0])
    X.append(i)
    print p

plotY1Y2((X,Y,GT),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper left',
                        xaxis="FNR %",
                        yaxis="Error",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = "Total Error: Tradeoff False Negatives",
                        filename="plot/simulation_results/total_error_1e.png",
                        gt=False
                        )

"""














#OLD



"""
config = [(1,0,0)]
#error = [(0.3,0),(0.6,0),(0.3,0.5),(0.6,0.5)]
#titles = ['(A) fpr=0.3, fnr=0.0','(B) fpr=0.6, fnr=0.0','(C) fpr=0.3, fnr=0.5','(D) fpr=0.6, fnr=0.5']
priotize = [True,False]
recall = [0.1,0.3]
precision = [0.6,0.8]
titles = ['(a) sample 10% at random','(b) sample 30% at random']
#titles = ['(a) 10% FN','(b) 20% FN','(c) 30% FN']
for c in config:
    #total error
    estimators = [wChao92,lambda x: sChao92(x,shift=1),vNominal]
    gt_list = [lambda x: gt]
    legend=["wChao92","vChao92(s=1)","Nominal (Voting)"]
    legend_gt=["Ground Truth (total)"]
    #switch 
    estimators2 = [lambda x: remain_switch(x) - sNominal(x)]
    gt_list2 = [lambda x: gt_switch(x,slist)]
    legend2 = ["Chao92-sNominal"]
    legend_gt2 = ["Ground Truth (rem. switch)"]
    #marginal
    estimators3 = [goodToulmin,lambda x:goodToulmin2(x,10), ]
    gt_list3 = [lambda x: gt_marginal(x,slist),lambda x: gt_remaining(x,slist)]
    legend3 = ["GT (m=sqrt(n))","GT (m=10)"]
    legend_gt3 = ["Ground Truth (rem. error)","Ground Truth (marg. error)"]
    #estimators = [chao92,vGoodToulmin,lambda x:sChao92(data=x,shift=1),wChao92,qaChao92,vNominal,nominal,lambda x: ext]
    #legend=["Chao92","GT","vChao92(s=1)","wChao92","qaChao92","Nominal(Voting)","Nominal","Extrap. (1%)"]
    n_worker = 42
    scale = 2
    dirty = .2
    testBias = False
  
    for prio in priotize:
        for prec in precision:
            for rec in recall: 
                d,gt, prec_, rec_ = simulatedData(dirty=dirty,precision=prec, recall=rec,err_skew=False,priotized=prio)
                #simulated data
                label = pickle.load(open('dataset/sim_label.p','rb'))
                slist = label
                #d, gt, prec_, rec_ = loadRestaurant(simulated=True, priotize=prio, n_workers=n_worker, precision=prec,recall=rec)
                #restaurant simulation
                #solutions = pickle.load(open('dataset/simulated_solution.p','rb'))
                #slist = [v[0] for v in solutions.values()]

                # extrapolation
                pair_sample = numpy.random.choice(range(len(slist)),len(d)/100,replace=False)
                ext = numpy.sum([slist[s] for s in pair_sample])
                q = len(pair_sample) / float(len(d))
                ext = ext/q
                                    
                plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        loc = 'upper right',
                        xaxis="Workers",
                        #title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        title = titles[recall.index(rec)],
                        filename="plot/sim_total_prec"+str(prec).replace('.','')+"_rec"+str(rec).replace('.','')+"_prio"+str(prio)+".png"#"_fn_only.png",
                        )
                
                plotY1Y2(holdoutRealWorkers(d,gt_list2,range(scale,n_worker,scale),estimators2),
                        legend = legend2,      
                        legend_gt = legend_gt2,  
                        loc = 'lower right',
                        xaxis="Workers",
                        title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        #filename="plot/simulation_alternate(skew)_prec%d"%precision.index(prec)+"_rec%d"%recall.index(rec)+".png",
                        filename="plot/sim_switch_prec"+str(prec).replace('.','')+"_rec"+str(rec).replace('.','')+"_prio"+str(prio)+".png",
                        )
                '''                
                plotY1Y2(holdoutRealWorkers(d,gt_list3,range(scale,n_worker,scale),estimators3),
                        legend = legend3,
                        legend_gt = legend_gt3,        
                        xaxis="Workers",
                        title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        #filename="plot/simulation_alternate(skew)_prec%d"%precision.index(prec)+"_rec%d"%recall.index(rec)+".png",
                        filename="plot/sim_marginal_prec"+str(prec).replace('.','')+"_rec"+str(rec).replace('.','')+"_prio"+str(prio)+".png",
                        )
                '''
                plotHist(holdoutRealWorkers(d,gt_list3,range(scale,n_worker,scale),estimators3),
                        legend = legend3,
                        legend_gt = legend_gt3,        
                        xaxis="Workers",
                        title='Restaurant Dataset, %d simulated workers'%n_worker+'\n' + '(rec=%f'%rec + ', prec=%f'%prec + '), priotize=' + str(prio), 
                        #filename="plot/simulation_alternate(skew)_prec%d"%precision.index(prec)+"_rec%d"%recall.index(rec)+".png",
                        filename="plot/sim_hist_prec"+str(prec).replace('.','')+"_rec"+str(rec).replace('.','')+"_prio"+str(prio)+".png",
                        )

"""
                 
"""
#total error
estimators = [wChao92,lambda x: sChao92(x,shift=1),vNominal]
gt_list = [lambda x: gt]
legend=["wChao92","vChao92(s=1)","Nominal (Voting)"]
legend_gt=["Ground Truth (total)"]
#switch 
estimators2 = [lambda x:remain_switch(x) - sNominal(x)]
gt_list2 = [lambda x: gt_switch(x,slist)]
legend2 = ["Chao92-sNominal"]
legend_gt2 = ["Ground Truth (rem. switch)"]
#compare switch and total error estimations
estimators2b = [lambda x: gt_switch(x,slist), vNominal]
legend2b = ["Ground truth(remaining switches)","Nominal(voting)"]
#marginal
estimators3 = [goodToulmin]
gt_list3 = [lambda x: gt_marginal(x,slist)]
legend3 = ["GT (m=sqrt(n))","GT (m=2*sqrt(n))"]
legend_gt3 = ["Ground Truth (marg. error)"]

n_worker = 400
scale = 20


#restaurant dataset
d,gt,prec,rec = loadRestaurant2('dataset/mostly_hard_new_4.csv',priotization=True)
pair_solution = pickle.load( open('dataset/pair_solution.p','rb') )
# Pair solution ground truth
slist = pair_solution.values()
print 'restaurant data:',len(slist),numpy.sum(numpy.array(slist) == 1)

plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (with all workers)', 
                        filename="plot/rest_mostly_hard_all.png",
                        )

plotY1Y2(holdoutRealWorkers(d,gt_list2,range(scale,n_worker,scale),estimators2),
                        legend = legend2,    
                        legend_gt = legend_gt2,    
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (with all workers)', 
                        filename="plot/rest_mostly_hard_all_switch.png",
                        )

plotHist(holdoutRealWorkers(d,gt_list3,range(scale,n_worker,scale),estimators3),
                        legend = legend3,    
                        legend_gt = legend_gt3,    
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (with all workers)', 
                        filename="plot/rest_mostly_hard_all_switch_marginal.png",
                        )


#product dataset
print 'loading product data'
d,gt,prec,rec = loadProduct()
pair_solution = pickle.load( open('dataset/products/pair_solution.p','rb'))
print 'loaded product data'

# extrapolation
'''
ilist = pair_solution.keys()
pair_sample = numpy.random.choice(range(len(ilist)),len(d)/20,replace=False)
ext = numpy.sum([pair_solution[ilist[s]] for s in pair_sample])
q = len(pair_sample) / float(len(d))
ext = ext/q
'''
# Pair solution ground truth
slist = pair_solution.values()
#print len(slist),numpy.sum(numpy.array(slist) == 1), slist

plotY1Y2(holdoutRealWorkers(d,gt_list,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        legend_gt=legend_gt,
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (with all workers)', 
                        filename="plot/product_mostly_hard_all.png",
                        )

plotY1Y2(holdoutRealWorkers(d,gt_list2,range(scale,n_worker,scale),estimators2),
                        legend = legend2,    
                        legend_gt = legend_gt2,    
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (with all workers)', 
                        filename="plot/product_mostly_hard_all_switch.png",
                        )

plotHist(holdoutRealWorkers(d,gt_list3,range(scale,n_worker,scale),estimators3),
                        legend = legend3,    
                        legend_gt = legend_gt3,    
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (with all workers)', 
                        filename="plot/product_mostly_hard_all_switch_marginal.png",
                        )


d,gt,prec,rec = loadRestaurant2('dataset/mostly_hard_new_4.csv',wq_assurance=True)
plotY1Y2(holdoutRealWorkers(d,gt,range(scale,n_worker,scale),estimators),
                        legend=legend,
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (without bad workers)', 
                        filename="plot/mostly_hard_no_bad.png",
                        ymax=50
                        )
plotHowMuchMore(holdoutRealWorkers(d,-10,range(scale,n_worker,scale),estimators2),
                        legend = legend2,        
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (without bad workers)', 
                        filename="plot/mostly_hard_yes_bad_switch.png",
                        )

pirs:  13022 #workers:  243
roduct dataset
plotHowMuchMore(holdoutRealWorkers(d,-10,range(scale,n_worker,scale),estimators3),
                        legend = legend3,        
                        xaxis="Workers",
                        title='mostly_hard_new_4.csv (without bad workers)', 
                        filename="plot/mostly_hard_yes_bad_switch_marginal.png",
                        )
"""
