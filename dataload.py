#!/usr/bin/python
import numpy as np
import pylab as P
import matplotlib.pyplot as plt
import math
import random
import csv
import pickle
from estimator import remain_switch, sNominal, gt_switch
#import editdistance

"""
    The CrowdFlower sentiment data asks the rater to judge the sentiment of a tweet
    discussing the weather. The data is comprised of 98,979 tweets. Each tweet was
    evaluated by at least 5 raters, for a total of approximately 500,000 answers.
"""
def loadCrowdFlowerData(filename='dataset/weather-non-agg-DFE.csv'):
    raterset = set()
    docset = set()
    tuplelist = []
    not_related = set() # predict number of not related tweets
    fpr,fnr,p,n = 0., 0.

    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            raterset.add(row[7]) #_worker_id
            docset.add(row[14]) #tweet_id

            # I can't tell / Negative / Neutral (author is just sharing information) 
            # / Positive / Tweet not related to weather condition
            if 'not' in row[12]:
                tuplelist.append((row[14],row[7]))
                not_related.add(row[14])
                p += 1
            else:
                n += 1

    # tweet-rater matrix
    data = np.zeros((len(docset),len(raterset)))
    ilist1 = list(docset)
    ilist2 = list(raterset)

    for t in tuplelist:
        index1 = ilist1.index(t[0])
        index2 = ilist2.index(t[1])
        data[index1,index2] = 1
        if t[0] not in not_related:
            fpr += 1/float(p)
    for t in not_related:
        index1 = ilist1.index(t)
        for r in range(len(ilist2)):
            if data[index1,r] == 0:
                fnr += 1/float(n)

    return data, fpr, fnr, len(not_related)

def simulatedData(items,fpr,fnr,workers,dirty,recall,err_skew=False,priotized=False,eps=0.,hdirty=0.1,error_type=0):
    # generate duplicates (positive examples)
    label = np.zeros(items)
    for i in range(int(items*dirty)):
        label[i] = 1
    random.shuffle(label)

    pickle.dump(label,open('dataset/sim_label.p','wb'))
    label = pickle.load(open('dataset/sim_label.p','rb'))

    similarity = np.zeros(items)
    dirty_heuristic = int(items*dirty*(1.-hdirty))
    cnt = 0
    for i in range(items):
        if label[i] == 1:
            cnt += 1
            similarity[i] = 0.7
        if cnt >= dirty_heuristic:
            break

    dist_fpr = np.zeros(items)
    dist_fnr = np.zeros(items)
    if err_skew:
        dist_fpr = np.random.normal(loc=1.-fpr,scale=0.03,size=items)
        dist_fpr = dist_fpr/float(np.max(dist_fpr))
        dist_fpr[dist_fpr < max(1.-fpr-0.03, 0.6)] = max(1.-fpr-0.03, 0.6)
        dist_fpr[dist_fpr > 1.0] = 1.0

        dist_fnr = np.random.normal(loc=1.-fnr,scale=0.03,size=items)
        dist_fnr = dist_fnr/float(np.max(dist_fnr))
        dist_fnr[dist_fnr < max(1.-fnr-0.03, 0.6)] = max(1.-fnr-0.03, 0.6)
        dist_fnr[dist_fnr > 1.0] = 1.0
    else:
        dist_fpr = np.zeros(items) + (1.-fpr) #uniform distribution
        dist_fnr = np.zeros(items) + (1.-fnr) #uniform distribution

    data = np.zeros((items,workers))
    for w in range(workers):
        for i in range(items):
            if label[i] == 1 and random.random() <= dist_fnr[i]:
                data[i,w] = label[i]
            elif label[i] == 0 and random.random() <= dist_fpr[i]:
                data[i,w] = label[i]
            else:
                if error_type == 0:
                    data[i,w] = (label[i] + 1)%2
                elif error_type == 1:
                    data[i,w] = 1
                elif error_type == 2:
                    data[i,w] = 0

    # skip over some pairs
    for w in range(workers):
        for i in range(items):
            if not priotized: # random
                if random.random() > recall: # uniform distribution
                    data[i,w] = -1
            else: # priotized
                if random.random() > recall: # uniform distribution
                    data[i,w] = -1
                # have the difficult pairs examined by workers more & randomly give out seemingly easier pairs
                if similarity[i] < 0.5 or similarity[i] > 0.9:
                    #easy pairs
                    if random.random() > eps:
                        data[i,w] = -1
                else:
                    if random.random() > (1.-eps):
                        data[i,w] = -1

    gt = np.sum(label)
    #if priotized:
        #gt = np.sum(label) - np.sum(np.logical_and(np.sum(data,axis=1) == -1 * workers,label == 1))
    return data, gt, 0., 0.

def simulatedData2(items=1000,workers=100,dirty=0.2,recall=1.0,precision=1.0,err_skew=False,a=2):
    # generate duplicates (positive examples)
    label = np.zeros(items)
    label[range(int(items*dirty))] = 1
    random.shuffle(label)  

    pickle.dump(label,open('dataset/sim_label.p','wb'))
    label = pickle.load(open('dataset/sim_label.p','rb'))

    if err_skew:
        # skewed difficulty for items
        import scipy.special as sps
        x = np.arange(1,items)
        px = x**(-a)/sps.zetac(a)
        px = px/max(px)
        dist = px
    if not err_skew:
        # uniform difficulty for items
        dist = np.zeros(items) + precision 

    precisions = np.zeros(workers)
    data = np.zeros((items,workers))
    for w in range(workers):
        for i in range(items):
            if random.random() <= dist[i]:
                data[i,w] = label[i]
                precisions[w] += 1.
            elif label[i] == 1:
                data[i,w] = 0 #math.fabs(label[i] - 1)
            elif label[i] == 0:
                data[i,w] = 1
    precisions /= items

    # skip over some pairs
    recalls = np.zeros(workers) + items
    for w in range(workers):
        for i in range(items):
            if random.random() > recall: # uniform distribution
                data[i,w] = -1
                recalls[w] -= 1.
    recalls /= items
    gt = np.sum(label)

    return data, gt, np.mean(precisions), np.mean(recalls)


def loadAddress():
    tasks_table = 'dataset/addr/address_dataset'
    workers_table = 'dataset/addr/address_workers.csv'
    
    task_id = set()
    task_sol = {}
    with open(tasks_table,'rb') as f:
        reader = csv.reader(f)
        for task in reader:
            index = task[0].split(" ")[0]
            if '*' in index:
                index = index.split("*")[0]
                task_sol[index] = 1 
            else:
                task_sol[index] = 0
            task_id.add(index)
    pickle.dump( task_sol, open('dataset/addr_solution.p','wb') )
    task_sol = pickle.load( open('dataset/addr_solution.p','rb') )

    worker_id = set()
    task_workers = {}
    with open(workers_table,'rb') as f:
        reader = csv.reader(f)
        for worker in reader:
            if worker[0] == 'id':
                continue

            wid = worker[4] #worker[3]
            worker_id.add(wid)
            for addr_resp in worker[1].split(","):
                addr_resp = addr_resp.replace("{","").replace("}","")
                resp = addr_resp.split(":")[1].strip()
                tid = addr_resp.split(":")[0]
                tid = tid.replace("\"","").split("Address")[1]
                
                if tid not in task_workers:
                    task_workers[tid] = [(wid,resp)]
                else:
                    task_workers[tid].append((wid,resp))

    #remove not well-formatted tasks that are inherently so difficult
    err_rates = []
    discards = set()
    for k,v in task_workers.iteritems():
        err = 0.
        for t in v:
            if float(t[1]) != float(task_sol[k]):
                err += 1.
        err_rates.append(err / len(v))
        if err / len(v) > 0.5:
            discards.add(k)
    print 'discards set size', len(discards)
    print 'previdous', len(task_workers)
    #for k in discards:
    #    del task_workers[k]
    #    task_id.discard(k)
    print 'current', len(task_workers)

    ilist_task = task_sol.keys() #list(task_id)
    ilist_worker = list(worker_id)
    print 'address dataset dimensions: ', len(ilist_task), len(ilist_worker) 
    data = np.zeros((len(ilist_task),len(ilist_worker))) + -1
    for i in range(len(task_sol)):
        tid = ilist_task[i]
        for t in task_workers[tid]:
            data[i, ilist_worker.index(t[0])] = t[1]

    return data, np.sum(task_sol.values()), 0., 0.


"""
    limited to the completed tasks
"""
def loadInstitution():    
    tasks_table = 'dataset/tasks.csv'
    workers_table = 'dataset/workers.csv'
    
    task_id = set()
    worker_id = set()
    task_sol = {}
    task_workers = {}
    completed_tasks, dup, non = 0, 0, 0
    with open(tasks_table,'rb') as f:
        reader = csv.reader(f)
        for task in reader:
            answers = task[6][1:-2].replace("\"","").split(",")
            if len(answers) >= 3:
                completed_tasks += 1
                task_id.add(task[3])
                ans_sum = 0.
                w = []
                for ans in answers:
                    w.append((ans.split(":")[0],ans.split(":")[1]))
                    worker_id.add(ans.split(":")[0])
                    ans_sum += float(ans.split(":")[1])
                task_workers[task[3]] = w
                if ans_sum > len(answers)/2:
                    dup += 1
                    task_sol[task[3]] = 1 #duplicate
                else:
                    non += 1
                    task_sol[task[3]] = 0
            else:
                continue

    ilist_task = list(task_id)
    ilist_worker = list(worker_id)
    data = np.zeros((len(ilist_task),len(ilist_worker))) + -1
    for k,v in task_workers.iteritems():
        for t in v:
            data[ilist_task.index(k),ilist_worker.index(t[0])] = t[1]

    return data, np.sum(task_sol.values()), 0., 0.


def loadProduct(filename,priotization=True):
    base_table = 'dataset/products/products.csv'
    gt_table = 'dataset/products/product_mapping.csv'
    #response_table1 = 'dataset/products/products_table1.csv'
    #response_table2 = 'dataset/products/products_table2.csv'
    response_table1 = 'dataset/jn_heur/jn_heur_products.csv'
    #dict {o1: easy case below threshold, o2: easy above threshold, h: heuristic}
    #pair_clusters = pickle.load(open('dataset/products/data.p','rb')) 
    pair_clusters = pickle.load(open('dataset/jn_heur/save-s5.p','rb'))
    hard_pairs_ = pair_clusters#pair_clusters['h']
    print 'hard_pairs_', len(hard_pairs_)
    hard_pairs = {}
    for p in hard_pairs_:
        rid1 = p[0][0].replace('\"','').strip() 
        rid2 = p[0][1].replace('\"','').strip()
        if (rid1,rid2) not in hard_pairs and (rid2,rid1) not in hard_pairs:
            hard_pairs[(rid1,rid2)] = float(p[1])
    matches = {}
    with open(gt_table,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            matches[(row[0],row[1])] = 1
    print 'matches : ',np.sum(matches.values())
    # match = 0
    ''' 
    records_amzn = {}
    records_goog = {}
    with open(base_table,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            source = row[0] # amazon | google
            if source == 'amazon':
                records_amzn[row[1].strip()] = row[2:5]
            else:
                records_goog[row[1].strip()] = row[2:5]
    pickle.dump( records_amzn, open('dataset/products/records_amzn.p','wb') )
    pickle.dump( records_goog, open('dataset/products/records_goog.p','wb') )
    '''
    records_amzn = pickle.load( open('dataset/products/records_amzn.p','rb') )
    records_goog = pickle.load( open('dataset/products/records_goog.p','rb') ) 

    """
    pair_solution = {}
    for rid1 in records_goog.keys():
        for rid2 in records_amzn.keys():
            # priotization == True?
            if priotization and ((rid1,rid2) not in hard_pairs and (rid2,rid1) not in hard_pairs):
                continue
            elif (rid1,rid2) not in pair_solution and (rid2,rid1) not in pair_solution:
                if (rid1,rid2) in matches or (rid2,rid1) in matches:
                    pair_solution[(rid1,rid2)] = 1
                else:
                    pair_solution[(rid1,rid2)] = 0
    #pickle.dump( pair_solution, open('dataset/products/pair_solution.p','wb') )
    #pair_solution = pickle.load ( open('dataset/products/pair_solution.p','rb') ) # dictionary
    pickle.dump( pair_solution, open('dataset/jn_heur/pair_solution.p','wb') )
    """
    pair_solution = pickle.load ( open('dataset/jn_heur/pair_solution.p','rb') ) # dictionary
    print 'pair_solution loaded, ground-truth: ', np.sum(pair_solution.values()) 

    #non-heuristic pairs
    """
    easy_pair_solution = {}
    for rid1 in records_goog.keys():
        for rid2 in records_amzn.keys():
            if rid1 == rid2: 
                continue
            # priotization == True?
            elif ((rid1,rid2) in hard_pairs or (rid2,rid1) in hard_pairs):
                continue
            if (rid1,rid2) not in easy_pair_solution and (rid2,rid1) not in easy_pair_solution:
                if (rid1,rid2) in matches or (rid2,rid1) in matches:
                    easy_pair_solution[(rid1,rid2)] = 1
                else:
                    easy_pair_solution[(rid1,rid2)] = 0
    #pickle.dump( easy_pair_solution, open('dataset/products/easy_pair_solution.p','wb') )
    #easy_pair_solution = pickle.load ( open('dataset/products/easy_pair_solution.p','rb') ) # dictionary
    pickle.dump( easy_pair_solution, open('dataset/jn_heur/easy_pair_solution.p','wb') )
    """
    easy_pair_solution = pickle.load ( open('dataset/jn_heur/easy_pair_solution.p','rb') ) # dictionary

    worker_resp = {}
    for asgn_table in filename:
        with open(asgn_table,'rb') as f:
            reader = csv.reader(f)
            for asgn in reader:
                if asgn[0] == 'id':
                    continue
                replacements = {'}':'','{':'','-':'','"':'','\t':'',',':''}
                pairs = asgn[1]
                for x,y in replacements.iteritems():
                    pairs = pairs.replace(x,y)
                pairs = pairs.split('Pair')
                w = asgn[4]#asgn[3] #worker id
                for p in pairs:
                     rid1_rid2_resp = p.split('\\')
                     if len(rid1_rid2_resp) < 5:
                         continue
                     rid1 = rid1_rid2_resp[1]
                     rid2 = rid1_rid2_resp[3]
                     resp = float(rid1_rid2_resp[4].replace(': ',''))
                     tup = ( (rid1,rid2), resp )
                     tup2 = ( (rid2,rid1), resp )
                     if priotization and tup[0] not in hard_pairs and tup2[0] not in hard_pairs:
                         #print tup[0], hard_pairs.keys()[0]
                         continue
                     if w in worker_resp:
                         if tup in worker_resp[w] or tup2 in worker_resp[w]:
                             continue
                         else:
                             worker_resp[w].append(tup)
                     else:
                         worker_resp[w] = [tup]
    print 'worker_resp loaded'

    # worker evaluation
    score = []
    for w in worker_resp.keys():
        correct = 0.
        for res in worker_resp[w]:
            if res[0] in pair_solution and pair_solution[res[0]] == res[1]:
                correct += 1.
            elif (res[0][1],res[0][0]) in pair_solution and pair_solution[(res[0][1],res[0][0])] == res[1]:
                correct += 1.
        # remove bad workers
        if False and score[-1] < 0.6:
            worker_resp.pop(w)
    score = np.array(score)
    print '#bad workers:',np.sum(score < 0.6)

    ilist_workers = worker_resp.keys()
    ilist_pairs = pair_solution.keys()
    lookup_tbl = {}
    for i in range(len(ilist_pairs)):
        lookup_tbl[ilist_pairs[i]] = i
    data = np.zeros((len(ilist_pairs),len(ilist_workers))) + -1
    print '#pairs: ', len(ilist_pairs), '#workers: ',len(ilist_workers)

    for k,v in worker_resp.iteritems():
        for pair_resp in v:
            #print 'check: ', pair_resp[0]
            if pair_resp[0] in pair_solution:
                data[lookup_tbl[pair_resp[0]],ilist_workers.index(k)] = pair_resp[1]
            elif (pair_resp[0][1],pair_resp[0][0]) in pair_solution:
                data[lookup_tbl[(pair_resp[0][1],pair_resp[0][0])],ilist_workers.index(k)] = pair_resp[1]

    print 'n=',np.sum(np.sum(data != -1,axis=1))
    print 'Ground truth (True Majority Error):', np.sum(np.array(pair_solution.values()) == 1)
    print 'False Majoirty Clean:', np.sum(np.logical_and(np.array(pair_solution.values()) == 1, np.sum(data == 0,axis=1) >= np.sum(data != -1, axis=1)/2))
    #data[np.logical_and(np.array(pair_solution.values()) == 1, np.sum(data == 0,axis=1) >= np.sum(data != -1, axis=1)/2),:] = 1
    print 'False Majority Error:', np.sum(np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2))
    #data[np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2),:] = 0
    print '+ votes on FME:', np.sum(data[np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2),:] == 1,axis=1)
    print '- votes on FME:', np.sum(data[np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2),:] == 0,axis=1) 
    #print np.sum(data[np.array(pair_solution.values()) == 1,:] == 1, axis=1), np.sum(data[np.array(pair_solution.values()) == 1,:] == 0, axis=1)    
    #print np.array(pair_solution.keys())[np.array(pair_solution.values()) == 1] #rids of ground truth
    print 'TN:', np.sum(np.array(pair_solution.values()) == 0), 'FP:', np.sum(data[np.array(pair_solution.values()) == 0,:] == 1)
    print 'TP:', np.sum(data[np.array(pair_solution.values()) == 0,:] == 1), 'FN:', np.sum(data[np.array(pair_solution.values()) == 1,:] == 0) 
    return data, np.sum(pair_solution.values()), 0., 0.
   

def loadRestaurant2(filename,wq_assurance=False,priotization=True):
    base_table = 'dataset/restaurant.csv'
    hard_pairs_ = pickle.load( open('dataset/hard_pairs.p','rb') ) # list of tuples
    print 'len of hard_pairs_', len(hard_pairs_)
    hard_pairs = {} #heuristic pairq = len(pair_sample) / float(len(d))s, in case of imperfect heuristics, we include random easy pairs during crowdsourcing
    for p in hard_pairs_:
        rid1 = int(p[0][0]) 
        rid2 = int(p[0][1]) 
        if rid1 < rid2:
            hard_pairs[(rid1,rid2)] = float(p[1])
        else:
            hard_pairs[(rid2,rid1)] = float(p[1])
    records = {}
    with open(base_table,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            gid = int(row[1]) # GT label
            rid = int(row[0]) # rid
            name = row[2]
            records[rid] = (gid,name)
                         
    pair_solution = {}
    for rid1 in records.keys():
        for rid2 in records.keys():
            if rid1 == rid2: 
                continue
            # priotization == True?
            elif priotization and ((rid1,rid2) not in hard_pairs and (rid2,rid1) not in hard_pairs):
                continue
            if (rid1,rid2) not in pair_solution and (rid2,rid1) not in pair_solution:
                if records[rid1][0] == records[rid2][0]:
                    pair_solution[(rid1,rid2)] = 1
                else:
                    pair_solution[(rid1,rid2)] = 0
    pickle.dump( pair_solution, open('dataset/pair_solution.p','wb') )
    
    #216 (1~217, except 8,21,93) records have duplicates (one each); 106 pairs.
    pair_solution = pickle.load ( open('dataset/pair_solution.p','rb') ) # dictionary
    print 'pair_solution loaded, ground-truth: ', np.sum(pair_solution.values()) 

    #non-heuristic pairs
    easy_pair_solution = {}
    for rid1 in records.keys():
        for rid2 in records.keys():
            if rid1 == rid2: 
                continue
            # priotization == True?
            elif ((rid1,rid2) in hard_pairs or (rid2,rid1) in hard_pairs):
                continue
            if (rid1,rid2) not in easy_pair_solution and (rid2,rid1) not in easy_pair_solution:
                if records[rid1][0] == records[rid2][0]:
                    easy_pair_solution[(rid1,rid2)] = 1
                else:
                    easy_pair_solution[(rid1,rid2)] = 0
    pickle.dump( easy_pair_solution, open('dataset/easy_pair_solution.p','wb') )
    easy_pair_solution = pickle.load ( open('dataset/easy_pair_solution.p','rb') ) # dictionary

    task_resp = {}
    worker_resp = {}
    for asgn_table in filename:    
        with open(asgn_table,'rb') as f:
            reader = csv.reader(f)
            for asgn in reader:
                if asgn[0] == 'id':
                    continue
                w = asgn[4] #asgn[3]
                task = asgn[2]
            
                answers = asgn[1][1:-2].replace("\"","").replace("Pair","").split(",")
                for ans in answers:
                    rids = ans.split(":")[0].strip().split("-")
                    resp = float(ans.split(":")[1])
                    sim = jaccard(records[int(rids[0])][1],records[int(rids[1])][1])
                    tup = ( (int(rids[0]),int(rids[1])), resp )
                    if tup[0] not in pair_solution:
                        tup = ( (int(rids[1]),int(rids[0])), resp )
                    if priotization and tup[0] not in hard_pairs:
                        continue
                    # for worker_resp
                    if w in worker_resp:
                        if tup in worker_resp[w]:
                            continue
                        else:
                            worker_resp[w].append(tup)
                    else:
                        worker_resp[w] = [tup]
                    # for task_resp
                    if task in task_resp:
                        if tup in task_resp[task]:
                            continue
                        else:
                            task_resp[task].append(tup)
                    else:
                        task_resp[task] = [tup]
        print '#workers:',len(worker_resp), '#tasks:',len(task_resp)
    print 'worker_resp loaded' 
    
    # worker evaluation
    score = []
    for w in worker_resp.keys():
        correct = 0.
        for res in worker_resp[w]:
            if res[0][0] < res[0][1]:
                if pair_solution[res[0]] == res[1]:
                    correct += 1.
            else:
                if pair_solution[(res[0][1],res[0][0])] == res[1]:
                    correct += 1.
        score.append(correct/len(worker_resp[w]))
        # remove bad workers
        if wq_assurance and score[-1] < 0.6:
            worker_resp.pop(w)
    score = np.array(score)
    print '#bad workers:',np.sum(score < 0.6)

    ilist_workers = worker_resp.keys()
    ilist_tasks = task_resp.keys()
    ilist_pairs = pair_solution.keys()
    lookup_tbl = {}
    for i in range(len(ilist_pairs)):
        lookup_tbl[ilist_pairs[i]] = i
    
    data = np.zeros((len(ilist_pairs),len(ilist_workers))) + -1
    print '*********', data.shape
    #data = np.zeros((len(ilist_pairs),len(ilist_tasks))) + -1
    print '#pairs: ', len(ilist_pairs), '#workers: ',len(ilist_workers)
    #print '#pairs: ', len(ilist_pairs), '#tasks: ',len(ilist_tasks)
    for k,v in worker_resp.iteritems():
    #for k,v in task_resp.iteritems():
        for pair_resp in v:
            #print 'check: ', pair_resp[0]
            if pair_resp[0][0] < pair_resp[0][1]:
                data[lookup_tbl[pair_resp[0]],ilist_workers.index(k)] = pair_resp[1]
                #data[lookup_tbl[pair_resp[0]],ilist_tasks.index(k)] = pair_resp[1]
            else:
                data[lookup_tbl[(pair_resp[0][1],pair_resp[0][0])],ilist_workers.index(k)] = pair_resp[1]

    print 'n=',np.sum(np.sum(data != -1,axis=1))
    print 'Ground truth (True Majority Error):', np.sum(np.array(pair_solution.values()) == 1)
    print 'False Majoirty Clean:', np.sum(np.logical_and(np.array(pair_solution.values()) == 1, np.sum(data == 0,axis=1) >= np.sum(data != -1, axis=1)/2))
    print 'False Majority Error:', np.sum(np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2))
    print '+ votes on FME:', np.sum(data[np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2),:] == 1,axis=1)
    print '- votes on FME:', np.sum(data[np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2),:] == 0,axis=1) 
    #print np.sum(data[np.array(pair_solution.values()) == 1,:] == 1, axis=1), np.sum(data[np.array(pair_solution.values()) == 1,:] == 0, axis=1)    
    #print np.array(pair_solution.keys())[np.array(pair_solution.values()) == 1] #rids of ground truth
    print 'TN:', np.sum(np.array(pair_solution.values()) == 0), 'FP:', np.sum(data[np.array(pair_solution.values()) == 0,:] == 1)
    print 'TP:', np.sum(data[np.array(pair_solution.values()) == 0,:] == 1), 'FN:', np.sum(data[np.array(pair_solution.values()) == 1,:] == 0) 
    return data, np.sum(pair_solution.values()), 0., 0.
    

def loadRestaurantExtSample(filename,priotization=True):
    base_table = 'dataset/restaurant.csv'
    hard_pairs_ = pickle.load( open('dataset/hard_pairs.p','rb') ) # list of tuples
    print 'len of hard_pairs_', len(hard_pairs_)
    hard_pairs = {} #heuristic pairq = len(pair_sample) / float(len(d))s, in case of imperfect heuristics, we include random easy pairs during crowdsourcing
    for p in hard_pairs_:
        rid1 = int(p[0][0]) 
        rid2 = int(p[0][1]) 
        if rid1 < rid2:
            hard_pairs[(rid1,rid2)] = float(p[1])
        else:
            hard_pairs[(rid2,rid1)] = float(p[1])
    records = {}
    with open(base_table,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            gid = int(row[1]) # GT label
            rid = int(row[0]) # rid
            name = row[2]
            records[rid] = (gid,name)
         
    pair_solution = {}
    for rid1 in records.keys():
        for rid2 in records.keys():
            if rid1 == rid2: 
                continue
            # priotization == True?
            elif priotization and ((rid1,rid2) not in hard_pairs and (rid2,rid1) not in hard_pairs):
                continue
            if (rid1,rid2) not in pair_solution and (rid2,rid1) not in pair_solution:
                if records[rid1][0] == records[rid2][0]:
                    pair_solution[(rid1,rid2)] = 1
                else:
                    pair_solution[(rid1,rid2)] = 0
    pickle.dump( pair_solution, open('dataset/pair_solution.p','wb') )
    
    #216 (1~217, except 8,21,93) records have duplicates (one each); 106 pairs.
    pair_solution = pickle.load ( open('dataset/pair_solution.p','rb') ) # dictionary
    print 'pair_solution loaded, ground-truth: ', np.sum(pair_solution.values()) 

    #non-heuristic pairs
     
    easy_pair_solution = {}
    for rid1 in records.keys():
        for rid2 in records.keys():
            if rid1 == rid2: 
                continue
            elif ((rid1,rid2) in hard_pairs or (rid2,rid1) in hard_pairs):
                continue
            if (rid1,rid2) not in easy_pair_solution and (rid2,rid1) not in easy_pair_solution:
                if records[rid1][0] == records[rid2][0]:
                    easy_pair_solution[(rid1,rid2)] = 1
                else:
                    easy_pair_solution[(rid1,rid2)] = 0
    pickle.dump( easy_pair_solution, open('dataset/easy_pair_solution.p','wb') )
    
    easy_pair_solution = pickle.load ( open('dataset/easy_pair_solution.p','rb') ) # dictionary

    samples = []
    for sample_table in filename:    
        worker_resp = {}
        with open(sample_table,'rb') as f:
            reader = csv.reader(f)
            for asgn in reader:
                if asgn[0] == 'id':
                    continue
                w = asgn[3] #asgn[3]
            
                answers = asgn[1][1:-2].replace("\"","").replace("Pair","").split(",")
                for ans in answers:
                    rids = ans.split(":")[0].strip().split("-")
                    resp = float(ans.split(":")[1])
                    sim = jaccard(records[int(rids[0])][1],records[int(rids[1])][1])
                    tup = ( (int(rids[0]),int(rids[1])), resp )
                    if tup[0] not in pair_solution:
                        tup = ( (int(rids[1]),int(rids[0])), resp )
                    if priotization and tup[0] not in hard_pairs:
                        # priotization shouldn't matter, because the sample is all heuristic pairs
                        continue
                    if w in worker_resp:
                        if tup in worker_resp[w]:
                            continue
                        else:
                            worker_resp[w].append(tup)
                    else:
                        worker_resp[w] = [tup]
    
        ilist_workers = worker_resp.keys()
        ilist_pairs = pair_solution.keys()
        lookup_tbl = {}
        for i in range(len(ilist_pairs)):
            lookup_tbl[ilist_pairs[i]] = i
    
        data = np.zeros((len(ilist_pairs),len(ilist_workers))) + -1
        print '#pairs: ', len(ilist_pairs), '#workers: ',len(ilist_workers)
        for k,v in worker_resp.iteritems():
            for pair_resp in v:
                #print 'check: ', pair_resp[0]
                if pair_resp[0][0] < pair_resp[0][1]:
                    data[lookup_tbl[pair_resp[0]],ilist_workers.index(k)] = pair_resp[1]
                else:
                    data[lookup_tbl[(pair_resp[0][1],pair_resp[0][0])],ilist_workers.index(k)] = pair_resp[1]

        samples.append(data)

        print '==============For the current sample==============='
        print 'w=',len(worker_resp)
        print 'n=',np.sum(np.sum(data != -1,axis=1))
        print 'False Majoirty Clean:', np.sum(np.logical_and(np.array(pair_solution.values()) == 1, np.sum(data == 0,axis=1) >= np.sum(data != -1, axis=1)/2))
        print 'False Majority Error:', np.sum(np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2))
        print '+ votes on FME:', np.sum(data[np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2),:] == 1,axis=1)
        print '- votes on FME:', np.sum(data[np.logical_and(np.array(pair_solution.values()) == 0, np.sum(data == 1,axis=1) > np.sum(data != -1, axis=1)/2),:] == 0,axis=1) 
        print 'TN:', np.sum(np.array(pair_solution.values()) == 0), 'FP:', np.sum(data[np.array(pair_solution.values()) == 0,:] == 1)
        print 'TP:', np.sum(data[np.array(pair_solution.values()) == 0,:] == 1), 'FN:', np.sum(data[np.array(pair_solution.values()) == 1,:] == 0) 
        print '==================================================='

    return samples

def jaccard(a,b):
    word_set_a = set(a.lower().split())
    word_set_b = set(b.lower().split())
    word_set_c = word_set_a.intersection(word_set_b)
    return float(len(word_set_c)) / (len(word_set_a) + len(word_set_b) - len(word_set_c))

if __name__ == "__main__":
    d,gt,prec,rec = loadAddress()
    print gt
    #d,gt,prec,rec = loadRestaurant2(['dataset/restaurant_new.csv','dataset/restaurant_new2.csv'],priotization=True)
