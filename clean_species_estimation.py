#!/usr/bin/env python

import csv
import random


#expects entities in a CSV file
FILENAME = 'restaurant.csv'
#clean 752, dirty 776, size 858 

#Right now just compares over one attribute
attr_to_compare = 2
ground_truth_attr = 1

#look-up table for edit distance
edit_distance_tbl = {}

"""
Loads the CSV into an array of arrays
"""
def loadFromFile():
    lines = []
    with open(FILENAME, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            lines.append(row)
    return lines

"""
Returns a Sample WITHOUT replacement of Lines
"""
def sample(lines, k): 
    return random.sample(lines,  k)

"""
Returns an integrated sample of size k,
sampled without replacement by w workers.
"""
def sampleByWorker(lines, k, w):
    s = k/w
    remainder = k%w
    sample = []

    fpr_w = 0. #false positive rate
    fnr_w = 0. #false negative rate
    adj_singleton_rate = 0.
    adj_doubleton_rate = 0.
    for i in range(w):
        sample += random.sample(lines, s + remainder)
        n = float(s + remainder)
        remainder = 0
        
        gt_cnt = {}
        obs_cnt = {} #observed unique entities
        for e in sample:
            if e[ground_truth_attr] in gt_cnt:
                gt_cnt[e[ground_truth_attr]] = gt_cnt[e[ground_truth_attr]] + 1
            else:
                gt_cnt[e[ground_truth_attr]] = 1

            if (e[attr_to_compare],e[ground_truth_attr]) in obs_cnt:
                obs_cnt[e[attr_to_compare],e[ground_truth_attr]] = obs_cnt[e[attr_to_compare],e[ground_truth_attr]] + 1
            else:
                obs_cnt[e[attr_to_compare],e[ground_truth_attr]] = 1
            
        neg = len([v for v in obs_cnt.values() if v == 1]) #uniques
        pos = sum(obs_cnt.values()) - neg #duplicates
        
        # compute fpr & fnr
        if i == 0: # for the first sample (worker) only
            fp, fn = 0, 0
            for k, v in obs_cnt.iteritems():
                if v > 1 and gt_cnt[k[1]] == 1: #false duplicates
                    fp += v
                    if v == 2:
                        adj_doubleton_rate -= 1
                    adj_singleton_rate += 1
                if v == 1 and gt_cnt[k[1]] > 1: #false uniques
                    fn += v
                    if gt_cnt[k[1]] == 2:
                        adj_doubleton_rate += 1
                    adj_singleton_rate -= 1
            if pos == 0:
                fpr_w = 0.
            else:        
                fpr_w = fp/pos
            if neg == 0:
                fnr_w = 0.
            else:
                fnr_w = fn/neg

            adj_singleton_rate /= n
            adj_doubleton_rate /= n

            print 'worker_%d:'%i,'fpr_w=',fpr_w,'fnr_w=',fnr_w,'adj_s_r:',adj_singleton_rate,'adj_d_r:',adj_doubleton_rate
    return sample, fpr_w, fnr_w, adj_singleton_rate, adj_doubleton_rate

"""
Returns an entity count dictionary over the projection defined above.
"""
def gtMatchER(lines):
    entity_counts = {}
    for i in lines:
        if i[ground_truth_attr] in entity_counts:
            entity_counts[i[ground_truth_attr]] = entity_counts[i[ground_truth_attr]] + 1
        else:
            entity_counts[i[ground_truth_attr]] = 1
    return entity_counts

"""
Returns an entity count dictionary over the ground truth projection defined above.
"""
def exactMatchER(lines):
    entity_counts = {}
    for i in lines:
        if i[attr_to_compare] in entity_counts:
            entity_counts[i[attr_to_compare]] = entity_counts[i[attr_to_compare]] + 1
        else:
            entity_counts[i[attr_to_compare]] = 1
    return entity_counts

"""
"""
def editMatchER(lines,threshold):
    entity_counts = {}
    for i in lines:
        candidate = i[attr_to_compare]
        if candidate in entity_counts:
            entity_counts[candidate] = entity_counts[candidate] + 1
        elif len(entity_counts) > 0:
            entities = entity_counts.keys()
            sim = [edit_distance(candidate,e) for e in entities] 
        
            max_idx = sim.index(max(sim))
            if sim[max_idx] <= threshold:
                entity_counts[entities[max_idx]] = entity_counts[entities[max_idx]] + 1
            else:
                entity_counts[candidate] = 1
        else:
            entity_counts[candidate] = 1
    return entity_counts
            

"""
Return pairs perfectly identified from the data set.
"""
def identifyPairs(lines):
    pairs = []
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            if lines[i][ground_truth_attr] == lines[j][ground_truth_attr]:
                pairs.append((lines[i][attr_to_compare],lines[j][attr_to_compare]))
    return pairs

"""
Returns the Good-Turing Estimate.
"""
def goodTuring(entity_counts):
    N = len(entity_counts.keys()) #entities
    N1 = len([entity for entity in entity_counts if entity_counts[entity] == 1]) #singletones
    N2 = len([entity for entity in entity_counts if entity_counts[entity] == 2])

    c_hat = 1.0 - (N1+0.)/N
    if c_hat == 0:
        return N
    else:
        return N/c_hat + N1*N1/(2*N2)

"""
Returns the Weighted Good-Turing Estimate.
"""
def weightedGoodTuring(entity_counts, weights):
    N = len(entity_counts.keys()) #entities
    N1 = len([entity for entity in entity_counts if entity_counts[entity] == 1])
    N2 = len([entity for entity in entity_counts if entity_counts[entity] == 2])

    S1 = [entity for entity in entity_counts if entity_counts[entity] == 1] #singletones
    S1P = [entity for entity in entity_counts if entity_counts[entity] != 1] #not singletones

    #print S1, entity_counts

    #calculate weights for singletons
    subtract_from_N1 = 0
    for s in S1:
        subtract_from_N1 = subtract_from_N1 + weights[s]

    #calculate weights for non-singletons
    add_to_N1 = 0
    for s in S1P:
        add_to_N1 = add_to_N1 + (1-weights[s])

    #update N1
    N1 = N1 - subtract_from_N1 + add_to_N1

    c_hat = 1.0 - float(N1)/N 
    if c_hat == 0:
        return N
    else:
        return N/c_hat + N1*N1/(2*N2)

"""
adjusted, based on the TP estimate, Good-Turing estimate.
"""
def aGoodTuring(entity_counts,fpr,fnr):
    N = len(entity_counts.keys()) #entities
    N1 = len([entity for entity in entity_counts if entity_counts[entity] == 1]) #singletones
    N2 = len([entity for entity in entity_counts if entity_counts[entity] == 2]) #doubletons 

    for k,v in entity_counts.iteritems():
        if v == 1 and random.random() < fnr:
            N1 -= 1
            N -= 1
        if v > 1 and random.random() < fpr:
            if v == 2:
                N2 -= 1
            N1 += v
            N += v
    print N
    c_hat = 1. - float(N1)/N
    if c_hat == 0:
        return N
    else:
        return N/c_hat + N1*N1/(2*N2)

"""
Relational Good-Turing Estimate.
"""
def rGoodTuring(entity_counts, pairs):
    c = len(entity_counts.keys())
    f1 = len([entity for entity in entity_counts if entity_counts[entity] == 1])
    f2 = len([entity for entity in entity_counts if entity_counts[entity] == 2])
    S = [p[0] for p in pairs] + [p[1] for p in pairs]

    # correct f1
    for entity, count in entity_counts.iteritems():
        if entity not in S:
            similarities = []
            for p in pairs:
                similarities.append(jaccard(entity,p[0]))
                similarities.append(jaccard(entity,p[1]))
            if max(similarities) > 0.9:
                f1 += 1
    c_hat = max(0.,1.0 - (f1+0.)/c)
    if c_hat == 0:
        return c
    else:
        return c/c_hat + f1*f1/(2*f2)

"""
Shifted Good-Turing Estimate.
"""
def sGoodTuring(entity_counts, shift):
    N = len(entity_counts.keys()) #entities
    N1 = len([entity for entity in entity_counts if entity_counts[entity] == shift]) #singletones
    N2 = len([entity for entity in entity_counts if entity_counts[entity] == shift+1]) 

    c_hat = 1.0 - (N1+0.)/N
    if c_hat == 0:
        return N
    elif N2 == 0:
        return N/c_hat
    else:
        return N/c_hat + N1*N1/(2*N2)

"""
Voting-based Good-Turing Estimate.
"""
def vGoodTuring(entity_counts):
    N = len(entity_counts.keys()) #entities
    f1 = len([entity for entity in entity_counts if entity_counts[entity] == 1]) + 0.
    f2 = len([entity for entity in entity_counts if entity_counts[entity] == 2]) + 0.
    weight = max(entity_counts.values())

    c_hat = 1.0 - (f1/weight)/N 
    if c_hat == 0:
        return N
    else:
        return N/c_hat + f1*f1/(2*f2)/weight


"""
edit distance
"""
def edit_distance(a,b):
    if (a,b) in edit_distance_tbl:
        return edit_distance_tbl[a,b]
    if (b,a) in edit_distance_tbl:
        return edit_distance_tbl[b,a]

    m = len(a)+1
    n = len(b)+1
    
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1,n):
            cost = 0 if a[i-1] == b[j-1] else 1
            tbl[i,j] = min(tbl[i,j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    edit_distance_tbl[a,b] = tbl[i,j]
    return tbl[i,j]

"""
Jaccard Similarity Between Strings
"""
def jaccard(a,b):
    word_set_a = set(a.lower().split())
    word_set_b = set(b.lower().split())
    word_set_c = word_set_a.intersection(word_set_b)
    return float(len(word_set_c)) / (len(word_set_a) + len(word_set_b) - len(word_set_c))

"""
Calculates the weights based on Jaccard Similarity
"""
def calculateJaccardWeights(lines):
    weights = {}
    for j in lines:
        maxW = 0

        #get the max weight over all neighbors
        for k in lines:
            wjk = jaccard(j[attr_to_compare], k[attr_to_compare])
            if wjk > maxW and j != k:
                maxW = wjk
        
        weights[j[attr_to_compare]] = maxW

    #print weights
    return weights

"""
Calculates the weights based on a Thresholded Jaccard Similarity
"""
def calculateTJaccardWeights(lines):
    weights = {}
    for j in lines:
        maxW = 0

        #get the max weight over all neighbors
        for k in lines:
            wjk = jaccard(j[attr_to_compare], k[attr_to_compare])
            if wjk > maxW and j != k:
                maxW = wjk
        
        if maxW <= 0.5:
            maxW = 0

        weights[j[attr_to_compare]] = maxW

    return weights

"""
Experiments Code
"""

"""
Runs the basic good turing estimate with exactMatchER over the set of params
"""
def doEMEstimation(sample_sizes):
    obs = [] # GT
    n = []
    gt = []
    results_em = []
    #results_ed = []
    #results_wem = []
    results_sem = []
    results_gt = []
    results_rem = []
    results_vem = []
    results_aem = []

    lines = loadFromFile()
    pairs = identifyPairs(lines)
    for s in sample_sizes:
        materialized_sample, fpr, fnr, adj_s_r, adj_d_r = sampleByWorker(lines,s,100)
        obs.append(len(exactMatchER(materialized_sample).keys()))
        n.append(len(materialized_sample))
        gt.append(752)
        weights = calculateTJaccardWeights(materialized_sample)
        results_em.append(goodTuring(exactMatchER(materialized_sample)))
        #results_ed.append(goodTuring(editMatchER(materialized_sample,2)))
        results_sem.append(sGoodTuring(exactMatchER(materialized_sample),2))
        #results_wem.append(weightedGoodTuring(exactMatchER(materialized_sample), weights))
        results_gt.append(goodTuring(gtMatchER(materialized_sample)))
        results_rem.append(rGoodTuring(exactMatchER(materialized_sample),pairs))
        results_vem.append(vGoodTuring(exactMatchER(materialized_sample)))
        results_aem.append(aGoodTuring(exactMatchER(materialized_sample),adj_s_r*s,adj_d_r*s))
    #good turing dirty, shifted good turing dirty, good turing clean 
    return (obs, gt, results_em, results_gt, #results_ed, 
            results_rem, results_sem, results_vem, results_aem)

def plotResults(domain, rangeTuple):
    import matplotlib.pyplot as plt
    plt.plot(domain, rangeTuple[0], 'mx-')
    plt.plot(domain, rangeTuple[1], 'gx-')
    plt.plot(domain, rangeTuple[2], 'bs-')
    plt.plot(domain, rangeTuple[3], 'rx-')
    plt.plot(domain, rangeTuple[4], 'kx-')
    plt.plot(domain, rangeTuple[5], 'bD-')
    plt.plot(domain, rangeTuple[6], 'ys-')
    plt.plot(domain, rangeTuple[7], 'rD-')
    #plt.plot(domain, rangeTuple[8], 'bs-')
    plt.xlabel('Sample Size |S|')
    plt.xlim([100,2900])
    plt.ylabel('Number of unique entities') # 'missing probability mass'
    plt.ylim([0,3000])
    plt.legend(['Observed(GT)','Ground Truth','Est. on Dirty',
                'Est. on GT', #'Est (w/edit dist) on Dirty',
                'Relational Est. on Dirty','Shifted Est on Dirty',
                'Voted Est on Dirty','Adjusted Est on Dirty'])
    plt.title('Species est on restaurant dataset (GT: 752), 100 workers.')
    plt.savefig('clean_species_results.png')

def main():
    a = doEMEstimation(range(100,3000,100))
    plotResults(range(100,3000,100),a)


"""
This script runs as the main
"""
if __name__ == "__main__":
    main()
