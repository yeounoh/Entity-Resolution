#!/usr/bin/python
import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson
import random
import copy
import pylab as P
import math
import pickle

def computeGamma(histogram):
    n = np.sum(histogram)
    c = np.sum(histogram > 0)
    f1 = float(np.sum(histogram == 1))
    c_hat = 1. - f1/n
    
    s = 0.
    for i in range(2,len(histogram)):
        s += np.sum(histogram == i) * i * (i-1)
    gamma = s * (c/c_hat) / n / (n-1) - 1.
    
    return max(gamma,0)
    

def sampleCoverage(data):
    hist  = np.sum(data == 1,axis=1)
    n = float(np.sum(hist))
    f1 = float(np.sum(hist == 1))
    if n == 0:
        return 0.
    
    return 1 - f1/n


"""
    at least 1 vote for all items
"""
def minTasks(data):
    for w in range(1,len(data[0])+1):
        if np.sum(np.sum(data[:,0:w]!=-1,axis=1) > 0) == len(data):
            return w
    return -1

def minTasks2(data,q):
    n = float(np.sum(np.sum(data != -1,axis=1)))
    for w in range(1,len(data[0])+1):
        if np.sum(np.sum(data[:,0:w]!=-1,axis=1) > 0) >= len(data)*q:
            return w
    return -1

def minTasks3(data,thresh=0.5):
    for w in range(1,len(data[0])+1):
        if sampleCoverage(data[:,0:w]) >= thresh:
            return w
    return -1

"""
    How many tasks would have been needed, 
    if we have thrown in a fixed number of workers for all?
"""
def minTasksToCleanAll(data,asgnPerTask=10,workers=3):
    n = len(data) # the total number of items
    return int(math.ceil(float(n)*3/asgnPerTask))


def positiveVotes(data):
    return np.sum(np.sum(data == 1))

def chao92(data,corrected=True):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    data_subset[data_subset == -1] = 0
    histogram = np.sum(data_subset,axis=1)
    n = float(np.sum(histogram))
    #n = float(np.sum(n_worker))
    n_bar = float(np.mean([i for i in histogram if i > 0]))
    #n_bar = float(np.mean([i for i in n_worker if i >0]))
    v_bar = float(np.var(histogram[histogram>0]))
    #v_bar = float(np.var([i for i in n_worker if i>0]))
    d = float(np.sum(histogram > 0)) #float(len([i for i in histogram if i > 0]))
    #d = np.sum(histogram > n_worker/2)
    f1 = float(np.sum(histogram == 1)) #float(len([i for i in histogram if i == 1]))
    if n == 0:
        return d
    c_hat = 1 - f1/n
    #gamma = v_bar/n_bar
    gamma = computeGamma(histogram)
    if c_hat == 0.:
        return d
    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat
    return est



def sjackknife(data):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    data_subset[data_subset == -1] = 0
    
    n_bar = float(np.mean([i for i in n_worker if i > 0]))
    v_bar = float(np.var(n_worker))
    d = np.sum(np.sum(data == 1,axis=1) > n_worker/2)

    f = []
    n = 0
    for w in range(len(data[0])):
        diff = np.sum(data == 1,axis=1) - np.sum(data == 0,axis=1)
        fx = np.sum(diff == w+1)
        fx += np.sum(diff == -(w+1))
        n += fx * (w+1)
        f.append(fx)
    f1 = f[0]

    histogram = np.sum(data_subset,axis=1)
    N = float(len(data))
    D_ = (d - (f1/n)) * (1 - (N-n+1)*f1/n/N)**-1
    N_ = float(N)/D_
    h_N_ = math.gamma(N-N_+1)*math.gamma(N-n+1)
    h_N_ = h_N_/math.gamma(N-n-N_+1)/math.gamma(N+1)
    
    g_n_1_N_ = np.sum((N-np.array(range(n-1))+1-n-N_)**-1)
    gamma_D_ = 0.
    for i in range(n):
        gamma_D_ += (i+1)*i*np.sum(histogram == i+1)
    gamma_D_ = gamma_D_ * (N-1)*D/N/n/(n-1) + D/N - 1
    D = (1 - (N-N_-n+1)*f1/n/N)**-1 * (d+N*g_n_1_N_*gamma_D_)
    return D



def unseen(data):
    grid_factor = 1.05 #x_i (the grid of prob) will be geometric with this ratio
    alpha = .5 #avoid overfitting, smaller value increase the risk
    max_itr = 1000

    #data pre-processing
    data_subset = copy.deepcopy(data)
    pos_idx = np.sum(data == 1, axis=1) > np.sum(data != -1, axis=1)/2
    
    #discard opposing votes
    for i in range(len(data_subset)):
        if pos_idx[i]:
            data_subset[i,data_subset[i,:] == 0] = -1    
        else:
            data_subset[i,data_subset[i,:] == 1] = -1

    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    
    #clean and no-ops are ignored; errorneous pairs are of different classes
    f = []
    n = 0
    for w in range(len(data_subset[0])):
        fx = np.sum(np.sum(data_subset == 1,axis=1) == w+1)
        f.append( fx )
        n += fx * (w+1)
    f = np.array(f)
    f1 = f[0] #zero-indexed

    #minimum allowable probability
    xLP_min = 1./(n*max(10,n))
    i_min = np.argmax(f > 0)
    if i_min > 0:
        xLP_min = (i_min+1.)/n

    #split the f-statistics into the dense portion and the sparse portion
    x= [0.]
    histx = [0.];
    fLP = np.zeros(len(f))
    for i in range(len(f)):
        if f[i] > 0:
            i_lower = max(0,i-math.ceil(math.sqrt(i)))
            i_upper = min(len(f)-1, i+math.ceil(math.sqrt(i))) 
            if np.sum(f[i_lower:i_upper+1]) < math.sqrt(i):
                # sparse region used the empirical histogram
                x.append((i+1)/n)
                histx.append(f[i])
                fLP[i] = 0
            else:
                # will use LP for dense region
                fLP[i] = f[i]
    x = np.array(x)
    histx = np.array(histx)

    # no LP portion
    if np.sum(fLP > 0) == 0:
        x = x[1:]
        histx = histx[1:]
        return np.sum(histx)

    # first LP problem
    LP_mass = 1 - np.sum(x*histx)
    f_max = len(f) - np.argmax(fLP[::-1] > 0) - 1
    fLP = np.append(fLP[0:f_max+1],np.zeros(math.ceil(math.sqrt(f_max))))
    szLPf = len(fLP)

    xLP_max = (f_max+1)/float(n)
    xLP = xLP_min*grid_factor**np.array( range( int(math.ceil(math.log(xLP_max/xLP_min)/math.log(grid_factor)))+1 ) )
    szLPx = len(xLP)

    objf = np.zeros(szLPx+2*szLPf)
    objf[szLPx::2] = 1./np.vectorize(math.sqrt)(fLP+1)
    objf[szLPx+1::2] = 1./np.vectorize(math.sqrt)(fLP+1)

    A = np.zeros((2*szLPf,szLPx+2*szLPf))
    b = np.zeros((2*szLPf,1))
    for i in range(szLPf):
        A[2*i,0:szLPx] = np.vectorize(lambda x:poisson.pmf(i+1,x))(n*xLP)
        A[2*i+1,0:szLPx] = -1 * A[2*i,0:szLPx]
        A[2*i,szLPx+2*i] = -1
        A[2*i+1,szLPx+2*i+1] = -1
        b[2*i] = fLP[i]
        b[2*i+1] = fLP[i]

    Aeq = np.zeros(szLPx + 2*szLPf)
    Aeq[0:szLPx] = xLP
    beq = LP_mass

    for i in range(szLPx):
        A[:,i] = A[:,i]/xLP[i]
        Aeq[i] = Aeq[i]/xLP[i]
    #result consists of x, slack, success, status, nit, message
    result = linprog(objf,A_ub=A,b_ub=b,A_eq=Aeq.reshape((1,len(Aeq))),b_eq=beq,options={'maxiter':max_itr}) 
    sol = result.x
    val = result.fun #objf_val = objf * sol
    #print 'first optimization result:',result.success,result.status,result.message

    # second LP problem
    objf2 = 0 * objf
    objf2[0:szLPx] = 1
    A2 = np.append(A,objf.reshape((1,len(objf))),axis=0)
    b2 = np.append(b, np.array(val)+alpha)
    for i in range(szLPx):
        objf2[i] = objf2[i]/xLP[i]
    result2 = linprog(objf2,A_ub=A2,b_ub=b2,A_eq=Aeq.reshape((1,len(Aeq))),b_eq=beq,options={'maxiter':max_itr})
    sol2 = result2.x
    #print 'second optimization result:',result2.success,result2.status,result2.message

    # combine the dense and sparse region solutions
    sol2[0:szLPx] = sol2[0:szLPx]/xLP
    x = np.append(x,xLP)
    histx = np.append(histx,sol2)
    idx = [i[0] for i in sorted(enumerate(x), key=lambda x:x[1])]
    x = x[idx]
    histx = histx[idx]
    x = x[histx > 0]
    histx = histx[histx > 0]
    
    return np.sum(histx)



def sChao92(data,corrected=True,shift=0):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    data_subset[data_subset == -1] = 0
    histogram = np.sum(data_subset==1,axis=1)
    n = float(np.sum(histogram))
    #n = float(np.sum(n_worker))
    n_bar = float(np.mean([i for i in histogram if i > 0]))
    #n_bar = float(np.mean([i for i in n_worker if i >0]))
    v_bar = float(np.var(histogram[histogram>0]))
    #v_bar = float(np.var([i for i in n_worker if i >0]))
    d = np.sum(histogram >(n_worker/2))
    
    f1 = float(np.sum(histogram == 1+shift)) #float(len([i for i in histogram if i == 1]))
    if n == 0:
        return d
    c_hat = 1 - f1/n
    #gamma = v_bar/n_bar
    gamma = computeGamma(histogram)
    if c_hat == 0.:
        return d
    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat

    return est


"""
    extrapolate from the data (worker-responses or tasks); 
    or from a fraction (0.0<= q <=1.0) of data.
"""
def extrapolateFromSample(data,labels,q=1.0,golden=True): 
    n = q * len(data) # we sample from the entire records, not those reviewed by workers.
    sampleIdx = np.random.choice(range(len(data)),int(n),replace=False)
    sample = data[sampleIdx,:]
    est = vNominal(sample)/q
    if golden:
        est = np.sum(np.array(labels)[sampleIdx])/q
    return est


"""
    extrapolation from a random n/len(data) sample - taken on the fly from the data
"""
def extrapolation(data,pair_solution,n):
    ilist = pair_solution.keys()
    ext_array = []
    pair_sample = np.random.choice(range(len(ilist)),n,replace=False)
    ext = np.sum([pair_solution[ilist[s]] for s in pair_sample])
    q = len(pair_sample) / float(len(data))
    ext_array.append(ext/q)
    ext = np.mean(ext_array) 
    return ext

"""
    extrapolation based on a golden sample data, for total error estimates.
"""
def extrapolation2(data,d,samples):
    ext_array = []
    for s in samples:
        ext = vNominal(s[:,0:len(data[0])])
        q = np.sum(np.sum(s == -1,axis=1) != -len(s[0])) / float(len(data))
        ext_array.append(ext/q)
    ext = np.mean(ext_array)
    return ext

"""
    extrapolation based on a golden sample data, for switch estimates.
"""
def extrapolation3(data,d,samples):
    ext_array = []
    for s in samples:
        ext = sNominal(s[:,0:len(data[0])])
        q = np.sum(np.sum(s == -1,axis=1) != -1) / float(np.sum(np.sum(d != -1)))
        ext_array.append(ext/q)
    ext = np.mean(ext_array)
    return ext



"""
    observed unique data items. if this converges to GT, then there exists no FP.
"""
def nominal(data):
    return np.sum(np.sum(data == 1,axis=1) > 0)

def nominalCov(data):
    return np.sum(np.sum(data != -1,axis=1) >0)

def nominalF1(data):
    return np.sum(np.sum(data==1,axis=1)==1)

def nominalF2(data):
    return np.sum(np.sum(data==1,axis=1)==2)

"""
    voting based nominal estimation.
"""
def vNominal(data):
    return np.sum(np.sum(data == 1,axis=1) > np.sum(data != -1,axis=1)/2)

def majority_fp(data, slist):
    return np.sum(np.logical_and(np.sum(data == 1,axis=1) > np.sum(data != -1,axis=1)/2,slist == 0))

def majority_fn(data, slist):
    print 'false negative:', np.sum(np.logical_and(np.sum(data == 0,axis=1) >= np.sum(data != -1,axis=1)/2,slist == 1))
    
    return np.sum(np.logical_and(np.sum(data == 0,axis=1) >= np.sum(data != -1,axis=1)/2,slist == 1))



def sNominal(data,pos_switch=True,neg_switch=True):
    data_subset = data # no copying
    majority = np.zeros((len(data_subset),len(data_subset[0])))
    switches = np.zeros((len(data_subset),len(data_subset[0])))
    for i in range(len(data_subset)):
        prev = 0
        for w in range(0,len(data_subset[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)

            maj = 0
            if n_pos == n_neg and n_pos != 0:
                # tie results in switch
                maj = (prev + 1)%2
            elif n_pos > n_w/2:
                maj = 1
            if prev != maj:
                if (maj == 1 and pos_switch) or (maj == 0 and neg_switch):
                    switches[i][w] = 1
            prev = maj
            majority[i][w] = maj
    
    return np.sum(np.logical_and(np.sum(switches,axis=1), np.sum(data,axis=1) != -1*len(data[0])))   



# true remaining errors; false positives 
def gt_remaining(data,slist):
    data_subset = copy.deepcopy(data)
    if data.size < 1:
        return np.sum(np.array(slist)==1)
    
    votes = []
    for i in range(len(data_subset)):
        n_worker = len([v for v in data_subset[i] if v != -1])
        maj = 0
        if np.sum(data_subset[i] == 1) > n_worker/2:
            maj = 1
        votes.append(maj)
    return np.sum(np.logical_and(np.array(slist) == 1, np.array(votes) != 1))



def gt_marginal(data,slist):
    data_subset = copy.deepcopy(data)
    if len(data[0]) < 2:
        return np.sum(np.array(slist)==1)

    votes_prev = []
    for i in range(len(data_subset)):
        n_w = np.sum(data_subset[i,:-2] != -1)
        maj = 0
        if np.sum(data_subset[i,:-2] == 1) > n_w/2:
            maj = 1
        votes_prev.append(maj)
    votes_prev = np.array(votes_prev)
    votes_cur = []
    for i in range(len(data_subset)):
        n_w = np.sum(data_subset[i,:-1] != -1)
        maj = 0
        if np.sum(data_subset[i,:-1] == 1) > n_w/2:
            maj = 1
        votes_cur.append(maj)
    votes_cur = np.array(votes_cur)

    margin = np.sum(np.logical_and(votes_cur == 1, - votes_prev == 0))
    return margin
     


def gt_switch(data,slist,pos_switch=True,neg_switch=True):
    if len(data[0]) < 1:
        # assuming all clean vector
        return np.sum(slist==1)
    '''
    votes = []
    for i in range(len(data)):
        n_w = np.sum(data[i] != -1)
        n_pos = np.sum(data[i] == 1)
        n_neg = np.sum(data[i] == 0)
        maj = 0#-1
        if n_pos == n_neg:
            prev = np.sum(data[i][0:-1] == 1) > np.sum(data[i][0:-1] != -1)/2
            maj = (prev+1)%2
        elif n_pos > n_w/2:
            maj = 1
        votes.append(maj)
    votes = np.array(votes)
    '''
    votes = np.zeros((len(data)))
    for i in range(len(data)):
        prev = 0
        for w in range(0,len(data[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)
            if True:
                maj = 0#-1
                if np.sum(data[i][0:w+1] == 1) > n_w/2:
                    maj = 1
                if n_pos == n_neg and n_pos != 0:
                    # tie results in switch
                    maj = (prev + 1)%2
                prev = maj
                votes[i] = maj 
    
    if pos_switch and not neg_switch:
        return np.sum(np.logical_and(np.logical_xor(votes,np.array(slist)), np.array(slist) == 1))
    elif not pos_switch and neg_switch:
        return np.sum(np.logical_and(np.logical_xor(votes,np.array(slist)), np.array(slist) == 0))
    else:
        # assuming all clean pairs initially for non-reviewed pairs
        return np.sum(np.logical_xor(votes,np.array(slist)))



"""
    weighted f1 score; follows nominal, but avoid overestimation.
"""
def wChao92(data,corrected=True):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    data_subset[data_subset == -1] = 0
    n_worker = np.array(n_worker)
    histogram = np.sum(data_subset,axis=1)
    n = float(np.sum(n_worker))
    n_bar = float(np.mean([i for i in n_worker if i >0]))
    v_bar = float(np.var(n_worker))

    d = np.sum(histogram > n_worker/2)
    f1 = np.sum(np.ones(len(histogram))[histogram == 1]/n_worker[histogram == 1])

    if n == 0:
        return d
    c_hat = 1 - f1/n
    gamma = v_bar/n_bar

    if c_hat == 0.:
        return d
    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat

    return est




"""
    remove FP more aggresively using majority voting.
"""
def fChao92(data,corrected=True):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    data_subset[data_subset == -1] = 0
    n_worker = np.array(n_worker)
    histogram = np.sum(data_subset,axis=1)
    n = float(np.sum(n_worker))
    n_bar = float(np.mean([i for i in n_worker if i >0]))
    v_bar = float(np.var(n_worker))

    d = np.sum(histogram > n_worker/2)
    f1 = np.sum(np.logical_and(np.sum(data == 0,axis=1) < 3, histogram == 1))

    if n == 0:
        return d
    c_hat = 1 - f1/n
    gamma = v_bar/n_bar

    if c_hat == 0.:
        return d
    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat

    return est



def qaChao92(data,corrected=True):
    # how many workers have voted for each record?
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    data_subset[data_subset == -1] = 0
    
    hist_ = np.sum(data_subset,axis=1)
    majority = np.zeros(len(data))
    for i in range(len(data_subset)):
        if hist_[i] > n_worker[i]/2:
            majority[i] = 1
        else:
            majority[i] = 0

    quality = np.zeros(len(data[0]))
    for w in range(len(data[0])):
        for i in range(len(data)):
            if majority[i] == data_subset[i][w]:
                quality[w] += 1
    quality /= len(data)

    hist = np.sum(data_subset[:,quality >= 0.5],axis=1)
    n = float(np.sum(n_worker))
    n_bar = float(np.mean([i for i in n_worker if i > 0]))
    v_bar = float(np.var(n_worker))

    d = np.sum(hist > n_worker/2)
    f1 = float(np.sum(hist == 1)) #np.sum(np.ones(len(hist))[hist == 1]/n_worker[hist == 1])

    if n == 0:
        return d
    
    c_hat = 1 - f1/n
    gamma = v_bar/n_bar

    if c_hat == 0.:
        return d

        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat

    return est


def dChao92(data):
    return 0.

def bChao92(data):
    return 0.

def vGoodToulmin(data,c=2):
    return vNominal(data) + goodToulmin(data,c)

def vRemainSwitch(data,pos=True,neg=True):
    pos_adj = max(0,remain_switch(data,pos_switch=True,neg_switch=False) - sNominal(data,pos_switch=True,neg_switch=False))
    neg_adj = max(0,remain_switch(data,pos_switch=False,neg_switch=True) - sNominal(data,pos_switch=False,neg_switch=True))

    if not pos:
        pos_adj = 0
    if not neg:
        neg_adj = 0

    return max(0,vNominal(data) + pos_adj - neg_adj)


"""
    If majority decreases, then use neg_adj for FPs; otherwise, use pos_adj for FNs.
"""
def vRemainSwitch2(data):
    n_worker = len(data[0])
    est = vNominal(data)
    thresh = np.max([vNominal(data[:,:n_worker/2]), vNominal(data[:,:n_worker/4]), vNominal(data[:,:n_worker/4*3]) ])
    pos_adj = 0
    neg_adj = 0
    if est - thresh < 0:
        neg_adj = max(0,remain_switch(data,pos_switch=False,neg_switch=True) - sNominal(data,pos_switch=False,neg_switch=True))
    else:
        pos_adj = max(0,remain_switch(data,pos_switch=True,neg_switch=False) - sNominal(data,pos_switch=True,neg_switch=False))
    return max(0,est + pos_adj - neg_adj)


"""
    First, estimate positive switches; second, estimate how many of those switches can flip back (negative switch).
"""
def twoPhase(data):
    pos_adj = max(0,remain_switch(data,pos_switch=True,neg_switch=False) - sNominal(data,pos_switch=True,neg_switch=False))
    pos_adj = min(len(data), pos_adj)
    neg_adj = max(0,remain_switch(data,pos_switch=False,neg_switch=True) - sNominal(data,pos_switch=False,neg_switch=True))
    neg_adj = min(pos_adj + np.sum(np.sum(data == 1,axis=1) > np.sum(data != -1,axis=1)/2), neg_adj)

    return vNominal(data) + pos_adj - neg_adj
    


"""
    good-toulmin estimator: how many more duplicates (errors) would we get for an additional worker?
"""
def goodToulmin(data,c=2):
    data_subset = copy.deepcopy(data)
    if data.size < 1:
        return len(data)
    #m_worker = []
    #for j in range(len(data[0])):
    #    m_worker.append(np.sum(data[:,j] == 1))
    #m_worker = np.array(m_worker)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)
    data_subset[data_subset == -1] = 0
    hist_records = np.sum(data_subset,axis=1)
    n = np.sum(n_worker) #np.sum(m_worker)
    m = min(c*math.sqrt(n),n)
    if n == 0:
        return len(hist_records)
    est = 0.
    for i in range(1,int(np.amax(hist_records))+1):
        est += ((-m/n)**i) * np.sum(hist_records == i)
    est *= -1

    return est



def remain_switch(data,corrected=False,pos_switch=True,neg_switch=True):
    data_subset = copy.deepcopy(data)
    #bug!!!
    #if len(data_subset[0]) < 2:
        #return 0
    majority = np.zeros((len(data_subset),len(data_subset[0])))
    switches = np.zeros((len(data_subset),len(data_subset[0])))
    for i in range(len(data_subset)):
        prev = 0
        for w in range(0,len(data_subset[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)

            maj = 0
            if n_pos == n_neg and n_pos != 0:
                # tie results in switch
                maj = (prev + 1)%2
            elif n_pos > n_w/2:
                maj = 1
            if prev != maj:
                if (maj == 1 and pos_switch) or (maj == 0 and neg_switch):
                    switches[i][w] = 1
            prev = maj
            majority[i][w] = maj

    n_worker = np.sum(data_subset != -1, axis=1)
    n_all = n_worker
    '''
    if pos_switch and not neg_switch:
        n_worker = np.sum(data_subset == 1, axis=1)
    elif not pos_switch and neg_switch:
        n_worker = np.sum(data_subset == 0, axis=1)
    '''
    data_subset[data_subset == -1] = 0
    
    histogram = n_worker 
    n = float(np.sum(n_worker))
    n_bar = float(np.mean(n_worker))
    v_bar = float(np.var(n_worker))
    #histogram = np.sum(switches == 1,axis=1)
    #n = float(np.sum(histogram))
    #n_bar = float(np.mean(histogram[histogram>0]))
    #v_bar = float(np.var(histogram[histogram>0]))
    d = np.sum(np.logical_and(np.sum(switches,axis=1), n_all != 0))#np.sum(data,axis=1) != -1*len(data[0])))   
    if n == 0:
        return d
    
    f1 = 0.
    for i in range(len(switches)):
        if n_worker[i] == 0:
            continue
        for k in range(len(switches[0])):
            j = len(switches[0]) -1 - k
            #n_w = np.sum(data[i][0:j+1] != -1)
            if data[i][j] == -1:
                continue
            # we don't count the confirmation switches
            #elif n_w == 1:
                #break
            elif switches[i][j] == 1:
                f1 += 1
                break
            else:
                break
    # remove no-ops
    for i in range(len(switches)):
        switch_idx= np.where(switches[i,:]==1)[0]
        if len(switch_idx) > 0:
            n -= np.sum(data[i,:np.amin(switch_idx)] != -1)
        elif len(switch_idx) == 0:
            n -= np.sum(data[i,:] != -1)
    if n == 0:
        return d
    c_hat = 1 - f1/n
    c_hat = max(0,c_hat)
    gamma = v_bar/n_bar
    #gamma = computeGamma(histogram)
    if c_hat == 0.:
        return d

    if corrected:
        est = d/c_hat + n*(1-c_hat)/c_hat*gamma
    else:
        est = d/c_hat

    return est
