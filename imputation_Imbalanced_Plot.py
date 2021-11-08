import pandas as pd 
import numpy as np 
import seaborn as sn 
from seaborn import distplot
from seaborn import heatmap
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import math
import time
import numpy as np
from tqdm import tqdm
import multiprocessing
import pickle
from math import isinf 
from scipy.interpolate import interp1d
from numpy import array, zeros, full, argmin, inf, ndim
from sklearn import preprocessing

import statistics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RB
#from sklearn.decomposition import KernelPCA
#from imblearn.over_sampling import SMOTE as SM
from sklearn.ensemble import RandomForestClassifier as RF,GradientBoostingRegressor as GB,ExtraTreesClassifier as ET
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neural_network import MLPClassifier as MP
from bisect import bisect_left

import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
 
# Scikit-Learn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_score, recall_score


df = pd.read_csv(r'EnergyConsumption_normalized.csv')

num_zeros = []
for i in tqdm(range(len(df))):
    num_zeros.append(df.iloc[i][1:].to_list().count(0))
    
df['num_zeros'] = num_zeros

def find_gaps(x):
    seq = False
    missing_seqs = []
    seq_start_idx = -1
    for i in range(len(x)):
        if seq == False and x[i] == 0:
            seq = True
            seq_start_idx = i
        
        elif seq == True and x[i] != 0:
            seq = False
            if seq_start_idx == -1:
                raise
            missing_seqs.append((seq_start_idx, i))
            seq_start_idx = -1
    
    return missing_seqs

def remove_micro_gaps(x, maxlen=5):
    x_filled = x.copy()
    missing_seqs = find_gaps(x)
    
    for seq in missing_seqs:
        if (seq[1] - seq[0]) <= maxlen:
            len_gap = seq[1] - seq[0]
            if seq[0]-len_gap < 0 or seq[1]+len_gap>len(x):
                continue
            interpfunk = interp1d(list(range(seq[0]-len_gap, seq[0])) + list(range(seq[1], seq[1]+len_gap)), x[seq[0]-len_gap:seq[0]] + x[seq[1]:seq[1]+len_gap]) 
            
            for i in range(len_gap):
                x_filled[seq[0]+i] = interpfunk(seq[0]+i).item()
            
    return x_filled

def tiny_impute(i):
    x = df.iloc[i][:-2].to_list()
    return remove_micro_gaps(x)

p = multiprocessing.Pool() 
new_rows = list(tqdm(p.imap(tiny_impute, list(range(len(df)))), total=len(df)))

df_interp = pd.DataFrame(new_rows)

df_interp.columns = df.columns[:-2]

df_interp['FLAG'] = df['FLAG']


df_0 = df_interp[df_interp['FLAG'] == 0].reset_index(drop=True)
df_1 = df_interp[df_interp['FLAG'] == 1].reset_index(drop=True)


df_imp = po.concat([df_0, df_1], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)

df_imp.to_csv(r'imp_raw.csv', index=False)

df_imp = pd.read_csv(r'imp_raw.csv')


spring_0 = df_imp.loc[:, '2015/04/01':'2015/06/01'] + df_imp.loc[:, '2015/04/01':'2015/06/01'] + df_imp.loc[:, '2016/04/01':'2016/06/01']
summer_0 = df_imp.loc[:, '2015/06/01':'2015/09/01'] + df_imp.loc[:, '2015/06/01':'2015/09/01'] + df_imp.loc[:, '2016/06/01':'2016/09/01']
autumn_0 = df_imp.loc[:, '2015/09/01':'2015/11/01'] + df_imp.loc[:, '2015/09/01':'2015/11/01'] + df_imp.loc[:, '2016/09/01':'2016/10/01']
winter_0 = df_imp.loc[:, '2014/11/01':'2015/04/01'] + df_imp.loc[:, '2015/11/01':'2016/04/01']



def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(i, j, x, y)
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def derivative_dtw_distance(i, j, x, y):
    if i+1 == len(x) or j+1 == len(y):
        dist = (x[i] - y[j])**2
    
    else:
        d_x_i = ((x[i] - x[i-1]) + (x[i+1] - x[i-1])/2)/2
        d_y_j = ((y[j] - y[j-1]) + (y[j+1] - y[j-1])/2)/2    

        dist = (d_x_i - d_y_j)**2

    return dist

def find_gaps(x):
    seq = False
    missing_seqs = []
    seq_start_idx = -1
    for i in range(len(x)):
        if seq == False and x[i] == 0:
            seq = True
            seq_start_idx = i
        
        elif seq == True and x[i] != 0:
            seq = False
            if seq_start_idx == -1:
                raise
            missing_seqs.append((seq_start_idx, i))
            seq_start_idx = -1
    
    return missing_seqs

def dtwbi(D, Q, len_gap, stride=20):
	
	min_dtw_cost = inf
	start_index = 0
	
	for i in range(0, len(D)-len_gap, stride):
		#print(i, i+len_gap)
		try:
			cost, cost_matrix, acc_cost_matrix, path = dtw(D[i:i+len_gap], Q, dist=derivative_dtw_distance)
		except:
			print(D[i*len_gap:(i+1)*len_gap])
			print('Q', Q)
			print('len_gap', len_gap)
			print('len(D)', len(D))
			print(i*len_gap, (i+1)*len_gap)
			raise
		if cost < min_dtw_cost and cost > dist:
			min_dtw_cost = cost
			start_index=i

	return start_index

def apply_dtwbi_after(x, start_index, end_index, size):
        if (size == 1):
            x = spring_0
        elif (size == 2):
            x = summer_0
        elif (size == 3):
            x = autumn_0
        else:
            x = winter_0
	len_gap = end_index - start_index
	
	Qa = x[end_index:end_index+len_gap]
	Da = x[end_index+len_gap:]
	
	Qas_start = dtwbi(Da, Qa, len_gap)
	#Qas = x[Qas_start:Qas_start+len_gap]
	
	if Qas_start-len_gap < 0:
		refA = x[Qas_start:Qas_start+len_gap] # = Qa
	else:
		refA = x[Qas_start-len_gap:Qas_start] # Previous window of Qas

	return refA

def apply_dtwbi_before(x, start_index, end_index, size):
        if (size == 1):
            x = spring_0
        elif (size == 2):
            x = summer_0
        elif (size == 3):
            x = autumn_0
        else:
            x = winter_0
	len_gap = end_index - start_index
	
	Qb = x[start_index-len_gap:start_index]
	Db = x[:start_index-len_gap]
	
	if len(Qb) == 0:
		print('start_index', start_index)
		print('len_gap', len_gap)
		print('end_index', end_index)
		raise

	Qbs_start = dtwbi(Db, Qb, len_gap)
	#Qbs = x[Qbs_start:Qbs_start+len_gap]
	
	if Qbs_start+2*len_gap > len(x):
		refB = x[Qbs_start:Qbs_start+len_gap] # = Qb
	
	else:
		refB = x[Qbs_start+len_gap:Qbs_start+2*len_gap] # Next window of Qbs

	return refB

def edtwbi(x, start_index, end_index, size):
	len_gap = end_index - start_index
	
	if end_index + len_gap >= len(x):
		refB = apply_dtwbi_before(x, start_index, end_index, size) # only dtwbi in other direction
		
		return refB
		
	elif start_index-len_gap <= 0:
		refA = apply_dtwbi_after(x, start_index, end_index, size) # only dtwbi in other direction
		
		return refA
	
	else: # both cannot simultaneously happen, so not keeping a case for that
		refA = apply_dtwbi_after(x, start_index, end_index, size)
		refB = apply_dtwbi_before(x, start_index, end_index, size)
		
		return np.mean([np.array(refA), np.array(refB)], axis = 0)

def impute_row(i):
	row = df_imp.iloc[i][:-2].to_list()
	if (row in spring_0):
            size = 1
        elif (row in summer_0):
            size = 2
        elif (row in autumn_0):
            size = 3
        else:
            size = 4
	row_filled = row.copy()
	
	gaps = find_gaps(row)
	for gap in gaps:
		if gap[1] - gap[0] < 0.3*len(row):
			row_filled[gap[0]:gap[1]] = edtwbi(row, gap[0], gap[1], size)

	return row_filled

p = multiprocessing.Pool() 
new_rows = list(tqdm(p.imap(impute_row, list(range(len(df_imp)))), total=len(df_imp)))
df_edtwbi_impute = pd.DataFrame(new_rows)
df_edtwbi_impute.columns = df1.columns
df_edtwbi_impute['FLAG'] = y
df_edtwbi_impute.to_csv(r'edtwbi_full.csv', index=False)

imputed = pd.read_csv(r'edtwbi_full.csv')
plt.figure(figsize=(20, 10))
#df = df.drop(['FLAG', 'CONS_NO'], axis = 1)
plt.plot(df.iloc[1500], color='Blue')
plt.xlabel('Days', fontsize=20)
plt.ylabel('Energy Consumption (kWh)', fontsize=20)
plt.legend(['Original Value'])
plt.savefig('missing.png')

plt.figure(figsize=(20, 10))
plt.plot(imputed.iloc[1500][:-1], color='red')
plt.plot(df.iloc[1500], color='Blue')
plt.xticks(range(0, len(df.iloc[28]), 100), range(0, len(df.iloc[1500]), 100)) 
plt.xlabel('Days', fontsize=20)
plt.ylabel('Energy Consumption (kWh)', fontsize=20)
plt.legend(['Imputed Value', 'Original Value'])
plt.savefig('imputed.png')

