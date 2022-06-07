#!/usr/bin/env python
# coding: utf-8

# # **Combine results**

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[2]:


from more_itertools import powerset

candidates = ['ANOVA', 'PCA_PC1', 'PCA_mixed', 'LDA', 'Tree', 'Forest', 'XGB']
combinations = list(powerset(candidates))
combinations.pop(0)
print(len(combinations))


# In[3]:


import math

t200 = dict()
for cand in candidates:
    fname = cand + '_31.csv'
    t200[cand] = pd.read_csv(f'scores/{fname}')    # Read CSV files
    
    # for sum of rankings
    n_entries = len(t200[cand].index)
    # scales = [200*(n_entries-i)/n_entries for i in range(n_entries)]    # linear scales
    scales = [200*math.exp(-10*i/n_entries) for i in range(n_entries)]  # exponential scales
    t200[cand]['score'] = scales


# ## Intersections

# In[4]:


intersections = dict()
for comb in combinations:
    key = '&'.join(comb)
    lists = []
    for model in comb:
        lists.append(t200[model]['Gene'].tolist())
    result = list(set.intersection(*map(set, lists)))
    if (result):
        intersections[key] = result


# ## Sum of Rankings

# In[5]:


df = pd.read_csv('simplify_combined_dataset.csv')
genes = df.columns.tolist()
genes.remove('ScanREF')
genes.remove('DataSet')
genes.remove('state')
genes.remove('train/test')


# In[6]:


# testing code for database query
series_add = t200['ANOVA'].loc[t200['ANOVA']['Gene']=='DHX30']['score']
if not series_add.empty:
    print(series_add.item())


# In[7]:


from operator import itemgetter

T20 = dict()
for comb in combinations:
    key = '&'.join(comb)
    scores = dict.fromkeys(genes, 0)
    for model in comb:
        for gene in genes:
            series_add = t200[model].loc[t200[model]['Gene']==gene]['score']
            if not series_add.empty:
                scores.update({gene: scores[gene] + series_add.item()})
    T20[key] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]


# ### Display and Export

# In[8]:


def display_dict(dictionary):
    """Displays dictionaries neatly."""
    for key, value in dictionary.items():
        print(key)
        print(value)
        print('\n')

def export_dict(dictionary, fname):
    with open(f'{fname}.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Succesfully exported {fname}.pickle.')


# In[9]:


# display_dict(intersections)
export_dict(T20, 'scales_EXP_31')


# In[ ]:




