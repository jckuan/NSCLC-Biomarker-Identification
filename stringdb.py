#!/usr/bin/env python
# coding: utf-8

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


with open('scales_LIN_31.pickle', 'rb') as handle:
    pkl = pickle.load(handle)

top_genes = pd.DataFrame(columns=['Model', 'Genes', 'average_node_degree', 'p_value', 'score'])
top_genes = top_genes.astype({'average_node_degree':'float', 'p_value':'float', 'score':'float'})
for k, v in pkl.items():
    new_entry = {'Model':k, 'Genes':v}
    top_genes = top_genes.append(new_entry, ignore_index=True)


# # **String API**

# In[4]:


import stringdb
import requests


# ## get_ppi_enrichment()
# 
# ## get_network()
# 
# - Retrieves the score of the edges in the network.
# - Check [this](http://version10.string-db.org/help/faq/#the-protein-interactions-from-the-string-website-via-web-api-calls-what-do-the-score-columns-mean-for-example-nscore-fscore-tscore-etc) for the meaning of the different scores.

# In[5]:


import time

n_entries = top_genes.shape[0]
print(f'Entries: {n_entries}')

for i in range(n_entries):
    genes = top_genes['Genes'].iloc[i]
    try:
        string_ids = stringdb.get_string_ids(genes)
    except:
        continue
    
    if (string_ids.empty):
        print(f'!! Entry {str(i).ljust(3)} EMPTY. !!')
    else:
        ppi = stringdb.get_ppi_enrichment(string_ids.queryItem)
        top_genes.loc[i, 'average_node_degree'] = ppi['average_node_degree'][0]
        top_genes.loc[i, 'p_value'] = ppi['p_value'][0]

        edges = stringdb.get_network(string_ids.queryItem)
        edges = edges.drop_duplicates() 
        top_genes.loc[i, 'score'] = edges['score'].sum()

        print(f'Entry {str(i).ljust(3)} completed.')
    
    if (i%10 == 0):
        time.sleep(1)    # Delay for 1 sec


# In[10]:


# print(top_genes.nsmallest(20, ['p_value'])[['Model', 'p_value', 'average_node_degree']])
print(top_genes.nlargest(20, ['score']))


# # **Experiment Results**

# In[8]:


base_url = "https://string-db.org/api/image/network?identifiers="
species = "&species=9606"    # homo sapiens
my_genes = top_genes.loc[2, 'Genes']

request_url = base_url + "%0d".join(my_genes)
request_url = request_url + species
# print(request_url)

response = requests.get(request_url)
with open('scales_LIN_91.png', 'wb') as fh:
    fh.write(response.content)


# index/Model/average_node_degree/p_value/score
# 
# inter_31:      2 /PCA_mixed     /6.59/0.000000    /344.104
# scales_LIN_31: 2 /PCA_mixed     /1.06/0.000009    /5.388
#                19/PCA_mixed&Tree/0.94/0.000085    /4.802
# scales_EXP_31: (same as scales_LIN_31)
# inter_91:      2 /PCA_mixed     /7.10/0.000000    /367.039
# scales_LIN_91: 2 /PCA_mixed     /2.11/9.770000e-10/12.204
#                19/PCA_mixed&Tree/2.12/1.010000e-09/11.606
# scales_EXP_91: (same as scales_LIN_31)

# In[ ]:




