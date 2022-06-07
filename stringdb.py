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


with open('scales_LIN_91.pickle', 'rb') as handle:
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


# In[7]:


print(top_genes.nsmallest(5, ['p_value']))
print(top_genes.nlargest(5, ['score']))


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


# index/Model/average_node_degree/p_value/score/training_acc/testing_acc
# 
# inter_31:      2 /PCA_mixed     /6.59/0.000000    /344.104/0.8634/0.8313
# inter_91:      2 /PCA_mixed     /7.10/0.000000    /367.039/0.8613/0.8056
# scales_LIN_31: 2 /PCA_mixed     /1.06/0.000009    /5.388  /0.8414/0.8313
#                19/PCA_mixed&Tree/0.94/0.000085    /4.802  /0.8414/0.8313
# scales_LIN_91: 2 /PCA_mixed     /2.11/9.770000e-10/12.204 /0.8431/0.8056
#                19/PCA_mixed&Tree/2.12/1.010000e-09/11.606 /0.8577/0.8056
# scales_EXP_31: (same as scales_LIN_31)
# scales_EXP_91: (same as scales_LIN_31)

# ___

# ## Testing with Selected Genes

# In[39]:


with open('scales_LIN_31.pickle', 'rb') as handle:
    top_genes = pickle.load(handle)
top_genes = top_genes['PCA_mixed&Tree']


# In[28]:


df = pd.read_csv('simplify_combined_dataset_v9.csv')    # shape: (310, 22882), 9/1 split
print(df.shape)


# In[30]:


x_cols = df.columns.tolist()
info_cols = ['ScanREF', 'DataSet']
y_cols = ['state']

x_cols.remove('ScanREF')
x_cols.remove('DataSet')
x_cols.remove('state')
x_cols.remove('type')    # train/test; type


# ### Split Data

# In[32]:


df_train = df[df['type']=='train']
df_test = df[df['type']=='test']
print(df_train.shape)
print(df_test.shape)


# In[33]:


X_train = df_train[x_cols]                           # shape: (274, 22878)
y_train = np.squeeze(df_train[y_cols].to_numpy())    # shape: (274,)
X_test = df_test[x_cols]                             # shape: (36, 22878)
y_test = np.squeeze(df_test[y_cols].to_numpy())      # shapeL (36,)


# In[40]:


from sklearn import svm

clf = svm.SVC()
clf.fit(X_train[top_genes], y_train)
print(clf.score(X_train[top_genes], y_train))
print(clf.score(X_test[top_genes], y_test))


# In[ ]:




