#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv


# In[2]:


df = pd.read_csv('simplify_combined_dataset_v9.csv')    # shape: (310, 22882)
# df = pd.read_csv('combined_data_v9.csv')
print(df.shape)


# In[3]:


print(df.columns[:15])
print(df.columns[-15:])


# In[4]:


x_cols = df.columns.tolist()
info_cols = ['ScanREF', 'DataSet']
y_cols = ['state']

x_cols.remove('ScanREF')
x_cols.remove('DataSet')
x_cols.remove('state')
x_cols.remove('type')    # train/test; type


# # **Split Data**

# In[5]:


df_train = df[df['type']=='train']
df_test = df[df['type']=='test']
print(df_train.shape)
print(df_test.shape)


# In[6]:


X_train = df_train[x_cols]                           # shape: (274, 22878)
y_train = np.squeeze(df_train[y_cols].to_numpy())    # shape: (274,)
X_test = df_test[x_cols]                             # shape: (36, 22878)
y_test = np.squeeze(df_test[y_cols].to_numpy())      # shapeL (36,)


# In[7]:


# print(np.shape(X_train))
# print(np.shape(y_train))
# print(np.shape(X_test))
# print(np.shape(y_test))


# # **Screening**

# ### Display and Export

# In[8]:


def display_dict(dictionary):
    """Displays dictionaries neatly."""
    for key, value in dictionary.items():
        print(f"{key.rjust(15)}: {'{0:.6f}'.format(value)}")

def export_dict(dictionary, fname):
    with open(f'{fname}.csv', 'w', newline='') as csvfile:
        header = ['Gene', fname]
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for key in dictionary:
            writer.writerow({'Gene': key, fname: dictionary[key]})
    print(f'Succesfully exported {fname}.csv.')


# ## Principal Component Analysis (PCA)

# In[9]:


from sklearn.decomposition import PCA

pca = PCA(n_components=200)
pca.fit(X_train)


# In[10]:


# print(pd.DataFrame(pca.components_, columns=X_train.columns))    # (n_components, n_features)
# print(np.shape(pca.components_))


# In[11]:


def reduced_weights(weights, thres=.005):
    """Get the number of PCs with a variance ratio greater than the threshold."""
    for i in range(len(weights)):
        if (weights[i] < thres):
            return i

# pc_weights = pca.explained_variance_ratio_
# print(np.shape(pc_weights))
# pca_score = pc_weights @ np.absolute(pca.components_)    # (1, k) * (k, 22878), absolute value for importance

# for getting separate PC1 score only
pca_score = np.absolute(pca.components_[0])


# In[12]:


pca_score_top = np.sort(pca_score)[::-1]
# print(pca_score_top[:20])

pca_top_ind = np.argsort(pca_score)[::-1]
# print(pca_top_ind[:20])


# In[13]:


pca_dict = {}
for i in range(200):
    pca_dict.update({X_train.columns[pca_top_ind[i]]: pca_score_top[i]})
export_dict(pca_dict, 'PCA_PC1_31')


# ## Linear Discriminant Analysis (LDA)

# In[9]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)


# In[10]:


lda_score = np.squeeze(clf.coef_)
lda_score_top = np.sort(lda_score)[::-1]
# print(lda_score_top[:20])

lda_top_ind = np.argsort(lda_score)[::-1]
# print(lda_top_ind[:20])


# In[11]:


lda_dict = {}
for i in range(200):
    lda_dict.update({X_train.columns[lda_top_ind[i]]: lda_score_top[i]})
export_dict(lda_dict, 'LDA_31')


# # **Tree-based Methods**

# ## Decision Tree

# In[8]:


from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

params = {'max_features':(None, 'sqrt', 'log2', 200), 'class_weight':(None, 'balanced')}
clf = DecisionTreeClassifier(random_state=0)
CVclf = GridSearchCV(clf, params)
CVclf.fit(X_train, y_train)


# In[9]:


col_display = ['param_class_weight', 'param_max_features', 'mean_test_score', 'rank_test_score']
print(pd.DataFrame.from_dict(CVclf.cv_results_)[col_display])


# In[11]:


# clf = DecisionTreeClassifier(random_state=0)    # 9/1 split; (None, None): 0.787879
clf = DecisionTreeClassifier(max_features=200, random_state=0)    # 3/1 split; (200, None): 0.797005
clf.fit(X_train, y_train)


# In[12]:


tree_score = clf.feature_importances_
tree_score_top = np.sort(tree_score)[::-1]
tree_top_ind = np.argsort(tree_score)[::-1]


# In[13]:


tree_dict = {}
for i in range(200):
    if (tree_score_top[i] != 0):
        tree_dict.update({X_train.columns[tree_top_ind[i]]: tree_score_top[i]})
export_dict(tree_dict, 'Tree_31')


# In[14]:


from sklearn import tree
from matplotlib import pyplot as plt

fig = plt.figure()
plt.figure(figsize=(12, 8))
tree.plot_tree(clf)
plt.show()


# ## Random Forest

# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

params = {'n_estimators':(50, 100, 200), 'class_weight':(None, 'balanced'), 'max_samples':(0.7, 0.9)}
clf = RandomForestClassifier(max_features=None, random_state=0)
CVclf = GridSearchCV(clf, params)
CVclf.fit(X_train, y_train)


# In[12]:


col_display = ['param_' + x for x in [*params]]
col_display.extend(['mean_test_score', 'rank_test_score'])
print(CVclf.best_params_)
print(CVclf.best_score_)


# In[13]:


# 9/1 split: ('balanced', 0.7, 100): 0.886667
# clf = RandomForestClassifier(class_weight='balanced', max_features=None, max_samples=0.7,
#                              n_estimators=100, random_state=0)
# 3/1 split: ()
clf = RandomForestClassifier(class_weight='balanced', max_features=None, max_samples=0.9,
                             n_estimators=200, random_state=0)
clf.fit(X_train, y_train)


# In[14]:


forest_score = clf.feature_importances_
forest_score_top = np.sort(forest_score)[::-1]
forest_top_ind = np.argsort(forest_score)[::-1]


# In[15]:


forest_dict = {}
for i in range(200):
    if (forest_score_top[i] != 0):
        forest_dict.update({X_train.columns[forest_top_ind[i]]: forest_score_top[i]})
export_dict(forest_dict, 'Forest_31')


# ## XGBoost

# In[9]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[10]:


clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)


# In[11]:


xgb_score = clf.feature_importances_
xgb_score_top = np.sort(xgb_score)[::-1]
xgb_top_ind = np.argsort(xgb_score)[::-1]


# In[12]:


xgb_dict = {}
for i in range(200):
    if (xgb_score_top[i] != 0):
        xgb_dict.update({X_train.columns[xgb_top_ind[i]]: xgb_score_top[i]})
export_dict(xgb_dict, 'XGB_31')


# ___

# # Mapping Gene Symbols

# In[6]:


import pandas as pd

# Extract 'ID' and 'Gene Symbol' from GPL570-55999.txt
df_probe = pd.read_table('GPL570-55999.txt', skiprows=16)    # shape: (54675, 16)

# Map the probes in the normalized gene expression data to genes.
df = pd.read_table('GSE19188_results.txt')    # shape: (probes, samples) = (54675, 56)
df = df.rename(columns={'Probesets': 'ID'})
df_joined = pd.merge(df_probe[['ID', 'Gene Symbol']], df, on='ID', how='inner')
df_merged = df_joined.groupby('Gene Symbol').mean()
print(df_merged.head(20))

