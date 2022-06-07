#!/usr/bin/env python
# coding: utf-8

# # **Testing with Selected Genes**

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[2]:


with open('inter_31.pickle', 'rb') as handle:
    top_genes = pickle.load(handle)
top_genes = top_genes['PCA_mixed']


# In[3]:


df = pd.read_csv('simplify_combined_dataset_v9.csv')    # shape: (310, 22882), 3/1 split
print(df.shape)


# In[4]:


x_cols = df.columns.tolist()
info_cols = ['ScanREF', 'DataSet']
y_cols = ['state']

x_cols.remove('ScanREF')
x_cols.remove('DataSet')
x_cols.remove('state')
x_cols.remove('type')    # train/test; type


# In[5]:


# Split Data
df_train = df[df['type']=='train']
df_test = df[df['type']=='test']
print(df_train.shape)
print(df_test.shape)


# In[6]:


X_train = df_train[x_cols]                           # shape: (227, 22878)
y_train = np.squeeze(df_train[y_cols].to_numpy())    # shape: (227,)
X_test = df_test[x_cols]                             # shape: (83, 22878)
y_test = np.squeeze(df_test[y_cols].to_numpy())      # shapeL (83,)


# # **Data Exploration**

# In[7]:


import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [7.5, 6]

def _to_pie(data):
    """ Converts raw data to wedge sizes and labels for pie chart input. """
    data__ser = pd.Series(data)
    data__summary = data__ser.value_counts(normalize=True)
    
    return list(data__summary.index), list(data__summary.array)

def plot_pie(labels, sizes, label_names, title=''):
    """ Plots a pie chart with the input data. """
    # fake data
    # x = [1, 2, 3, 4]
    
    # plot
    plt.pie(sizes, labels=labels, autopct='%.2f%%', startangle=90,
            wedgeprops={'linewidth': 2, 'alpha': 0.9}, textprops={'size': 'large'})
    plt.legend(label_names, bbox_to_anchor=(1, 1), loc="upper left")
    plt.title(title, pad=10)
    plt.show()


# In[8]:


labels, sizes = _to_pie(y_train)
plot_pie(labels, sizes, labels, 'Labels (Training)')


# In[10]:


labels = top_genes[:20]
express = X_test[labels]

plt.figure(figsize=(12, 7.2))
plt.boxplot(express, vert=True,
            patch_artist=True, labels=labels,
            medianprops={'linestyle': 'solid', 'linewidth': 2, 'color': 'firebrick'})
plt.xlabel('Gene')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Gene Expression')
plt.title('Gene Expression (Testing)', pad=10)
plt.show()


# # **Results**

# In[142]:


from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# clf = svm.SVC()
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train[top_genes], y_train)

y_train_pred = clf.predict(X_train[top_genes])
y_test_pred = clf.predict(X_test[top_genes])


# In[143]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
print(f'Training Accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f'Testing Accuracy: {accuracy_score(y_test, y_test_pred)}')
print('')

# Precision & recall
print(classification_report(y_test, y_test_pred))

# Confusion matrix
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [7.5, 6]

cm = confusion_matrix(y_test, y_test_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title('Confusion Matrix', pad=10)
plt.show()

# ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Tree')
display.plot()
plt.show()

