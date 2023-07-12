#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.dpi'] = 300


# In[ ]:


#Importing Data
dataset = pd.read_csv("copd vs Control.csv")
dataset.head()


# In[ ]:


dataset["Condition"].value_counts()


# In[ ]:


X = dataset.iloc[:,1:]
X.head()


# In[ ]:


Y=dataset.iloc[:,:1]
Y.head()


# In[ ]:


#Randomly Splitting Data into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:


#Classification_Report
print(classification_report(y_test, y_pred))


# In[ ]:


import xgboost as xgb


# In[ ]:


#Classifier
xgb_mod=xgb.XGBClassifier(random_state=42) 
xgb_mod=xgb_mod.fit(X_train,y_train.values.ravel()) 


# In[ ]:


y_pred = xgb_mod.predict(X_test)

# Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


#Classification_Report
print(classification_report(y_test, y_pred))


# In[ ]:


import shap
explainer = shap.TreeExplainer(xgb_mod)
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value

############## visualizations #############
# Generate summary dot plot
shap.summary_plot(shap_values, X,title="SHAP summary plot") 


# In[ ]:


shap_values.shape


# In[ ]:


#Bar plot 
shap.summary_plot(shap_values, X,plot_type="bar",max_display=14) 

