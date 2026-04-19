#!/usr/bin/env python
# coding: utf-8

# # Training 🚂

# ### 0. First submission baseline

# In[1]:


# 0.0 
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
df = pd.read_csv('../processed/training_data.csv')
test_df = pd.read_csv('../processed/test_data.csv')

test_df.head()


# In[2]:


# 0.1. Split data
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']
X_test = test_df # Do not need to drop target class as this is already done in Kaggle dataset

X_test = X_test.reindex(columns=X.columns, fill_value=0)


# In[3]:


# 0.2. train on XGBoost
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X, y)
pred = xgb.predict(X_test)


# In[4]:


# 0.3. FIRST results for Kaggle submission
output = pd.DataFrame({
    'Id' : X_test.Id,
    'SalePrice' : pred
})


# In[5]:


# 0.4. FIRST Result output for submission
# output.to_csv('../Submission.csv', index=False)
# Commented out post submission 12.4.26


# ### 1. Hyperparameter tuning post-first submission 

# In[6]:


# 1.0 Split training data and get new baseline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_log_error

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_val)
print(f"RMSLE: {np.sqrt(mean_squared_log_error(y_val, y_pred))}")
# First result RMSLE: 0.135
# Second result after feature engineering: 0.144
# Third result after removing this again: 0.134


# In[7]:


# 1.1 Hyperparameter tune XGBoost
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5]
}

rand_search = RandomizedSearchCV(
    XGBRegressor(random_state = 42),
    param_dist,
        n_iter=30,
    cv=5,
    scoring='neg_mean_squared_log_error',
    random_state=42,
    n_jobs=-1
)

rand_search.fit(X_train, y_train)
best_xgb = rand_search.best_estimator_

print(f"Best params: {rand_search.best_params_}")

y_pred = best_xgb.predict(X_val)
print(f"Tuned RMSLE: {np.sqrt(mean_squared_log_error(y_val, y_pred))}")

# Best params: {'subsample': 0.8, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 4, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
# Tuned RMSLE: 0.096
# Thid run after trialling and removing feature engineering: 0.113


# ## Model export

# In[8]:


import pickle
with open('../models/trained_xgb.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)


# In[ ]:




