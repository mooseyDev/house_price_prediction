#!/usr/bin/env python
# coding: utf-8

# # Processing

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../raw_data/train.csv')
test_df = pd.read_csv('../raw_data/test.csv')


# ## Cleaning
# Missing Features that need imputation: \
# Alley	93.77	str \
# LotFrontage	17.74	float64 \
# MasVnrType	59.73	str \
# MasVnrArea	0.55	float64 \
# BsmtQual	2.53	str \
# BsmtCond	2.53	str \
# BsmtExposure	2.60	str \
# BsmtFinType1	2.53	str \
# BsmtFinType2	2.60	str \
# Electrical	0.07	str \
# FireplaceQu	47.26	str \
# GarageType	5.55	str \
# GarageYrBlt	5.55	float64 \
# GarageFinish	5.55	str \
# GarageQual	5.55	str \
# GarageCond	5.55	str \
# PoolQC	99.52	str \
# Fence	80.75	str \
# MiscFeature	96.30	str \
# #### 1.01 Missing imputation: Alley

# In[3]:


# Fill missing values with an integer flag.
df['Alley'] = df['Alley'].notna().astype(int)
test_df['Alley'] = test_df['Alley'].notna().astype(int)


# #### 1.02 Missing imputation: LotFrontage

# In[4]:


# Check correlational features with LotFrontage to understand plausability of imputation methods.
df.select_dtypes(include=['number']).corr()['LotFrontage'].sort_values(ascending=False)


# In[5]:


# .02 Numerical imputation: LotFrontage
# Check overlapping rows between LotFrontage and features for potential imputation
missing_lotfrontage = df['LotFrontage'].isna()
features_for_lotfrontage_imputation = ['1stFlrSF', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
for feature in features_for_lotfrontage_imputation:
    overlap = missing_lotfrontage & df[feature].isna()
    if not overlap.any():
        print(f'{feature} doesnt contain any missing overlapping rows with LotFrontage ✅')
    else: print(f'{feature} has overlapping rows with LotFrontage')


# In[6]:


# .02 Numerical imputation: LotFrontage
# Linear regression imputation of LotFrontage using highly correlational features.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = df[df['LotFrontage'].notna()][features_for_lotfrontage_imputation]
y = df[df['LotFrontage'].notna()]['LotFrontage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print(f'R^2 score: {r2_score(y_test, model.predict(X_test))}')


# In[7]:


# Poor R2 score from linear regression. 
# Investigating...
# How much does just predicting the mean explain?
y_mean = df[df['LotFrontage'].notna()]['LotFrontage'].mean()
y_actual = df[df['LotFrontage'].notna()]['LotFrontage']
baseline_r2 = 1 - (((y_actual - y_mean) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum())
print(f"Baseline R² (just predicting mean): {baseline_r2}")


# In[8]:


# Check relationships of highly correlational features and the target class. 
import matplotlib.pyplot as plt

features_for_lotfrontage_imputation = ['1stFlrSF', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'MSSubClass']

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i, feature in enumerate(features_for_lotfrontage_imputation):
    axes[i].scatter(df[feature], df['LotFrontage'], alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('LotFrontage')
plt.tight_layout()
plt.show()


# In[9]:


# Due to visualisations clearly showing a non-linear relationship, use RF instead and check outcomes. 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

X = df[df['LotFrontage'].notna()][features_for_lotfrontage_imputation]
y = df[df['LotFrontage'].notna()]['LotFrontage']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_dist = {
    'max_depth': [5,10,15,20,30],
    'n_estimators': [50,100,200,300],
    'min_samples_leaf': [1,2,4,8]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
model = random_search.best_estimator_

print(f"Best params: {random_search.best_params_}")
print(f"R² Score: {r2_score(y_val, model.predict(X_val))}")


# In[10]:


# RF performs well with 0.57 score.. thus, impute on the training and test data.
# Impute on training
missing_mask = df['LotFrontage'].isna()
X_missing = df[missing_mask][features_for_lotfrontage_imputation]
df.loc[missing_mask, 'LotFrontage'] = model.predict(X_missing)

# Imputing on test
test_missing_mask = test_df['LotFrontage'].isna()
X_test_missing = test_df[test_missing_mask][features_for_lotfrontage_imputation]
test_df.loc[test_missing_mask, 'LotFrontage'] = model.predict(X_test_missing)

print(f"LotFrontage missing: {df['LotFrontage'].isna().sum()}")
print(f"test data LotFrontage missing: {test_df['LotFrontage'].isna().sum()}")


# #### 1.03 Missing imputation: MasVnrType

# In[11]:


# In the data_description.txt, a valid input was None, thus, missing values imputed with None.
df['MasVnrType'] = df['MasVnrType'].fillna('None')
test_df['MasVnrType'] = test_df['MasVnrType'].fillna('None')


# #### 1.04 Missing imputation: All string columns below 5% missing

# In[12]:


# All missing are very low, thus imputed with N/A as some houses don't have a 
# basement or garage
columns_for_NA_imputation = [
    'BsmtQual', 
    'BsmtCond', 
    'BsmtExposure', 
    'BsmtFinType1',
    'BsmtFinType2',
    'Electrical', 
    'GarageType', 
    'GarageFinish', 
    'GarageQual', 
    'GarageCond'
    ]

df[columns_for_NA_imputation] = df[columns_for_NA_imputation].fillna('N/A')
test_df[columns_for_NA_imputation] = test_df[columns_for_NA_imputation].fillna('N/A')


# #### 1.05 Missing imputation: MasVnrArea

# In[13]:


# Very+ low missing percentage. Will impute with Mean. 
MasVnrArea_training_mean = df['MasVnrArea'].mean()
df['MasVnrArea'] = df['MasVnrArea'].fillna(MasVnrArea_training_mean)
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(MasVnrArea_training_mean)


# #### 1.06 Missing imputation: FireplaceQu

# In[14]:


# Missing values are supposed to be NA. e.g. No fireplace.
df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna('NA')


# #### 1.05 Missing imputation: GarageYrBlt

# In[15]:


# Missing values are supposed to be NA. e.g. No Garage.
df['GarageYrBlt'] = df['GarageYrBlt'].fillna('NA')
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna('NA')


# #### 1.06 Missing imputation: PoolQC

# In[16]:


# Missing values are supposed to be NA. e.g. No Pool.
df['PoolQC'] = df['PoolQC'].fillna('NA')
test_df['PoolQC'] = test_df['PoolQC'].fillna('NA')


# #### 1.07 Missing imputation: Fence

# In[17]:


# Missing values are supposed to be NA. e.g. No Fence.
df['Fence'] = df['Fence'].fillna('NA')
test_df['Fence'] = test_df['Fence'].fillna('NA')


# #### 1.08 Missing imputation: MiscFeature

# In[18]:


# Missing values are supposed to be NA. e.g. No Misc feature.
df['MiscFeature'] = df['MiscFeature'].fillna('NA')
test_df['MiscFeature'] = test_df['MiscFeature'].fillna('NA')


# In[19]:


# Final check for null instances
null_info = pd.DataFrame({
    'null_percent': (df.isnull().sum() / len(df) * 100).round(2),
    'dtype': df.dtypes
})
null_info[null_info['null_percent'] > 0]


# In[20]:


df['Neighborhood']


# # Post-training feature engineering
# 
# Commented out due to reduction in prediction quality

# In[21]:


# new_features = pd.DataFrame({
#     # 
#     'HouseAge': df['YrSold'] - df['YearBuilt'],
#     'RemodAge': df['YrSold'] - df['YearRemodAdd'],
#     #
#     'QualityArea' : df['OverallQual'] * df['GrLivArea'],
#     'QualityBsmt' : df['OverallQual'] * df['TotalBsmtSF'],
#     #
#     'TotalArea' : df['GrLivArea'] + df['TotalBsmtSF'],
#     #
#     'NeighborhoodMedianPrice' : df.groupby('Neighborhood')['SalePrice'].transform('median')
# }, index=df.index)

# df = pd.concat([df, new_features], axis = 1)


# ### Encoding of categorical columns

# In[22]:


# Encode categorical cols with get_dummies
df = pd.get_dummies(df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)
test_df = test_df.reindex(columns=df.columns, fill_value=0)
test_df = test_df.drop('SalePrice', axis=1)


# In[23]:


df.shape


# In[24]:


df.describe()


# In[25]:


df.columns = df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Exporting data to processed
df.to_csv('../processed/training_data.csv', index=False)
test_df.to_csv('../processed/test_data.csv', index=False)


# In[ ]:




