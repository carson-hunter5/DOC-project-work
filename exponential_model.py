#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:10:52 2024

@author: dylansacks
"""
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(url):
    category = str(url)
    response = requests.get(category)
    data = response.json()
    data_dict = data["items"]
    data = pd.DataFrame.from_records(data_dict)
    return data

# gettinh data
demographics = get_data('https://api.unhcr.org/population/v1/demographics/?&yearFrom=2010&yearTo=2025&coo_all=TRUE&limit=10000000&coa_all=TRUE')

# Cleaning
demographics[['year','f_0_4', 'f_5_11', 'f_12_17', 'f_18_59', 'f_60', 'f_other', 'f_total', 'm_0_4', 'm_5_11', 'm_12_17', 'm_18_59', 'm_60', 'm_other', 'm_total', 'total']] = demographics[['year', 'f_0_4', 'f_5_11', 'f_12_17', 'f_18_59', 'f_60', 'f_other', 'f_total', 'm_0_4', 'm_5_11', 'm_12_17', 'm_18_59', 'm_60', 'm_other', 'm_total', 'total']].astype(int)
demographics = demographics.drop(['coo', 'coo_iso', 'coa', 'coa_iso'], axis=1)
demographics = demographics.dropna()

# Filtering 
demographics = demographics[demographics["coo_id"] != demographics["coa_id"]]

# Melting the data
df_melted = pd.melt(demographics, id_vars=["year", "coa_name"], value_vars=["f_0_4", "f_5_11", "f_12_17", "f_18_59", "f_60", "m_0_4", "m_5_11", "m_12_17", "m_18_59", "m_60"],
                    var_name="age_group", value_name="number")

# Splitting 
df_melted['gender'] = df_melted['age_group'].str[0].replace({'f': 'F', 'm': 'M'})
df_melted['age_group'] = df_melted['age_group'].str[2:]
df_melted = df_melted[["year", "coa_name", "gender", "age_group", "number"]]

#  log transformation
df_melted['log_number'] = np.log(df_melted['number'] + 1)  # Adding 1 to handle zeros

# dummy variables
dummy = pd.get_dummies(df_melted, columns=['coa_name', 'gender', 'age_group'], drop_first=True, dtype='int')

# predictors vs response
X = dummy.drop(['number', 'log_number'], axis=1).values
y = dummy['log_number'].values

# Check the shape of X and y to ensure consistency
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# bias
bias = np.ones((X.shape[0], 1))
X = np.hstack((bias, X))

# Check the shape of X after adding bias term
print("Shape of X after adding bias:", X.shape)

# coefficients
b = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))

# predictions
pred_log = np.matmul(X, b)

# Check the shape of b and pred_log to ensure consistency
print("Shape of b:", b.shape)
print("Shape of pred_log:", pred_log.shape)

# residuals
residuals = y - pred_log

# r2
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - pred_log) ** 2)
r2 = 1 - (ss_residual / ss_total)

# MSE
mse = np.mean((y - pred_log) ** 2)

print("R-squared:", r2)
print("MSE:", mse)

# residuals vs predicted scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(pred_log, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Log-Transformed Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

years = df_melted['year'].values

# residuals vs year scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(years, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.title('Residuals vs Year')
plt.show()

# Q-Q plot
import scipy.stats as stats
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()