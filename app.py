"""
model01.py is an example of how to access model parameter values that you are storing
in the database and use them to make a prediction when a route associated with prediction is
accessed. 
"""
from turtle import pd
from backend.db_connection import db
import numpy as np
import logging

def get_data(url):
    category = str(url)
    response = requests.get(category)
    data = response.json()
    data_dict = data["items"]
    data = pd.DataFrame.from_records(data_dict)
    
    # Cleaning
    data[['year','f_0_4', 'f_5_11', 'f_12_17', 'f_18_59', 'f_60', 'f_other', 'f_total', 'm_0_4', 'm_5_11', 'm_12_17', 'm_18_59', 'm_60', 'm_other', 'm_total', 'total']] = data[['year', 'f_0_4', 'f_5_11', 'f_12_17', 'f_18_59', 'f_60', 'f_other', 'f_total', 'm_0_4', 'm_5_11', 'm_12_17', 'm_18_59', 'm_60', 'm_other', 'm_total', 'total']].astype(int)
    data = data.drop(['coo', 'coo_iso', 'coa', 'coa_iso'], axis=1)
    data = data.dropna()
    
    # Filtering
    data = data[data["coo_id"] != data["coa_id"]]
    
    # melting data
    df_melted = pd.melt(demographics, id_vars=["year", "coa_name"], value_vars=["f_0_4", "f_5_11", "f_12_17", "f_18_59", "f_60", "m_0_4", "m_5_11", "m_12_17", "m_18_59", "m_60"],
                        var_name="age_group", value_name="number")

      # splitting 
    df_melted['gender'] = df_melted['age_group'].str[0].replace({'f': 'F', 'm': 'M'})
    df_melted['age_group'] = df_melted['age_group'].str[2:]
    df_melted = df_melted[["year", "coa_name", "gender", "age_group", "number"]]
    print(df_melted)
     #  log transformation
    df_melted['log_number'] = np.log(df_melted['number'] + 1)
    return data

def train(X, y):
  """
  You could have a function that performs training from scratch as well as testing (see below).
  It could be activated from a route for an "administrator role" or something similar. 
  """
  # bias term
  bias = np.ones((X.shape[0], 1))
  X = np.hstack((bias, X))
    
  # coefficients 
  b = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))

  return b

def test(X, y, b):
  bias = np.ones((X.shape[0], 1))
  X = np.hstack((bias, X))
    
    # predictions
  y_hat = np.matmul(X, b)
    
    # residuals
  residuals = y - y_hat
    
    # r2
  ss_total = np.sum((y - np.mean(y)) ** 2)
  ss_residual = np.sum((y - y_hat) ** 2)
  r2 = 1 - (ss_residual / ss_total)
    
    # mse
  mse = np.mean((y - y_hat) ** 2)
    
  return y_hat, residuals, r2, mse

def predict(var01=None, var02=None, X=None):
  """
  Retreives model parameters from the database and uses them for real-time prediction
  """

  # get a database cursor 
  cursor = db.get_db().cursor()
  # get the model params from the database
  query = 'SELECT beta_vals FROM model1_params ORDER BY sequence_number DESC LIMIT 1'
  cursor.execute(query)
  return_val = cursor.fetchone()



  params = return_val['beta_vals']
  logging.info(f'params = {params}')
  logging.info(f'params datatype = {type(params)}')

  # turn the values from the database into a numpy array
  params_array = np.array(list(map(float, params[1:-1].split(','))))
  logging.info(f'params array = {params_array}')
  logging.info(f'params_array datatype = {type(params_array)}')

  if var01 is not None and var02 is not None:
        # turn the variables sent from the UI into a numpy array
        input_array = np.array([1.0, float(var01), float(var02)])
        
        # calculate the dot product for prediction
        prediction = np.dot(params_array, input_array)
        return prediction
  elif X is not None:
        # Add bias term to input features
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))
        
        # calculate the dot product for predictions
        predictions = np.dot(X, params_array)
        return predictions
  else:
        raise ValueError("Either var01 and var02 or X must be provided.")

  # turn the variables sent from the UI into a numpy array
  # input_array = np.array([1.0, float(var01), float(var02)])

 # bias = np.ones((X.shape[0], 1))
 # X = np.hstack((bias, X))

  # calculate the dot product (since this is a fake regression)
 # prediction = np.dot(params_array, input_array)

  #return prediction

  ##############################################################
  
  # # retreive the parameters from the appropriate table
  # cursor.execute('select beta_0, beta_1, beta_2 from model1_param_vals')
  # # fetch the first row from the cursor
  # data = cursor.fetchone()
  # # calculate the predicted result using this functions arguments as well as the model parameter values
  # result = data[0] + int(var01) * data[1] + int(var02) * data[2]

  # # return the result 
  # return result