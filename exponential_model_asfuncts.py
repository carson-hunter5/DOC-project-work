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
    
    return data

def train_model(X, y):
    #  bias term
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))
    
    #  coefficients 
    b = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
    
    return b

def evaluate_model(X, y, b):
    # adding bias 
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))
    
    # predictions
    pred_log = np.matmul(X, b)
    
    # residuals
    residuals = y - pred_log
    
    # r2
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - pred_log) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    # mse
    mse = np.mean((y - pred_log) ** 2)
    
    return pred_log, residuals, r2, mse

def plot_residuals(pred_log, residuals):
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_log, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Log-Transformed Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.show()

def plot_residuals_vs_year(years, residuals):
    plt.figure(figsize=(10, 6))
    plt.scatter(years, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Year')
    plt.show()

def predict_log(X, b):
    # bias term
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))
    
    pred_log = np.matmul(X, b)
    
    return pred_log

def plot_qq_plot(residuals):
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.show()