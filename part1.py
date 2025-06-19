import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    'https://raw.githubusercontent.com/JeffChou32/wine-dataset/refs/heads/main/winequality-white.csv',
    sep=';'
)
df = df.dropna().drop_duplicates() #Remove rows with missing values and duplicates
X = df.drop('quality', axis=1) #Separate features and target 
y = df['quality']  
scaler = StandardScaler() #Standardize feature values
X_scaled = scaler.fit_transform(X)
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled] #Add a column of ones to X 
y = y.to_numpy().reshape(-1, 1) #Convert target variable to column vector
X_train, X_test, y_train, y_test = train_test_split( #80/20 training/testing
    X_scaled, y, test_size=0.2, random_state=42
)
def compute_mse(X, y, theta): #compute MSE
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    return (1 / (2 * m)) * np.sum(error ** 2)

def gradient_descent(X, y, alpha, iterations): #gradient descent
    m, n = X.shape
    theta = np.zeros((n, 1))  #weights zero
    cost_history = []
    
    for i in range(iterations): #Loop to update weights
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * X.T.dot(errors)  #Compute gradients
        theta -= alpha * gradients  #Update weights
        cost = compute_mse(X, y, theta)  #Log cost
        cost_history.append(cost)

    return theta, cost_history

params_log = [] #Experiment with different learning rates and iteration counts
alphas = [0.001, 0.005, 0.01]  #Learning rates to test
iterations_list = [500, 1000, 2000]  #Iteration counts to test

#For each combination of alpha and iteration count, run training and log results
for alpha in alphas:
    for iterations in iterations_list:
        theta, cost_hist = gradient_descent(X_train, y_train, alpha, iterations)
        train_mse = compute_mse(X_train, y_train, theta)
        params_log.append((alpha, iterations, train_mse))
        with open("log.txt", "a") as f:
            f.write(f"alpha={alpha}, iterations={iterations}, train_mse={train_mse:.4f}\n")

best = min(params_log, key=lambda x: x[2]) #Select the best parameter combination based on lowest training MSE
best_alpha, best_iters, _ = best

#Retrain model using best parameters and compute test MSE
theta, _ = gradient_descent(X_train, y_train, best_alpha, best_iters)
test_mse = compute_mse(X_test, y_test, theta)

#Output best parameters and performance on test set
print(f"Best alpha: {best_alpha}, Best iterations: {best_iters}")
print(f"Test MSE: {test_mse:.4f}")
