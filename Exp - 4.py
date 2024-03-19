import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert to a Pandas DataFrame for easier manipulation
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Compute covariance between features and target
cov_matrix = np.cov(X.T, y)

# Calculate deviation about the mean for each feature
mean_X = np.mean(X, axis=0)
deviation_X = X - mean_X
# Compute regression coefficients
coefficients = np.linalg.inv(cov_matrix[:-1, :-1]) @ cov_matrix[:-1, -1]

import matplotlib.pyplot as plt

def plot_regression_line():
    plt.scatter(X[:, 0], y, label="Data points")
    plt.plot(X[:, 0], X[:, 0] * coefficients[0] + coefficients[1], color='red', label="Regression line")
    plt.xlabel("Feature 1")
    plt.ylabel("Target")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()

# Call the function to plot the regression line
plot_regression_line()
