import numpy as np

# Gradient Descent Function
def gradient_descent(X, y, iterations, learning_rate, stopping_threshold):
    n = len(X)
    w = 0
    b = 0
    for i in range(iterations):
        y_pred = w * X + b
        error = y_pred - y
        loss = np.mean(np.square(error))
        
        dw = (2/n) * np.dot(X.T, error)
        db = (2/n) * np.sum(error)
        
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if loss < stopping_threshold:
            break
    
    return w, b, loss

# Actual data
X_actual = np.array([1, 2, 3, 4, 5])
y_actual = np.array([5, 7, 9, 11, 13])

# Modified data
X_modified = np.array([1, 2, 3, 4, 5])
y_modified = np.array([6, 10, 14, 18, 22])

# Hyperparameters
iterations = 1000
learning_rate = 0.01
stopping_threshold = 0.0001

# Estimation of optimal parameters with actual data
w_optimal_actual, b_optimal_actual, loss_optimal_actual = gradient_descent(X_actual, y_actual, iterations, learning_rate, stopping_threshold)

# Estimation of optimal parameters with modified data
w_optimal_modified, b_optimal_modified, loss_optimal_modified = gradient_descent(X_modified, y_modified, iterations, learning_rate, stopping_threshold)

# Results for actual data
print("Optimal weight with actual data:", w_optimal_actual)
print("Optimal bias with actual data:", b_optimal_actual)
print("Final loss with actual data:", loss_optimal_actual)

# Results for modified data
print("\nOptimal weight with modified data:", w_optimal_modified)
print("Optimal bias with modified data:", b_optimal_modified)
print("Final loss with modified data:", loss_optimal_modified)
