import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    mean_x, mean_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    cross_deviation = np.sum(y * x) - n * mean_y * mean_x
    deviation_x = np.sum(x * x) - n * mean_x * mean_x

    # calculating regression coefficients
    b = cross_deviation / deviation_x
    a = mean_y - b * mean_x

    return a, b

def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color='b', marker='o', s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color='r')

    # adding labels
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # showing grid
    plt.grid(True)

    # showing plot
    plt.show()

def main():
    # observations
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 6, 7])

    # estimating coefficients
    a, b = estimate_coef(x, y)
    print("Estimated coefficients:\nIntercept:", a, "\nSlope:", b)

    # plotting regression line
    plot_regression_line(x, y, (a, b))

if __name__ == "__main__":
    main()
