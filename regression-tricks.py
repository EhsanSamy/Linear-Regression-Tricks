import numpy as np
import random
from matplotlib import pyplot as plt

# -------- Simple Trick --------
def simple_trick(x, y, weight, bias, eta1, eta2):
    y_predicted = weight * x + bias
    
    if y > y_predicted:
        if x > 0:
            weight += eta1
            bias += eta2
        else:
            weight -= eta1
            bias += eta2
    else:
        if x > 0:
            weight -= eta1
            bias -= eta2
        else:
            weight += eta1
            bias -= eta2
    return weight, bias

# -------- Square Trick --------
def square_trick(x, y, weight, bias, eta1, eta2):
    y_predicted = weight * x + bias
    weight += eta1 * x * (y - y_predicted)
    bias += eta2 * (y - y_predicted)
    return weight, bias

# -------- Absolute Trick --------
def absolute_trick(x, y, weight, bias, eta1, eta2):
    y_predicted = weight * x + bias
    if y > y_predicted:
        weight += eta1 * x
        bias += eta2
    else:
        weight -= eta1 * x
        bias -= eta2
    return weight, bias

# -------- Gradient Descent Trick with Loss Tracking --------
def gradient_descent(x, y, weight, bias, eta, num_iterations):
    m = len(x)
    losses = []

    for _ in range(num_iterations):
        y_predicted = weight * x + bias
        error = y_predicted - y

        dw = (1/m) * np.dot(error, x)
        db = (1/m) * np.sum(error)

        weight -= eta * dw
        bias -= eta * db

        cost = (1/(2*m)) * np.sum(error ** 2)
        losses.append(cost)

    return weight, bias, losses

# -------- Plotting Function --------
def plot_data(x, y, weight, bias, trick_type):
    plt.scatter(x, y, color='blue', label='Data Points')
    x_line = np.linspace(min(x), max(x), 100)
    y_line = weight * x_line + bias
    plt.plot(x_line, y_line, color='red', label='Regression Line')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Data Points and Regression Line {trick_type} trick')
    plt.legend()
    plt.show()

# -------- Main Function (updated) --------
def main(trick_type, num_iterations, eta1, eta2=None):
    np.random.seed(0)

    x = np.random.uniform(-10, 10, 100)
    y = 2 * x + 3 + np.random.normal(0, 1, 100)  

    weight = random.uniform(-1, 1)
    bias = random.uniform(-1, 1)

    if trick_type == 'gd':
        weight, bias, losses = gradient_descent(x, y, weight, bias, eta1, num_iterations)
        plot_data(x, y, weight, bias, trick_type)

        # Plot loss curve
        plt.plot(range(num_iterations), losses, color='green')
        plt.xlabel('Iterations')
        plt.ylabel('Cost (MSE)')
        plt.title('Loss Curve - Gradient Descent')
        plt.show()
    else:
        for _ in range(num_iterations):
            for xi, yi in zip(x, y):
                if trick_type == 'simple':
                    weight, bias = simple_trick(xi, yi, weight, bias, eta1, eta2)
                elif trick_type == 'square':
                    weight, bias = square_trick(xi, yi, weight, bias, eta1, eta2)
                elif trick_type == 'absolute':
                    weight, bias = absolute_trick(xi, yi, weight, bias, eta1, eta2)

        plot_data(x, y, weight, bias, trick_type)

if __name__ == "__main__":
    main(trick_type='simple', num_iterations=100, eta1=0.005, eta2=0.005)
    main(trick_type='square', num_iterations=100, eta1=0.005, eta2=0.005)
    main(trick_type='absolute', num_iterations=100, eta1=0.005, eta2=0.005)
    main(trick_type='gd', num_iterations=500, eta1=0.01)  
