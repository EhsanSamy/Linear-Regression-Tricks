# Linear-Regression-Tricks

This project demonstrates various techniques ("tricks") to fit a linear regression model on synthetic data using Python. It explores different parameter update methods and visualizes the results using Matplotlib.

## 📌 Project Overview

The project implements four approaches to train a linear regression model:
- **Simple Trick**: Updates weight and bias based on conditional rules and error direction.
- **Square Trick**: Adjusts parameters using squared error (`y - y_predicted`).
- **Absolute Trick**: Updates parameters based on the absolute error.
- **Gradient Descent**: Optimizes parameters by minimizing Mean Squared Error (MSE) and tracks the loss curve.

### 📊 Dataset
Synthetic data is generated as:
$$ y = 2x + 3 + \text{Gaussian noise} $$
- **Size**: 100 data points.
- **Features**: `x` (random values between -10 and 10), `y` (target values with noise).

## 📊 Visualizations
- Scatter plots of data points with fitted regression lines for each trick.
- Loss curve for Gradient Descent to show convergence.

**Example Outputs**:
- [Simple Trick Plot](https://example.com/simple_trick_plot.png)
- [Gradient Descent Loss Curve](https://example.com/gd_loss_curve.png)

## ⚙️ How It Works
1. **Data Generation**: Creates synthetic data with Gaussian noise.
2. **Parameter Initialization**: Randomly initializes `weight` and `bias`.
3. **Training**:
   - Iterates through data points for Simple, Square, and Absolute Tricks.
   - Uses batch Gradient Descent for optimization in the GD method.
4. **Visualization**: Plots data points, regression lines, and loss curve (for GD).

## 📈 Results
After 100 iterations (500 for GD), typical performance (R² scores, approximate):
- **Simple Trick**: ~0.70
- **Square Trick**: ~0.75
- **Absolute Trick**: ~0.73
- **Gradient Descent**: ~0.76

*Note*: Results depend on hyperparameters (`eta1`, `eta2`) and random seed.

## ▶️ Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/regression-tricks-demo.git
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
3. Run the script:
   ```bash
   python regression_tricks.py
   ```
4. Check outputs: Visualizations will display for each trick.

## 📦 Requirements
- Python 3.7+
- Libraries: `numpy`, `matplotlib`

## 📚 File Structure
```
regression_tricks/
├── regression_tricks.py  # Main script
├── README.md             # Project documentation
```

## 🔗 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/ehsan-samy/)
- 📧 [Gmail](mailto:ehsansamy9@gmail.com)
- 🗃️ [GitHub](https://github.com/EhsanSamy)
