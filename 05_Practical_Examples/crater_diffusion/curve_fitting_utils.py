import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim

def generate_sample_data(n_samples=100, noise=0.1):
    """
    Generate synthetic data for demonstration
    
    Parameters:
    n_samples (int): Number of data points
    noise (float): Standard deviation of Gaussian noise
    
    Returns:
    tuple: X (input) and y (target) arrays
    """
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 0.5 * np.sin(X.ravel()) + 0.5 * X.ravel() + np.random.normal(0, noise, n_samples)
    return X, y

def polynomial_regression(X, y, degree=3):
    """
    Polynomial Regression implementation
    
    This method transforms the input features into polynomial features
    and then applies linear regression. Useful for fitting curves that
    follow polynomial patterns, like simplified orbital mechanics.
    
    Parameters:
    X (array): Input features
    y (array): Target values
    degree (int): Degree of the polynomial
    
    Returns:
    tuple: Fitted model and transformed features
    """
    # Transform features to polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly_features

def regularized_regression(X, y, method='ridge', alpha=1.0):
    """
    Regularized Regression implementation (Ridge or Lasso)
    
    These methods add regularization terms to prevent overfitting:
    - Ridge: L2 regularization (sum of squared coefficients)
    - Lasso: L1 regularization (sum of absolute coefficients)
    
    Parameters:
    X (array): Input features
    y (array): Target values
    method (str): 'ridge' or 'lasso'
    alpha (float): Regularization strength
    
    Returns:
    object: Fitted model
    """
    if method.lower() == 'ridge':
        model = Ridge(alpha=alpha)
    else:
        model = Lasso(alpha=alpha)
    
    model.fit(X, y)
    return model

def gaussian_process_regression(X, y):
    """
    Gaussian Process Regression implementation
    
    GPR is particularly useful for astronomical data as it:
    1. Provides uncertainty estimates
    2. Works well with noisy data
    3. Makes fewer assumptions about the underlying function
    
    Parameters:
    X (array): Input features
    y (array): Target values
    
    Returns:
    object: Fitted GPR model
    """
    # Define the kernel (RBF with a constant kernel)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    
    # Initialize and fit the model
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X, y)
    
    return gpr

def support_vector_regression(X, y):
    """
    Support Vector Regression implementation
    
    SVR is useful for:
    1. Handling non-linear relationships
    2. Being robust to outliers
    3. Finding global minima
    
    Parameters:
    X (array): Input features
    y (array): Target values
    
    Returns:
    object: Fitted SVR model
    """
    # Initialize and fit the model with RBF kernel
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(X, y)
    
    return svr

class NeuralNetworkRegressor(nn.Module):
    """
    Neural Network implementation for regression
    
    Neural networks can capture complex, non-linear relationships
    and are particularly useful when the underlying function is unknown
    or very complex.
    """
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_neural_network(X, y, model, epochs=1000):
    """
    Train the neural network
    
    Parameters:
    X (array): Input features
    y (array): Target values
    model (nn.Module): Neural network model
    epochs (int): Number of training epochs
    
    Returns:
    object: Trained neural network model
    """
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

