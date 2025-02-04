import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def np_to_th(x):
    """
    Convert a numpy array to a PyTorch tensor and move it to the appropriate device.
    
    Parameters:
    x (numpy.ndarray): Input numpy array.
    
    Returns:
    torch.Tensor: PyTorch tensor on the appropriate device.
    """
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)

class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        """
        Initialize the neural network.
        
        Parameters:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        n_units (int): Number of units in each hidden layer.
        epochs (int): Number of training epochs.
        loss (torch.nn.Module): Loss function for training.
        lr (float): Learning rate for the optimizer.
        loss2 (torch.nn.Module, optional): Additional loss function for regularization.
        loss2_weight (float, optional): Weight for the additional loss function.
        """
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        # Define the layers of the neural network
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Output tensor.
        """
        h = self.layers(x)
        out = self.out(h)
        return out

    def fit(self, X, y):
        """
        Train the neural network.
        
        Parameters:
        X (numpy.ndarray): Input features for training.
        y (numpy.ndarray): Target values for training.
        
        Returns:
        list: List of loss values for each epoch.
        """
        Xt = np_to_th(X)
        yt = np_to_th(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        """
        Make predictions using the trained neural network.
        
        Parameters:
        X (numpy.ndarray): Input features for prediction.
        
        Returns:
        numpy.ndarray: Predicted values.
        """
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()

class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        """
        Initialize the neural network for parameter discovery.
        
        Parameters:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        n_units (int): Number of units in each hidden layer.
        epochs (int): Number of training epochs.
        loss (torch.nn.Module): Loss function for training.
        lr (float): Learning rate for the optimizer.
        loss2 (torch.nn.Module, optional): Additional loss function for regularization.
        loss2_weight (float, optional): Weight for the additional loss function.
        """
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        # Add a learnable parameter for radius
        self.r = nn.Parameter(data=torch.tensor([0.]))
