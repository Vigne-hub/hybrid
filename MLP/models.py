import torch
from torch import nn
import lightning as L


class MeanRelativeError(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
        Initialize the Mean Relative Error Loss Module.

        Parameters:
        - epsilon: A small value added to the denominator to avoid division by zero.
        """
        super(MeanRelativeError, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Calculate the Mean Relative Error between predictions and targets.

        Parameters:
        - predictions: Predicted values (torch.Tensor).
        - targets: Actual values (torch.Tensor).

        Returns:
        - Mean Relative Error (torch.Tensor).
        """
        relative_errors = torch.abs(predictions - targets) / (torch.abs(targets) + self.epsilon)
        return torch.mean(relative_errors)


class MLPModule(L.LightningModule):
    def __init__(self, num_features):
        super().__init__()
        # Your model definition
        self.accuracy = None
        self.predictions = []
        self.targets = []
        self.test_features = []

        self.layers = nn.Sequential(
            nn.Linear(num_features, 2 * num_features),
            nn.LeakyReLU(),
            nn.Linear(2 * num_features, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1)
        )
        # self.loss = nn.MSELoss()
        #self.loss = nn.L1Loss()

        self.loss = MeanRelativeError()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.flatten(), y.flatten())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.flatten(), y.flatten())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat.flatten(), y.flatten())
        self.log("test_loss", loss)

        self.predictions.append(y_hat.flatten().cpu())
        self.test_features.append(x)
        self.targets.append(y.cpu())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
