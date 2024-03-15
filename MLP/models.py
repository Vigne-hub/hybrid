import torch
from torch import nn
import lightning as L
from post_processing import write_features_predictions_to_csv
from visualizations import linearplot_transition_colored_dynamic


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
    def __init__(self, features, outputs, hidden_dims, scaler, learning_rate=1e-3):
        super().__init__()

        # Model definition
        self.predictions = []
        self.targets = []
        self.test_features = []
        self.scaler = scaler
        self.features = features
        self.outputs = outputs

        # Model architecture input and output dimentions
        input_dim = len(features)
        output_dim = len(outputs)

        # Build the network
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.extend([nn.Linear(hidden_dims[i - 1], hidden_dims[i]), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.model = nn.Sequential(*layers)

        self.loss = MeanRelativeError()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

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

        # Convert features to NumPy, unstandardize, then convert back to a tensor
        x_cpu_numpy = x.cpu().numpy()  # Convert to NumPy array for inverse_transform
        x_unstandardized_numpy = self.scaler.inverse_transform(x_cpu_numpy)  # Unstandardize
        x_unstandardized_tensor = torch.tensor(x_unstandardized_numpy, dtype=torch.float32)  # Convert back to tensor

        self.test_features.append(x_unstandardized_tensor)
        self.predictions.append(y_hat.flatten().cpu())
        self.targets.append(y.flatten().cpu())

    def on_test_end(self) -> None:

        # Access the logger from the trainer
        logger = self.trainer.logger

        # Example: Print and use the logger version
        print(f"The logger dir for this run is: {logger.log_dir}")

        file_name = f"{logger.log_dir}/data_with_predictions_{logger.version}.csv"

        write_features_predictions_to_csv(self.test_features, self.predictions, self.targets,
                                          file_name=file_name,
                                          target_names=list(self.outputs),
                                          columns=list(self.features))

        linearplot_transition_colored_dynamic(file_name,f"{logger.log_dir}/transition_colored_{logger.version}.png")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
