import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from models import MLPModule
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
import os


def main(csv="mlp_dataset_simple_nonbeads.csv"):
    train_loader, val_loader, num_features = get_datasets(csv)

    # Initialize the module
    model = MLPModule(num_features)

    # Initialize the trainer
    trainer = L.Trainer(deterministic=True, max_epochs=100,

                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)])
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, val_loader)

    write_features_predictions_to_csv(model.test_features, model.predictions, model.targets,
                                      file_name=f"data_with_predictions_{trainer.logger.version}.csv",
                                      columns=["state_i", "state_j", "rc1", "rc2"])


def get_datasets(csv='mlp_dataset_simple_allnbeads.csv'):
    # set seed
    seed_everything(42, workers=True)
    # load data for the mlp regression
    df = pd.read_csv(csv)
    # Assuming 's_bias_mean' as the target
    features = df.drop(columns=['s_bias_mean']).values
    targets = df[['s_bias_mean']].values
    # Standardize features
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    return train_loader, val_loader, X_train.shape[1]


import torch
import pandas as pd

import torch
import pandas as pd


def write_features_predictions_to_csv(features, predictions, actuals, file_name="data_with_predictions.csv",
                                      columns=None):
    """
    Writes features, predictions, and actual values from lists of tensors to a CSV file.

    Parameters:
    - features: List[torch.Tensor] - A list of tensors containing the features.
    - predictions: List[torch.Tensor] - A list of tensors containing the predictions.
    - actuals: List[torch.Tensor] - A list of tensors containing the actual values.
    - file_name: str - The name of the CSV file to write.
    """
    # Concatenate the lists of tensors into single tensors
    features_tensor = torch.cat(features, dim=0)
    predictions_tensor = torch.cat(predictions, dim=0)
    actuals_tensor = torch.cat(actuals, dim=0)

    # Convert tensors to numpy arrays
    features_array = features_tensor.numpy()
    predictions_array = predictions_tensor.numpy()
    actuals_array = actuals_tensor.numpy()

    # Flatten the predictions and actuals if they are not already 1D
    predictions_flat = predictions_array.flatten()
    actuals_flat = actuals_array.flatten()

    if columns is None:
        # Create a DataFrame from the features array
        df_features = pd.DataFrame(features_array, columns=[f'Feature_{i}' for i in range(features_array.shape[1])])

    df_features = pd.DataFrame(features_array, columns=columns)

    # Add predictions and actuals to the DataFrame
    df_features['Predictions'] = predictions_flat
    df_features['Actuals'] = actuals_flat

    # Write the DataFrame to a CSV file
    df_features.to_csv(file_name, index=False)


def post_process(checkpoint_path, version=4):
    train_loader, val_loader, num_features = get_datasets()

    model = MLPModule.load_from_checkpoint(checkpoint_path=checkpoint_path, num_features=num_features)

    # Initialize the trainer
    trainer = L.Trainer(deterministic=True, max_epochs=100,

                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)])
    # Test the model
    trainer.test(model, val_loader)

    write_features_predictions_to_csv(model.test_features, model.predictions, model.targets,
                                      file_name=f"data_with_predictions_{version}.csv")


if __name__ == '__main__':
    main()
    print('Done')
