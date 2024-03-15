import numpy as np

from visualizations import linearplot_transition_colored_dynamic
import torch
import pandas as pd


def write_features_predictions_to_csv(features, predictions, actuals, file_name, target_names, columns=None):
    """
    Writes features, predictions, and actual values from lists of tensors to a CSV file.

    Parameters:
    - features: List[torch.Tensor] - A list of tensors containing the features.
    - predictions: List[torch.Tensor] - A list of tensors containing the predictions for each target.
    - actuals: List[torch.Tensor] - A list of tensors containing the actual values for each target.
    - file_name: str - The name of the CSV file to write.
    - target_names: List[str] - A list of strings representing the names of each target.
    - columns: List[str] - Optional. A list of column names for the features.
    """

    # Concatenate the lists of tensors into single tensors for features
    features_tensor = torch.cat(features, dim=0)

    # Create numpy arrays
    features_array = features_tensor.numpy()
    predictions_array = np.array(predictions)
    actuals_array = np.array(actuals)

    # Create a DataFrame from the features array
    if columns is None:
        df_features = pd.DataFrame(features_array, columns=[f'Feature_{i}' for i in range(features_array.shape[1])])
    else:
        df_features = pd.DataFrame(features_array, columns=columns)

    # Process each target to correct column
    for i, target_name in enumerate(target_names):

        # Add predictions and actuals to the DataFrame
        df_features[f'{target_name}_pred'] = predictions_array[:, i]
        df_features[f'{target_name}_actual'] = actuals_array[:, i]

    # Write the DataFrame to a CSV file
    df_features.to_csv(file_name, index=False)


