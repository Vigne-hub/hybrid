import numpy as np
import pandas as pd
import sqlite3
import json
from scipy.stats import skew, kurtosis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from functools import cached_property
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, TensorDataset
import torch


class FoldingTransitionEntropyData:
    """
    This class contains the base raw data generated for a bunch of hybridmc simulations.
    """

    def __init__(self, database_file='../generate_input/simulation_configs_1/configurations.db'):
        self._data = None
        self._database_file = database_file
        self.database_file = self._database_file

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # process the data to obtain the bond lists for the two states that are in question
        data = value.dropna()
        data['state_i'] = data.apply(lambda row: self.mask_fn(row['nonlocal_bonds'], row['state_i_bits']), axis=1)
        data['state_j'] = data.apply(lambda row: self.mask_fn(row['nonlocal_bonds'], row['state_j_bits']), axis=1)

        self._data = data

    @property
    def database_file(self):
        return self._database_file

    @database_file.setter
    def database_file(self, database_file):
        self._database_file = database_file
        self.data = self.load_data_from_sql()

    @staticmethod
    def mask_fn(arr, mask):
        """
        mask function takes a list of bonds, and applies a mask specified by a string of 0 and 1
        corresponding to teh truth value of each index in the list.
        So if a list is [bond1, bonds2], and a mask is '01', the function returns [bonds2]
        :param arr: any array like input
        :param mask: str: a bitstring of 0s and 1s
        :return: a np.Array with the mask applied
        """

        # Convert the mask string into a boolean array
        bool_mask = np.array(list(mask)) == '1'

        # Apply the boolean mask to the array
        masked_arr = np.array(arr)[bool_mask]

        return masked_arr

    def load_data_from_sql(self):
        """
        Load simulation data from an SQL lite database
        :param db_config:
        :return:
        """
        # Assuming your SQLite database is named 'diff_s_bias.db' and the table is 'diff_s_bias'
        conn = sqlite3.connect(self.database_file)
        df = pd.read_sql_query("SELECT * FROM diff_s_bias", conn)
        conn.close()
        # Convert the 'nonlocal_bonds' column from JSON-formatted string to list
        df['nonlocal_bonds'] = df['nonlocal_bonds'].apply(lambda x: json.loads(x))

        return df


class FoldingMLPData(FoldingTransitionEntropyData):
    """
    This is the base class for using the hybridmc raw data for an MLP based ML method
    """

    def __init__(self, database_file='../generate_input/simulation_configs_1/configurations.db', feature_names=None,
                 existing_csv=None):
        super(FoldingMLPData, self).__init__(database_file)
        self.generated_feature_names = feature_names
        self.existing_csv = existing_csv

    def calculate_features(self, row):

        state_i_int = int(row['state_i_bits'], base=2)
        state_j_int = int(row['state_j_bits'], base=2)
        rc1, rc2 = (float(el[2]) for el in row.nonlocal_bonds)

        self.generated_feature_names = ['state_i_int', 'state_j_int', 'rc1', 'rc2']

        return pd.Series(
            [state_i_int, state_j_int, rc1, rc2],

            index=self.generated_feature_names)

    @staticmethod
    def calculate_normalized_degree_centrality(state, nbeads):
        degree_counts = {i: 0 for i in range(nbeads)}  # assuming bead indices start at 0

        for bond in state:
            degree_counts[bond[0]] += 1
            degree_counts[bond[1]] += 1

        total_degree = sum(degree_counts.values())

        normalized_degree_centrality = total_degree / (nbeads * 2) if nbeads else 0

        return normalized_degree_centrality

    @cached_property
    def mlp_data(self):

        if self.existing_csv:
            mlp_data = pd.read_csv(self.existing_csv)
        else:
            features = self.data.apply(self.calculate_features, axis=1)

            mlp_data = pd.concat([self.data, features], axis=1)

        mlp_data = mlp_data.dropna()

        return mlp_data

    def write_mlp_dataset_to_csv(self, file_path):
        """
        Writes the generated MLP dataset to a CSV file.

        :param file_path: The path of the CSV file where the dataset will be saved.
        """

        # Write the dataset to a CSV file
        try:
            self.mlp_data.to_csv(file_path, index=False)
            print(f"MLP dataset successfully written to {file_path}")
        except Exception as e:
            print(f"An error occurred while writing the MLP dataset to CSV: {e}")


class FoldingMLPDataMFPT(FoldingMLPData):
    """
    This class is used to generate an MLP dataset for a many target MLP architecture, the two targets are the sbias and
    the MFPT. This class also provides the dataloaders to be used with pytorch using the MLP dataset.
    """

    def __init__(self, database_file='../generate_input/simulation_configs_2/configurations_2.db',
                 table_name="merged_table",
                 feature_names=None,
                 existing_csv=None,
                 target_columns=tuple(['s_bias_mean', 'outer_fpt', 'inner_fpt'])):

        self.target_columns = target_columns
        self.table_name = table_name

        super().__init__(database_file, feature_names, existing_csv)

    def load_data_from_sql(self):
        """
        Load simulation data from an SQL lite database
        :param db_config:
        :return:
        """
        # Assuming SQLite database has tables
        conn = sqlite3.connect(self.database_file)
        df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
        conn.close()
        # Convert the 'nonlocal_bonds' column from JSON-formatted string to list
        df['nonlocal_bonds'] = df['nonlocal_bonds'].apply(lambda x: json.loads(x))

        # convert the state i and j bits to proper representation if they are not already

        if 'int' in str(df.state_i_bits.dtype):
            df["state_i_bits"] = df.apply(lambda row:
                                          str(row["state_i_bits"]).rjust(len(row["nonlocal_bonds"]), '0'),
                                          axis=1)

        if 'int' in str(df.state_j_bits.dtype):
            df["state_j_bits"] = df.apply(lambda row:
                                          str(row["state_j_bits"]).rjust(len(row["nonlocal_bonds"]), '0'),
                                          axis=1)

        return df

    def get_datasets(self, csv=None, data=None, query='nbeads == 25', val_size=0.2, batch_size=1, seed=42,
                     second_split=0):

        # convert the target columns ot list to work better as pandas col index
        target_columns = list(self.target_columns)

        # set seed
        seed_everything(seed, workers=True)

        if csv is not None:
            # load data for the mlp regression
            df = pd.read_csv(csv)
        else:
            df = self.mlp_data.copy()

        # apply any query to filter out data in df
        if query is not None:
            df = df.query(query)

        if data is not None:
            df = data

        # Split based on unique configuration IDs
        config_ids = df['Config_Id'].unique()
        train_ids, val_ids = train_test_split(config_ids, test_size=val_size, random_state=seed)

        train_df = df[df['Config_Id'].isin(train_ids)]
        val_df = df[df['Config_Id'].isin(val_ids)]

        if second_split:
            # Split based on unique configuration IDs
            config_ids = train_df['Config_Id'].unique()
            train_ids, val_ids = train_test_split(config_ids, test_size=0.2, random_state=seed)

            train_df = df[df['Config_Id'].isin(train_ids)]
            val_df = df[df['Config_Id'].isin(val_ids)]

        # Process training set
        X_train = train_df[self.generated_feature_names].values
        y_train = train_df[target_columns].values

        # Process validation set
        X_val = val_df[self.generated_feature_names].values
        y_val = val_df[target_columns].values

        # Standardize features (fit on training data, then transform both training and validation data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Convert to PyTorch tensors
        train_dataset = TensorDataset(torch.Tensor(X_train_scaled), torch.Tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val_scaled), torch.Tensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=1)

        return train_loader, val_loader, self.generated_feature_names, target_columns, scaler


class FoldingMLPVaryPattern(FoldingMLPDataMFPT):
    """
    This class extends the FoldingMLPDataMFPT class to provide features with varying bond patterns
    """
    def calculate_features(self, row):
        state_i_int = int(row['state_i_bits'], base=2)
        state_j_int = int(row['state_j_bits'], base=2)
        rc1, rc2 = (float(el[2]) for el in row.nonlocal_bonds)

        # obtain the 0th and 1st columns in nonlocal bonds (the bead indices for i and j resepctively)
        bead_indices = list(np.array(row.nonlocal_bonds)[:, 0:2].flatten())

        # the bead indices is of the form i1, j1, i2, j2 where nonlocal bonds = [[i1, j1, rc1], [i2, j2, rc2]]

        # define the feature names
        self.generated_feature_names = ['state_i_int', 'state_j_int', 'rc1', 'rc2', 'i1', 'j1', 'i2', 'j2']

        # create pandas series with with the features in approporiate positions as in the index list
        return pd.Series(
            [state_i_int, state_j_int, rc1, rc2] + bead_indices,

            index=self.generated_feature_names)


if __name__ == '__main__':
    # Assuming the correct database file path
    mlp_dataset = FoldingMLPVaryPattern(database_file='../generate_input/simulation_configs_2/configurations_2.db')
    print(mlp_dataset.mlp_data.head())
    mlp_dataset.write_mlp_dataset_to_csv('mlpdata_nbeads=30_vary_pattern_more_data.csv')
