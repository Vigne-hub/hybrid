import numpy as np
import pandas as pd
import sqlite3
import json
from scipy.stats import skew, kurtosis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class FoldingTransitionEntropyData:
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

    def calculate_features(self, row):
        # Jaccard similarity between state_i and state_j
        state_i_set = set(tuple(x) for x in row['state_i'])
        state_j_set = set(tuple(x) for x in row['state_j'])
        jaccard_sim = len(state_i_set.intersection(state_j_set)) / len(
            state_i_set.union(state_j_set)) if state_i_set or state_j_set else 0

        # Preparing vectors for cosine similarity calculations with the final state
        weights_final = np.array([bond[2] for bond in row['nonlocal_bonds']])
        bond_index_map = {tuple(bond[:2]): idx for idx, bond in enumerate(row['nonlocal_bonds'])}
        weights_vector_i = np.zeros(len(row['nonlocal_bonds']))
        weights_vector_j = np.zeros(len(row['nonlocal_bonds']))

        for bond in row['state_i']:
            idx = bond_index_map.get(tuple(bond[:2]))
            if idx is not None:
                weights_vector_i[idx] = bond[2]
        for bond in row['state_j']:
            idx = bond_index_map.get(tuple(bond[:2]))
            if idx is not None:
                weights_vector_j[idx] = bond[2]

        # Normalizing weight vectors for cosine similarity
        weights_vector_i_norm = normalize([weights_vector_i])[0]
        weights_vector_j_norm = normalize([weights_vector_j])[0]
        final_bond_weights_norm = normalize([weights_final])[0]

        # Cosine similarity differences between states and the final state
        cosine_diff_i_final = cosine_similarity([weights_vector_i_norm], [final_bond_weights_norm])[0][0]
        cosine_diff_j_final = cosine_similarity([weights_vector_j_norm], [final_bond_weights_norm])[0][0]
        cosine_diff_i_j = cosine_similarity([weights_vector_i_norm], [weights_vector_j_norm])[0][0]

        # Normalized Degree Centrality for state_i and state_j
        normalized_degree_cent_i = self.calculate_normalized_degree_centrality(row['state_i'], row['nbeads'])
        normalized_degree_cent_j = self.calculate_normalized_degree_centrality(row['state_j'], row['nbeads'])
        diff_normalized_degree_cent = abs(normalized_degree_cent_i - normalized_degree_cent_j)

        return pd.Series(
            [jaccard_sim, cosine_diff_i_final, cosine_diff_j_final, cosine_diff_i_j, diff_normalized_degree_cent],

            index=['jaccard_sim', 'cosine_diff_i_final', 'cosine_diff_j_final', 'cosine_diff_i_j',
                   'diff_normalized_degree_cent'])

    @staticmethod
    def calculate_normalized_degree_centrality(state, nbeads):
        degree_counts = {i: 0 for i in range(nbeads)}  # assuming bead indices start at 0

        for bond in state:
            degree_counts[bond[0]] += 1
            degree_counts[bond[1]] += 1

        total_degree = sum(degree_counts.values())

        normalized_degree_centrality = total_degree / (nbeads * 2) if nbeads else 0

        return normalized_degree_centrality

    def generate_MLP_dataset(self):
        features = self.data.apply(self.calculate_features, axis=1)

        mlp_data = pd.concat([self.data, features], axis=1)

        mlp_data = mlp_data[['jaccard_sim', 'cosine_diff_i_final', 'cosine_diff_j_final', 'cosine_diff_i_j',
                             'diff_normalized_degree_cent', 's_bias_mean']].dropna()
        return mlp_data


if __name__ == '__main__':
    # Assuming the correct database file path
    fted = FoldingTransitionEntropyData()
    mlp_dataset = fted.generate_MLP_dataset()
    print(mlp_dataset.head())
