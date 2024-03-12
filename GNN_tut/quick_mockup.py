import sqlite3
import pandas as pd
import json
import torch
from torch_geometric.data import Data, DataLoader

from torch_geometric import loader
import torch.optim as optim


# Function to convert a row to a PyG Data object
def row_to_graph(row, max_nbeads):
    nbeads = row["nbeads"]
    x = torch.eye(nbeads)  # Simple identity matrix for node features

    # Create an identity matrix for existing nodes and pad the rest
    x = torch.eye(max_nbeads)  # Use max_nbeads for the dimension
    if nbeads < max_nbeads:
        # Pad the feature matrix with zeros if nbeads is less than max_nbeads
        padding = torch.zeros((max_nbeads - nbeads, max_nbeads))
        x = torch.cat((x[:nbeads], padding), dim=0)

    edge_index = []
    edge_attr = []

    # Add implicit edges for nearest and next-nearest neighbors with a size of 1.5 units
    for i in range(nbeads):  # Assuming indices start at 1
        for j in range(i + 1, min(i + 3, nbeads)):  # Nearest and next-nearest
            edge_index.append([i, j])
            edge_attr.append([1.5])  # Connection size for nearest and next-nearest neighbors
            edge_index.append([j, i])  # Add edge in the other direction
            edge_attr.append([1.5])

    # Add edges from nonlocal_bonds
    for bond in row['nonlocal_bonds']:
        edge_index.append([bond[0] - 1, bond[1] - 1])
        edge_attr.append([bond[2]])
        edge_index.append([bond[1] - 1, bond[0] - 1])
        edge_attr.append([bond[2]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([float(row['s_bias_mean'])], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def get_training_sets(graphs, batch_size=1, train_frac=0.8):
    # define some model parameters :
    train_size = int(len(graphs) * train_frac)
    train_dataset = graphs[:train_size]
    test_dataset = graphs[train_size:]

    # load data using torch dataloader
    trainloader = loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return testloader, trainloader





if __name__ == "__main__":
    from models import GATModel

    node_features = 31  # Adjust based on your dataset
    hidden_channels = 16
    model = GATModel(node_features=node_features, hidden_channels=hidden_channels)

    # get data from sql-lite
    df = process_data_from_sql()

    # Convert all rows in the dataframe to PyG Data objects
    graphs = [row_to_graph(row, df.nbeads.max()) for index, row in df.iterrows()]


    def enhance_node_features_with_edge_attributes(data):
        # Example: Summing edge attributes for each node
        edge_attr_sum = torch.zeros((data.num_nodes, 1))

        for i, (source, target) in enumerate(data.edge_index.t()):
            edge_attr_sum[source] += data.edge_attr[i]
            edge_attr_sum[target] += data.edge_attr[i]  # Assuming undirected edges for simplicity

        # Concatenating the original node features with the aggregated edge attributes
        data.x = torch.cat([data.x, edge_attr_sum], dim=-1)

        return data


    graphs = [enhance_node_features_with_edge_attributes(graph) for graph in graphs]

    # Splitting the dataset into train and test sets
    test_loader, train_loader = get_training_sets(graphs, batch_size=3)

    # Assuming a regression problem with your s_bias_mean as the target
    loss_function = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training the model for a certain number of epochs
    epochs = 100

    # training loop
    for epoch in range(epochs):
        loss = model.train_model(train_loader=train_loader, optimizer=optimizer, loss_function=loss_function)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    # validation set check
    test_loss = model.test_model(test_loader, loss_function)
    print(f'Test Loss: {test_loss:.4f}')
