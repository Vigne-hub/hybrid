import sqlite3
import pandas as pd
import json
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear


# Function to convert a row to a PyG Data object
def row_to_graph(row, nbeads):
    x = torch.eye(nbeads)  # Simple identity matrix for node features

    edge_index = []
    edge_attr = []

    # Add implicit edges for nearest and next-nearest neighbors with a size of 1.5 units
    for i in range(1, nbeads + 1):  # Assuming indices start at 1
        for j in range(i + 1, min(i + 3, nbeads + 1)):  # Nearest and next-nearest
            edge_index.append([i, j])
            edge_attr.append([1.5])  # Connection size for nearest and next-nearest neighbors
            edge_index.append([j, i])  # Add edge in the other direction
            edge_attr.append([1.5])

    # Add edges from nonlocal_bonds
    for bond in row['nonlocal_bonds']:
        edge_index.append([bond[0], bond[1]])
        edge_attr.append([bond[2]])
        edge_index.append([bond[1], bond[0]])
        edge_attr.append([bond[2]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([float(row['s_bias_mean'])], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class GNN(MessagePassing):
    def __init__(self, node_features, hidden_channels):
        super(GNN, self).__init__(aggr='mean')  # "Mean" aggregation.
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch_index):
        # First Graph Convolution
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second Graph Convolution
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        # Readout layer
        x = global_mean_pool(x, batch_index)  # Aggregate node features to graph-level

        # Apply a final classifier
        x = F.relu(x)
        x = self.lin(x)

        return x


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()  # Clear gradients
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Forward pass
        loss = loss_function(output, data.y)  # Compute the loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = loss_function(output, data.y)  # Compute the loss
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    # Assuming your SQLite database is named 'diff_s_bias.db' and the table is 'diff_s_bias'
    conn = sqlite3.connect('../generate_input/simulation_configs_1/configurations.db')
    df = pd.read_sql_query("SELECT * FROM diff_s_bias", conn)
    conn.close()

    # Convert the 'nonlocal_bonds' column from JSON-formatted string to list
    df['nonlocal_bonds'] = df['nonlocal_bonds'].apply(lambda x: json.loads(x))

    # Adjust nbeads as necessary for each row, this is just an example for the first row
    sample_graph = row_to_graph(df.iloc[0], nbeads=df.iloc[0]['nbeads'])

    # Convert all rows in the dataframe to PyG Data objects
    graphs = [row_to_graph(row, nbeads=row['nbeads']) for index, row in df.iterrows()]

    # Splitting the dataset into train and test sets
    # Let's say you want to use 80% of your data for training and 20% for testing
    train_size = int(len(graphs) * 0.8)
    train_dataset = graphs[:train_size]
    test_dataset = graphs[train_size:]

    node_features = sample_graph.num_node_features
    hidden_channels = 16  # Example size, adjust as needed

    model = GNN(node_features, hidden_channels)

    from torch_geometric.data import DataLoader

    batch_size = 32  # You can adjust this according to your GPU memory

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    import torch.optim as optim

    # Assuming a regression problem with your s_bias_mean as the target
    loss_function = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Learning rate can be adjusted

    # Training the model for a certain number of epochs
    epochs = 100
    for epoch in range(epochs):
        loss = train()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    test_loss = test(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
