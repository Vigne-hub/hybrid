from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool, GATConv
import torch.nn.functional as F
from torch.nn import Linear
import torch

class GNN(MessagePassing):
    def __init__(self, node_features, hidden_channels):
        super(GNN, self).__init__(aggr='mean')  # "Mean" aggregation.
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 20)
        self.conv3 = GCNConv(20, 10)
        self.conv4 = GCNConv(10, 5)
        self.lin = Linear(5, 1)

    def forward(self, x, edge_index, edge_attr, batch_index):
        # First Graph Convolution
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index, edge_attr)

        # Readout layer
        x = global_mean_pool(x, batch_index)  # Aggregate node features to graph-level

        # Apply a final classifier
        x = F.relu(x)
        x = self.lin(x)

        return x


class NNModel(torch.nn.module):

    def __init__(self):
        super().__init__()


class GATModel(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):

        super(GATModel, self).__init__()

        self.conv1 = GATConv(node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch_index)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        return x

    def train_model(self, train_loader, optimizer, loss_function):
        self.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = self(data.x, data.edge_index, data.batch)
            loss = loss_function(output.squeeze(1), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)

    def test_model(self, test_loader, loss_function):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for data in test_loader:
                output = self(data.x, data.edge_index, data.batch)
                loss = loss_function(output.squeeze(), data.y)
                total_loss += loss.item() * data.num_graphs
        return total_loss / len(test_loader.dataset)
