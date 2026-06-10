"""
Graph Neural Networks for Quantum Error Propagation Modeling
Implements GNN architectures for entanglement stability and error spread prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class GNNConfig:
    """Configuration for GNN models"""
    # Graph properties
    num_nodes: int = 10  # Number of qubits
    node_features: int = 8  # Features per qubit
    edge_features: int = 3  # Features per connection
    
    # GNN architecture
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    aggregation: str = 'mean'  # 'mean', 'sum', 'max'
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    weight_decay: float = 1e-5


class GraphConvolution(nn.Module):
    """Simple graph convolutional layer"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return self.dropout(output)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network (GAT) layer"""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4,
                 dropout: float = 0.0, alpha: float = 0.2, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features, 1))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_nodes, in_features = x.shape
        h = torch.matmul(x.unsqueeze(1), self.W)  # (batch, heads, nodes, out_features)
        
        # Use actual num_nodes from input
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, -1, num_nodes, -1, -1)
        
        a_input = torch.cat([h_i, h_j], dim=-1)
        e = torch.matmul(a_input, self.a.unsqueeze(0)).squeeze(-1)
        e = self.leakyrelu(e)
        
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        else:
            # Ensure adj has correct shape
            if adj.shape[-1] != num_nodes or adj.shape[-2] != num_nodes:
                adj_trimmed = adj[:, :num_nodes, :num_nodes]
            else:
                adj_trimmed = adj
            e = e.masked_fill(adj_trimmed.unsqueeze(1) == 0, -1e9)
            
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)
        
        if self.concat:
            output = h_prime.reshape(batch_size, num_nodes, -1)
        else:
            output = h_prime.mean(dim=1)
            
        return output


class QuantumGNN(nn.Module):
    """Graph Neural Network for quantum error propagation modeling"""
    
    def __init__(self, config: GNNConfig):
        super(QuantumGNN, self).__init__()
        self.config = config
        
        self.node_embedding = nn.Linear(config.node_features, config.hidden_channels)
        
        if config.edge_features > 0:
            self.edge_embedding = nn.Linear(config.edge_features, config.hidden_channels)
        else:
            self.edge_embedding = None
            
        self.conv_layers = nn.ModuleList()
        in_channels = config.hidden_channels
        
        for i in range(config.num_layers):
            out_channels = config.hidden_channels // (2 ** i) if i < config.num_layers - 1 else config.hidden_channels
            conv = GraphConvolution(in_features=in_channels, out_features=out_channels, dropout=config.dropout)
            self.conv_layers.append(conv)
            in_channels = out_channels
            
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(config.hidden_channels // (2 ** i) if i < config.num_layers - 1 else config.hidden_channels)
            for i in range(config.num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.node_classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels * 2, 1),
            nn.Sigmoid()
        )
        
        self.graph_classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels * 2, 1),
            nn.Sigmoid()
        )
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, num_nodes, _ = x.shape
        h = F.relu(self.node_embedding(x))
        
        if edge_features is not None and self.edge_embedding is not None:
            edge_emb = self.edge_embedding(edge_features)
            adj_weighted = adj * edge_emb.mean(dim=-1, keepdim=True)
        else:
            adj_weighted = adj
            
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            h_prev = h
            h = conv(h, adj_weighted)
            
            if i > 0 and h.shape == h_prev.shape:
                h = h + h_prev
                
            h = h.transpose(1, 2)
            h = bn(h)
            h = h.transpose(1, 2)
            h = F.relu(h)
            
        node_preds = self.node_classifier(h)
        h_graph = h.transpose(1, 2)
        h_graph = self.global_pool(h_graph).squeeze(-1)
        graph_pred = self.graph_classifier(h_graph)
        
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        edge_input = torch.cat([h_i, h_j], dim=-1)
        edge_preds = self.edge_predictor(edge_input)
        
        return {
            'node_predictions': node_preds.squeeze(-1),
            'graph_prediction': graph_pred.squeeze(-1),
            'edge_predictions': edge_preds.squeeze(-1),
            'node_embeddings': h
        }


class EntanglementGNN(nn.Module):
    """Specialized GNN for entanglement stability analysis"""
    
    def __init__(self, config: GNNConfig):
        super(EntanglementGNN, self).__init__()
        self.config = config
        
        # Use simpler GCN layers instead of GAT to avoid shape issues
        self.conv_layers = nn.ModuleList()
        in_channels = config.node_features
        
        for i in range(config.num_layers):
            out_channels = config.hidden_channels // (2 ** i) if i < config.num_layers - 1 else config.hidden_channels
            conv = GraphConvolution(in_features=in_channels, out_features=out_channels, dropout=config.dropout)
            self.conv_layers.append(conv)
            in_channels = out_channels
            
        self.entanglement_stability = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )
        
        self.fidelity_predictor = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(in_channels, 1)
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = x
        
        for conv in self.conv_layers:
            h = conv(h, adj)
            h = F.elu(h)
            
        stability = self.entanglement_stability(h)
        fidelity = self.fidelity_predictor(h)
        
        return {
            'stability': stability.squeeze(-1),
            'fidelity': fidelity.squeeze(-1),
            'embeddings': h
        }


class GNNTrainer:
    """Trainer for GNN models"""
    
    def __init__(self, model: nn.Module, config: GNNConfig, device: torch.device = DEVICE):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.bce_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.history = {'train_loss': [], 'val_loss': [], 'node_acc': [], 'graph_acc': []}
        
    def prepare_data(self, graphs: List[Dict], val_split: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        num_graphs = len(graphs)
        num_val = int(num_graphs * val_split)
        indices = np.random.permutation(num_graphs)
        
        train_graphs = [graphs[i] for i in indices[num_val:]]
        val_graphs = [graphs[i] for i in indices[:num_val]]
        
        return train_graphs, val_graphs
    
    def train_epoch(self, graphs: List[Dict]) -> float:
        self.model.train()
        total_loss = 0
        
        for graph in graphs:
            x = torch.FloatTensor(graph['node_features']).unsqueeze(0).to(self.device)
            adj = torch.FloatTensor(graph['adjacency']).unsqueeze(0).to(self.device)
            
            node_labels = torch.FloatTensor(graph.get('node_labels', np.zeros(x.shape[1]))).unsqueeze(0).to(self.device)
            graph_label = torch.FloatTensor([graph.get('graph_label', 0)]).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x, adj)
            
            loss = 0
            
            # Handle different output types for different models
            if 'node_predictions' in outputs and outputs['node_predictions'].shape == node_labels.shape:
                node_loss = self.bce_loss(outputs['node_predictions'], node_labels)
                loss += node_loss
                
            if 'stability' in outputs:
                # Entanglement GNN - use stability as node-level prediction
                if node_labels.shape[-1] == outputs['stability'].shape[-1]:
                    node_loss = self.bce_loss(outputs['stability'], node_labels)
                    loss += node_loss
                    
            if graph_label is not None:
                graph_pred = outputs.get('graph_prediction', outputs.get('stability', torch.zeros(1)).mean())
                if len(graph_pred.shape) > 1:
                    graph_pred = graph_pred.mean(dim=-1)
                if len(graph_pred.shape) == 2:
                    graph_pred = graph_pred.squeeze(0)
                if graph_pred.dim() == 0:
                    graph_pred = graph_pred.unsqueeze(0)
                if graph_pred.shape == graph_label.shape:
                    graph_loss = self.bce_loss(graph_pred, graph_label)
                    loss += graph_loss
                
            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                
        return total_loss / len(graphs) if graphs else 0
    
    def validate(self, graphs: List[Dict]) -> Tuple[float, float, float]:
        self.model.eval()
        node_correct = 0
        node_total = 0
        graph_correct = 0
        graph_total = 0
        
        with torch.no_grad():
            for graph in graphs:
                x = torch.FloatTensor(graph['node_features']).unsqueeze(0).to(self.device)
                adj = torch.FloatTensor(graph['adjacency']).unsqueeze(0).to(self.device)
                
                node_labels = graph.get('node_labels', np.zeros(x.shape[1]))
                graph_label = graph.get('graph_label', 0)
                
                outputs = self.model(x, adj)
                
                # Handle different output types
                if 'node_predictions' in outputs and len(node_labels) > 0:
                    node_preds = outputs['node_predictions'].cpu().numpy()[0]
                    node_correct += np.sum((node_preds > 0.5) == node_labels)
                    node_total += len(node_labels)
                elif 'stability' in outputs and len(node_labels) > 0:
                    node_preds = outputs['stability'].cpu().numpy()[0]
                    node_correct += np.sum((node_preds > 0.5) == node_labels)
                    node_total += len(node_labels)
                    
                # Graph accuracy
                if 'graph_prediction' in outputs:
                    graph_pred = outputs['graph_prediction'].cpu().numpy()
                    if isinstance(graph_pred, np.ndarray):
                        graph_pred = graph_pred.item() if graph_pred.size == 1 else graph_pred.mean()
                    graph_correct += (graph_pred > 0.5) == graph_label
                    graph_total += 1
                elif 'stability' in outputs:
                    graph_pred = outputs['stability'].cpu().numpy().mean()
                    graph_correct += (graph_pred > 0.5) == graph_label
                    graph_total += 1
                
        node_acc = node_correct / node_total if node_total > 0 else 0
        graph_acc = graph_correct / graph_total if graph_total > 0 else 0
        
        return 0, node_acc, graph_acc
    
    def train(self, train_graphs: List[Dict], val_graphs: List[Dict], num_epochs: Optional[int] = None) -> Dict:
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"Training GNN on {self.device}")
        print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")
        print("-" * 60)
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_graphs)
            _, node_acc, graph_acc = self.validate(val_graphs)
            
            self.history['train_loss'].append(train_loss)
            self.history['node_acc'].append(node_acc)
            self.history['graph_acc'].append(graph_acc)
            
            self.scheduler.step(-graph_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {train_loss:.4f}, Node Acc: {node_acc:.4f}, Graph Acc: {graph_acc:.4f}")
            
            if graph_acc > best_val_acc:
                best_val_acc = graph_acc
                torch.save(self.model.state_dict(), 'best_gnn_model.pth')
                
        print("-" * 60)
        print(f"Best graph accuracy: {best_val_acc:.4f}")
        
        return self.history


def generate_quantum_graphs(num_graphs: int = 100, num_qubits: int = 10, seed: Optional[int] = None) -> List[Dict]:
    """Generate synthetic quantum connectivity graphs for training"""
    if seed is not None:
        np.random.seed(seed)
        
    graphs = []
    
    for _ in range(num_graphs):
        adj = np.zeros((num_qubits, num_qubits))
        
        for i in range(num_qubits - 1):
            adj[i, i+1] = 1
            adj[i+1, i] = 1
            
        num_long_range = np.random.randint(1, 4)
        for _ in range(num_long_range):
            i, j = np.random.choice(num_qubits, 2, replace=False)
            adj[i, j] = 1
            adj[j, i] = 1
            
        node_features = np.random.uniform(0, 1, (num_qubits, 8))
        node_features[:, 0] = 0.5 + 0.5 * node_features[:, 1]
        node_features[:, 4] = 1 - node_features[:, 3]
        
        error_prob = 1 - node_features[:, 2]
        noise_factor = node_features[:, 7]
        node_labels = (error_prob + noise_factor > 1.0).astype(float)
        graph_label = float(np.mean(node_labels) > 0.3)
        
        graphs.append({
            'node_features': node_features,
            'adjacency': adj,
            'node_labels': node_labels,
            'graph_label': graph_label,
            'num_qubits': num_qubits
        })
        
    return graphs


if __name__ == "__main__":
    print("=" * 60)
    print("Graph Neural Networks for Quantum Error Propagation")
    print("=" * 60)
    
    config = GNNConfig(num_nodes=10, node_features=8, edge_features=0, hidden_channels=64, num_layers=3, dropout=0.3, batch_size=16, num_epochs=30)
    
    print(f"\nGNN Configuration:")
    print(f"  Num qubits: {config.num_nodes}")
    print(f"  Node features: {config.node_features}")
    print(f"  Hidden channels: {config.hidden_channels}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Device: {DEVICE}")
    
    print("\nGenerating synthetic quantum graphs...")
    graphs = generate_quantum_graphs(num_graphs=200, num_qubits=10, seed=42)
    
    print(f"Generated {len(graphs)} graphs")
    print(f"Positive graph labels: {sum(g['graph_label'] for g in graphs)}")
    
    trainer = GNNTrainer(QuantumGNN(config), config)
    train_graphs, val_graphs = trainer.prepare_data(graphs, val_split=0.2)
    
    print(f"\nTrain: {len(train_graphs)} graphs, Val: {len(val_graphs)} graphs")
    
    print("\n" + "=" * 60)
    print("Training Quantum GNN")
    print("=" * 60)
    
    history = trainer.train(train_graphs, val_graphs, num_epochs=30)
    
    print("\n" + "=" * 60)
    print("Testing Entanglement GNN")
    print("=" * 60)
    
    ent_config = GNNConfig(num_nodes=10, node_features=8, hidden_channels=32, num_layers=2, dropout=0.3)
    ent_model = EntanglementGNN(ent_config)
    ent_trainer = GNNTrainer(ent_model, ent_config)
    
    print("\nTraining Entanglement GNN...")
    ent_history = ent_trainer.train(train_graphs, val_graphs, num_epochs=20)
    
    print("\n" + "=" * 60)
    print("Demo Inference")
    print("=" * 60)
    
    try:
        trainer.model.load_state_dict(torch.load('best_gnn_model.pth', map_location=DEVICE, weights_only=True))
        print("Loaded best GNN model")
    except:
        print("Using trained model directly")
        
    sample_graph = graphs[0]
    x = torch.FloatTensor(sample_graph['node_features']).unsqueeze(0)
    adj = torch.FloatTensor(sample_graph['adjacency']).unsqueeze(0)
    
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(x, adj)
        
    print(f"\nSample graph predictions:")
    print(f"  Graph stability: {outputs['graph_prediction'].item():.4f}")
    print(f"  Node error probabilities: {outputs['node_predictions'].cpu().numpy()[0]}")
    print(f"  Actual node labels: {sample_graph['node_labels']}")
    
    print("\n" + "=" * 60)
    print("GNN Training Complete")
    print("=" * 60)
