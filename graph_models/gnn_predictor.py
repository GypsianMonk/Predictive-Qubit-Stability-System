"""
Graph Neural Networks for Quantum Error Propagation Modeling
Models qubit connectivity and error spread in quantum systems
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class GraphConfig:
    """Configuration for GNN models"""
    num_nodes: int = 5
    node_features: int = 17
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001


class QuantumGraph:
    """Represents the quantum system as a graph"""
    
    def __init__(self, num_qubits: int, connectivity: Optional[np.ndarray] = None):
        self.num_nodes = num_qubits
        self.node_features = np.zeros((num_qubits, 17))  # 17 telemetry features
        
        # Default: nearest neighbor connectivity
        if connectivity is None:
            self.adjacency_matrix = self._create_nearest_neighbor_graph(num_qubits)
        else:
            self.adjacency_matrix = connectivity
        
        self.edge_index = self._adjacency_to_edge_index(self.adjacency_matrix)
    
    def _create_nearest_neighbor_graph(self, n: int) -> np.ndarray:
        """Create nearest neighbor connectivity matrix"""
        adj = np.zeros((n, n))
        for i in range(n):
            if i > 0:
                adj[i, i-1] = 1
                adj[i-1, i] = 1
            if i < n - 1:
                adj[i, i+1] = 1
                adj[i+1, i] = 1
        return adj
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> np.ndarray:
        """Convert adjacency matrix to edge index format"""
        edges = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] > 0:
                    edges.append([i, j])
        return np.array(edges).T if edges else np.array([]).reshape(2, 0)
    
    def update_node_features(self, qubit_id: int, features: np.ndarray):
        """Update features for a specific qubit"""
        if 0 <= qubit_id < self.num_nodes:
            self.node_features[qubit_id] = features
    
    def get_error_propagation_probability(self, source_qubit: int) -> np.ndarray:
        """Calculate probability of error spreading from source to other qubits"""
        probs = np.zeros(self.num_nodes)
        probs[source_qubit] = 1.0
        
        # Simple diffusion model
        for _ in range(3):  # 3 propagation steps
            new_probs = probs.copy()
            for i in range(self.num_nodes):
                neighbors = np.where(self.adjacency_matrix[i] > 0)[0]
                if len(neighbors) > 0:
                    new_probs[i] += 0.3 * np.mean(probs[neighbors])
            probs = new_probs / np.max(new_probs)
        
        return probs


class GraphConvolutionalLayer:
    """Simple Graph Convolutional Network layer (fallback implementation)"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)
        self.dropout_rate = dropout
    
    def forward(self, features: np.ndarray, edge_index: np.ndarray, 
                num_nodes: int) -> np.ndarray:
        """Forward pass through GCN layer"""
        # Normalize adjacency
        adj_norm = self._normalize_adjacency(edge_index, num_nodes)
        
        # Message passing: aggregate neighbor features
        aggregated = adj_norm @ features
        
        # Linear transformation
        output = aggregated @ self.weights + self.bias
        
        # ReLU activation
        output = np.maximum(0, output)
        
        # Dropout
        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, output.shape)
            output = output * mask / (1 - self.dropout_rate)
        
        return output
    
    def _normalize_adjacency(self, edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        """Compute normalized adjacency matrix with self-loops"""
        adj = np.zeros((num_nodes, num_nodes))
        
        # Add self-loops
        for i in range(num_nodes):
            adj[i, i] = 1
        
        # Add edges
        if edge_index.size > 0:
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                adj[src, dst] = 1
        
        # Row-normalize
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        adj_norm = adj / row_sum
        
        return adj_norm


class QuantumGNN:
    """Graph Neural Network for quantum error prediction"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.layers = []
        
        # Build GNN layers
        in_dim = config.node_features
        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else 1
            self.layers.append(GraphConvolutionalLayer(
                in_dim, out_dim, config.dropout
            ))
            in_dim = config.hidden_dim
    
    def predict_error_probability(self, graph: QuantumGraph) -> np.ndarray:
        """Predict error probability for each qubit"""
        features = graph.node_features
        edge_index = graph.edge_index
        
        # Forward pass through all layers
        for layer in self.layers[:-1]:
            features = layer.forward(features, edge_index, graph.num_nodes)
        
        # Final layer for binary classification
        final_layer = self.layers[-1]
        probabilities = final_layer.forward(features, edge_index, graph.num_nodes)
        
        # Sigmoid activation for probability
        probabilities = 1 / (1 + np.exp(-probabilities))
        
        return probabilities.flatten()
    
    def predict_error_propagation(self, graph: QuantumGraph, 
                                  source_qubit: int) -> Dict:
        """Predict how errors propagate from a source qubit"""
        base_probs = self.predict_error_probability(graph)
        propagation_probs = graph.get_error_propagation_probability(source_qubit)
        
        # Combine base error probability with propagation
        combined_probs = 0.7 * base_probs + 0.3 * propagation_probs * base_probs[source_qubit]
        
        return {
            'base_error_prob': base_probs,
            'propagation_prob': propagation_probs,
            'combined_prob': combined_probs,
            'source_qubit': source_qubit
        }


def create_quantum_graph_from_telemetry(telemetry_data: Dict[int, np.ndarray],
                                        num_qubits: int) -> QuantumGraph:
    """Create a quantum graph from telemetry data"""
    graph = QuantumGraph(num_qubits)
    
    for qubit_id, features in telemetry_data.items():
        if 0 <= qubit_id < num_qubits:
            graph.update_node_features(qubit_id, features)
    
    return graph


def simulate_error_cascade(graph: QuantumGraph, initial_error_qubit: int,
                          threshold: float = 0.5) -> List[int]:
    """Simulate error cascade through the quantum system"""
    affected_qubits = [initial_error_qubit]
    current_probs = graph.get_error_propagation_probability(initial_error_qubit)
    
    # Iteratively find affected qubits
    for _ in range(5):
        new_affected = []
        for qubit in range(graph.num_nodes):
            if qubit not in affected_qubits and current_probs[qubit] > threshold:
                new_affected.append(qubit)
        
        if not new_affected:
            break
        
        affected_qubits.extend(new_affected)
        
        # Update probabilities based on newly affected qubits
        for qubit in new_affected:
            prop_probs = graph.get_error_propagation_probability(qubit)
            current_probs = np.maximum(current_probs, prop_probs * 0.8)
    
    return affected_qubits


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH NEURAL NETWORKS FOR QUANTUM ERROR MODELING")
    print("=" * 60)
    
    # Configuration
    config = GraphConfig(num_nodes=5, node_features=17)
    
    # Create quantum graph
    print("\nInitializing quantum graph...")
    graph = QuantumGraph(num_qubits=config.num_nodes)
    
    # Simulate realistic telemetry data
    np.random.seed(42)
    for i in range(config.num_nodes):
        features = np.random.randn(17) * 0.1
        # Add some degradation patterns
        if i == 2:  # Qubit 2 has higher error rate
            features[0] -= 0.5  # T1 degradation
            features[1] -= 0.3  # T2 degradation
            features[2] -= 0.4  # Gate fidelity
        graph.update_node_features(i, features)
    
    print(f"Graph created with {graph.num_nodes} qubits")
    print(f"Connectivity: Nearest neighbor")
    print(f"Edge count: {graph.edge_index.shape[1] if graph.edge_index.size > 0 else 0}")
    
    # Initialize GNN
    print("\nInitializing Graph Neural Network...")
    gnns = [QuantumGNN(config) for _ in range(3)]  # Ensemble of 3 GNNs
    
    # Predict error probabilities
    print("\nPredicting error probabilities...")
    all_predictions = []
    for gnn in gnns:
        preds = gnn.predict_error_probability(graph)
        all_predictions.append(preds)
    
    # Ensemble average
    ensemble_preds = np.mean(all_predictions, axis=0)
    
    print("\nError Probability by Qubit:")
    print("-" * 40)
    for i, prob in enumerate(ensemble_preds):
        risk_level = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"
        print(f"  Qubit {i}: {prob:.4f} [{risk_level}]")
    
    # Error propagation analysis
    print("\n" + "=" * 60)
    print("ERROR PROPAGATION ANALYSIS")
    print("=" * 60)
    
    source_qubit = 2  # Assume qubit 2 has an error
    gnn = gnns[0]
    propagation_result = gnn.predict_error_propagation(graph, source_qubit)
    
    print(f"\nSource: Qubit {source_qubit}")
    print("\nPropagation Probabilities:")
    for i, (base, prop, combined) in enumerate(zip(
        propagation_result['base_error_prob'],
        propagation_result['propagation_prob'],
        propagation_result['combined_prob']
    )):
        print(f"  Qubit {i}: Base={base:.4f}, Propagation={prop:.4f}, Combined={combined:.4f}")
    
    # Simulate error cascade
    print("\nSimulating error cascade...")
    cascade_qubits = simulate_error_cascade(graph, source_qubit, threshold=0.3)
    print(f"Affected qubits in cascade: {cascade_qubits}")
    print(f"Cascade size: {len(cascade_qubits)} / {config.num_nodes} qubits")
    
    # Topology optimization suggestion
    print("\n" + "=" * 60)
    print("TOPOLOGY OPTIMIZATION SUGGESTIONS")
    print("=" * 60)
    
    high_risk_qubits = np.where(ensemble_preds > 0.5)[0]
    if len(high_risk_qubits) > 0:
        print(f"\nHigh-risk qubits detected: {high_risk_qubits.tolist()}")
        print("Recommendations:")
        for qubit in high_risk_qubits:
            neighbors = np.where(graph.adjacency_matrix[qubit] > 0)[0]
            print(f"  - Qubit {qubit}: Consider isolating from neighbors {neighbors.tolist()}")
            print(f"    or migrating state to lower-connectivity qubit")
    else:
        print("\nNo high-risk qubits detected. System topology is stable.")
    
    print("\n" + "=" * 60)
    print("GNN ANALYSIS COMPLETE")
    print("=" * 60)
