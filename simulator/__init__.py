"""
Quantum Simulator for PQSS

Synthetic quantum data generation including:
- Noise simulator
- Decoherence simulator
- Multi-qubit system simulation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

# Try to import optional quantum computing libraries
try:
    import qiskit
    from qiskit import QuantumCircuit, Aer, execute
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Using simplified simulation backend.")


@dataclass
class SimulationConfig:
    """Configuration for quantum simulation."""
    
    num_qubits: int = 10
    base_t1: float = 100.0  # microseconds
    base_t2: float = 80.0   # microseconds
    base_gate_fidelity: float = 0.99
    temperature: float = 0.015  # Kelvin
    
    # Noise parameters
    white_noise_strength: float = 0.01
    thermal_noise_strength: float = 0.005
    crosstalk_strength: float = 0.01
    
    # Decoherence parameters
    decoherence_rate: float = 0.001
    environmental_perturbation: float = 0.002


class NoiseSimulator:
    """
    Simulates various types of quantum noise.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def generate_white_noise(self, size: int) -> np.ndarray:
        """Generate white noise samples."""
        return np.random.normal(0, self.config.white_noise_strength, size)
    
    def generate_thermal_noise(self, size: int) -> np.ndarray:
        """Generate thermal noise based on temperature."""
        # Thermal noise scales with temperature
        strength = self.config.thermal_noise_strength * (self.config.temperature / 0.015)
        return np.random.exponential(strength, size)
    
    def generate_crosstalk(self, source_qubit: int, target_qubits: List[int],
                           distance_matrix: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Generate crosstalk noise from one qubit affecting others.
        
        Parameters
        ----------
        source_qubit : int
            The qubit causing crosstalk
        target_qubits : List[int]
            Qubits potentially affected
        distance_matrix : Optional[np.ndarray]
            Matrix of distances between qubits
            
        Returns
        -------
        Dict[int, float]
            Crosstalk levels for each affected qubit
        """
        crosstalk_effects = {}
        
        for target in target_qubits:
            if target == source_qubit:
                continue
            
            # Base crosstalk
            effect = self.config.crosstalk_strength * np.random.exponential()
            
            # Reduce with distance if matrix provided
            if distance_matrix is not None:
                distance = distance_matrix[source_qubit, target]
                effect *= np.exp(-distance / 2.0)  # Exponential decay with distance
            
            crosstalk_effects[target] = effect
        
        return crosstalk_effects
    
    def generate_environmental_perturbation(self) -> float:
        """Generate random environmental perturbation."""
        return np.random.normal(0, self.config.environmental_perturbation)


class DecoherenceSimulator:
    """
    Simulates qubit decoherence processes.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.noise_sim = NoiseSimulator(config)
        
        # Track qubit states
        self.coherence_levels = np.ones(config.num_qubits)
        self.phase_coherence = np.ones(config.num_qubits)
        
    def apply_t1_relaxation(self, qubit_id: int, time_step: float) -> float:
        """
        Apply T1 relaxation (energy decay).
        
        Parameters
        ----------
        qubit_id : int
            Qubit identifier
        time_step : float
            Duration of the step
            
        Returns
        -------
        float
            Remaining coherence level
        """
        t1 = self.config.base_t1 * (1 + np.random.normal(0, 0.05))
        decay_factor = np.exp(-time_step / t1)
        
        self.coherence_levels[qubit_id] *= decay_factor
        
        # Add noise
        noise = self.noise_sim.generate_thermal_noise(1)[0]
        self.coherence_levels[qubit_id] -= noise * time_step
        self.coherence_levels[qubit_id] = max(0, min(1, self.coherence_levels[qubit_id]))
        
        return self.coherence_levels[qubit_id]
    
    def apply_t2_dephasing(self, qubit_id: int, time_step: float) -> float:
        """
        Apply T2 dephasing (phase coherence loss).
        
        Parameters
        ----------
        qubit_id : int
            Qubit identifier
        time_step : float
            Duration of the step
            
        Returns
        -------
        float
            Remaining phase coherence
        """
        t2 = self.config.base_t2 * (1 + np.random.normal(0, 0.05))
        decay_factor = np.exp(-time_step / t2)
        
        self.phase_coherence[qubit_id] *= decay_factor
        
        # Add noise
        noise = self.noise_sim.generate_white_noise(1)[0]
        self.phase_coherence[qubit_id] -= abs(noise) * time_step * 0.1
        self.phase_coherence[qubit_id] = max(0, min(1, self.phase_coherence[qubit_id]))
        
        return self.phase_coherence[qubit_id]
    
    def apply_gate_error(self, qubit_id: int) -> float:
        """
        Simulate gate operation error.
        
        Returns
        -------
        float
            Actual gate fidelity achieved
        """
        base_fidelity = self.config.base_gate_fidelity
        
        # Reduce fidelity based on current coherence
        coherence_factor = (self.coherence_levels[qubit_id] + 
                           self.phase_coherence[qubit_id]) / 2
        
        actual_fidelity = base_fidelity * coherence_factor
        
        # Add random variation
        actual_fidelity += np.random.normal(0, 0.01)
        
        return max(0.5, min(1.0, actual_fidelity))
    
    def check_decoherence_event(self, qubit_id: int, 
                                 threshold: float = 0.3) -> bool:
        """
        Check if a qubit has decohered.
        
        Parameters
        ----------
        qubit_id : int
            Qubit identifier
        threshold : float
            Coherence level below which decoherence occurs
            
        Returns
        -------
        bool
            True if decoherence event occurred
        """
        combined_coherence = (self.coherence_levels[qubit_id] * 0.6 +
                             self.phase_coherence[qubit_id] * 0.4)
        return combined_coherence < threshold
    
    def reset_qubit(self, qubit_id: int) -> None:
        """Reset a qubit to full coherence."""
        self.coherence_levels[qubit_id] = 1.0
        self.phase_coherence[qubit_id] = 1.0


class QuantumSystemSimulator:
    """
    Complete quantum system simulator for PQSS.
    
    Generates synthetic telemetry data and simulates
    multi-qubit quantum systems with realistic noise
    and decoherence patterns.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        if config is None:
            config = SimulationConfig()
        
        self.config = config
        self.decoherence_sim = DecoherenceSimulator(config)
        self.noise_sim = NoiseSimulator(config)
        
        # Time tracking
        self.current_time = 0
        self.time_step = 1.0  # microseconds
        
        # Operation tracking
        self.gate_counts = np.zeros(config.num_qubits, dtype=int)
        self.circuit_depths = np.zeros(config.num_qubits, dtype=int)
        
    def simulate_step(self) -> Dict[int, Dict]:
        """
        Advance simulation by one time step.
        
        Returns
        -------
        Dict[int, Dict]
            Telemetry data for each qubit
        """
        telemetry = {}
        
        for qubit_id in range(self.config.num_qubits):
            # Apply decoherence processes
            t1_remaining = self.decoherence_sim.apply_t1_relaxation(
                qubit_id, self.time_step
            )
            t2_remaining = self.decoherence_sim.apply_t2_dephasing(
                qubit_id, self.time_step
            )
            
            # Get gate fidelity
            gate_fid = self.decoherence_sim.apply_gate_error(qubit_id)
            
            # Generate noise metrics
            white_noise = self.noise_sim.generate_white_noise(1)[0]
            thermal_noise = self.noise_sim.generate_thermal_noise(1)[0]
            env_perturbation = self.noise_sim.generate_environmental_perturbation()
            
            # Generate crosstalk from other active qubits
            active_qubits = [i for i in range(self.config.num_qubits) 
                           if i != qubit_id and self.gate_counts[i] > 0]
            crosstalk_effects = self.noise_sim.generate_crosstalk(
                qubit_id, active_qubits
            )
            total_crosstalk = sum(crosstalk_effects.values()) if crosstalk_effects else 0
            
            # Compute derived metrics
            t1_time = self.config.base_t1 * t1_remaining
            t2_time = self.config.base_t2 * t2_remaining
            
            # Readout fidelity (slightly better than gate fidelity)
            readout_fid = min(1.0, gate_fid * 1.02)
            
            # Entanglement strength (depends on both coherences)
            entanglement = (t1_remaining * t2_remaining) ** 0.5
            
            # Error frequency (higher when coherence is low)
            error_freq = (1 - (t1_remaining + t2_remaining) / 2) * 0.1
            error_freq += abs(white_noise) * 0.01
            
            # Check for decoherence event
            is_decohering = self.decoherence_sim.check_decoherence_event(qubit_id)
            if is_decohering:
                error_freq *= 5  # Spike in errors
            
            telemetry[qubit_id] = {
                'timestamp': self.current_time,
                'qubit_id': qubit_id,
                
                # Hardware Metrics
                'temperature': self.config.temperature + np.random.normal(0, 0.001),
                'frequency_drift': np.random.normal(0, 100),
                'resonance_instability': np.random.normal(0, 50),
                'voltage_variation': np.random.normal(0, 0.001),
                
                # Quantum Metrics
                't1_relaxation_time': t1_time,
                't2_dephasing_time': t2_time,
                'gate_fidelity': gate_fid,
                'readout_fidelity': readout_fid,
                'entanglement_strength': entanglement,
                
                # Noise Metrics
                'white_noise_level': abs(white_noise),
                'thermal_noise_level': thermal_noise,
                'environmental_noise': abs(env_perturbation),
                'crosstalk_level': abs(total_crosstalk),
                
                # Operational Metrics
                'gate_count': int(self.gate_counts[qubit_id]),
                'circuit_depth': int(self.circuit_depths[qubit_id]),
                'qubit_utilization': min(1.0, self.gate_counts[qubit_id] / 1000),
                'error_frequency': error_freq,
                
                # Event flags
                'is_decohering': is_decohering,
            }
        
        # Advance time
        self.current_time += self.time_step
        
        return telemetry
    
    def apply_gate_operation(self, qubit_ids: List[int], 
                              gate_type: str = 'single') -> None:
        """
        Record a gate operation on specified qubits.
        
        Parameters
        ----------
        qubit_ids : List[int]
            Qubits involved in the gate
        gate_type : str
            Type of gate ('single', 'two_qubit', etc.)
        """
        for qid in qubit_ids:
            self.gate_counts[qid] += 1
            
            # Two-qubit gates increase circuit depth more
            if gate_type == 'two_qubit':
                self.circuit_depths[qid] += 2
            else:
                self.circuit_depths[qid] += 1
    
    def reset_qubit(self, qubit_id: int) -> None:
        """Reset a specific qubit."""
        self.decoherence_sim.reset_qubit(qubit_id)
        self.gate_counts[qubit_id] = 0
        self.circuit_depths[qubit_id] = 0
    
    def reset_system(self) -> None:
        """Reset the entire system."""
        for qid in range(self.config.num_qubits):
            self.reset_qubit(qid)
        self.current_time = 0
    
    def introduce_anomaly(self, qubit_id: int, 
                          anomaly_type: str = 'degradation',
                          severity: float = 0.5) -> None:
        """
        Introduce an anomaly for testing prediction models.
        
        Parameters
        ----------
        qubit_id : int
            Target qubit
        anomaly_type : str
            Type of anomaly ('degradation', 'noise_spike', 'frequency_drift')
        severity : float
            Severity level (0-1)
        """
        if anomaly_type == 'degradation':
            # Artificially reduce coherence
            self.decoherence_sim.coherence_levels[qubit_id] *= (1 - severity)
            self.decoherence_sim.phase_coherence[qubit_id] *= (1 - severity)
            
        elif anomaly_type == 'noise_spike':
            # Increase noise parameters temporarily
            self.config.white_noise_strength *= (1 + severity)
            self.config.thermal_noise_strength *= (1 + severity)
            
        elif anomaly_type == 'frequency_drift':
            # This would be handled in telemetry generation
            pass
    
    def run_simulation(self, num_steps: int, 
                       operations_schedule: Optional[List[Dict]] = None
                       ) -> Dict[int, List[Dict]]:
        """
        Run a complete simulation.
        
        Parameters
        ----------
        num_steps : int
            Number of time steps to simulate
        operations_schedule : Optional[List[Dict]]
            Schedule of gate operations to apply
            
        Returns
        -------
        Dict[int, List[Dict]]
            Full telemetry history for each qubit
        """
        history = {i: [] for i in range(self.config.num_qubits)}
        
        for step in range(num_steps):
            # Apply scheduled operations
            if operations_schedule and step < len(operations_schedule):
                op = operations_schedule[step]
                if 'qubits' in op:
                    gate_type = op.get('gate_type', 'single')
                    self.apply_gate_operation(op['qubits'], gate_type)
            
            # Simulate this time step
            telemetry = self.simulate_step()
            
            # Store results
            for qubit_id, data in telemetry.items():
                history[qubit_id].append(data)
        
        return history


def generate_synthetic_dataset(num_qubits: int = 10,
                                num_samples: int = 1000,
                                include_anomalies: bool = True
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset for training ML models.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system
    num_samples : int
        Number of samples to generate
    include_anomalies : bool
        Whether to include decoherence events
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature sequences and labels
    """
    config = SimulationConfig(num_qubits=num_qubits)
    simulator = QuantumSystemSimulator(config)
    
    feature_dim = 17  # Number of features per timestep
    sequence_length = 10
    
    all_features = []
    all_labels = []
    
    for sample_idx in range(num_samples):
        # Run simulation for this sample
        if include_anomalies and np.random.random() < 0.3:
            # Introduce anomaly in random qubit
            anomaly_qubit = np.random.randint(num_qubits)
            simulator.introduce_anomaly(anomaly_qubit, 'degradation', 
                                       np.random.uniform(0.3, 0.7))
        
        # Generate sequence
        history = simulator.run_simulation(sequence_length + 5)
        
        # Extract features and labels for each qubit
        for qubit_id in range(num_qubits):
            qubit_history = history[qubit_id]
            
            # Build feature sequence
            features = []
            for timestep_data in qubit_history[:sequence_length]:
                feature_vec = np.array([
                    timestep_data['temperature'],
                    timestep_data['frequency_drift'],
                    timestep_data['resonance_instability'],
                    timestep_data['voltage_variation'],
                    timestep_data['t1_relaxation_time'],
                    timestep_data['t2_dephasing_time'],
                    timestep_data['gate_fidelity'],
                    timestep_data['readout_fidelity'],
                    timestep_data['entanglement_strength'],
                    timestep_data['white_noise_level'],
                    timestep_data['thermal_noise_level'],
                    timestep_data['environmental_noise'],
                    timestep_data['crosstalk_level'],
                    timestep_data['gate_count'],
                    timestep_data['circuit_depth'],
                    timestep_data['qubit_utilization'],
                    timestep_data['error_frequency'],
                ])
                features.append(feature_vec)
            
            # Label: did decoherence occur in the next few steps?
            label = 0
            for future_data in qubit_history[sequence_length:]:
                if future_data['is_decohering']:
                    label = 1
                    break
            
            all_features.append(np.array(features))
            all_labels.append(label)
        
        # Reset periodically to avoid cumulative effects
        if sample_idx % 100 == 0:
            simulator.reset_system()
    
    return np.array(all_features), np.array(all_labels)


__all__ = [
    "SimulationConfig",
    "NoiseSimulator",
    "DecoherenceSimulator",
    "QuantumSystemSimulator",
    "generate_synthetic_dataset",
]
