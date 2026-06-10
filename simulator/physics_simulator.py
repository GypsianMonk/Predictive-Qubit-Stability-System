"""
Advanced Physics-Based Quantum Simulator for PQSS
Implements realistic noise models, decoherence dynamics, and multi-qubit interactions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings


class NoiseType(Enum):
    """Types of quantum noise"""
    DEPOLARIZING = "depolarizing"
    DEPHASING = "dephasing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    THERMAL = "thermal"
    CROSSTALK = "crosstalk"
    CONTROL_ERROR = "control_error"
    DRIFT = "drift"


@dataclass
class NoiseParameters:
    """Parameters for noise models"""
    # Depolarizing noise
    p_depolarizing: float = 0.001
    
    # Dephasing noise (T2)
    t2_time: float = 100e-6  # seconds
    
    # Amplitude damping (T1)
    t1_time: float = 150e-6  # seconds
    
    # Thermal noise
    temperature: float = 0.015  # Kelvin
    thermal_excitation_rate: float = 0.001
    
    # Crosstalk
    crosstalk_strength: float = 0.01
    crosstalk_range: int = 2  # number of neighboring qubits
    
    # Control errors
    gate_error_rate: float = 0.005
    pulse_amplitude_error: float = 0.01
    pulse_timing_error: float = 1e-9  # seconds
    
    # Drift parameters
    frequency_drift_rate: float = 100  # Hz/s
    amplitude_drift_rate: float = 0.001 / 3600  # per hour


@dataclass
class QubitState:
    """Represents the state of a single qubit"""
    # Bloch sphere coordinates
    x: float = 0.0
    y: float = 0.0
    z: float = 1.0  # Start in |0⟩ state
    
    # Coherence properties
    phase: float = 0.0
    population_0: float = 1.0
    population_1: float = 0.0
    
    # Error tracking
    accumulated_error: float = 0.0
    error_history: List[float] = field(default_factory=list)
    
    # Time tracking
    coherence_start_time: float = 0.0
    last_operation_time: float = 0.0


@dataclass
class GateOperation:
    """Represents a quantum gate operation"""
    gate_type: str
    target_qubits: List[int]
    duration: float
    ideal_matrix: Optional[np.ndarray] = None
    actual_matrix: Optional[np.ndarray] = None


class PhysicsBasedQuantumSimulator:
    """
    High-fidelity physics-based quantum simulator
    Implements realistic noise, decoherence, and multi-qubit dynamics
    """
    
    def __init__(self, 
                 num_qubits: int = 10,
                 noise_params: Optional[NoiseParameters] = None,
                 dt: float = 1e-9,  # 1 ns time step
                 seed: Optional[int] = None):
        """
        Initialize the quantum simulator
        
        Args:
            num_qubits: Number of qubits to simulate
            noise_params: Noise parameters
            dt: Simulation time step in seconds
            seed: Random seed for reproducibility
        """
        self.num_qubits = num_qubits
        self.noise_params = noise_params or NoiseParameters()
        self.dt = dt
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize qubit states
        self.qubit_states: List[QubitState] = [QubitState() for _ in range(num_qubits)]
        
        # Qubit frequencies (with slight variations)
        base_frequency = 5e9  # 5 GHz
        self.qubit_frequencies = base_frequency + np.random.uniform(-50e6, 50e6, num_qubits)
        
        # Coupling strengths between qubits
        self.coupling_matrix = self._initialize_coupling_matrix()
        
        # Time tracking
        self.current_time = 0.0
        self.operation_history: List[GateOperation] = []
        
        # Precursor signals for ML detection
        self.precursor_signals: Dict[str, List[List[float]]] = {
            'noise_fluctuations': [],
            'fidelity_drift': [],
            'frequency_instability': [],
            'entanglement_degradation': [],
            'phase_diffusion': []
        }
        
        # Ground truth labels for ML training
        self.decoherence_events: List[Tuple[float, int, str]] = []
        
    def _initialize_coupling_matrix(self) -> np.ndarray:
        """Initialize qubit coupling matrix based on topology"""
        coupling = np.zeros((self.num_qubits, self.num_qubits))
        
        # Simple linear chain topology with nearest neighbor coupling
        for i in range(self.num_qubits - 1):
            strength = self.noise_params.crosstalk_strength * np.exp(-0.5)
            coupling[i, i+1] = strength
            coupling[i+1, i] = strength
            
        return coupling
    
    def _apply_depolarizing_noise(self, qubit_idx: int) -> None:
        """Apply depolarizing noise to a qubit"""
        state = self.qubit_states[qubit_idx]
        p = self.noise_params.p_depolarizing
        
        if np.random.random() < p:
            # Random Pauli error
            error_type = np.random.choice(['x', 'y', 'z'])
            
            if error_type == 'x':
                state.x *= -1
            elif error_type == 'y':
                state.y *= -1
            else:
                state.z *= -1
                
            state.accumulated_error += p
            
    def _apply_dephasing_noise(self, qubit_idx: int, dt: float) -> None:
        """Apply dephasing noise (T2 decay)"""
        state = self.qubit_states[qubit_idx]
        t2 = self.noise_params.t2_time
        
        # Exponential decay of off-diagonal elements
        decay_factor = np.exp(-dt / t2)
        
        state.x *= decay_factor
        state.y *= decay_factor
        
        # Add stochastic phase diffusion
        phase_noise = np.random.normal(0, np.sqrt(dt / t2))
        state.phase += phase_noise
        
        # Track precursor signal
        phase_diffusion_rate = abs(phase_noise) / dt if dt > 0 else 0
        self.precursor_signals['phase_diffusion'].append([self.current_time, qubit_idx, phase_diffusion_rate])
        
    def _apply_amplitude_damping(self, qubit_idx: int, dt: float) -> None:
        """Apply amplitude damping (T1 decay)"""
        state = self.qubit_states[qubit_idx]
        t1 = self.noise_params.t1_time
        
        # Decay from |1⟩ to |0⟩
        decay_prob = 1 - np.exp(-dt / t1)
        
        if state.population_1 > 0 and np.random.random() < decay_prob:
            # Transition from |1⟩ to |0⟩
            transition_prob = decay_prob * state.population_1
            state.population_1 *= (1 - decay_prob)
            state.population_0 = 1 - state.population_1
            
            # Update Bloch sphere
            state.z = state.population_0 - state.population_1
            
        state.accumulated_error += decay_prob * state.population_1
        
    def _apply_thermal_noise(self, qubit_idx: int, dt: float) -> None:
        """Apply thermal excitation noise"""
        state = self.qubit_states[qubit_idx]
        temp = self.noise_params.temperature
        excitation_rate = self.noise_params.thermal_excitation_rate
        
        # Thermal excitation probability (simplified model)
        thermal_prob = excitation_rate * np.exp(-1/temp) * dt * 1e6
        
        if np.random.random() < thermal_prob:
            # Thermal excitation from |0⟩ to |1⟩
            state.population_0 -= thermal_prob
            state.population_1 += thermal_prob
            state.z = state.population_0 - state.population_1
            
    def _apply_crosstalk(self, qubit_idx: int) -> None:
        """Apply crosstalk from neighboring qubits"""
        state = self.qubit_states[qubit_idx]
        
        for j in range(self.num_qubits):
            if j != qubit_idx:
                distance = abs(j - qubit_idx)
                if distance <= self.noise_params.crosstalk_range:
                    coupling = self.coupling_matrix[qubit_idx, j]
                    
                    if coupling > 0:
                        neighbor_state = self.qubit_states[j]
                        
                        # Induced rotation due to neighbor's state
                        induced_rotation = coupling * neighbor_state.z
                        
                        # Apply small rotation to current qubit
                        angle = induced_rotation * self.dt * 1e9
                        state.x += angle * state.y
                        state.y -= angle * state.x
                        
                        # Normalize Bloch vector
                        norm = np.sqrt(state.x**2 + state.y**2 + state.z**2)
                        if norm > 0:
                            state.x /= norm
                            state.y /= norm
                            state.z /= norm
                            
    def _apply_control_errors(self, gate: GateOperation) -> None:
        """Apply control errors to gate operations"""
        # Pulse amplitude error
        amplitude_error = np.random.normal(0, self.noise_params.pulse_amplitude_error)
        
        # Pulse timing error
        timing_error = np.random.normal(0, self.noise_params.pulse_timing_error)
        
        # Frequency drift during operation
        freq_drift = self.noise_params.frequency_drift_rate * gate.duration
        
        # Update gate's actual matrix if ideal matrix exists
        if gate.ideal_matrix is not None:
            # Simplified error model: scale and add noise
            error_matrix = gate.ideal_matrix * (1 + amplitude_error)
            error_matrix += np.random.normal(0, 0.01, gate.ideal_matrix.shape)
            gate.actual_matrix = error_matrix
            
        # Track precursor signals
        self.precursor_signals['fidelity_drift'].append([
            self.current_time, 
            gate.target_qubits[0] if gate.target_qubits else 0,
            abs(amplitude_error)
        ])
        
    def _track_frequency_instability(self) -> None:
        """Track frequency instability as precursor signal"""
        for i in range(self.num_qubits):
            # Simulate frequency drift
            drift = self.noise_params.frequency_drift_rate * self.dt
            measured_freq = self.qubit_frequencies[i] + np.random.normal(0, drift)
            
            self.precursor_signals['frequency_instability'].append([
                self.current_time, i, abs(measured_freq - self.qubit_frequencies[i])
            ])
            
    def apply_gate(self, gate: GateOperation) -> bool:
        """
        Apply a quantum gate with realistic noise and errors
        
        Returns:
            True if gate applied successfully, False if decoherence occurred
        """
        self.operation_history.append(gate)
        
        # Apply control errors
        self._apply_control_errors(gate)
        
        # Simulate gate evolution with time steps
        gate_steps = int(gate.duration / self.dt)
        
        for step in range(gate_steps):
            # Evolve each affected qubit
            for qubit_idx in gate.target_qubits:
                if qubit_idx >= self.num_qubits:
                    continue
                    
                state = self.qubit_states[qubit_idx]
                
                # Apply ideal gate evolution (simplified)
                if gate.gate_type == 'X':
                    # Rotation around X axis
                    angle = np.pi * self.dt / gate.duration
                    old_y, old_z = state.y, state.z
                    state.y = old_y * np.cos(angle) - old_z * np.sin(angle)
                    state.z = old_y * np.sin(angle) + old_z * np.cos(angle)
                    
                elif gate.gate_type == 'Y':
                    # Rotation around Y axis
                    angle = np.pi * self.dt / gate.duration
                    old_x, old_z = state.x, state.z
                    state.x = old_x * np.cos(angle) + old_z * np.sin(angle)
                    state.z = -old_x * np.sin(angle) + old_z * np.cos(angle)
                    
                elif gate.gate_type == 'Z':
                    # Rotation around Z axis
                    angle = np.pi * self.dt / gate.duration
                    old_x, old_y = state.x, state.y
                    state.x = old_x * np.cos(angle) - old_y * np.sin(angle)
                    state.y = old_x * np.sin(angle) + old_y * np.cos(angle)
                    
                elif gate.gate_type == 'H':
                    # Hadamard (simplified evolution)
                    angle = np.pi/2 * self.dt / gate.duration
                    old_x, old_z = state.x, state.z
                    state.x = old_x * np.cos(angle) + old_z * np.sin(angle)
                    state.z = old_x * np.sin(angle) + old_z * np.cos(angle)
                
                # Apply noise processes
                self._apply_dephasing_noise(qubit_idx, self.dt)
                self._apply_amplitude_damping(qubit_idx, self.dt)
                self._apply_thermal_noise(qubit_idx, self.dt)
                self._apply_crosstalk(qubit_idx)
                
                # Check for decoherence event
                if state.accumulated_error > 0.1:  # Threshold for decoherence
                    self.decoherence_events.append((
                        self.current_time, 
                        qubit_idx, 
                        f"Decoherence during {gate.gate_type} gate"
                    ))
                    return False
                    
            # Apply depolarizing noise periodically
            if step % 10 == 0:
                for qubit_idx in gate.target_qubits:
                    if qubit_idx < self.num_qubits:
                        self._apply_depolarizing_noise(qubit_idx)
                        
            self.current_time += self.dt
            
        # Track frequency instability
        self._track_frequency_instability()
        
        # Update last operation time
        for qubit_idx in gate.target_qubits:
            if qubit_idx < self.num_qubits:
                self.qubit_states[qubit_idx].last_operation_time = self.current_time
                
        return True
    
    def measure_qubit(self, qubit_idx: int) -> int:
        """
        Measure a qubit in the computational basis
        
        Returns:
            Measurement result (0 or 1)
        """
        if qubit_idx >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit_idx} out of range")
            
        state = self.qubit_states[qubit_idx]
        
        # Calculate measurement probability
        prob_1 = state.population_1
        
        # Perform measurement
        result = 1 if np.random.random() < prob_1 else 0
        
        # Collapse the state
        if result == 0:
            state.x = 0
            state.y = 0
            state.z = 1
            state.population_0 = 1
            state.population_1 = 0
        else:
            state.x = 0
            state.y = 0
            state.z = -1
            state.population_0 = 0
            state.population_1 = 1
            
        return result
    
    def get_state_tomography(self) -> Dict[int, Dict]:
        """
        Get full state tomography for all qubits
        
        Returns:
            Dictionary containing state information for each qubit
        """
        tomography = {}
        
        for i in range(self.num_qubits):
            state = self.qubit_states[i]
            
            # Calculate fidelity with ideal state
            ideal_z = 1.0  # Assuming ideal state is |0⟩
            fidelity = (1 + state.z * ideal_z) / 2
            
            tomography[i] = {
                'bloch_vector': [state.x, state.y, state.z],
                'population_0': state.population_0,
                'population_1': state.population_1,
                'phase': state.phase,
                'fidelity': fidelity,
                'accumulated_error': state.accumulated_error,
                'coherence_time': self.current_time - state.coherence_start_time,
                't1_effective': self.noise_params.t1_time,
                't2_effective': self.noise_params.t2_time
            }
            
        return tomography
    
    def get_precursor_features(self, window_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Extract precursor features for ML prediction
        
        Args:
            window_size: Number of recent time steps to consider
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        for signal_name, signal_data in self.precursor_signals.items():
            if len(signal_data) == 0:
                features[signal_name] = np.array([])
                continue
                
            # Get recent data
            recent_data = signal_data[-window_size:]
            
            # Convert to array
            data_array = np.array([d[2] for d in recent_data])  # Just the values
            
            # Compute statistics
            if len(data_array) > 0:
                features[signal_name] = np.array([
                    np.mean(data_array),
                    np.std(data_array),
                    np.max(data_array),
                    np.min(data_array),
                    np.gradient(data_array).mean(),  # Trend
                    np.abs(np.gradient(data_array)).max()  # Max rate of change
                ])
            else:
                features[signal_name] = np.zeros(6)
                
        return features
    
    def reset(self) -> None:
        """Reset the simulator to initial state"""
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.qubit_states = [QubitState() for _ in range(self.num_qubits)]
        self.current_time = 0.0
        self.operation_history = []
        self.decoherence_events = []
        
        for key in self.precursor_signals:
            self.precursor_signals[key] = []
            
    def run_random_circuit(self, 
                          num_gates: int = 50,
                          gate_types: List[str] = ['X', 'Y', 'Z', 'H'],
                          verbose: bool = False) -> Dict:
        """
        Run a random quantum circuit for testing
        
        Returns:
            Results dictionary
        """
        success_count = 0
        failure_count = 0
        
        for i in range(num_gates):
            # Random gate selection
            gate_type = np.random.choice(gate_types)
            target = np.random.randint(0, self.num_qubits)
            
            # Random gate duration (10-100 ns)
            duration = np.random.uniform(10e-9, 100e-9)
            
            gate = GateOperation(
                gate_type=gate_type,
                target_qubits=[target],
                duration=duration
            )
            
            success = self.apply_gate(gate)
            
            if success:
                success_count += 1
            else:
                failure_count += 1
                if verbose:
                    print(f"Decoherence event at gate {i}: {gate_type} on qubit {target}")
                    
        # Final measurements
        measurements = [self.measure_qubit(i) for i in range(self.num_qubits)]
        
        return {
            'total_gates': num_gates,
            'successful_gates': success_count,
            'failed_gates': failure_count,
            'success_rate': success_count / num_gates if num_gates > 0 else 0,
            'measurements': measurements,
            'final_tomography': self.get_state_tomography(),
            'precursor_features': self.get_precursor_features(),
            'decoherence_events': self.decoherence_events
        }


def generate_training_dataset(num_samples: int = 1000,
                             num_qubits: int = 5,
                             gates_per_sample: int = 100,
                             seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training dataset for ML models
    
    Args:
        num_samples: Number of samples to generate
        num_qubits: Number of qubits per sample
        gates_per_sample: Number of gates per sample
        seed: Random seed
        
    Returns:
        X: Feature array (num_samples, num_features)
        y: Labels (num_samples,) - 1 if decoherence occurred, 0 otherwise
    """
    if seed is not None:
        np.random.seed(seed)
        
    X_list = []
    y_list = []
    
    simulator = PhysicsBasedQuantumSimulator(num_qubits=num_qubits, seed=seed)
    
    for sample_idx in range(num_samples):
        # Reset simulator with different noise parameters for variety
        noise_params = NoiseParameters(
            t1_time=np.random.uniform(50e-6, 200e-6),
            t2_time=np.random.uniform(30e-6, 150e-6),
            p_depolarizing=np.random.uniform(0.0001, 0.01),
            crosstalk_strength=np.random.uniform(0.001, 0.05),
            gate_error_rate=np.random.uniform(0.001, 0.02)
        )
        
        simulator.noise_params = noise_params
        simulator.reset()
        
        # Run random circuit
        results = simulator.run_random_circuit(num_gates=gates_per_sample)
        
        # Extract features
        features = results['precursor_features']
        feature_vector = np.concatenate([features[key] for key in sorted(features.keys())])
        
        # Label: 1 if any decoherence occurred, 0 otherwise
        label = 1 if results['failed_gates'] > 0 else 0
        
        X_list.append(feature_vector)
        y_list.append(label)
        
        if (sample_idx + 1) % 100 == 0:
            print(f"Generated {sample_idx + 1}/{num_samples} samples")
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    
    return X, y


if __name__ == "__main__":
    # Demo the physics-based simulator
    print("=" * 60)
    print("Physics-Based Quantum Simulator Demo")
    print("=" * 60)
    
    # Create simulator
    sim = PhysicsBasedQuantumSimulator(num_qubits=5, seed=42)
    
    print(f"\nInitialized {sim.num_qubits} qubit simulator")
    print(f"T1 time: {sim.noise_params.t1_time*1e6:.1f} μs")
    print(f"T2 time: {sim.noise_params.t2_time*1e6:.1f} μs")
    
    # Run random circuit
    print("\nRunning random circuit with 100 gates...")
    results = sim.run_random_circuit(num_gates=100, verbose=True)
    
    print(f"\nResults:")
    print(f"  Success rate: {results['success_rate']*100:.1f}%")
    print(f"  Successful gates: {results['successful_gates']}")
    print(f"  Failed gates: {results['failed_gates']}")
    print(f"  Decoherence events: {len(results['decoherence_events'])}")
    
    # Show final state tomography for first qubit
    print(f"\nFinal state tomography for qubit 0:")
    tomo = results['final_tomography'][0]
    print(f"  Bloch vector: [{tomo['bloch_vector'][0]:.3f}, {tomo['bloch_vector'][1]:.3f}, {tomo['bloch_vector'][2]:.3f}]")
    print(f"  Fidelity: {tomo['fidelity']*100:.1f}%")
    print(f"  Accumulated error: {tomo['accumulated_error']:.4f}")
    
    # Show precursor features
    print(f"\nPrecursor features detected:")
    features = results['precursor_features']
    for name, feat in features.items():
        if len(feat) > 0:
            print(f"  {name}: mean={feat[0]:.4f}, std={feat[1]:.4f}, max_rate={feat[5]:.4f}")
    
    # Generate small training dataset
    print("\n" + "=" * 60)
    print("Generating training dataset...")
    print("=" * 60)
    
    X, y = generate_training_dataset(num_samples=100, num_qubits=5, gates_per_sample=50, seed=42)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Positive samples: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
