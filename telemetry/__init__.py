"""
Telemetry Acquisition Layer for PQSS

Collects and processes quantum hardware telemetry data including:
- Hardware metrics (temperature, frequency drift, voltage)
- Quantum metrics (T1, T2, gate fidelity, readout fidelity)
- Noise metrics (white noise, thermal noise, crosstalk)
- Operational metrics (gate count, circuit depth, qubit utilization)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class TelemetryData:
    """Container for quantum telemetry measurements."""
    
    timestamp: datetime
    qubit_id: int
    
    # Hardware Metrics
    temperature: float  # Kelvin
    frequency_drift: float  # Hz
    resonance_instability: float  # Hz
    voltage_variation: float  # V
    
    # Quantum Metrics
    t1_relaxation_time: float  # microseconds
    t2_dephasing_time: float  # microseconds
    gate_fidelity: float  # 0-1
    readout_fidelity: float  # 0-1
    entanglement_strength: float  # 0-1
    
    # Noise Metrics
    white_noise_level: float
    thermal_noise_level: float
    environmental_noise: float
    crosstalk_level: float
    
    # Operational Metrics
    gate_count: int
    circuit_depth: int
    qubit_utilization: float  # 0-1
    error_frequency: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert telemetry data to a feature vector for ML models."""
        return np.array([
            self.temperature,
            self.frequency_drift,
            self.resonance_instability,
            self.voltage_variation,
            self.t1_relaxation_time,
            self.t2_dephasing_time,
            self.gate_fidelity,
            self.readout_fidelity,
            self.entanglement_strength,
            self.white_noise_level,
            self.thermal_noise_level,
            self.environmental_noise,
            self.crosstalk_level,
            self.gate_count,
            self.circuit_depth,
            self.qubit_utilization,
            self.error_frequency,
        ])


class TelemetryCollector:
    """
    Collects and manages telemetry data from quantum hardware.
    
    In a real implementation, this would interface with actual
    quantum hardware control systems. For simulation purposes,
    it can generate synthetic telemetry data.
    """
    
    def __init__(self, num_qubits: int = 10):
        self.num_qubits = num_qubits
        self.telemetry_history: Dict[int, List[TelemetryData]] = {
            i: [] for i in range(num_qubits)
        }
        self.current_state: Dict[int, TelemetryData] = {}
        
    def collect(self, qubit_id: int, data: TelemetryData) -> None:
        """Store new telemetry data for a qubit."""
        if qubit_id >= self.num_qubits:
            raise ValueError(f"Invalid qubit ID: {qubit_id}")
        
        self.telemetry_history[qubit_id].append(data)
        self.current_state[qubit_id] = data
        
    def get_history(self, qubit_id: int, 
                    last_n: Optional[int] = None) -> List[TelemetryData]:
        """Retrieve telemetry history for a qubit."""
        history = self.telemetry_history[qubit_id]
        if last_n is not None:
            return history[-last_n:]
        return history
    
    def get_current_state(self, qubit_id: int) -> Optional[TelemetryData]:
        """Get the most recent telemetry data for a qubit."""
        return self.current_state.get(qubit_id)
    
    def get_all_features(self, qubit_id: int, 
                         last_n: int = 10) -> Optional[np.ndarray]:
        """
        Get feature matrix for ML models.
        
        Returns a 2D array of shape (last_n, num_features).
        """
        history = self.get_history(qubit_id, last_n)
        if len(history) == 0:
            return None
        
        return np.array([data.to_feature_vector() for data in history])


def generate_synthetic_telemetry(qubit_id: int, 
                                  decoherence_probability: float = 0.1
                                  ) -> TelemetryData:
    """
    Generate synthetic telemetry data for simulation purposes.
    
    Parameters
    ----------
    qubit_id : int
        The qubit identifier
    decoherence_probability : float
        Probability that the qubit is approaching decoherence
        
    Returns
    -------
    TelemetryData
        Synthetic telemetry measurement
    """
    # Base values for stable qubit
    base_t1 = 100.0  # microseconds
    base_t2 = 80.0   # microseconds
    base_fidelity = 0.99
    
    # Introduce degradation if approaching decoherence
    degradation_factor = 1.0 - (decoherence_probability * 0.3)
    
    return TelemetryData(
        timestamp=datetime.now(),
        qubit_id=qubit_id,
        
        # Hardware Metrics (with some noise)
        temperature=0.015 + np.random.normal(0, 0.001),  # ~15 mK
        frequency_drift=np.random.normal(0, 100),  # Hz
        resonance_instability=np.random.normal(0, 50),  # Hz
        voltage_variation=np.random.normal(0, 0.001),  # V
        
        # Quantum Metrics (degraded if approaching decoherence)
        t1_relaxation_time=base_t1 * degradation_factor + np.random.normal(0, 5),
        t2_dephasing_time=base_t2 * degradation_factor + np.random.normal(0, 5),
        gate_fidelity=max(0.5, base_fidelity * degradation_factor + np.random.normal(0, 0.01)),
        readout_fidelity=max(0.5, base_fidelity + np.random.normal(0, 0.01)),
        entanglement_strength=max(0.0, 0.95 * degradation_factor + np.random.normal(0, 0.05)),
        
        # Noise Metrics (increased if approaching decoherence)
        white_noise_level=np.random.exponential(0.01),
        thermal_noise_level=np.random.exponential(0.005) * (1 + decoherence_probability),
        environmental_noise=np.random.exponential(0.002),
        crosstalk_level=np.random.exponential(0.01) * (1 + decoherence_probability * 0.5),
        
        # Operational Metrics
        gate_count=np.random.randint(10, 1000),
        circuit_depth=np.random.randint(5, 100),
        qubit_utilization=np.random.uniform(0.1, 0.9),
        error_frequency=np.random.exponential(0.01) * (1 + decoherence_probability * 2),
    )


__all__ = [
    "TelemetryData",
    "TelemetryCollector",
    "generate_synthetic_telemetry",
]
