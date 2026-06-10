"""
Digital Twin Architecture for PQSS

Virtual counterparts for physical qubits that track:
- State evolution
- Error trends
- Noise exposure
- Stability probability

Benefits:
- Faster prediction
- Safer experimentation
- Real-time diagnostics
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy


@dataclass
class DigitalTwinState:
    """Represents the current state of a qubit's digital twin."""
    
    qubit_id: int
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # State tracking
    current_state_vector: Optional[np.ndarray] = None
    historical_states: List[np.ndarray] = field(default_factory=list)
    
    # Error tracking
    cumulative_error_rate: float = 0.0
    error_trend: float = 0.0  # Rate of change
    error_history: List[float] = field(default_factory=list)
    
    # Noise exposure
    total_noise_exposure: float = 0.0
    noise_profile: Dict[str, float] = field(default_factory=dict)
    
    # Stability metrics
    stability_probability: float = 1.0
    predicted_lifetime: float = float('inf')
    degradation_rate: float = 0.0
    
    # Metadata
    update_count: int = 0
    intervention_count: int = 0


class DigitalTwin:
    """
    Digital twin for a single qubit.
    
    Maintains a virtual representation that mirrors the physical qubit
    and enables predictive analysis without risking actual quantum states.
    """
    
    def __init__(self, qubit_id: int):
        self.qubit_id = qubit_id
        self.state = DigitalTwinState(qubit_id=qubit_id)
        
        # Model parameters (learned over time)
        self.degradation_model = None
        self.noise_sensitivity = {}
        
    def update(self, telemetry_data) -> None:
        """
        Update the digital twin with new telemetry data.
        
        Parameters
        ----------
        telemetry_data : TelemetryData
            Latest measurements from the physical qubit
        """
        # Update timestamp
        self.state.last_updated = datetime.now()
        self.state.update_count += 1
        
        # Store state vector if available
        if hasattr(telemetry_data, 'to_feature_vector'):
            feature_vec = telemetry_data.to_feature_vector()
            self.state.historical_states.append(feature_vec.copy())
            
            # Keep only recent history (last 100 measurements)
            if len(self.state.historical_states) > 100:
                self.state.historical_states.pop(0)
        
        # Update error tracking
        error_rate = getattr(telemetry_data, 'error_frequency', 0)
        self.state.error_history.append(error_rate)
        
        if len(self.state.error_history) > 2:
            # Compute error trend
            recent = self.state.error_history[-5:]
            self.state.error_trend = (recent[-1] - recent[0]) / len(recent)
        
        # Update cumulative error
        self.state.cumulative_error_rate = np.mean(self.state.error_history)
        
        # Update noise profile
        noise_metrics = {
            'white_noise': getattr(telemetry_data, 'white_noise_level', 0),
            'thermal_noise': getattr(telemetry_data, 'thermal_noise_level', 0),
            'environmental_noise': getattr(telemetry_data, 'environmental_noise', 0),
            'crosstalk': getattr(telemetry_data, 'crosstalk_level', 0),
        }
        
        for noise_type, level in noise_metrics.items():
            self.state.total_noise_exposure += level
            if noise_type not in self.state.noise_profile:
                self.state.noise_profile[noise_type] = []
            self.state.noise_profile[noise_type].append(level)
            
            # Keep rolling average
            if len(self.state.noise_profile[noise_type]) > 50:
                self.state.noise_profile[noise_type].pop(0)
        
        # Learn noise sensitivity
        self._update_noise_sensitivity(noise_metrics, error_rate)
        
        # Update stability probability
        self._update_stability_probability(telemetry_data)
        
        # Estimate remaining lifetime
        self._estimate_lifetime()
    
    def _update_noise_sensitivity(self, noise_metrics: Dict[str, float],
                                   error_rate: float) -> None:
        """Learn how sensitive this qubit is to different noise types."""
        for noise_type, level in noise_metrics.items():
            if noise_type not in self.noise_sensitivity:
                self.noise_sensitivity[noise_type] = []
            
            # Store correlation between noise level and errors
            if level > 0:
                sensitivity = error_rate / (level + 1e-6)
                self.noise_sensitivity[noise_type].append(sensitivity)
                
                # Keep recent values
                if len(self.noise_sensitivity[noise_type]) > 100:
                    self.noise_sensitivity[noise_type].pop(0)
    
    def _update_stability_probability(self, telemetry_data) -> None:
        """Update the probability that the qubit remains stable."""
        # Use telemetry metrics to estimate stability
        gate_fid = getattr(telemetry_data, 'gate_fidelity', 1.0)
        t1 = getattr(telemetry_data, 't1_relaxation_time', 100)
        t2 = getattr(telemetry_data, 't2_dephasing_time', 80)
        
        # Base stability on fidelity and coherence times
        fidelity_factor = gate_fid
        coherence_factor = min(1.0, t2 / (2 * t1)) if t1 > 0 else 0.5
        
        # Factor in error trend
        trend_factor = max(0, 1 - abs(self.state.error_trend) * 10)
        
        # Combined stability probability
        self.state.stability_probability = (
            0.4 * fidelity_factor +
            0.3 * coherence_factor +
            0.3 * trend_factor
        )
    
    def _estimate_lifetime(self) -> None:
        """Estimate remaining useful lifetime based on degradation rate."""
        if len(self.state.error_history) < 5:
            self.state.predicted_lifetime = float('inf')
            return
        
        # Fit linear degradation model
        errors = np.array(self.state.error_history[-20:])  # Last 20 measurements
        
        if len(errors) > 1:
            # Simple linear regression
            x = np.arange(len(errors))
            slope = np.polyfit(x, errors, 1)[0]
            self.state.degradation_rate = slope
            
            # Estimate when error rate will exceed threshold
            if slope > 0:
                current_error = errors[-1]
                threshold = 0.1  # Error rate threshold for failure
                steps_to_failure = (threshold - current_error) / slope
                self.state.predicted_lifetime = max(1, steps_to_failure)
            else:
                self.state.predicted_lifetime = float('inf')
    
    def predict_future_state(self, steps_ahead: int = 10) -> Dict[str, float]:
        """
        Predict future state based on current trends.
        
        Parameters
        ----------
        steps_ahead : int
            Number of time steps to predict
            
        Returns
        -------
        Dict[str, float]
            Predicted metrics
        """
        predictions = {
            'error_rate': None,
            'stability_probability': None,
            'predicted_lifetime': None,
        }
        
        if self.state.degradation_rate != 0:
            current_error = self.state.error_history[-1] if self.state.error_history else 0
            predicted_error = current_error + self.state.degradation_rate * steps_ahead
            predictions['error_rate'] = max(0, predicted_error)
        
        # Project stability probability
        if self.state.predicted_lifetime != float('inf'):
            decay_factor = steps_ahead / self.state.predicted_lifetime
            predictions['stability_probability'] = max(
                0, self.state.stability_probability * (1 - decay_factor)
            )
        else:
            predictions['stability_probability'] = self.state.stability_probability
        
        predictions['predicted_lifetime'] = max(
            0, self.state.predicted_lifetime - steps_ahead
        )
        
        return predictions
    
    def record_intervention(self) -> None:
        """Record that an intervention was performed on this qubit."""
        self.state.intervention_count += 1
        
        # Interventions may reset some metrics
        if len(self.state.error_history) > 5:
            # Clear recent error history to reflect fresh start
            self.state.error_history = self.state.error_history[:-5]


class DigitalTwinManager:
    """
    Manages digital twins for all qubits in the system.
    
    Provides centralized access to twin states and enables
    system-wide analysis and experimentation.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.twins: Dict[int, DigitalTwin] = {
            i: DigitalTwin(i) for i in range(num_qubits)
        }
        
    def update_twin(self, qubit_id: int, telemetry_data) -> None:
        """Update a specific qubit's twin with new data."""
        if qubit_id not in self.twins:
            raise ValueError(f"Invalid qubit ID: {qubit_id}")
        
        self.twins[qubit_id].update(telemetry_data)
    
    def get_twin(self, qubit_id: int) -> DigitalTwin:
        """Get the digital twin for a qubit."""
        if qubit_id not in self.twins:
            raise ValueError(f"Invalid qubit ID: {qubit_id}")
        
        return self.twins[qubit_id]
    
    def get_system_health(self) -> Dict:
        """
        Get overall system health summary.
        
        Returns
        -------
        Dict
            System-wide health metrics
        """
        stability_probs = [
            twin.state.stability_probability for twin in self.twins.values()
        ]
        
        lifetimes = [
            twin.state.predicted_lifetime for twin in self.twins.values()
            if twin.state.predicted_lifetime != float('inf')
        ]
        
        error_rates = [
            twin.state.cumulative_error_rate for twin in self.twins.values()
        ]
        
        return {
            'average_stability': np.mean(stability_probs),
            'min_stability': np.min(stability_probs),
            'max_stability': np.max(stability_probs),
            'average_predicted_lifetime': np.mean(lifetimes) if lifetimes else float('inf'),
            'average_error_rate': np.mean(error_rates),
            'total_interventions': sum(
                twin.state.intervention_count for twin in self.twins.values()
            ),
            'qubits_at_risk': sum(1 for p in stability_probs if p < 0.7),
        }
    
    def run_simulation(self, scenario: str, 
                       steps: int = 100) -> Dict[int, List[Dict]]:
        """
        Run a what-if simulation on digital twins.
        
        This allows testing intervention strategies without
        affecting real quantum hardware.
        
        Parameters
        ----------
        scenario : str
            Scenario type ('increased_noise', 'pulse_optimization', etc.)
        steps : int
            Number of simulation steps
            
        Returns
        -------
        Dict[int, List[Dict]]
            Simulation results for each qubit
        """
        results = {i: [] for i in range(self.num_qubits)}
        
        # Save current state
        saved_states = {
            i: deepcopy(twin.state) for i, twin in self.twins.items()
        }
        
        try:
            for step in range(steps):
                for qubit_id, twin in self.twins.items():
                    # Apply scenario effects
                    if scenario == 'increased_noise':
                        # Simulate increasing noise
                        twin.state.total_noise_exposure *= 1.01
                        twin.state.stability_probability *= 0.995
                    
                    elif scenario == 'optimal_conditions':
                        # Simulate ideal environment
                        twin.state.stability_probability = min(1.0, 
                            twin.state.stability_probability * 1.001)
                    
                    # Record state
                    results[qubit_id].append({
                        'step': step,
                        'stability': twin.state.stability_probability,
                        'error_rate': twin.state.cumulative_error_rate,
                    })
            
            return results
        
        finally:
            # Restore original state
            for qubit_id, saved_state in saved_states.items():
                self.twins[qubit_id].state = saved_state


__all__ = [
    "DigitalTwinState",
    "DigitalTwin",
    "DigitalTwinManager",
]
