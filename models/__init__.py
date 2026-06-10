"""
Predictive AI Engine for PQSS

Machine learning models for predicting qubit decoherence:
- LSTM Networks for temporal pattern recognition
- Transformer models for long-range dependencies
- Temporal Convolution Networks for efficiency
- Graph Neural Networks for quantum topology
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DecoherenceRisk(Enum):
    """Risk levels for qubit decoherence."""
    STABLE = "stable"           # PSS 90-100
    WARNING = "warning"         # PSS 70-89
    CRITICAL = "critical"       # PSS 50-69
    IMMEDIATE = "immediate"     # PSS < 50


class PredictiveStabilityScore:
    """
    Computes the Predictive Stability Score (PSS) for a qubit.
    
    The PSS is a composite metric ranging from 0-100 that indicates
    the current stability state of a qubit and likelihood of 
    impending decoherence.
    """
    
    def __init__(self):
        # Weights for different metric categories
        self.weights = {
            't1_t2_ratio': 0.15,
            'gate_fidelity': 0.20,
            'readout_fidelity': 0.15,
            'entanglement': 0.15,
            'noise_levels': 0.20,
            'error_frequency': 0.15,
        }
        
    def compute(self, feature_vector: np.ndarray) -> Tuple[float, DecoherenceRisk]:
        """
        Compute PSS from a telemetry feature vector.
        
        Parameters
        ----------
        feature_vector : np.ndarray
            Array of telemetry features in expected order
            
        Returns
        -------
        Tuple[float, DecoherenceRisk]
            PSS score (0-100) and risk classification
        """
        # Extract features (matching TelemetryData.to_feature_vector order)
        t1 = feature_vector[4]
        t2 = feature_vector[5]
        gate_fid = feature_vector[6]
        readout_fid = feature_vector[7]
        entanglement = feature_vector[8]
        white_noise = feature_vector[9]
        thermal_noise = feature_vector[10]
        env_noise = feature_vector[11]
        crosstalk = feature_vector[12]
        error_freq = feature_vector[16]
        
        # Normalize and score each component
        
        # T1/T2 ratio (ideal is close to theoretical max, typically T2 <= 2*T1)
        t1_t2_score = max(0, min(1, (t2 / (2 * t1)) * 100)) if t1 > 0 else 0
        
        # Gate fidelity (already 0-1, scale to 0-100)
        gate_score = gate_fid * 100
        
        # Readout fidelity
        readout_score = readout_fid * 100
        
        # Entanglement strength
        entanglement_score = entanglement * 100
        
        # Noise levels (lower is better)
        total_noise = white_noise + thermal_noise + env_noise + crosstalk
        noise_score = max(0, 100 - (total_noise * 1000))  # Scale appropriately
        
        # Error frequency (lower is better)
        error_score = max(0, 100 - (error_freq * 500))
        
        # Compute weighted sum
        pss = (
            self.weights['t1_t2_ratio'] * t1_t2_score +
            self.weights['gate_fidelity'] * gate_score +
            self.weights['readout_fidelity'] * readout_score +
            self.weights['entanglement'] * entanglement_score +
            self.weights['noise_levels'] * noise_score +
            self.weights['error_frequency'] * error_score
        )
        
        # Clamp to 0-100
        pss = max(0, min(100, pss))
        
        # Classify risk
        risk = self._classify_risk(pss)
        
        return pss, risk
    
    def _classify_risk(self, pss: float) -> DecoherenceRisk:
        """Classify risk level based on PSS."""
        if pss >= 90:
            return DecoherenceRisk.STABLE
        elif pss >= 70:
            return DecoherenceRisk.WARNING
        elif pss >= 50:
            return DecoherenceRisk.CRITICAL
        else:
            return DecoherenceRisk.IMMEDIATE


class DecoherencePredictor:
    """
    Base class for decoherence prediction models.
    
    Subclasses implement specific ML architectures:
    - LSTMDecoherencePredictor
    - TransformerDecoherencePredictor
    - TCNDecoherencePredictor
    - GNNDecoherencePredictor
    """
    
    def __init__(self, num_qubits: int, sequence_length: int = 10):
        self.num_qubits = num_qubits
        self.sequence_length = sequence_length
        self.is_trained = False
        
    def train(self, training_data: Dict[int, np.ndarray], 
              labels: Dict[int, np.ndarray]) -> None:
        """
        Train the model on historical data.
        
        Parameters
        ----------
        training_data : Dict[int, np.ndarray]
            Feature sequences for each qubit
        labels : Dict[int, np.ndarray]
            Binary labels (1=decoherence event, 0=stable)
        """
        raise NotImplementedError
        
    def predict(self, qubit_id: int, 
                feature_sequence: np.ndarray) -> Tuple[float, float]:
        """
        Predict decoherence probability and time-to-event.
        
        Parameters
        ----------
        qubit_id : int
            Qubit identifier
        feature_sequence : np.ndarray
            Recent telemetry sequence of shape (seq_len, num_features)
            
        Returns
        -------
        Tuple[float, float]
            (probability of decoherence, estimated time steps until event)
        """
        raise NotImplementedError
    
    def get_remaining_useful_lifetime(self, qubit_id: int,
                                       feature_sequence: np.ndarray) -> float:
        """
        Estimate remaining useful coherence lifetime (RUL).
        
        Parameters
        ----------
        qubit_id : int
            Qubit identifier
        feature_sequence : np.ndarray
            Recent telemetry sequence
            
        Returns
        -------
        float
            Estimated time steps before decoherence
        """
        prob, time_steps = self.predict(qubit_id, feature_sequence)
        return time_steps


class SimpleThresholdPredictor(DecoherencePredictor):
    """
    Simple baseline predictor using threshold-based rules.
    
    This serves as a baseline for comparing more sophisticated ML models.
    """
    
    def __init__(self, num_qubits: int, sequence_length: int = 10):
        super().__init__(num_qubits, sequence_length)
        self.stability_scorer = PredictiveStabilityScore()
        
        # Thresholds learned from training data or set heuristically
        self.pss_threshold_warning = 70
        self.pss_threshold_critical = 50
        self.trend_threshold = -0.5  # Negative trend indicates degradation
        
    def train(self, training_data: Dict[int, np.ndarray],
              labels: Dict[int, np.ndarray]) -> None:
        """
        Learn optimal thresholds from training data.
        
        For this simple model, we analyze the PSS values at which
        decoherence events occurred.
        """
        pss_at_decoherence = []
        pss_stable = []
        
        for qubit_id, features in training_data.items():
            label_seq = labels.get(qubit_id, np.zeros(len(features)))
            
            for i, (feat, label) in enumerate(zip(features, label_seq)):
                pss, _ = self.stability_scorer.compute(feat)
                
                if label == 1:  # Decoherence event
                    pss_at_decoherence.append(pss)
                else:
                    pss_stable.append(pss)
        
        # Set thresholds based on observed distributions
        if pss_at_decoherence:
            self.pss_threshold_critical = np.percentile(pss_at_decoherence, 50)
            self.pss_threshold_warning = np.percentile(pss_at_decoherence, 75)
        
        self.is_trained = True
        
    def predict(self, qubit_id: int,
                feature_sequence: np.ndarray) -> Tuple[float, float]:
        """
        Predict based on current PSS and trend analysis.
        """
        if len(feature_sequence) == 0:
            return 0.0, float('inf')
        
        # Compute PSS for each timestep
        pss_values = []
        for features in feature_sequence:
            pss, _ = self.stability_scorer.compute(features)
            pss_values.append(pss)
        
        current_pss = pss_values[-1]
        
        # Compute trend (rate of change)
        if len(pss_values) > 1:
            trend = (pss_values[-1] - pss_values[0]) / len(pss_values)
        else:
            trend = 0
        
        # Estimate probability based on PSS and trend
        # Lower PSS and negative trend increase probability
        base_prob = 1.0 - (current_pss / 100.0)
        trend_factor = max(0, -trend / 10)  # Negative trends increase probability
        probability = min(1.0, base_prob + trend_factor)
        
        # Estimate time to event
        if probability > 0.5:
            # Linear extrapolation based on trend
            if trend < 0:
                steps_to_threshold = (current_pss - self.pss_threshold_critical) / abs(trend)
                time_steps = max(1, steps_to_threshold)
            else:
                time_steps = 10  # Default near-term
        else:
            time_steps = float('inf')  # No imminent event
        
        return probability, time_steps


class EnsemblePredictor(DecoherencePredictor):
    """
    Ensemble predictor combining multiple models.
    """
    
    def __init__(self, predictors: List[DecoherencePredictor],
                 weights: Optional[List[float]] = None):
        if len(predictors) == 0:
            raise ValueError("Must provide at least one predictor")
        
        # Use first predictor's parameters
        super().__init__(
            predictors[0].num_qubits,
            predictors[0].sequence_length
        )
        
        self.predictors = predictors
        
        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            if len(weights) != len(predictors):
                raise ValueError("Weights length must match predictors length")
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def train(self, training_data: Dict[int, np.ndarray],
              labels: Dict[int, np.ndarray]) -> None:
        """Train all constituent models."""
        for predictor in self.predictors:
            predictor.train(training_data, labels)
        self.is_trained = True
    
    def predict(self, qubit_id: int,
                feature_sequence: np.ndarray) -> Tuple[float, float]:
        """Combine predictions from all models."""
        probabilities = []
        time_steps_list = []
        
        for predictor, weight in zip(self.predictors, self.weights):
            prob, ts = predictor.predict(qubit_id, feature_sequence)
            probabilities.append(prob * weight)
            if ts != float('inf'):
                time_steps_list.append(ts)
        
        final_probability = sum(probabilities)
        
        if time_steps_list:
            final_time = np.median(time_steps_list)
        else:
            final_time = float('inf')
        
        return final_probability, final_time


__all__ = [
    "DecoherenceRisk",
    "PredictiveStabilityScore",
    "DecoherencePredictor",
    "SimpleThresholdPredictor",
    "EnsemblePredictor",
]
