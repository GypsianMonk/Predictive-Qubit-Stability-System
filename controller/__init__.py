"""
Intervention Controller for PQSS

Adaptive intervention engine that generates and executes
stabilization strategies based on AI predictions:
- Pulse Optimization
- State Migration
- Dynamic Scheduling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class InterventionType(Enum):
    """Types of stabilization interventions."""
    PULSE_OPTIMIZATION = "pulse_optimization"
    STATE_MIGRATION = "state_migration"
    DYNAMIC_SCHEDULING = "dynamic_scheduling"
    ERROR_CORRECTION_INCREASE = "error_correction_increase"
    GATE_DENSITY_REDUCTION = "gate_density_reduction"
    QUBET_IDLE = "qubit_idle"


@dataclass
class InterventionAction:
    """Represents a stabilization action to be executed."""
    
    intervention_type: InterventionType
    qubit_id: int
    priority: float  # 0-1, higher is more urgent
    parameters: Dict[str, float]
    estimated_duration: int  # time steps
    expected_improvement: float  # Expected PSS improvement
    
    def __str__(self) -> str:
        return (f"Intervention({self.intervention_type.value} on qubit {self.qubit_id}, "
                f"priority={self.priority:.2f})")


@dataclass
class InterventionResult:
    """Result of executing an intervention."""
    
    success: bool
    actual_improvement: float
    execution_time: int
    message: str


class InterventionStrategy:
    """Base class for intervention strategies."""
    
    def generate_action(self, 
                        qubit_id: int,
                        current_pss: float,
                        predicted_decoherence_prob: float,
                        time_to_event: float,
                        system_state: Dict) -> Optional[InterventionAction]:
        """
        Generate an intervention action based on current conditions.
        
        Parameters
        ----------
        qubit_id : int
            Target qubit identifier
        current_pss : float
            Current Predictive Stability Score
        predicted_decoherence_prob : float
            Probability of impending decoherence
        time_to_event : float
            Estimated time steps until decoherence
        system_state : Dict
            Current system-wide state information
            
        Returns
        -------
        Optional[InterventionAction]
            Action to execute, or None if no intervention needed
        """
        raise NotImplementedError


class PulseOptimizationStrategy(InterventionStrategy):
    """
    Strategy A: Pulse Optimization
    
    Adjusts control pulse parameters to counteract detected instability.
    """
    
    def __init__(self):
        self.pulse_adjustment_factors = {
            'amplitude': 0.05,
            'frequency': 0.02,
            'phase': 0.01,
            'duration': 0.03,
        }
    
    def generate_action(self, qubit_id: int,
                        current_pss: float,
                        predicted_decoherence_prob: float,
                        time_to_event: float,
                        system_state: Dict) -> Optional[InterventionAction]:
        
        if predicted_decoherence_prob < 0.3:
            return None  # No intervention needed
        
        # Calculate required adjustments based on degradation patterns
        telemetry = system_state.get('telemetry', {})
        frequency_drift = telemetry.get('frequency_drift', 0)
        resonance_instability = telemetry.get('resonance_instability', 0)
        
        # Generate pulse parameter adjustments
        params = {
            'frequency_shift': -frequency_drift * 0.8,  # Counteract drift
            'amplitude_adjustment': 1.0 + (1 - current_pss/100) * 0.1,
            'phase_correction': resonance_instability * 0.01,
            'pulse_duration_factor': 1.0 + (1 - current_pss/100) * 0.05,
        }
        
        # Estimate improvement based on severity
        expected_improvement = min(20, predicted_decoherence_prob * 30)
        
        return InterventionAction(
            intervention_type=InterventionType.PULSE_OPTIMIZATION,
            qubit_id=qubit_id,
            priority=predicted_decoherence_prob,
            parameters=params,
            estimated_duration=5,
            expected_improvement=expected_improvement,
        )


class StateMigrationStrategy(InterventionStrategy):
    """
    Strategy B: State Migration
    
    Transfers quantum information from high-risk qubits to healthier ones.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.migration_threshold = 0.6  # Probability threshold for migration
    
    def generate_action(self, qubit_id: int,
                        current_pss: float,
                        predicted_decoherence_prob: float,
                        time_to_event: float,
                        system_state: Dict) -> Optional[InterventionAction]:
        
        if predicted_decoherence_prob < self.migration_threshold:
            return None
        
        # Find suitable target qubit
        qubit_health = system_state.get('qubit_health', {})
        target_qubit = self._find_healthy_target(qubit_id, qubit_health)
        
        if target_qubit is None:
            return None  # No suitable target available
        
        params = {
            'source_qubit': qubit_id,
            'target_qubit': target_qubit,
            'transfer_fidelity_threshold': 0.99,
            'verification_required': True,
        }
        
        # State migration can significantly improve stability
        expected_improvement = min(40, predicted_decoherence_prob * 50)
        
        return InterventionAction(
            intervention_type=InterventionType.STATE_MIGRATION,
            qubit_id=qubit_id,
            priority=predicted_decoherence_prob,
            parameters=params,
            estimated_duration=20,  # Migration takes time
            expected_improvement=expected_improvement,
        )
    
    def _find_healthy_target(self, source_qubit: int,
                              qubit_health: Dict[int, float]) -> Optional[int]:
        """Find the healthiest available qubit for migration."""
        candidates = []
        
        for qid, health in qubit_health.items():
            if qid != source_qubit and health > 0.8:  # Only consider healthy qubits
                candidates.append((qid, health))
        
        if not candidates:
            return None
        
        # Return the healthiest candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


class DynamicSchedulingStrategy(InterventionStrategy):
    """
    Strategy C: Dynamic Scheduling
    
    Reduces gate operations on vulnerable qubits to prevent further degradation.
    """
    
    def __init__(self):
        self.reduction_factors = {
            'warning': 0.7,    # Reduce to 70% of normal operations
            'critical': 0.4,   # Reduce to 40%
            'immediate': 0.1,  # Reduce to 10% (nearly idle)
        }
    
    def generate_action(self, qubit_id: int,
                        current_pss: float,
                        predicted_decoherence_prob: float,
                        time_to_event: float,
                        system_state: Dict) -> Optional[InterventionAction]:
        
        # Determine reduction level based on PSS
        if current_pss >= 90:
            return None  # No reduction needed
        
        if current_pss >= 70:
            level = 'warning'
        elif current_pss >= 50:
            level = 'critical'
        else:
            level = 'immediate'
        
        reduction_factor = self.reduction_factors[level]
        
        params = {
            'max_gate_rate': reduction_factor,
            'priority_operations_only': level in ['critical', 'immediate'],
            'cooldown_duration': int(50 * (1 - reduction_factor)),
        }
        
        # Scheduling changes help preserve coherence
        expected_improvement = min(15, (1 - reduction_factor) * 25)
        
        return InterventionAction(
            intervention_type=InterventionType.DYNAMIC_SCHEDULING,
            qubit_id=qubit_id,
            priority=1 - current_pss/100,
            parameters=params,
            estimated_duration=params['cooldown_duration'],
            expected_improvement=expected_improvement,
        )


class InterventionController:
    """
    Central controller for managing all intervention strategies.
    
    Coordinates prediction-based intervention selection and execution.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        
        # Initialize strategies
        self.strategies = [
            PulseOptimizationStrategy(),
            StateMigrationStrategy(num_qubits),
            DynamicSchedulingStrategy(),
        ]
        
        # Track intervention history
        self.intervention_history: Dict[int, List[InterventionAction]] = {
            i: [] for i in range(num_qubits)
        }
        
        # Current active interventions
        self.active_interventions: Dict[int, InterventionAction] = {}
    
    def evaluate_and_intervene(self,
                                qubit_id: int,
                                current_pss: float,
                                risk_level: str,
                                predicted_prob: float,
                                time_to_event: float,
                                system_state: Dict) -> Optional[InterventionAction]:
        """
        Evaluate all strategies and execute the best intervention.
        
        Parameters
        ----------
        qubit_id : int
            Target qubit
        current_pss : float
            Current stability score
        risk_level : str
            Risk classification (stable/warning/critical/immediate)
        predicted_prob : float
            Decoherence probability from predictor
        time_to_event : float
            Estimated time until decoherence
        system_state : Dict
            Global system state
            
        Returns
        -------
        Optional[InterventionAction]
            Selected intervention action
        """
        # Collect candidate actions from all strategies
        candidate_actions = []
        
        for strategy in self.strategies:
            action = strategy.generate_action(
                qubit_id=qubit_id,
                current_pss=current_pss,
                predicted_decoherence_prob=predicted_prob,
                time_to_event=time_to_event,
                system_state=system_state,
            )
            
            if action is not None:
                candidate_actions.append(action)
        
        if not candidate_actions:
            return None
        
        # Select action with highest priority (urgency * expected improvement)
        best_action = max(
            candidate_actions,
            key=lambda a: a.priority * (1 + a.expected_improvement / 100)
        )
        
        # Record intervention
        self.intervention_history[qubit_id].append(best_action)
        self.active_interventions[qubit_id] = best_action
        
        return best_action
    
    def execute_intervention(self, action: InterventionAction) -> InterventionResult:
        """
        Execute an intervention action.
        
        In a real implementation, this would interface with
        quantum hardware control systems.
        """
        # Simulate execution (in real system, this would send commands to hardware)
        import random
        
        # Simulate success probability based on intervention type and priority
        base_success_rate = 0.95
        
        if action.intervention_type == InterventionType.STATE_MIGRATION:
            # State migration is more complex, slightly lower success rate
            success_prob = base_success_rate - 0.05
        else:
            success_prob = base_success_rate
        
        success = random.random() < success_prob
        
        if success:
            actual_improvement = action.expected_improvement * random.uniform(0.8, 1.2)
            message = f"Successfully executed {action.intervention_type.value}"
        else:
            actual_improvement = 0
            message = f"Failed to execute {action.intervention_type.value}"
        
        result = InterventionResult(
            success=success,
            actual_improvement=actual_improvement,
            execution_time=action.estimated_duration,
            message=message,
        )
        
        # Clear active intervention
        if action.qubit_id in self.active_interventions:
            del self.active_interventions[action.qubit_id]
        
        return result
    
    def get_intervention_statistics(self, qubit_id: int) -> Dict:
        """Get statistics about interventions for a qubit."""
        history = self.intervention_history[qubit_id]
        
        if not history:
            return {
                'total_interventions': 0,
                'by_type': {},
                'average_priority': 0,
                'average_expected_improvement': 0,
            }
        
        by_type = {}
        for action in history:
            type_name = action.intervention_type.value
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1
        
        return {
            'total_interventions': len(history),
            'by_type': by_type,
            'average_priority': np.mean([a.priority for a in history]),
            'average_expected_improvement': np.mean([a.expected_improvement for a in history]),
        }


__all__ = [
    "InterventionType",
    "InterventionAction",
    "InterventionResult",
    "InterventionStrategy",
    "PulseOptimizationStrategy",
    "StateMigrationStrategy",
    "DynamicSchedulingStrategy",
    "InterventionController",
]
