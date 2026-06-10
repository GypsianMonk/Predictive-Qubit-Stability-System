"""
Example: End-to-End PQSS Demonstration

This script demonstrates the complete Predictive Quantum Stability System
workflow from telemetry collection through prediction to intervention.
"""

import numpy as np
from telemetry import TelemetryCollector, generate_synthetic_telemetry
from models import PredictiveStabilityScore, SimpleThresholdPredictor, DecoherenceRisk
from controller import InterventionController
from digital_twin import DigitalTwinManager
from simulator import QuantumSystemSimulator, SimulationConfig


def run_pqss_demo(num_qubits=5, num_steps=20):
    """
    Run a complete PQSS demonstration.
    
    Parameters
    ----------
    num_qubits : int
        Number of qubits in the system
    num_steps : int
        Number of simulation steps
    """
    print("=" * 60)
    print("PREDICTIVE QUANTUM STABILITY SYSTEM (PQSS) DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Initialize components
    print("Initializing PQSS components...")
    config = SimulationConfig(num_qubits=num_qubits)
    simulator = QuantumSystemSimulator(config)
    collector = TelemetryCollector(num_qubits=num_qubits)
    predictor = SimpleThresholdPredictor(num_qubits=num_qubits, sequence_length=5)
    controller = InterventionController(num_qubits=num_qubits)
    twin_manager = DigitalTwinManager(num_qubits=num_qubits)
    scorer = PredictiveStabilityScore()
    
    print(f"  - Simulator: {num_qubits} qubits")
    print(f"  - Telemetry Collector: Ready")
    print(f"  - Predictor: Simple Threshold Model")
    print(f"  - Controller: 3 intervention strategies")
    print(f"  - Digital Twins: Active")
    print()
    
    # Introduce an anomaly for demonstration
    print("Introducing degradation anomaly on qubit 2...")
    simulator.introduce_anomaly(qubit_id=2, anomaly_type='degradation', severity=0.6)
    print()
    
    # Track statistics
    interventions_triggered = 0
    predictions_made = 0
    decoherence_events_detected = 0
    
    print("Starting simulation loop...")
    print("-" * 60)
    
    for step in range(num_steps):
        # Simulate one time step
        raw_telemetry = simulator.simulate_step()
        
        # Process each qubit
        for qubit_id in range(num_qubits):
            data_dict = raw_telemetry[qubit_id]
            
            # Convert to TelemetryData object
            from telemetry import TelemetryData
            from datetime import datetime
            
            telemetry_obj = TelemetryData(
                timestamp=datetime.now(),
                qubit_id=qubit_id,
                temperature=data_dict['temperature'],
                frequency_drift=data_dict['frequency_drift'],
                resonance_instability=data_dict['resonance_instability'],
                voltage_variation=data_dict['voltage_variation'],
                t1_relaxation_time=data_dict['t1_relaxation_time'],
                t2_dephasing_time=data_dict['t2_dephasing_time'],
                gate_fidelity=data_dict['gate_fidelity'],
                readout_fidelity=data_dict['readout_fidelity'],
                entanglement_strength=data_dict['entanglement_strength'],
                white_noise_level=data_dict['white_noise_level'],
                thermal_noise_level=data_dict['thermal_noise_level'],
                environmental_noise=data_dict['environmental_noise'],
                crosstalk_level=data_dict['crosstalk_level'],
                gate_count=data_dict['gate_count'],
                circuit_depth=data_dict['circuit_depth'],
                qubit_utilization=data_dict['qubit_utilization'],
                error_frequency=data_dict['error_frequency'],
            )
            
            # Collect telemetry
            collector.collect(qubit_id, telemetry_obj)
            
            # Update digital twin
            twin_manager.update_twin(qubit_id, telemetry_obj)
            
            # Get feature sequence for prediction
            features = collector.get_all_features(qubit_id, last_n=5)
            
            # Initialize pss and risk for later use
            pss = 100.0
            risk = DecoherenceRisk.STABLE
            
            if features is not None and len(features) >= 3:
                # Compute PSS
                pss, risk = scorer.compute(features[-1])
                
                # Make prediction
                prob, time_to_event = predictor.predict(qubit_id, features)
                predictions_made += 1
                
                # Check for decoherence
                if data_dict['is_decohering']:
                    decoherence_events_detected += 1
                
                # Trigger intervention if needed
                if risk != DecoherenceRisk.STABLE or prob > 0.4:
                    system_state = {
                        'telemetry': {
                            'frequency_drift': data_dict['frequency_drift'],
                            'resonance_instability': data_dict['resonance_instability'],
                        },
                        'qubit_health': {
                            i: twin_manager.get_twin(i).state.stability_probability
                            for i in range(num_qubits)
                        }
                    }
                    
                    action = controller.evaluate_and_intervene(
                        qubit_id=qubit_id,
                        current_pss=pss,
                        risk_level=risk.value,
                        predicted_prob=prob,
                        time_to_event=time_to_event,
                        system_state=system_state,
                    )
                    
                    if action:
                        interventions_triggered += 1
                        result = controller.execute_intervention(action)
                        
                        if step % 5 == 0 or qubit_id == 2:  # Print key events
                            print(f"Step {step:2d} | Qubit {qubit_id} | "
                                  f"PSS: {pss:5.1f} | Risk: {risk.value:9s} | "
                                  f"Prob: {prob:.2f} | → {action.intervention_type.value}")
            
            # Special monitoring for anomalous qubit
            if qubit_id == 2 and step % 3 == 0:
                twin = twin_manager.get_twin(qubit_id)
                print(f"Step {step:2d} | Qubit 2 | "
                      f"PSS: {pss:5.1f} | Twin Stability: {twin.state.stability_probability:.2f} | "
                      f"Pred. Lifetime: {twin.state.predicted_lifetime:.1f}")
    
    print("-" * 60)
    print()
    
    # Final statistics
    print("SIMULATION STATISTICS")
    print("=" * 60)
    print(f"Total Steps:              {num_steps}")
    print(f"Qubits Monitored:         {num_qubits}")
    print(f"Predictions Made:         {predictions_made}")
    print(f"Interventions Triggered:  {interventions_triggered}")
    print(f"Decoherence Events:       {decoherence_events_detected}")
    print()
    
    # System health summary
    health = twin_manager.get_system_health()
    print("SYSTEM HEALTH SUMMARY")
    print("=" * 60)
    print(f"Average Stability:        {health['average_stability']:.2%}")
    print(f"Min Stability:            {health['min_stability']:.2%}")
    print(f"Max Stability:            {health['max_stability']:.2%}")
    print(f"Qubits at Risk:           {health['qubits_at_risk']}")
    print(f"Total Interventions:      {health['total_interventions']}")
    print()
    
    # Per-qubit intervention stats
    print("INTERVENTION STATISTICS BY QUBIT")
    print("=" * 60)
    for qubit_id in range(num_qubits):
        stats = controller.get_intervention_statistics(qubit_id)
        if stats['total_interventions'] > 0:
            print(f"Qubit {qubit_id}:")
            print(f"  Total: {stats['total_interventions']}")
            print(f"  By Type: {stats['by_type']}")
            print(f"  Avg Priority: {stats['average_priority']:.2f}")
    print()
    
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("The PQSS successfully:")
    print("  ✓ Monitored qubit telemetry in real-time")
    print("  ✓ Computed Predictive Stability Scores (PSS)")
    print("  ✓ Predicted decoherence probabilities")
    print("  ✓ Triggered adaptive interventions")
    print("  ✓ Maintained digital twins for all qubits")
    print()
    print("This demonstrates the shift from reactive error correction")
    print("to proactive error prevention in quantum computing.")


if __name__ == "__main__":
    run_pqss_demo()
