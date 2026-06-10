"""
RL Training Experiment Script

Fast training demonstration with reduced episodes for validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reinforcement_learning import (
    QuantumStabilityEnv, 
    DQNAgent, 
    MultiAgentRLController,
    AdaptiveInterventionEngine,
    InterventionAction,
    train_rl_system
)
import numpy as np
import json


def test_environment():
    """Test the quantum stability environment"""
    print("=" * 70)
    print("TEST 1: Quantum Stability Environment")
    print("=" * 70)
    
    env = QuantumStabilityEnv(n_qubits=5, max_steps=100)
    state = env.reset()
    
    print(f"✓ Environment initialized with {env.n_qubits} qubits")
    print(f"✓ State dimension: {env.state_dim}")
    print(f"✓ Action space: {env.n_actions} actions")
    print(f"✓ Initial state shape: {state.shape}")
    print(f"✓ Initial avg PSS: {np.mean(env.pss_scores):.3f}")
    
    # Run some steps
    rewards = []
    for i in range(20):
        action = np.random.randint(0, env.n_actions)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        
    print(f"✓ Ran 20 steps, avg reward: {np.mean(rewards):.2f}")
    print(f"✓ Final avg PSS: {info['avg_pss']:.3f}")
    print(f"✓ Final avg fidelity: {info['avg_fidelity']:.4f}\n")
    
    return True


def test_dqn_agent():
    """Test DQN agent training"""
    print("=" * 70)
    print("TEST 2: DQN Agent Training")
    print("=" * 70)
    
    state_dim = 16  # Augmented state
    n_actions = len(InterventionAction)
    
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.95,
        buffer_size=5000,
        batch_size=32,
        target_update=5
    )
    
    print(f"✓ DQN agent created")
    print(f"✓ State dim: {state_dim}, Actions: {n_actions}")
    print(f"✓ Initial epsilon: {agent.epsilon:.2f}")
    
    # Quick training loop
    env = QuantumStabilityEnv(n_qubits=5)
    state = env.reset()
    
    episode_rewards = []
    for episode in range(10):
        episode_reward = 0
        for step in range(50):
            # Construct augmented state
            qubit_idx = step % 5
            qubit_state = state[qubit_idx*8:(qubit_idx+1)*8]
            neighbor_states = state[(qubit_idx+1)%5*8:(qubit_idx+2)%5*8][:8]
            augmented_state = np.concatenate([qubit_state, neighbor_states])
            
            action = agent.select_action(augmented_state)
            next_state, reward, done, _ = env.step(action)
            
            next_qubit_state = next_state[qubit_idx*8:(qubit_idx+1)*8]
            next_neighbor_states = next_state[(qubit_idx+1)%5*8:(qubit_idx+2)%5*8][:8]
            next_augmented_state = np.concatenate([next_qubit_state, next_neighbor_states])
            
            agent.store_transition(augmented_state, action, next_augmented_state, reward)
            agent.optimize_model()
            
            episode_reward += reward
            state = next_state
            
            if done:
                state = env.reset()
        
        episode_rewards.append(episode_reward)
        state = env.reset()
    
    print(f"✓ Trained for 10 episodes")
    print(f"✓ Avg reward (first 5 eps): {np.mean(episode_rewards[:5]):.2f}")
    print(f"✓ Avg reward (last 5 eps): {np.mean(episode_rewards[5:]):.2f}")
    print(f"✓ Final epsilon: {agent.epsilon:.2f}\n")
    
    return True


def test_multi_agent():
    """Test multi-agent RL controller"""
    print("=" * 70)
    print("TEST 3: Multi-Agent RL Controller (Quick Test)")
    print("=" * 70)
    
    controller = MultiAgentRLController(n_qubits=5, agent_type='dqn')
    
    print(f"✓ Multi-agent controller created with {controller.n_qubits} qubits")
    print(f"✓ Number of agents: {len(controller.agents)}")
    
    # Train for a few episodes
    results = []
    for episode in range(5):
        result = controller.train_episode(max_steps=100)
        results.append(result)
        print(f"  Episode {episode}: Reward={result['avg_reward']:.2f}, "
              f"PSS={result['final_pss']:.3f}, Errors={result['total_errors']}")
    
    avg_reward = np.mean([r['avg_reward'] for r in results])
    avg_pss = np.mean([r['final_pss'] for r in results])
    
    print(f"\n✓ Completed 5 episodes")
    print(f"✓ Average reward: {avg_reward:.2f}")
    print(f"✓ Average PSS: {avg_pss:.3f}\n")
    
    return True


def test_intervention_engine():
    """Test adaptive intervention engine"""
    print("=" * 70)
    print("TEST 4: Adaptive Intervention Engine")
    print("=" * 70)
    
    engine = AdaptiveInterventionEngine(n_qubits=10)
    
    # Simulate various telemetry scenarios
    scenarios = [
        {
            'name': 'Critical Qubit',
            'data': {'qubit_0': {'pss': 0.45, 't1': 50e-6, 't2': 30e-6, 
                                'gate_fidelity': 0.990, 'noise_level': 0.08, 'error_count': 5}}
        },
        {
            'name': 'Warning State',
            'data': {'qubit_3': {'pss': 0.72, 't1': 75e-6, 't2': 55e-6,
                                'gate_fidelity': 0.996, 'noise_level': 0.04, 'error_count': 2}}
        },
        {
            'name': 'Stable Qubit',
            'data': {'qubit_7': {'pss': 0.94, 't1': 110e-6, 't2': 90e-6,
                                'gate_fidelity': 0.999, 'noise_level': 0.01, 'error_count': 0}}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['name']}")
        recommendations = engine.recommend_intervention(scenario['data'])
        for rec in recommendations:
            print(f"    → {rec['action']} (Priority: {rec['priority']}, Confidence: {rec['confidence']:.2f})")
    
    print("\n✓ Intervention engine tested successfully\n")
    return True


def evaluate_performance():
    """Evaluate trained policy performance"""
    print("=" * 70)
    print("TEST 5: Policy Evaluation")
    print("=" * 70)
    
    controller = MultiAgentRLController(n_qubits=5, agent_type='dqn')
    
    # Quick evaluation with proper state handling
    all_rewards = []
    all_pss = []
    all_fidelities = []
    all_errors = []
    
    for _ in range(10):
        state = controller.env.reset()
        episode_reward = 0
        
        for step in range(100):
            actions = []
            for i in range(controller.n_qubits):
                qubit_state = state[i*8:(i+1)*8]
                neighbor_states = []
                for neighbor_idx in controller.neighbors[i]:
                    neighbor_state = state[neighbor_idx*8:(neighbor_idx+1)*8]
                    neighbor_states.extend(neighbor_state[:4])
                
                augmented_state = np.concatenate([qubit_state, np.array(neighbor_states)])
                action = controller.agents[i].select_action(augmented_state, training=False)
                actions.append(action)
            
            state, reward, done, info = controller.env.step(sum(actions) % controller.env.n_actions)
            episode_reward += reward
            
            if done:
                break
        
        all_rewards.append(episode_reward)
        all_pss.append(info['avg_pss'])
        all_fidelities.append(info['avg_fidelity'])
        all_errors.append(info['total_errors'])
    
    eval_results = {
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'avg_pss': np.mean(all_pss),
        'avg_fidelity': np.mean(all_fidelities),
        'avg_errors': np.mean(all_errors),
        'success_rate': np.mean([p > 0.8 for p in all_pss])
    }
    
    print(f"Evaluation Results (untrained policy):")
    print(f"  Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Average PSS: {eval_results['avg_pss']:.3f}")
    print(f"  Average Fidelity: {eval_results['avg_fidelity']:.4f}")
    print(f"  Average Errors: {eval_results['avg_errors']:.1f}")
    print(f"  Success Rate: {eval_results['success_rate']*100:.1f}%\n")
    
    return True


def test_dqn_agent_fixed():
    """Test DQN agent training with fixed state handling"""
    print("=" * 70)
    print("TEST 6: DQN Agent Training (Fixed)")
    print("=" * 70)
    
    state_dim = 16  # Augmented state
    n_actions = len(InterventionAction)
    
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.95,
        buffer_size=5000,
        batch_size=32,
        target_update=5
    )
    
    print(f"✓ DQN agent created")
    print(f"✓ State dim: {state_dim}, Actions: {n_actions}")
    print(f"✓ Initial epsilon: {agent.epsilon:.2f}")
    
    # Quick training loop with proper state handling
    env = QuantumStabilityEnv(n_qubits=5)
    
    episode_rewards = []
    for episode in range(10):
        state = env.reset()
        episode_reward = 0
        
        for step in range(50):
            # Construct augmented state properly
            qubit_idx = step % 5
            qubit_state = state[qubit_idx*8:(qubit_idx+1)*8].copy()
            neighbor_idx = (qubit_idx + 1) % 5
            neighbor_states = state[neighbor_idx*8:(neighbor_idx+1)*8][:8].copy()
            augmented_state = np.concatenate([qubit_state, neighbor_states])
            
            action = agent.select_action(augmented_state)
            next_state, reward, done, _ = env.step(action)
            
            # Next state
            next_qubit_state = next_state[qubit_idx*8:(qubit_idx+1)*8].copy()
            next_neighbor_states = next_state[neighbor_idx*8:(neighbor_idx+1)*8][:8].copy()
            next_augmented_state = np.concatenate([next_qubit_state, next_neighbor_states])
            
            agent.store_transition(augmented_state, action, next_augmented_state, reward)
            agent.optimize_model()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    print(f"✓ Trained for 10 episodes")
    print(f"✓ Avg reward (first 5 eps): {np.mean(episode_rewards[:5]):.2f}")
    print(f"✓ Avg reward (last 5 eps): {np.mean(episode_rewards[5:]):.2f}")
    print(f"✓ Final epsilon: {agent.epsilon:.2f}\n")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PQSS REINFORCEMENT LEARNING - VALIDATION SUITE")
    print("=" * 70 + "\n")
    
    tests = [
        ("Environment", test_environment),
        ("Multi-Agent Control", test_multi_agent),
        ("Intervention Engine", test_intervention_engine),
        ("Policy Evaluation", evaluate_performance),
        ("DQN Agent Fixed", test_dqn_agent_fixed)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            results[name] = f"✗ ERROR: {str(e)}"
            print(f"Error in {name}: {e}\n")
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        print(f"{name:.<50} {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
