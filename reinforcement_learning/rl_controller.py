"""
Reinforcement Learning for Adaptive Quantum Stabilization
Implements Q-learning and Policy Gradient methods for intervention strategy optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings


@dataclass
class RLConfig:
    """Configuration for RL agents"""
    state_dim: int = 20  # PSS + qubit features
    action_dim: int = 5  # 5 intervention strategies
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 32


class QuantumEnvironment:
    """Simulated quantum environment for RL training"""
    
    def __init__(self, num_qubits: int = 5):
        self.num_qubits = num_qubits
        self.qubit_states = np.ones(num_qubits) * 100  # PSS scores
        self.decoherence_rates = np.random.uniform(0.01, 0.05, num_qubits)
        self.step_count = 0
        self.max_steps = 100
        
        # Intervention effectiveness
        self.intervention_effects = {
            'pulse_optimization': 0.15,
            'state_migration': 0.25,
            'dynamic_scheduling': 0.10,
            'error_correction': 0.20,
            'cooling': 0.18
        }
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.qubit_states = np.ones(self.num_qubits) * 100
        self.decoherence_rates = np.random.uniform(0.01, 0.05, self.num_qubits)
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state"""
        # State: [mean_PSS, std_PSS, min_PSS, max_PSS, step_ratio] + qubit_states
        state = np.array([
            np.mean(self.qubit_states),
            np.std(self.qubit_states),
            np.min(self.qubit_states),
            np.max(self.qubit_states),
            self.step_count / self.max_steps
        ])
        # Pad or truncate qubit states to fixed size
        qubit_part = np.zeros(15)
        qubit_part[:min(self.num_qubits, 15)] = self.qubit_states[:15]
        return np.concatenate([state, qubit_part])
    
    def step(self, action: int, target_qubit: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in environment"""
        self.step_count += 1
        
        # Natural decoherence
        decay = np.exp(-self.decoherence_rates * 2)
        self.qubit_states *= decay
        
        # Apply intervention
        action_names = ['pulse_optimization', 'state_migration', 
                       'dynamic_scheduling', 'error_correction', 'cooling']
        action_name = action_names[action % len(action_names)]
        
        # Intervention effect with some randomness
        effect = self.intervention_effects[action_name] * (0.8 + np.random.random() * 0.4)
        self.qubit_states[target_qubit] = min(100, self.qubit_states[target_qubit] + effect * 100)
        
        # Also help neighboring qubits slightly
        for i in range(self.num_qubits):
            if i != target_qubit and abs(i - target_qubit) == 1:
                self.qubit_states[i] = min(100, self.qubit_states[i] + effect * 30)
        
        # Calculate reward
        reward = self._calculate_reward(action_name)
        
        # Check termination
        done = self.step_count >= self.max_steps or np.min(self.qubit_states) < 0
        
        info = {
            'mean_pss': np.mean(self.qubit_states),
            'min_pss': np.min(self.qubit_states),
            'action_name': action_name,
            'target_qubit': target_qubit
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, action_name: str) -> float:
        """Calculate reward based on system stability and intervention cost"""
        # Base reward: average stability
        base_reward = np.mean(self.qubit_states) / 100
        
        # Penalty for low stability qubits
        penalty = sum(max(0, 50 - q) for q in self.qubit_states) / 100
        
        # Cost of intervention
        intervention_costs = {
            'pulse_optimization': 0.02,
            'state_migration': 0.05,
            'dynamic_scheduling': 0.01,
            'error_correction': 0.03,
            'cooling': 0.04
        }
        cost = intervention_costs.get(action_name, 0.02)
        
        # Bonus for maintaining high stability
        bonus = 0.1 if np.min(self.qubit_states) > 70 else 0
        
        return base_reward - penalty - cost + bonus


class QLearningAgent:
    """Tabular Q-Learning agent for discrete state spaces"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.q_table = defaultdict(lambda: np.zeros(config.action_dim))
        self.epsilon = config.epsilon
    
    def discretize_state(self, state: np.ndarray, bins: int = 10) -> tuple:
        """Convert continuous state to discrete representation"""
        discretized = []
        for i, val in enumerate(state[:5]):  # Only use summary statistics
            if i == 2 or i == 3:  # min/max PSS
                bins_i = np.linspace(0, 100, bins)
            elif i == 4:  # step ratio
                bins_i = np.linspace(0, 1, bins)
            else:  # mean/std
                bins_i = np.linspace(0, 100, bins)
            
            digitized = np.digitize([val], bins_i)[0]
            discretized.append(digitized)
        
        return tuple(discretized)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        disc_state = self.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.config.action_dim)
        else:
            return np.argmax(self.q_table[disc_state])
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update Q-table using Q-learning update rule"""
        disc_state = self.discretize_state(state)
        disc_next_state = self.discretize_state(next_state)
        
        best_next_action = np.argmax(self.q_table[disc_next_state])
        
        if done:
            target = reward
        else:
            target = reward + self.config.discount_factor * \
                     self.q_table[disc_next_state][best_next_action]
        
        # Q-learning update
        self.q_table[disc_state][action] += self.config.learning_rate * \
            (target - self.q_table[disc_state][action])
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)


class PolicyGradientAgent:
    """Simple policy gradient agent for continuous state spaces"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        # Simple linear policy: action_probs = softmax(W @ state + b)
        self.W = np.random.randn(config.action_dim, config.state_dim) * 0.01
        self.b = np.zeros(config.action_dim)
        self.gamma = config.discount_factor
        
        # Training buffers
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action based on policy"""
        logits = self.W @ state + self.b
        probs = self.softmax(logits)
        return np.random.choice(self.config.action_dim, p=probs)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float):
        """Store transition for batch update"""
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
    
    def update(self):
        """Update policy using REINFORCE algorithm"""
        if len(self.reward_buffer) == 0:
            return
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.reward_buffer):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Policy gradient update
        for state, action, G in zip(self.state_buffer, self.action_buffer, returns):
            logits = self.W @ state + self.b
            probs = self.softmax(logits)
            
            # Gradient of log probability
            grad_log_probs = np.zeros_like(logits)
            grad_log_probs[action] = 1 - probs[action]
            for a in range(self.config.action_dim):
                if a != action:
                    grad_log_probs[a] = -probs[a]
            
            # Update weights
            self.W += self.config.learning_rate * G * np.outer(grad_log_probs, state)
            self.b += self.config.learning_rate * G * grad_log_probs
        
        # Clear buffers
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()


def train_agent(agent, env: QuantumEnvironment, episodes: int = 100, 
                verbose: bool = True) -> Dict:
    """Train RL agent"""
    history = {
        'episode_rewards': [],
        'mean_pss': [],
        'epsilon': [] if hasattr(agent, 'epsilon') else None
    }
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_pss = []
        
        while True:
            # Select action
            action = agent.select_action(state)
            target_qubit = np.argmin(env.qubit_states)  # Target weakest qubit
            
            # Execute action
            next_state, reward, done, info = env.step(action, target_qubit)
            
            # Update agent
            if isinstance(agent, QLearningAgent):
                agent.update(state, action, reward, next_state, done)
            elif isinstance(agent, PolicyGradientAgent):
                agent.store_transition(state, action, reward)
                if done:
                    agent.update()
            
            state = next_state
            episode_reward += reward
            episode_pss.append(info['mean_pss'])
            
            if done:
                break
        
        # Record metrics
        history['episode_rewards'].append(episode_reward)
        history['mean_pss'].append(np.mean(episode_pss))
        
        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
            history['epsilon'].append(agent.epsilon)
        
        if verbose and (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Reward={episode_reward:.2f}, Mean PSS={np.mean(episode_pss):.2f}")
    
    return history


if __name__ == "__main__":
    print("=" * 60)
    print("REINFORCEMENT LEARNING FOR QUANTUM STABILIZATION")
    print("=" * 60)
    
    # Configuration
    config = RLConfig()
    env = QuantumEnvironment(num_qubits=5)
    
    print("\nInitializing environments and agents...")
    print(f"  Environment: {env.num_qubits} qubits")
    print(f"  Action space: {config.action_dim} interventions")
    print(f"  State space: {config.state_dim} dimensions")
    
    # Train Q-Learning Agent
    print("\n" + "=" * 60)
    print("TRAINING Q-LEARNING AGENT")
    print("=" * 60)
    
    ql_agent = QLearningAgent(config)
    ql_history = train_agent(ql_agent, env, episodes=100, verbose=True)
    
    # Train Policy Gradient Agent
    print("\n" + "=" * 60)
    print("TRAINING POLICY GRADIENT AGENT")
    print("=" * 60)
    
    pg_agent = PolicyGradientAgent(config)
    pg_history = train_agent(pg_agent, env, episodes=100, verbose=True)
    
    # Compare performance
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Last 20 episodes average
    ql_avg_reward = np.mean(ql_history['episode_rewards'][-20:])
    pg_avg_reward = np.mean(pg_history['episode_rewards'][-20:])
    
    ql_avg_pss = np.mean(ql_history['mean_pss'][-20:])
    pg_avg_pss = np.mean(pg_history['mean_pss'][-20:])
    
    print(f"\nQ-Learning:")
    print(f"  Average Reward (last 20 eps): {ql_avg_reward:.2f}")
    print(f"  Average PSS (last 20 eps): {ql_avg_pss:.2f}")
    
    print(f"\nPolicy Gradient:")
    print(f"  Average Reward (last 20 eps): {pg_avg_reward:.2f}")
    print(f"  Average PSS (last 20 eps): {pg_avg_pss:.2f}")
    
    best_agent = "Q-Learning" if ql_avg_reward > pg_avg_reward else "Policy Gradient"
    print(f"\nBest performing agent: {best_agent}")
    
    # Test trained policy
    print("\n" + "=" * 60)
    print("TESTING TRAINED POLICY")
    print("=" * 60)
    
    test_env = QuantumEnvironment(num_qubits=5)
    state = test_env.reset()
    test_rewards = []
    test_pss = []
    
    for step in range(50):
        # Use Q-Learning agent for testing (no exploration)
        old_epsilon = ql_agent.epsilon
        ql_agent.epsilon = 0  # Pure exploitation
        action = ql_agent.select_action(state)
        ql_agent.epsilon = old_epsilon
        
        target_qubit = np.argmin(test_env.qubit_states)
        next_state, reward, done, info = test_env.step(action, target_qubit)
        
        test_rewards.append(reward)
        test_pss.append(info['mean_pss'])
        state = next_state
        
        if done:
            break
    
    print(f"\nTest Results (50 steps):")
    print(f"  Total Reward: {sum(test_rewards):.2f}")
    print(f"  Average PSS: {np.mean(test_pss):.2f}")
    print(f"  Min PSS: {np.min(test_pss):.2f}")
    print(f"  Final PSS by Qubit: {test_env.qubit_states}")
    
    # Intervention usage statistics
    print("\n" + "=" * 60)
    print("INTERVENTION STRATEGY ANALYSIS")
    print("=" * 60)
    
    action_counts = defaultdict(int)
    action_names = ['pulse_optimization', 'state_migration', 
                   'dynamic_scheduling', 'error_correction', 'cooling']
    
    # Analyze Q-table for Q-Learning
    for state_key, actions in ql_agent.q_table.items():
        best_action = np.argmax(actions)
        action_counts[action_names[best_action]] += 1
    
    total_actions = sum(action_counts.values())
    print("\nQ-Learning Preferred Interventions:")
    for action_name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"  {action_name}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("RL TRAINING COMPLETE")
    print("=" * 60)
