"""
Step 2: Reinforcement Learning & Adaptive Control System

This module implements advanced reinforcement learning algorithms for 
adaptive quantum stabilization strategies.

Components:
- Q-Learning Agent for discrete action spaces
- Deep Q-Network (DQN) for high-dimensional state spaces
- Policy Gradient methods for continuous control
- Multi-Agent RL for coordinated multi-qubit stabilization
- Adaptive intervention engine with learned policies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import random
from enum import Enum
import json


# ============================================================================
# Action Space Definition
# ============================================================================

class InterventionAction(Enum):
    """Available intervention actions for quantum stabilization"""
    NO_OP = 0
    PULSE_OPTIMIZATION_LOW = 1
    PULSE_OPTIMIZATION_MED = 2
    PULSE_OPTIMIZATION_HIGH = 3
    STATE_MIGRATION_NEAR = 4
    STATE_MIGRATION_FAR = 5
    DYNAMIC_SCHEDULING_REDUCE = 6
    DYNAMIC_SCHEDULING_PAUSE = 7
    ERROR_CORRECTION_INCREASE = 8
    ERROR_CORRECTION_DECREASE = 9
    FREQUENCY_TUNING = 10
    TEMPERATURE_ADJUSTMENT = 11
    GATE_CALIBRATION = 12
    ENTANGLEMENT_SWAP = 13
    DECOUPLING_SEQUENCE = 14


@dataclass
class ActionEffect:
    """Represents the effect of an intervention action"""
    action: InterventionAction
    success_probability: float
    coherence_gain: float
    fidelity_improvement: float
    resource_cost: float
    latency_ms: float
    side_effects: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Environment Model
# ============================================================================

class QuantumStabilityEnv:
    """
    Reinforcement Learning Environment for Quantum Stability Control
    
    State Space:
    - PSS scores for all qubits
    - T1, T2 times
    - Gate fidelities
    - Noise levels
    - Recent error history
    
    Action Space:
    - 15 discrete intervention actions per qubit
    
    Reward Function:
    - Extended coherence time
    - Reduced error rate
    - Improved fidelity
    - Penalized resource usage
    """
    
    def __init__(self, n_qubits: int = 10, max_steps: int = 1000):
        self.n_qubits = n_qubits
        self.max_steps = max_steps
        self.current_step = 0
        
        # State dimensions
        self.state_dim = n_qubits * 8  # 8 features per qubit
        
        # Action space
        self.n_actions = len(InterventionAction)
        
        # Initialize qubit states
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        
        # Initialize qubit parameters (normalized 0-1)
        self.pss_scores = np.ones(self.n_qubits) * 0.95
        self.t1_times = np.ones(self.n_qubits) * 100e-6  # 100 microseconds
        self.t2_times = np.ones(self.n_qubits) * 80e-6
        self.gate_fidelities = np.ones(self.n_qubits) * 0.999
        self.noise_levels = np.ones(self.n_qubits) * 0.01
        self.error_counts = np.zeros(self.n_qubits)
        self.coherence_history = deque(maxlen=10)
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector from qubit parameters"""
        state = np.zeros(self.state_dim)
        for i in range(self.n_qubits):
            offset = i * 8
            state[offset] = self.pss_scores[i]
            state[offset + 1] = self.t1_times[i] / 100e-6  # Normalize
            state[offset + 2] = self.t2_times[i] / 80e-6
            state[offset + 3] = self.gate_fidelities[i]
            state[offset + 4] = self.noise_levels[i] * 100
            state[offset + 5] = min(self.error_counts[i] / 10, 1.0)
            state[offset + 6] = np.mean(list(self.coherence_history)[-5:]) if self.coherence_history else 0.95
            state[offset + 7] = self.current_step / self.max_steps
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Decode action
        qubit_idx = action % self.n_qubits
        action_type = InterventionAction(action // self.n_qubits % self.n_actions)
        
        # Apply action effects
        effect = self._apply_action(qubit_idx, action_type)
        
        # Simulate natural decoherence
        self._simulate_decoherence()
        
        # Calculate reward
        reward = self._calculate_reward(effect)
        
        # Check termination
        done = self.current_step >= self.max_steps or np.mean(self.pss_scores) < 0.3
        
        # Update history
        self.coherence_history.append(np.mean(self.pss_scores))
        
        info = {
            'effect': effect,
            'avg_pss': np.mean(self.pss_scores),
            'avg_fidelity': np.mean(self.gate_fidelities),
            'total_errors': np.sum(self.error_counts)
        }
        
        return self._get_state(), reward, done, info
    
    def _apply_action(self, qubit_idx: int, action: InterventionAction) -> ActionEffect:
        """Apply intervention action to specific qubit"""
        # Base effects for each action type
        base_effects = {
            InterventionAction.NO_OP: ActionEffect(action, 1.0, 0.0, 0.0, 0.0, 0.0),
            InterventionAction.PULSE_OPTIMIZATION_LOW: ActionEffect(action, 0.85, 5e-6, 0.001, 0.1, 1.0),
            InterventionAction.PULSE_OPTIMIZATION_MED: ActionEffect(action, 0.80, 10e-6, 0.002, 0.2, 2.0),
            InterventionAction.PULSE_OPTIMIZATION_HIGH: ActionEffect(action, 0.75, 15e-6, 0.003, 0.3, 3.0),
            InterventionAction.STATE_MIGRATION_NEAR: ActionEffect(action, 0.90, 20e-6, 0.005, 0.5, 5.0, {'crosstalk': 0.02}),
            InterventionAction.STATE_MIGRATION_FAR: ActionEffect(action, 0.85, 25e-6, 0.006, 0.7, 8.0, {'crosstalk': 0.01}),
            InterventionAction.DYNAMIC_SCHEDULING_REDUCE: ActionEffect(action, 0.95, 8e-6, 0.001, 0.05, 0.5, {'throughput': -0.1}),
            InterventionAction.DYNAMIC_SCHEDULING_PAUSE: ActionEffect(action, 0.98, 15e-6, 0.002, 0.1, 1.0, {'throughput': -0.3}),
            InterventionAction.ERROR_CORRECTION_INCREASE: ActionEffect(action, 0.90, 3e-6, 0.004, 0.4, 2.0, {'overhead': 0.2}),
            InterventionAction.ERROR_CORRECTION_DECREASE: ActionEffect(action, 0.85, -2e-6, -0.001, -0.2, 1.0, {'overhead': -0.1}),
            InterventionAction.FREQUENCY_TUNING: ActionEffect(action, 0.80, 12e-6, 0.003, 0.25, 4.0),
            InterventionAction.TEMPERATURE_ADJUSTMENT: ActionEffect(action, 0.75, 18e-6, 0.004, 0.5, 10.0),
            InterventionAction.GATE_CALIBRATION: ActionEffect(action, 0.88, 6e-6, 0.005, 0.15, 3.0),
            InterventionAction.ENTANGLEMENT_SWAP: ActionEffect(action, 0.82, 10e-6, 0.002, 0.35, 6.0),
            InterventionAction.DECOUPLING_SEQUENCE: ActionEffect(action, 0.92, 20e-6, 0.006, 0.3, 2.5),
        }
        
        effect = base_effects.get(action, base_effects[InterventionAction.NO_OP])
        
        # Apply with success probability
        if random.random() < effect.success_probability:
            # Update qubit parameters
            self.pss_scores[qubit_idx] = min(1.0, self.pss_scores[qubit_idx] + effect.coherence_gain / 100e-6)
            self.t1_times[qubit_idx] = min(150e-6, self.t1_times[qubit_idx] + effect.coherence_gain)
            self.t2_times[qubit_idx] = min(120e-6, self.t2_times[qubit_idx] + effect.coherence_gain * 0.8)
            self.gate_fidelities[qubit_idx] = min(0.9999, self.gate_fidelities[qubit_idx] + effect.fidelity_improvement)
            self.noise_levels[qubit_idx] = max(0.001, self.noise_levels[qubit_idx] * 0.95)
            
            # Apply side effects
            for key, value in effect.side_effects.items():
                if key == 'crosstalk':
                    neighbor_idxs = [(qubit_idx + 1) % self.n_qubits, (qubit_idx - 1) % self.n_qubits]
                    for idx in neighbor_idxs:
                        self.noise_levels[idx] = min(0.1, self.noise_levels[idx] + value)
                elif key == 'throughput':
                    pass  # Handled in reward
                elif key == 'overhead':
                    pass  # Handled in reward
        else:
            # Action failed, small penalty
            self.pss_scores[qubit_idx] = max(0.5, self.pss_scores[qubit_idx] - 0.02)
        
        return effect
    
    def _simulate_decoherence(self):
        """Simulate natural decoherence processes"""
        decay_rate = 0.001
        
        for i in range(self.n_qubits):
            # T1/T2 decay
            self.t1_times[i] *= (1 - decay_rate)
            self.t2_times[i] *= (1 - decay_rate * 1.2)
            
            # PSS decay based on noise
            self.pss_scores[i] -= self.noise_levels[i] * decay_rate * 10
            
            # Random noise fluctuations
            self.noise_levels[i] += np.random.normal(0, 0.001)
            self.noise_levels[i] = np.clip(self.noise_levels[i], 0.001, 0.1)
            
            # Error accumulation
            if self.pss_scores[i] < 0.7:
                self.error_counts[i] += 1
            
            # Clamp values
            self.pss_scores[i] = np.clip(self.pss_scores[i], 0.0, 1.0)
            self.t1_times[i] = np.clip(self.t1_times[i], 10e-6, 150e-6)
            self.t2_times[i] = np.clip(self.t2_times[i], 5e-6, 120e-6)
            self.gate_fidelities[i] = np.clip(self.gate_fidelities[i], 0.9, 0.9999)
    
    def _calculate_reward(self, effect: ActionEffect) -> float:
        """Calculate reward based on system performance"""
        # Base reward from average PSS
        avg_pss = np.mean(self.pss_scores)
        reward = avg_pss * 10
        
        # Bonus for high fidelity
        avg_fidelity = np.mean(self.gate_fidelities)
        reward += avg_fidelity * 5
        
        # Penalty for errors
        total_errors = np.sum(self.error_counts)
        reward -= total_errors * 0.5
        
        # Penalty for resource cost
        reward -= effect.resource_cost * 2
        
        # Penalty for low PSS qubits
        low_pss_count = np.sum(self.pss_scores < 0.5)
        reward -= low_pss_count * 3
        
        # Bonus for stability (low variance)
        pss_variance = np.var(self.pss_scores)
        reward -= pss_variance * 5
        
        return reward


# ============================================================================
# Deep Q-Network (DQN)
# ============================================================================

class DQN(nn.Module):
    """Deep Q-Network for quantum stability control"""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = [256, 256, 128]):
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # Add dropout only between layers, not before output
                layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    """
    DQN Agent with experience replay and target network
    """
    
    def __init__(self, state_dim: int, n_actions: int, 
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 10):
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.policy_net = DQN(state_dim, n_actions)
        self.target_net = DQN(state_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=buffer_size)
        self.Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])
        
        self.steps = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, 
                        next_state: np.ndarray, reward: float):
        """Store transition in replay buffer"""
        self.memory.append(self.Transition(state, action, next_state, reward))
    
    def optimize_model(self):
        """Optimize policy network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        transitions = random.sample(self.memory, self.batch_size)
        batch = self.Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        reward_batch = torch.FloatTensor(batch.reward)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_q_values_max = next_q_values.max(1)[0]
            expected_q_values = reward_batch + self.gamma * next_q_values_max
        
        # Loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


# ============================================================================
# Policy Gradient (PPO-style)
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network for continuous control"""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        policy = self.actor(shared)
        value = self.critic(shared)
        return policy, value


class PolicyGradientAgent:
    """
    Policy Gradient Agent with baseline (Actor-Critic)
    """
    
    def __init__(self, state_dim: int, n_actions: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.model = ActorCritic(state_dim, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, _ = self.model(state_tensor)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            return action.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, done: bool):
        """Store transition for batch update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def optimize(self):
        """Optimize policy using collected trajectories"""
        if len(self.states) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.FloatTensor(self.dones)
        
        # Compute advantages using GAE
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze()
            
            # Generalized Advantage Estimation
            advantages = torch.zeros_like(rewards)
            delta = rewards + self.gamma * values * (1 - dones) - values
            
            running_advantage = 0
            for t in reversed(range(len(rewards))):
                running_advantage = delta[t] + self.gamma * self.gae_lambda * running_advantage * (1 - dones[t])
                advantages[t] = running_advantage
        
        # Get policy log probabilities
        policy, _ = self.model(states)
        log_probs = torch.log(policy.gather(1, actions.unsqueeze(1)).squeeze(1))
        
        # Policy loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        _, values = self.model(states)
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        
        return loss.item()
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))


# ============================================================================
# Multi-Agent RL for Coordinated Stabilization
# ============================================================================

class MultiAgentRLController:
    """
    Multi-Agent RL controller for coordinated multi-qubit stabilization
    
    Each qubit has its own agent, with communication between neighboring agents
    """
    
    def __init__(self, n_qubits: int, agent_type: str = 'dqn'):
        self.n_qubits = n_qubits
        self.agent_type = agent_type
        
        # Create environment
        self.env = QuantumStabilityEnv(n_qubits=n_qubits)
        
        # State dimension includes neighbor information (8 + 8 from neighbors)
        augmented_state_dim = 8 + 8  # per-qubit state + neighbor summary
        
        # Create agents (one per qubit)
        self.agents = []
        for i in range(n_qubits):
            if agent_type == 'dqn':
                agent = DQNAgent(
                    state_dim=augmented_state_dim,  # Augmented with neighbor info
                    n_actions=len(InterventionAction)
                )
            else:
                agent = PolicyGradientAgent(
                    state_dim=augmented_state_dim,
                    n_actions=len(InterventionAction)
                )
            self.agents.append(agent)
        
        # Communication graph (neighboring qubits)
        self.neighbors = {i: [(i-1) % n_qubits, (i+1) % n_qubits] for i in range(n_qubits)}
        
    def train_episode(self, max_steps: int = 500) -> Dict[str, float]:
        """Train one episode with all agents"""
        state = self.env.reset()
        total_reward = 0
        episode_rewards = []
        
        for step in range(max_steps):
            # Select actions for all qubits
            actions = []
            for i in range(self.n_qubits):
                qubit_state = state[i*8:(i+1)*8]
                
                # Incorporate neighbor information
                neighbor_states = []
                for neighbor_idx in self.neighbors[i]:
                    neighbor_state = state[neighbor_idx*8:(neighbor_idx+1)*8]
                    neighbor_states.extend(neighbor_state[:4])  # First 4 features
                
                augmented_state = np.concatenate([qubit_state, np.array(neighbor_states)])
                
                action = self.agents[i].select_action(augmented_state)
                actions.append(action)
            
            # Execute actions
            next_state, reward, done, info = self.env.step(sum(actions) % self.env.n_actions)
            total_reward += reward
            episode_rewards.append(reward)
            
            # Store transitions
            for i in range(self.n_qubits):
                qubit_state = state[i*8:(i+1)*8]
                next_qubit_state = next_state[i*8:(i+1)*8]
                
                neighbor_states = []
                for neighbor_idx in self.neighbors[i]:
                    neighbor_state = next_state[neighbor_idx*8:(neighbor_idx+1)*8]
                    neighbor_states.extend(neighbor_state[:4])
                
                augmented_next_state = np.concatenate([next_qubit_state, np.array(neighbor_states)])
                
                self.agents[i].store_transition(
                    np.concatenate([qubit_state, np.array(neighbor_states)]),
                    actions[i],
                    augmented_next_state,
                    reward / self.n_qubits  # Shared reward
                )
            
            # Optimize agents
            for agent in self.agents:
                if isinstance(agent, DQNAgent):
                    agent.optimize_model()
            
            state = next_state
            
            if done:
                break
        
        # Periodic policy gradient update
        if self.agent_type == 'pg':
            for agent in self.agents:
                agent.optimize()
        
        return {
            'total_reward': total_reward,
            'avg_reward': np.mean(episode_rewards),
            'final_pss': info['avg_pss'],
            'final_fidelity': info['avg_fidelity'],
            'total_errors': info['total_errors']
        }
    
    def train(self, n_episodes: int = 1000, eval_interval: int = 50) -> List[Dict]:
        """Train for multiple episodes"""
        results = []
        
        for episode in range(n_episodes):
            episode_result = self.train_episode()
            episode_result['episode'] = episode
            results.append(episode_result)
            
            if episode % eval_interval == 0:
                print(f"Episode {episode}: "
                      f"Reward={episode_result['avg_reward']:.2f}, "
                      f"PSS={episode_result['final_pss']:.3f}, "
                      f"Fidelity={episode_result['final_fidelity']:.4f}")
        
        return results
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """Evaluate trained policy"""
        all_rewards = []
        all_pss = []
        all_fidelities = []
        all_errors = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(500):
                actions = []
                for i in range(self.n_qubits):
                    qubit_state = state[i*8:(i+1)*8]
                    action = self.agents[i].select_action(qubit_state, training=False)
                    actions.append(action)
                
                state, reward, done, info = self.env.step(sum(actions) % self.env.n_actions)
                episode_reward += reward
                
                if done:
                    break
            
            all_rewards.append(episode_reward)
            all_pss.append(info['avg_pss'])
            all_fidelities.append(info['avg_fidelity'])
            all_errors.append(info['total_errors'])
        
        return {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_pss': np.mean(all_pss),
            'avg_fidelity': np.mean(all_fidelities),
            'avg_errors': np.mean(all_errors),
            'success_rate': np.mean([p > 0.8 for p in all_pss])
        }


# ============================================================================
# Adaptive Intervention Engine
# ============================================================================

class AdaptiveInterventionEngine:
    """
    Production-ready intervention engine using trained RL policies
    """
    
    def __init__(self, n_qubits: int, model_path: Optional[str] = None):
        self.n_qubits = n_qubits
        self.controller = MultiAgentRLController(n_qubits, agent_type='dqn')
        
        if model_path:
            self.load_models(model_path)
    
    def load_models(self, model_dir: str):
        """Load pre-trained models"""
        import os
        for i, agent in enumerate(self.controller.agents):
            model_path = os.path.join(model_dir, f'agent_{i}.pth')
            if os.path.exists(model_path):
                agent.load(model_path)
    
    def recommend_intervention(self, telemetry_data: Dict) -> List[Dict]:
        """
        Recommend interventions based on current telemetry
        
        Args:
            telemetry_data: Dictionary with qubit telemetry
            
        Returns:
            List of recommended interventions
        """
        recommendations = []
        
        for qubit_id, data in telemetry_data.items():
            # Construct state vector (8 features)
            qubit_state = np.array([
                data.get('pss', 0.95),
                data.get('t1', 100e-6) / 100e-6,
                data.get('t2', 80e-6) / 80e-6,
                data.get('gate_fidelity', 0.999),
                data.get('noise_level', 0.01) * 100,
                min(data.get('error_count', 0) / 10, 1.0),
                data.get('recent_coherence', 0.95),
                0.0  # Step placeholder
            ])
            
            # Get action from agent
            qubit_idx = int(qubit_id.split('_')[1]) if '_' in qubit_id else int(qubit_id)
            agent = self.controller.agents[qubit_idx % self.n_qubits]
            
            # Create augmented state with zeros for neighbors (since we don't have that info)
            neighbor_state = np.zeros(8)
            augmented_state = np.concatenate([qubit_state, neighbor_state])
            
            action_idx = agent.select_action(augmented_state, training=False)
            
            action = InterventionAction(action_idx)
            
            recommendations.append({
                'qubit_id': qubit_id,
                'action': action.name,
                'confidence': 1.0 - agent.epsilon,
                'priority': 'high' if data.get('pss', 1.0) < 0.7 else 'medium' if data.get('pss', 1.0) < 0.85 else 'low'
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def execute_intervention(self, recommendation: Dict) -> bool:
        """Execute recommended intervention"""
        # In production, this would interface with quantum hardware
        print(f"Executing {recommendation['action']} on qubit {recommendation['qubit_id']}")
        return True


# ============================================================================
# Training Script
# ============================================================================

def train_rl_system(n_qubits: int = 10, n_episodes: int = 500, save_path: str = 'rl_models'):
    """
    Train the complete RL system
    
    Args:
        n_qubits: Number of qubits to simulate
        n_episodes: Number of training episodes
        save_path: Directory to save trained models
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Training Multi-Agent RL system with {n_qubits} qubits...")
    print("=" * 60)
    
    controller = MultiAgentRLController(n_qubits, agent_type='dqn')
    results = controller.train(n_episodes=n_episodes, eval_interval=50)
    
    # Save models
    for i, agent in enumerate(controller.agents):
        agent.save(os.path.join(save_path, f'agent_{i}.pth'))
    
    # Save training results
    with open(os.path.join(save_path, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    eval_results = controller.evaluate(n_episodes=50)
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print(f"\nModels saved to: {save_path}")
    
    return controller, results


if __name__ == '__main__':
    # Run training
    controller, results = train_rl_system(n_qubits=10, n_episodes=200)
    
    # Test intervention engine
    print("\n" + "=" * 60)
    print("Testing Adaptive Intervention Engine:")
    
    engine = AdaptiveInterventionEngine(n_qubits=10)
    
    # Simulate telemetry
    telemetry = {
        'qubit_0': {'pss': 0.65, 't1': 80e-6, 't2': 60e-6, 'gate_fidelity': 0.995, 'noise_level': 0.05, 'error_count': 3},
        'qubit_1': {'pss': 0.82, 't1': 95e-6, 't2': 75e-6, 'gate_fidelity': 0.998, 'noise_level': 0.02, 'error_count': 1},
        'qubit_2': {'pss': 0.91, 't1': 105e-6, 't2': 85e-6, 'gate_fidelity': 0.999, 'noise_level': 0.01, 'error_count': 0},
    }
    
    recommendations = engine.recommend_intervention(telemetry)
    for rec in recommendations:
        print(f"  Qubit {rec['qubit_id']}: {rec['action']} (Priority: {rec['priority']}, Confidence: {rec['confidence']:.2f})")
