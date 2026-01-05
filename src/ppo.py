from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base_agent import BaseAgent


# --- 1. Define the Actor Network (Policy) ---
# The Actor decides which action to take given a state.
# For a discrete action space, it outputs probabilities over actions.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        # Output logits of actions
        return self.network(state)


# --- 2. Define the Critic Network (Value Function) ---
# The Critic estimates the value (expected return) of being in a given state.
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value (state-value)
        )

    def forward(self, state):
        return self.network(state)

# The Actor decides which action to take given a state.
# For a discrete action space, it outputs probabilities over actions.
# The Critic estimates the value (expected return) of being in a given state.
# They both share a feature encoder.
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        features = self.encoder(state)
        return self.actor(features), self.critic(features)

# --- 3. Define the PPO Agent ---
# This class combines the Actor and Critic, handles training, and action selection.
class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon_clip=0.2,
        K_epochs=4,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_clip=0.2,
        max_grad_norm=0.5,
        batch_size=5,
        minibatch_size=64,
        device="cuda"
    ):
        """
        PPO agent with additional improvements.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            lr_actor: Learning rate for policy network
            lr_critic: Learning rate for value network
            gamma: Discount factor for future rewards
            epsilon_clip: Clipping parameter for PPO objective (0.1-0.3)
            K_epochs: Number of optimization epochs per batch
            gae_lambda: Lambda for GAE (0=TD, 1=MC)
            entropy_coef: Coefficient for entropy bonus (exploration)
            value_clip: Clipping parameter for value function loss
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Number of episodes to collect before updating
            minibatch_size: Size of mini-batches for SGD updates
            device: Device to run the training/inference
        """
        # Initialize actor and critic networks
        # self.actor = Actor(state_dim, action_dim).to(device)
        # self.critic = Critic(state_dim).to(device)
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)

        # Define optimizers for both networks
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor) 
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor_critic.encoder.parameters()) +
            list(self.actor_critic.actor.parameters()),
            lr=lr_actor,
            #eps=1e-5
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.actor_critic.encoder.parameters()) +
            list(self.actor_critic.critic.parameters()),
            lr=lr_critic,
            #eps=1e-5
        )

        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        # Entropy coefficient for exploration bonus
        # Prevents premature convergence to deterministic policies
        self.entropy_coef = entropy_coef
        # Value clipping to prevent large value function updates
        self.value_clip = value_clip
        # Gradient clipping to prevent exploding gradients
        self.max_grad_norm = max_grad_norm
        # Batch size - collect multiple episodes before updating
        self.batch_size = batch_size
        # Mini-batch size for SGD updates during optimization
        self.minibatch_size = minibatch_size

        # Buffers for current episode
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_dones = []

        # Batch buffers for collecting multiple episodes before update
        # Previous implementation updated after every single episode
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_dones = []
        self.batch_next_values = []

        self.episodes_collected = 0
        self.training_mode = True
        
        self.device = device

    def set_training(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        self.actor_critic.train(training)

    def select_action(self, state, greedy=False):
        # Convert state (numpy array) to PyTorch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Get log probabilities from the actor (current policy)
        with torch.set_grad_enabled(self.training_mode):
            logits = self.actor_critic(state_tensor)[0]
            dist = torch.distributions.Categorical(logits=logits)

            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

            # Store experience during training
            if self.training_mode:
                value = self.actor_critic(state_tensor)[1]
                self.episode_states.append(state_tensor)
                self.episode_actions.append(action)
                self.episode_log_probs.append(dist.log_prob(action))
                self.episode_values.append(value)

        return int(action.item())

    def store_reward(self, reward):
        """Store reward for current step."""
        if self.training_mode:
            self.episode_rewards.append(reward)

    def end_episode(self, next_state=None, done=True):
        """
        New method for proper episode management and batch collection.
        Called at end of episode to finalize episode data and add to batch.
        When batch is full, triggers learning. This enables collecting multiple
        episodes before updating, improving sample efficiency.
        
        Args:
            next_state: Final state (for bootstrapping value if not done)
            done: Whether episode terminated naturally
        """
        if not self.training_mode or len(self.episode_states) == 0:
            return

        # Calculate next state value for GAE
        if done:
            next_value = torch.tensor([[0.0]])
        elif next_state is not None:
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
            with torch.no_grad():
                next_value = self.actor_critic(next_state_tensor)[1]
        else:
            next_value = torch.tensor([[0.0]])

        # Add episode to batch
        self.batch_states.extend(self.episode_states)
        self.batch_actions.extend(self.episode_actions)
        self.batch_log_probs.extend(self.episode_log_probs)
        self.batch_rewards.extend(self.episode_rewards)
        self.batch_values.extend(self.episode_values)
        self.batch_next_values.append(next_value)
        
        # Add done flags for each step in episode
        episode_len = len(self.episode_rewards)
        self.batch_dones.extend([False] * (episode_len - 1) + [done])

        # Clear episode buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_dones = []

        self.episodes_collected += 1

        # Learn only when batch is full (not after every episode)
        # This improves sample efficiency and training stability
        if self.episodes_collected >= self.batch_size:
            self.learn()
            self.episodes_collected = 0

    def store_transition(self, reward, done):
        """Legacy method for compatibility. Use store_reward() instead."""
        self.store_reward(reward)

    def calculate_gae(self):
        """
        Calculate Generalized Advantage Estimation (GAE).
        More stable than simple TD error by balancing bias and variance.
        """
        advantages = []
        returns = []
        last_advantage = 0

        # Convert lists to tensors
        rewards = torch.tensor(self.batch_rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.batch_dones, dtype=torch.float32, device=self.device)

        # Concatenate all next values (last value of each episode)
        values = torch.cat(self.batch_values).squeeze().detach().to(self.device)
        next_values = torch.cat(self.batch_next_values).squeeze().detach().to(self.device)
        
        # Build extended values array
        ext_values = torch.zeros(len(rewards) + 1)
        ext_values[:-1] = values
        
        # Set next values for episode boundaries
        episode_end_idx = 0
        for i, is_done in enumerate(dones):
            if is_done or i == len(dones) - 1:
                ext_values[i + 1] = next_values[episode_end_idx]
                episode_end_idx += 1
            else:
                ext_values[i + 1] = values[i + 1] if i + 1 < len(values) else 0

        # Calculate GAE in reverse
        for t in reversed(range(len(rewards))):
            # TD error for current step
            delta = (
                rewards[t]
                + self.gamma * ext_values[t + 1] * (1 - dones[t])
                - values[t]
            )
            # GAE formula: detach to prevent gradients through value
            last_advantage = (
                delta.detach()
                + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )
            advantages.insert(0, last_advantage)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values
        return advantages, returns

    def learn(self):
        """Update policy and value networks using collected batch."""
        if len(self.batch_states) == 0:
            return

        # Calculate advantages and returns using GAE
        advantages, returns = self.calculate_gae()

        # Convert stored lists to tensors for batch processing
        old_states = torch.cat(self.batch_states).squeeze(1)
        old_actions = torch.cat(self.batch_actions)
        old_log_probs = torch.cat(self.batch_log_probs)
        old_values = torch.cat(self.batch_values).squeeze()

        # Normalize advantages (important for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get total batch size
        batch_size = old_states.size(0)

        # PPO Optimization with Mini-Batches
        # Standard PPO shuffles data and processes in mini-batches for each epoch
        for epoch in range(self.K_epochs):
            # Generate random permutation for shuffling
            indices = torch.randperm(batch_size)
            
            # Process data in mini-batches
            for start_idx in range(0, batch_size, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, batch_size)
                minibatch_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch
                mb_states = old_states[minibatch_indices]
                mb_actions = old_actions[minibatch_indices]
                mb_old_log_probs = old_log_probs[minibatch_indices]
                mb_old_values = old_values[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                
                # Get new log probabilities and values for mini-batch
                new_log_probs_dist = torch.distributions.Categorical(
                    logits=self.actor_critic(mb_states)[0]
                )
                new_log_probs = new_log_probs_dist.log_prob(mb_actions)
                new_values = self.actor_critic(mb_states)[1].squeeze(-1)
                
                # Entropy bonus encourages exploration and prevents
                # premature convergence to deterministic policies
                entropy = new_log_probs_dist.entropy().mean()

                # --- Critic Loss with Value Clipping ---
                # Value function clipping (similar to policy clipping)
                # Prevents destructively large updates to the value function
                # Takes the maximum of clipped and unclipped loss for conservatism
                value_pred_clipped = mb_old_values.detach() + torch.clamp(
                    new_values - mb_old_values.detach(),
                    -self.value_clip,
                    self.value_clip
                )
                value_loss_unclipped = F.mse_loss(new_values, mb_returns.detach())
                value_loss_clipped = F.mse_loss(
                    value_pred_clipped, mb_returns.detach()
                )
                critic_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                # --- Actor Loss (Clipped Surrogate Objective) ---
                # Ratio of new policy probability to old policy probability
                ratio = torch.exp(new_log_probs - mb_old_log_probs.detach())

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                    * mb_advantages
                )
                # Add entropy bonus to actor loss
                # Encourages exploration by rewarding policy diversity
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                )

                # --- Update Networks with Gradient Clipping ---
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()

                critic_loss.backward(retain_graph=True)  # keep graph for shared encoder
                actor_loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                self.critic_optimizer.step()
                self.actor_optimizer.step()

        # Clear batch buffers
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_values = []
        self.batch_dones = []
        self.batch_next_values = []

    def save(self, path):
        torch.save(
            {
                # "actor_state_dict": self.actor.state_dict(),
                # "critic_state_dict": self.critic.state_dict(),
                "actor_critic_state_dict": self.actor_critic.state_dict()
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        # self.actor.load_state_dict(checkpoint["actor_state_dict"])
        # self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])


# --- 4. Training Loop ---
def train(
    env, state_dim, action_dim, num_episodes, max_steps_per_episode, target_score
):
    # Initialize the PPO agent
    agent = PPOAgent(state_dim, action_dim)

    scores_deque = deque(maxlen=100)
    scores = []

    print("Starting PPO training...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)

            state = next_state
            episode_reward += reward

            if done:
                break

        # End episode with next state for bootstrapping
        agent.end_episode(next_state=state if not done else None, done=done)

        scores_deque.append(episode_reward)
        scores.append(episode_reward)

        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores_deque):.2f}")

        # Check if solved (average over 100 episodes >= target)
        if np.mean(scores_deque) >= target_score:
            avg_score = np.mean(scores_deque)
            print(f"\nSolved in {episode} episodes! Average Score: {avg_score:.2f}")
            break

    print("\nTraining complete.")

    return agent, scores
