import random
from datetime import datetime
from itertools import count
from timeit import default_timer as timer

import numpy as np
import torch
from factory import loss_function_factory, model_factory, optimizer_factory
from logger import TensorBoardLogger
from memory import ReplayMemory, Transition
from networks import DQN_CNN, DQN_LSTM
from torch.optim.lr_scheduler import StepLR


class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.action_size = env.action_space.n
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger = TensorBoardLogger(self._get_log_path())
        self.memory = ReplayMemory(self.config["replay_buffer_size"])
        self.step_count = 0
        model = model_factory(self.config["model_type"])
        self.policy_network = model(
            self.action_size, config["hidden_size"]).to(self.device)
        self.target_network = model(
            self.action_size, config["hidden_size"]).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        optimizer_type = self.config["optimizer_type"]
        self.optimizer = optimizer_factory(optimizer_type)(
            self.policy_network.parameters(), lr=self.config["learning_rate"])
        self.loss_function = loss_function_factory(self.config["loss_type"])
        self.scheduler = StepLR(
            self.optimizer, step_size=self.config["scheduler_step_size"], gamma=self.config["scheduler_lr_decay_factor"])

    def _get_log_path(self):
        return self.config["logs_dir"] + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    def epsilon_greedy_action(self, state):
        self.step_count += 1
        if random.random() > self.config["epsilon_end"] + (self.config["epsilon_start"] - self.config["epsilon_end"]) * np.exp(-1. * self.step_count / self.config["epsilon_decay"]):
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def compute_q_values(self, state_batch, action_batch):
        return self.policy_network(state_batch).gather(1, action_batch)

    def compute_expected_q_values(self, next_states, rewards):
        mask = torch.tensor([s is not None for s in next_states],
                            device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in next_states if s is not None])
        v_values = torch.zeros(self.config["batch_size"], device=self.device)
        v_values[mask] = self.target_network(
            non_final_next_states).max(1)[0].detach()
        return (v_values * self.config["gamma"]) + rewards

    def optimize(self):
        if len(self.memory) < self.config["batch_size"]:
            return 0.0

        transitions = self.memory.sample(self.config["batch_size"])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).float().to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).float().to(self.device)

        q_values = self.compute_q_values(state_batch, action_batch)
        expected_q_values = self.compute_expected_q_values(
            batch.next_state, reward_batch)

        loss = self.loss_function(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.config["scheduler"]:
            self.scheduler.step()
        episode_loss = loss.detach().item()
        return episode_loss

    def train(self):
        cumulated_reward = 0.0
        best_cumulative_reward = -float('inf')
        no_improvement_count = 0
        episode_loss = 0.0

        for episode in range(self.config["num_episodes"]):
            state = self.env.reset()
            state = torch.Tensor(state).unsqueeze(0).to(self.device)

            episode_reward = 0
            episode_start = timer()
            for time_step in count():
                self.env.render()
                action = self.epsilon_greedy_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = torch.Tensor(
                    next_state).unsqueeze(0).to(self.device)
                reward = torch.tensor([reward], device=self.device)

                episode_reward += reward.item()

                self.memory.save_transition(state, action, next_state, reward)
                state = next_state
                step_loss = self.optimize()
                episode_loss += step_loss

                if done:
                    episode_end = timer()
                    episode_length = round(episode_end - episode_start, 6)
                    episode_timesteps = time_step
                    break

            cumulated_reward += episode_reward

            # Early stopping condition
            if cumulated_reward > best_cumulative_reward + self.config["reward_improvement_threshold"]:
                best_cumulative_reward = cumulated_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.config["early_stopping_patience"]:
                    print("Stopping early due to no improvement in cumulative reward.")
                    break

            print("Episode:", episode, "|| Timesteps:", time_step,
                  "|| Episode Length:", episode_length, " s", "|| Reward:", episode_reward)

            self.logger.log_scalar("episode_rewards", episode_reward, episode)
            self.logger.log_scalar("episode_lengths", episode_length, episode)
            self.logger.log_scalar("episode_timesteps",
                                   episode_timesteps, episode)
            self.logger.log_scalar("cumulated_rewards",
                                   cumulated_reward, episode)
            self.logger.log_scalar("loss",
                                   episode_loss, episode)

            if episode % self.config["target_update"] == 0:
                self.target_network.load_state_dict(
                    self.policy_network.state_dict())
