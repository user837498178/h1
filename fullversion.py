import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class HybridPredictor(nn.Module):
    def __init__(self, input_channels=3, sequence_length=300, prediction_length=48, d_model=512, n_heads=8, num_layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, 5, padding=2)
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv1d(16, 32, 5, stride=2, padding=2)
        self.norm2 = nn.GroupNorm(4, 32)
        reduced_len = sequence_length // 2
        self.num_tokens = 7
        self.linear_proj = nn.Linear(32 * reduced_len, self.num_tokens * d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, 1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_load = nn.Linear(d_model, prediction_length)
        self.fc_renew = nn.Linear(d_model, prediction_length)
        self.apply(_init_weights)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        b = x.size(0)
        x = x.reshape(b, -1)
        x = self.linear_proj(x).view(b, self.num_tokens, -1)
        x = self.transformer(x).mean(1)
        load = self.fc_load(x)
        renew = self.fc_renew(x)
        return load, renew

class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        self.apply(_init_weights)
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar.exp()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(_init_weights)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self.apply(_init_weights)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *transition):
        self.buffer.append(tuple(transition))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(lambda x: torch.stack(x, 0), zip(*batch))
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target1 = Critic(state_dim, action_dim).to(device)
        self.target2 = Critic(state_dim, action_dim).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), 3e-4)
        self.optim_critic = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), 3e-4)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], 3e-4)
        self.target_entropy = -action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.device = device
    def select_action(self, state):
        dist = self.actor(state)
        a = dist.rsample()
        return torch.tanh(a)
    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        s, a, r, s_, d = replay_buffer.sample(batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        d = d.to(self.device)
        with torch.no_grad():
            dist_ = self.actor(s_)
            a_ = torch.tanh(dist_.rsample())
            log_p = dist_.log_prob(a_).sum(-1, keepdim=True) - torch.log(1 - a_.pow(2) + 1e-6).sum(-1, keepdim=True)
            q1_target = self.target1(s_, a_)
            q2_target = self.target2(s_, a_)
            q_target = torch.min(q1_target, q2_target) - self.alpha * log_p
            y = r + self.gamma * (1 - d) * q_target
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        dist = self.actor(s)
        a_new = torch.tanh(dist.rsample())
        log_p_new = dist.log_prob(a_new).sum(-1, keepdim=True) - torch.log(1 - a_new.pow(2) + 1e-6).sum(-1, keepdim=True)
        q1_new = self.critic1(s, a_new)
        q2_new = self.critic2(s, a_new)
        actor_loss = (self.alpha * log_p_new - torch.min(q1_new, q2_new)).mean()
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        alpha_loss = -(self.log_alpha * (log_p_new + self.target_entropy).detach()).mean()
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        self.alpha = self.log_alpha.exp()
        for param, target_param in zip(self.critic1.parameters(), self.target1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__ == "__main__":
    predictor = HybridPredictor()
    bayes_lstm = BayesianLSTM(1)
    agent = SACAgent(512 + 48 * 2, 10) 