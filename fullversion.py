import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

def _w(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class SpatioTemporalPredictor(nn.Module):
    def __init__(self, seq_len=300, pred_len=48):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.norm2 = nn.GroupNorm(4, 32)
        flat_dim = 32 * 75 * 75
        self.num_tokens = 7
        self.d_model = 512
        self.proj = nn.Linear(flat_dim, self.num_tokens * self.d_model)
        enc_layer = nn.TransformerEncoderLayer(self.d_model, 8, 1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, 2)
        self.head_load = nn.Linear(self.d_model, pred_len)
        self.head_renew = nn.Linear(self.d_model, pred_len)
        self.apply(_w)
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        b = x.size(0)
        x = x.view(b, -1)
        x = self.proj(x).view(b, self.num_tokens, self.d_model)
        x = self.transformer(x).mean(1)
        load = self.head_load(x)
        renew = self.head_renew(x)
        return load, renew

class BayesianLSTM(nn.Module):
    def __init__(self, in_dim, hid=128, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, 2, batch_first=True, dropout=0.2)
        self.mu = nn.Linear(hid, out_dim)
        self.var = nn.Linear(hid, out_dim)
        self.apply(_w)
    def forward(self, x):
        h, _ = self.lstm(x)
        h = h[:, -1]
        return self.mu(h), F.softplus(self.var(h)) + 1e-6

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.ms_enc = nn.Linear(64, 128)
        self.node_enc = nn.Linear(64, 128)
        self.gru = nn.GRU(128, 256, batch_first=True)
        self.fc = nn.Linear(384, action_dim)
        self.apply(_w)
    def forward(self, ms, node):
        ms = F.relu(self.ms_enc(ms))
        node = F.relu(self.node_enc(node)).unsqueeze(1)
        _, h = self.gru(node)
        h = h[-1]
        z = torch.cat([ms, h], -1)
        return F.softmax(self.fc(z), -1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.q = nn.Linear(512, 1)
        self.apply(_w)
    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

class Buffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)
    def push(self, *t):
        self.buf.append(tuple(t))
    def sample(self, n):
        b = random.sample(self.buf, n)
        return [torch.stack(i) for i in zip(*b)]
    def __len__(self):
        return len(self.buf)

class SAC:
    def __init__(self, s_dim, a_dim, device="cpu"):
        self.act_dim = a_dim
        self.actor = Actor(s_dim, a_dim).to(device)
        self.critic1 = Critic(s_dim, a_dim).to(device)
        self.critic2 = Critic(s_dim, a_dim).to(device)
        self.t1 = Critic(s_dim, a_dim).to(device)
        self.t2 = Critic(s_dim, a_dim).to(device)
        self.t1.load_state_dict(self.critic1.state_dict())
        self.t2.load_state_dict(self.critic2.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), 3e-4)
        self.opt_c = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), 3e-4)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], 3e-4)
        self.alpha = self.log_alpha.exp()
        self.ent_target = -a_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.d = device
    def select(self, s):
        with torch.no_grad():
            p = self.actor(s[:, :64], s[:, 64:])
            dist = torch.distributions.Categorical(p)
            a = F.one_hot(dist.sample(), self.act_dim).float()
            return a
    def train_step(self, buf, bs):
        if len(buf) < bs:
            return
        s, a, r, s2, d = [x.to(self.d) for x in buf.sample(bs)]
        with torch.no_grad():
            p2 = self.actor(s2[:, :64], s2[:, 64:])
            a2 = F.one_hot(torch.distributions.Categorical(p2).sample(), self.act_dim).float()
            logp2 = torch.log((p2 * a2).sum(-1, keepdim=True) + 1e-6)
            q1t = self.t1(s2, a2)
            q2t = self.t2(s2, a2)
            qt = torch.min(q1t, q2t) - self.alpha * logp2
            y = r + self.gamma * (1 - d) * qt
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad()
        c_loss.backward()
        self.opt_c.step()
        p = self.actor(s[:, :64], s[:, 64:])
        a_new = F.one_hot(torch.distributions.Categorical(p).sample(), self.act_dim).float()
        logp = torch.log((p * a_new).sum(-1, keepdim=True) + 1e-6)
        q1n = self.critic1(s, a_new)
        q2n = self.critic2(s, a_new)
        a_loss = (self.alpha * logp - torch.min(q1n, q2n)).mean()
        self.opt_a.zero_grad()
        a_loss.backward()
        self.opt_a.step()
        a_loss_alpha = -(self.log_alpha * (logp + self.ent_target).detach()).mean()
        self.opt_alpha.zero_grad()
        a_loss_alpha.backward()
        self.opt_alpha.step()
        self.alpha = self.log_alpha.exp()
        for p, tp in zip(self.critic1.parameters(), self.t1.parameters()):
            tp.data.copy_(self.tau * p + (1 - self.tau) * tp)
        for p, tp in zip(self.critic2.parameters(), self.t2.parameters()):
            tp.data.copy_(self.tau * p + (1 - self.tau) * tp)

class IntegratedController:
    def __init__(self):
        self.pred = SpatioTemporalPredictor()
        self.bayes = BayesianLSTM(1)
        self.sac = SAC(192, 10)
        self.buf = Buffer(100000)
    def forward(self, x, sys_state):
        l, r = self.pred(x)
        ms = torch.cat([l, r], -1)
        node = sys_state
        action = self.sac.select(torch.cat([ms, node], -1))
        return action

if __name__ == "__main__":
    model = IntegratedController() 