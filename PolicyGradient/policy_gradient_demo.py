"""pip install torch
python policy_gradient_demo.py --method 6 --episodes 300
Six estimators, one finite-horizon on-policy toy environment.
"""
import argparse
import torch
from torch import nn
from torch.distributions import Categorical

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=int, choices=range(1, 7), default=6)
parser.add_argument('--episodes', type=int, default=300)
args = parser.parse_args()
torch.manual_seed(7)
gamma, horizon = 0.9, 3
# State is a one-hot time index (finite-horizon Markov state).
def net(out):
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, out))
actor, value, qnet = net(2), net(1), net(2)
actor_opt = torch.optim.Adam(actor.parameters(), lr=0.01)
critic_opt = torch.optim.Adam(list(value.parameters()) + list(qnet.parameters()), lr=0.01)
states = torch.eye(horizon)
for episode in range(args.episodes):
    # Complete fresh episode, with parameters fixed while collecting.
    with torch.no_grad():
        actions = Categorical(logits=actor(states)).sample()
        # Action 1 has higher expected reward at every step.
        rewards = (2 * actions.float() - 1) + 0.3 * torch.randn(horizon)
        returns = torch.zeros(horizon)
        running = 0.0
        for t in reversed(range(horizon)):
            running = rewards[t] + gamma * running
            returns[t] = running
    logp = Categorical(logits=actor(states)).log_prob(actions)
    v = value(states).squeeze(-1)
    q = qnet(states).gather(1, actions[:, None]).squeeze(1)
    discounts = gamma ** torch.arange(horizon)
    # All actor weights use the pre-update critic, detached below.
    if args.method == 1:
        weights = returns[0].expand(horizon)  # NO extra gamma**t
    elif args.method == 2:
        weights = discounts * returns
    elif args.method == 3:
        weights = discounts * (returns - v)
    elif args.method == 4:
        weights = discounts * q
    elif args.method == 5:
        weights = discounts * (q - v)
    else:
        # Last next state is truly terminal, so its value is zero.
        target = rewards + gamma * torch.cat([v.detach()[1:], v.new_zeros(1)])
        weights = discounts * (target - v)
    actor_loss = -(weights.detach() * logp).sum()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    # MC targets for methods 3–5; detached one-step TD target for 6.
    critic_loss = None
    if args.method in (3, 5):
        critic_loss = 0.5 * (v - returns).square().mean()
    if args.method in (4, 5):
        q_loss = 0.5 * (q - returns).square().mean()
        critic_loss = q_loss if critic_loss is None else critic_loss + q_loss
    if args.method == 6:
        critic_loss = 0.5 * (v - target.detach()).square().mean()
    if critic_loss is not None:
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
    if episode % 50 == 0 or episode == args.episodes - 1:
        with torch.no_grad():
            p = actor(states).softmax(-1)[:, 1].mean().item()
        print(f'episode={episode:4d} mean P(action=1)={p:.3f}')
# Educational example: one episode/update, no replay, no entropy bonus.
# Approximate Q / TD critics can bias the actor gradient; convergence is not guaranteed.
