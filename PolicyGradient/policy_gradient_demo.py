"""Six on-policy policy gradients. Install: pip install -r requirements.txt."""
import argparse
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Categorical


def mlp(out):
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, out))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(2)

    def forward(self, states):
        return Categorical(logits=self.net(states))


class ValueCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(1)

    def forward(self, states):
        return self.net(states).squeeze(-1)


class QCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(2)

    def forward(self, states, actions):
        return self.net(states).gather(-1, actions.unsqueeze(-1)).squeeze(-1)


class ToyEnv:
    """Three-step finite-horizon task; time is part of the Markov state.

    The horizon is a genuine task termination, not a sampler truncation.
    Action 1 has larger expected reward. No Gym dependency.
    """
    def reset(self):
        self.t = 0
        return torch.eye(3)[0]

    def step(self, action):
        reward = 2.0 * action - 1.0 + 0.3 * torch.randn(()).item()
        self.t += 1
        terminated = self.t == 3
        next_state = torch.zeros(3) if terminated else torch.eye(3)[self.t]
        return next_state, reward, terminated, False


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    discounts: torch.Tensor
    returns: torch.Tensor


@torch.no_grad()
def discounted_returns(rewards, gamma):
    """Complete single episode only: G_t = r_t + gamma G_{t+1}."""
    result = torch.zeros_like(rewards)
    running = rewards.new_zeros(())
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        result[t] = running
    return result


@torch.no_grad()
def rollout(actor, env, gamma):
    """Freeze parameters, collect one fresh complete episode, never replay it."""
    state = env.reset()
    states, actions, rewards, next_states, ends = [], [], [], [], []
    while True:
        action = actor(state).sample()
        next_state, reward, terminated, truncated = env.step(action.item())
        # MC methods need the full suffix. Do not silently turn a truncation
        # into a zero-tail MC target. A continuing-task adapter needs changes.
        if truncated and not terminated:
            raise ValueError('Incomplete episode: MC returns require a full suffix')
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        ends.append(terminated)
        state = next_state
        if terminated:
            break
    rewards = torch.tensor(rewards, dtype=torch.float32)
    return Batch(torch.stack(states), torch.stack(actions), rewards,
                 torch.stack(next_states), torch.tensor(ends),
                 gamma ** torch.arange(len(rewards)),
                 discounted_returns(rewards, gamma))


@torch.no_grad()
def td_target(batch, value, gamma):
    # Only true termination masks bootstrap; evaluate no terminal observations.
    next_v = torch.zeros_like(batch.rewards)
    live = ~batch.terminated
    next_v[live] = value(batch.next_states[live])
    return batch.rewards + gamma * next_v


def mse(prediction, target):
    return 0.5 * (prediction - target.detach()).square().mean()


def compute_losses(args, b, value, qnet, gamma):
    # 公用输入：完整、连续的一条轨迹；最后一步真正终止。
    if not b.terminated[-1] or b.terminated[:-1].any():
        raise ValueError('Expected one complete episode ending in true termination')
    rewards = b.rewards
    returns = b.returns.detach()  # MC 采样回报，不带梯度。
    discounts = b.discounts
    horizon = len(rewards)
    v = value(b.states) if args.method in (3, 5, 6) else None
    q = qnet(b.states, b.actions) if args.method in (4, 5) else None

    # Actor：六种方法只改变 weights。
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
        # 完整轨迹中 v[1:] 就是下一状态预测；最后真正终止，尾值为 0。
        target = rewards + gamma * torch.cat([v.detach()[1:], v.new_zeros(1)])
        weights = discounts * (target - v)  # TD error 作为优势估计。

    # Critic：1、2 无 Critic；3、5 训练 V；4、5 训练 Q；6 用 TD 训练 V。
    critic_loss = None
    if args.method in (3, 5):
        critic_loss = 0.5 * (v - returns).square().mean()
    if args.method in (4, 5):
        q_loss = 0.5 * (q - returns).square().mean()
        critic_loss = q_loss if critic_loss is None else critic_loss + q_loss
    if args.method == 6:
        critic_loss = 0.5 * (v - target.detach()).square().mean()
    return weights, critic_loss


def actor_loss(actor, batch, weights):
    # Sum within an episode. For B episodes: sum their losses / B.
    # Do not backpropagate into Q, V, targets, sampling, or the environment.
    return -(weights.detach() * actor(batch.states).log_prob(batch.actions)).sum()


def update_networks(actor, b, weights, critic_loss, actor_opt, critic_opt):
    # Actor Loss：权重作常数，只对 Actor 的 log probability 反传。
    log_prob = actor(b.states).log_prob(b.actions)
    actor_loss = -(weights.detach() * log_prob).sum()
    if not torch.isfinite(actor_loss) or (critic_loss is not None and
                                         not torch.isfinite(critic_loss)):
        raise FloatingPointError('Non-finite loss')
    # 更新 Actor。
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()
    # 更新 Critic（方法 1、2 跳过；方法 5 的优化器包含 Q 和 V 参数）。
    if critic_loss is not None:
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()


def train(method=6, episodes=300, gamma=0.9, seed=7, verbose=True):
    if method not in range(1, 7) or episodes < 1 or not 0 <= gamma <= 1:
        raise ValueError('method: 1..6; episodes >= 1; gamma: 0..1')
    torch.manual_seed(seed)
    actor = Actor()
    value = ValueCritic() if method in (3, 5, 6) else None
    qnet = QCritic() if method in (4, 5) else None
    actor_opt = torch.optim.Adam(actor.parameters(), lr=0.01)
    critic_params = [p for net in (value, qnet) if net is not None
                     for p in net.parameters()]
    critic_opt = torch.optim.Adam(critic_params, lr=0.01) if critic_params else None
    args = argparse.Namespace(method=method)
    env = ToyEnv()
    for episode in range(episodes):
        b = rollout(actor, env, gamma)
        # Both losses use pre-update predictions. No fitting before weighting.
        weights, critic_loss = compute_losses(args, b, value, qnet, gamma)
        update_networks(actor, b, weights, critic_loss, actor_opt, critic_opt)
        # Discard b, collect again with the updated policy.
        if verbose and (episode % 50 == 0 or episode == episodes - 1):
            with torch.no_grad():
                p = actor(torch.eye(3)).probs[:, 1].mean().item()
            print(f'method={method} episode={episode:4d} mean P(action=1)={p:.3f}')
    return actor, value, qnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    train(**vars(args))
