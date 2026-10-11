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


def method_1(b, value, qnet, gamma):
    # Actor：整条轨迹共用 G_0，不再乘 gamma**t。
    weights = b.returns[0].expand_as(b.returns)
    critic_loss = None  # 无 Critic 网络，也没有 Critic 更新。
    return weights, critic_loss


def method_2(b, value, qnet, gamma):
    # Actor：每一步只看从该步开始的回报。
    weights = b.discounts * b.returns
    critic_loss = None  # 无 Critic 网络，也没有 Critic 更新。
    return weights, critic_loss


def method_3(b, value, qnet, gamma):
    # 1. Critic 预测：V 网络估计当前策略的平均回报。
    v = value(b.states)
    # 2. Critic target：完整轨迹的 MC 回报，不是网络预测。
    critic_target = b.returns.detach()
    # 3. Actor 权重：实际回报减去更新前的 V；停止整个权重的梯度。
    weights = (b.discounts * (b.returns - v)).detach()
    # 4. Critic Loss：只对预测 v 反传，让 v 接近 MC 回报。
    critic_loss = 0.5 * (v - critic_target).square().mean()
    return weights, critic_loss


def method_4(b, value, qnet, gamma):
    # 1. Critic 预测：Q 网络取出实际采样动作的 Q 值。
    q = qnet(b.states, b.actions)
    # 2. Critic target：MC 回报 G_t；不是 DQN 的 max target。
    critic_target = b.returns.detach()
    # 3. Actor 权重：使用更新前的 Q，停止 Q 支路梯度。
    weights = (b.discounts * q).detach()
    # 4. Critic Loss：只对预测 q 反传。
    critic_loss = 0.5 * (q - critic_target).square().mean()
    return weights, critic_loss


def method_5(b, value, qnet, gamma):
    # 1. 两个独立 Critic：Q 预测该动作回报；V 预测平均回报。
    q = qnet(b.states, b.actions)
    v = value(b.states)
    # 2. 两个 Critic 都用完整 MC 回报作 target。
    critic_target = b.returns.detach()
    # 3. Actor 权重：Q - V 是优势估计；两条 Critic 支路都停止梯度。
    weights = (b.discounts * (q - v)).detach()
    # 4. 分别计算两个 Critic Loss，再相加。
    q_loss = 0.5 * (q - critic_target).square().mean()
    v_loss = 0.5 * (v - critic_target).square().mean()
    critic_loss = q_loss + v_loss
    return weights, critic_loss


def method_6(b, value, qnet, gamma):
    # 1. Critic 当前预测：需要梯度，用于训练 V 网络。
    v = value(b.states)
    # 2. Critic target：真实 reward + 下一状态预测；整个目标停止梯度。
    with torch.no_grad():
        next_v = torch.zeros_like(b.rewards)
        live = ~b.terminated  # 真正终止时 next_v = 0；不计算终止状态的 V。
        next_v[live] = value(b.next_states[live])
        critic_target = b.rewards + gamma * next_v
    # 3. Actor 权重：TD error 作为优势估计，整个权重停止梯度。
    advantage = critic_target - v
    weights = (b.discounts * advantage).detach()
    # 4. Critic Loss：只对当前预测 v 反传，不对下一状态 V 反传。
    critic_loss = 0.5 * (v - critic_target).square().mean()
    return weights, critic_loss


METHODS = [method_1, method_2, method_3, method_4, method_5, method_6]


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
    env = ToyEnv()
    for episode in range(episodes):
        b = rollout(actor, env, gamma)
        # Both losses use pre-update predictions. No fitting before weighting.
        weights, critic_loss = METHODS[method - 1](b, value, qnet, gamma)
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
