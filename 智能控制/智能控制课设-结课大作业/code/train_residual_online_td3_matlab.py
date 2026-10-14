"""Online TD3 residual training through MATLAB Engine.

This is real actor-critic training, not ES:
  state:  residual_feature_vector from MATLAB A*+DWA policy
  action: continuous residual [du, dv] applied as base_action + residual
  reward: MATLAB residual_rl_reward.m

The MATLAB side deliberately does not safety-filter the residual. The Arena
agent still applies its own physical saturation in Agent.step().
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        return torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_l1(xu))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        q2 = F.relu(self.q2_l1(xu))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_l1(xu))
        q1 = F.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


class ReplayBuffer:
    def __init__(self, state_dim: int, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, 2), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.s[self.ptr] = state
        self.a[self.ptr] = action
        self.r[self.ptr] = reward
        self.ns[self.ptr] = next_state
        self.d[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.s[idx], device=self.device),
            torch.as_tensor(self.a[idx], device=self.device),
            torch.as_tensor(self.r[idx], device=self.device),
            torch.as_tensor(self.ns[idx], device=self.device),
            torch.as_tensor(self.d[idx], device=self.device),
        )


@dataclass
class TrainState:
    total_it: int = 0
    best_score: float = -math.inf
    stable_windows: int = 0


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(source.parameters(), target.parameters()):
        tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def matlab_features_to_np(features) -> np.ndarray:
    return np.asarray(features, dtype=np.float32).reshape(-1)


def matlab_info_to_dict(info) -> dict:
    out = {}
    for key in info.keys():
        value = info[key]
        if isinstance(value, str):
            out[key] = value
        else:
            arr = np.asarray(value).reshape(-1)
            out[key] = float(arr[0]) if arr.size else value
    return out


def select_action(actor: Actor, state: np.ndarray, device: torch.device, noise_std: float) -> np.ndarray:
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = actor(s).cpu().numpy()[0]
    if noise_std > 0:
        action = action + np.random.normal(0.0, noise_std, size=2)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def train_td3_step(
    actor: Actor,
    actor_target: Actor,
    critic: Critic,
    critic_target: Critic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    replay: ReplayBuffer,
    args: argparse.Namespace,
    state: TrainState,
) -> dict:
    state.total_it += 1
    s, a, r, ns, d = replay.sample(args.batch_size)

    with torch.no_grad():
        noise = (torch.randn_like(a) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
        next_action = (actor_target(ns) + noise).clamp(-1.0, 1.0)
        target_q1, target_q2 = critic_target(ns, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = r + (1.0 - d) * args.gamma * target_q

    current_q1, current_q2 = critic(s, a)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()

    actor_loss_value = float("nan")
    if state.total_it % args.policy_delay == 0:
        actor_loss = -critic.q1(s, actor(s)).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()
        soft_update(actor, actor_target, args.tau)
        soft_update(critic, critic_target, args.tau)
        actor_loss_value = float(actor_loss.item())

    return {"critic_loss": float(critic_loss.item()), "actor_loss": actor_loss_value}


def export_actor(actor: Actor, output_path: Path, args: argparse.Namespace) -> None:
    actor_cpu = actor.to("cpu")
    mat = {
        "W1": actor_cpu.l1.weight.detach().numpy(),
        "b1": actor_cpu.l1.bias.detach().numpy().reshape(-1, 1),
        "W2": actor_cpu.l2.weight.detach().numpy(),
        "b2": actor_cpu.l2.bias.detach().numpy().reshape(-1, 1),
        "W3": actor_cpu.l3.weight.detach().numpy(),
        "b3": actor_cpu.l3.bias.detach().numpy().reshape(-1, 1),
        "x_mean": np.zeros((args.state_dim, 1), dtype=np.float32),
        "x_std": np.ones((args.state_dim, 1), dtype=np.float32),
        "max_du": np.array([[args.max_du]], dtype=np.float32),
        "max_dv": np.array([[args.max_dv]], dtype=np.float32),
        "feature_dim": np.array([[args.state_dim]], dtype=np.float32),
        "hidden_dim": np.array([[args.hidden_dim]], dtype=np.float32),
    }
    sio.savemat(output_path, mat)
    actor.to(args.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--maps", nargs="+", default=["amap", "a1map", "a2map", "bmap", "b1map", "b2map", "cmap", "c1map"])
    parser.add_argument("--agents", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--min-episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--reward-tolerance", type=float, default=25.0)
    parser.add_argument("--reach-tolerance", type=float, default=0.02)
    parser.add_argument("--state-dim", type=int, default=26)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--policy-noise", type=float, default=0.20)
    parser.add_argument("--noise-clip", type=float, default=0.50)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.35)
    parser.add_argument("--final-exploration-noise", type=float, default=0.08)
    parser.add_argument("--noise-decay-episodes", type=int, default=600)
    parser.add_argument("--max-du", type=float, default=0.25)
    parser.add_argument("--max-dv", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--export-name", default="residual_policy_weights.mat")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.codex_dir = args.codex_dir.resolve()
    args.device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.run_dir or (args.codex_dir / "online_td3_runs" / time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.txt"
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    actor = Actor(args.state_dim, args.hidden_dim).to(args.device)
    actor_target = Actor(args.state_dim, args.hidden_dim).to(args.device)
    actor_target.load_state_dict(actor.state_dict())
    critic = Critic(args.state_dim, 2, args.hidden_dim).to(args.device)
    critic_target = Critic(args.state_dim, 2, args.hidden_dim).to(args.device)
    critic_target.load_state_dict(critic.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(args.state_dim, args.buffer_size, args.device)
    train_state = TrainState()
    returns = deque(maxlen=args.window * 2)
    reached_flags = deque(maxlen=args.window * 2)

    import matlab.engine  # Server dependency: MATLAB Engine for Python.

    eng = matlab.engine.start_matlab()
    eng.cd(str(args.codex_dir), nargout=0)
    eng.addpath(str(args.codex_dir), nargout=0)

    header = "episode\tmap\tagents\treturn\tscore\tsteps\treached\tnoise\tcritic_loss\tactor_loss"
    metrics_path.write_text(header + "\n", encoding="utf-8")

    try:
        for episode in range(1, args.max_episodes + 1):
            map_name = random.choice(args.maps)
            agent_count = int(random.choice(args.agents))
            features, info = eng.residual_td3_env_reset(map_name, agent_count, args.max_steps, nargout=2)
            state_np = matlab_features_to_np(features)

            frac = min(1.0, episode / max(1, args.noise_decay_episodes))
            noise_std = args.exploration_noise + frac * (args.final_exploration_noise - args.exploration_noise)
            episode_return = 0.0
            last_info = matlab_info_to_dict(info)
            losses = {"critic_loss": float("nan"), "actor_loss": float("nan")}

            for _ in range(args.max_steps):
                if replay.size < args.warmup_steps:
                    action_norm = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                else:
                    action_norm = select_action(actor, state_np, args.device, noise_std)

                residual = np.array([action_norm[0] * args.max_du, action_norm[1] * args.max_dv], dtype=np.float64)
                next_features, reward, done, info = eng.residual_td3_env_step(
                    matlab.double([residual.tolist()]), nargout=4
                )
                next_state_np = matlab_features_to_np(next_features)
                reward_float = float(np.asarray(reward).reshape(-1)[0])
                done_bool = bool(np.asarray(done).reshape(-1)[0])
                replay.add(state_np, action_norm, reward_float, next_state_np, done_bool)
                episode_return += reward_float
                state_np = next_state_np
                last_info = matlab_info_to_dict(info)

                if replay.size >= args.warmup_steps:
                    for _update in range(args.updates_per_step):
                        losses = train_td3_step(
                            actor, actor_target, critic, critic_target, actor_opt, critic_opt,
                            replay, args, train_state
                        )

                if done_bool:
                    break

            score = float(last_info.get("score", 0.0))
            reached = float(last_info.get("reached", 0.0))
            steps = int(last_info.get("step", 0.0))
            returns.append(episode_return)
            reached_flags.append(reached)

            line = (
                f"{episode}\t{map_name}\t{agent_count}\t{episode_return:.3f}\t{score:.3f}\t"
                f"{steps}\t{int(reached)}\t{noise_std:.4f}\t"
                f"{losses['critic_loss']:.6f}\t{losses['actor_loss']:.6f}"
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(line, flush=True)

            if score > train_state.best_score:
                train_state.best_score = score
                export_actor(actor, run_dir / "best_residual_policy_weights.mat", args)
                export_actor(actor, args.codex_dir / args.export_name, args)

            if episode % args.checkpoint_every == 0:
                export_actor(actor, run_dir / f"episode_{episode:05d}_residual_policy_weights.mat", args)
                export_actor(actor, args.codex_dir / args.export_name, args)

            if episode >= args.min_episodes and len(returns) == args.window * 2:
                prev_return = np.mean(list(returns)[: args.window])
                recent_return = np.mean(list(returns)[args.window :])
                prev_reach = np.mean(list(reached_flags)[: args.window])
                recent_reach = np.mean(list(reached_flags)[args.window :])
                reward_stable = abs(recent_return - prev_return) <= args.reward_tolerance
                reach_stable = abs(recent_reach - prev_reach) <= args.reach_tolerance
                if reward_stable and reach_stable:
                    train_state.stable_windows += 1
                else:
                    train_state.stable_windows = 0

                if train_state.stable_windows >= args.patience:
                    reason = {
                        "episode": episode,
                        "recent_return": recent_return,
                        "previous_return": prev_return,
                        "recent_reach": recent_reach,
                        "previous_reach": prev_reach,
                        "stable_windows": train_state.stable_windows,
                    }
                    (run_dir / "converged.json").write_text(json.dumps(reason, indent=2), encoding="utf-8")
                    print("reward converged: " + json.dumps(reason), flush=True)
                    break

        export_actor(actor, run_dir / "final_residual_policy_weights.mat", args)
        export_actor(actor, args.codex_dir / args.export_name, args)
    finally:
        try:
            eng.residual_td3_env_close(nargout=0)
        finally:
            eng.quit()


if __name__ == "__main__":
    main()
