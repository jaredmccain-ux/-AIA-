"""End-to-end TD3 training for residual control on the A*+PD baseline.

Training loop:
  MATLAB Env -> A*+PD baseline action -> TD3 residual -> Env.step -> reward

The learned actor is exported as astar_residual_policy_weights.mat and can be
loaded by Policy_AStarPD_RLResidual. The original Policy.m is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from train_residual_online_td3_matlab import (
    Actor,
    Critic,
    ReplayBuffer,
    TrainState,
    export_actor,
    matlab_features_to_np,
    matlab_info_to_dict,
    select_action,
    train_td3_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--maps", nargs="+", default=["amap", "a1map", "a2map", "bmap", "b1map", "b2map", "cmap", "c1map"])
    parser.add_argument("--agents", nargs="+", type=int, default=[20])
    parser.add_argument("--max-episodes", type=int, default=1200)
    parser.add_argument("--min-episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--reward-tolerance", type=float, default=25.0)
    parser.add_argument("--score-tolerance", type=float, default=20.0)
    parser.add_argument("--reach-tolerance", type=float, default=0.02)
    parser.add_argument("--target-reach-rate", type=float, default=0.95)
    parser.add_argument("--min-score-improvement", type=float, default=5.0)
    parser.add_argument("--state-dim", type=int, default=34)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=2500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--policy-noise", type=float, default=0.18)
    parser.add_argument("--noise-clip", type=float, default=0.45)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.32)
    parser.add_argument("--final-exploration-noise", type=float, default=0.06)
    parser.add_argument("--noise-decay-episodes", type=int, default=500)
    parser.add_argument("--max-du", type=float, default=0.10)
    parser.add_argument("--max-dv", type=float, default=0.08)
    parser.add_argument("--gate-index", type=int, default=29)
    parser.add_argument("--seed", type=int, default=20240527)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--export-name", default="astar_residual_policy_weights.mat")
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

    run_dir = args.run_dir or (args.codex_dir / "astar_residual_td3_runs" / time.strftime("%Y%m%d_%H%M%S"))
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
    scores = deque(maxlen=args.window * 2)
    reached_flags = deque(maxlen=args.window * 2)
    residual_norms = deque(maxlen=args.window * 2)
    best_recent_score = -math.inf

    import matlab.engine

    eng = matlab.engine.start_matlab()
    eng.cd(str(args.codex_dir), nargout=0)
    eng.addpath(str(args.codex_dir), nargout=0)

    header = "episode\tmap\tagents\treturn\tscore\tsteps\treached\tnoise\tgate_rate\tavg_residual\tcritic_loss\tactor_loss"
    metrics_path.write_text(header + "\n", encoding="utf-8")
    convergence_path = run_dir / "convergence.txt"
    convergence_path.write_text(
        "episode\tprev_return\trecent_return\tprev_score\trecent_score\t"
        "prev_reach\trecent_reach\trecent_residual\tstable_windows\tstatus\n",
        encoding="utf-8",
    )

    try:
        for episode in range(1, args.max_episodes + 1):
            map_name = random.choice(args.maps)
            agent_count = int(random.choice(args.agents))
            episode_seed = int(args.seed + episode * 9973)
            features, info = eng.astar_residual_td3_env_reset(
                map_name, agent_count, args.max_steps, False, episode_seed, nargout=2
            )
            state_np = matlab_features_to_np(features)

            frac = min(1.0, episode / max(1, args.noise_decay_episodes))
            noise_std = args.exploration_noise + frac * (args.final_exploration_noise - args.exploration_noise)
            episode_return = 0.0
            episode_residual_norms: list[float] = []
            episode_gates: list[float] = []
            last_info = matlab_info_to_dict(info)
            losses = {"critic_loss": math.nan, "actor_loss": math.nan}

            for _ in range(args.max_steps):
                gate_value = float(state_np[args.gate_index]) if len(state_np) > args.gate_index else 1.0
                episode_gates.append(gate_value)
                if gate_value < 0.5:
                    action_norm = np.zeros(2, dtype=np.float32)
                elif replay.size < args.warmup_steps:
                    action_norm = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                else:
                    action_norm = select_action(actor, state_np, args.device, noise_std)

                residual = np.array([action_norm[0] * args.max_du, action_norm[1] * args.max_dv], dtype=np.float64)
                episode_residual_norms.append(float(np.linalg.norm(residual)))
                next_features, reward, done, info = eng.astar_residual_td3_env_step(
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
                            actor,
                            actor_target,
                            critic,
                            critic_target,
                            actor_opt,
                            critic_opt,
                            replay,
                            args,
                            train_state,
                        )

                if done_bool:
                    break

            score = float(last_info.get("score", 0.0))
            reached = float(last_info.get("reached", 0.0))
            steps = int(last_info.get("step", 0.0))
            avg_residual = float(np.mean(episode_residual_norms)) if episode_residual_norms else 0.0
            gate_rate = float(np.mean(episode_gates)) if episode_gates else 0.0
            returns.append(episode_return)
            scores.append(score)
            reached_flags.append(reached)
            residual_norms.append(avg_residual)

            line = (
                f"{episode}\t{map_name}\t{agent_count}\t{episode_return:.3f}\t{score:.3f}\t"
                f"{steps}\t{int(reached)}\t{noise_std:.4f}\t"
                f"{gate_rate:.4f}\t{avg_residual:.6f}\t"
                f"{losses['critic_loss']:.6f}\t{losses['actor_loss']:.6f}"
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(line, flush=True)

            if score > train_state.best_score:
                train_state.best_score = score
                export_actor(actor, run_dir / "best_astar_residual_policy_weights.mat", args)
                export_actor(actor, args.codex_dir / args.export_name, args)

            if episode % args.checkpoint_every == 0:
                export_actor(actor, run_dir / f"episode_{episode:05d}_astar_residual_policy_weights.mat", args)
                export_actor(actor, args.codex_dir / args.export_name, args)

            if episode >= args.min_episodes and len(returns) == args.window * 2:
                prev_return = np.mean(list(returns)[: args.window])
                recent_return = np.mean(list(returns)[args.window :])
                prev_score = np.mean(list(scores)[: args.window])
                recent_score = np.mean(list(scores)[args.window :])
                prev_reach = np.mean(list(reached_flags)[: args.window])
                recent_reach = np.mean(list(reached_flags)[args.window :])
                recent_residual = np.mean(list(residual_norms)[args.window :])

                reward_stable = abs(recent_return - prev_return) <= args.reward_tolerance
                score_stable = abs(recent_score - prev_score) <= args.score_tolerance
                reach_stable = abs(recent_reach - prev_reach) <= args.reach_tolerance
                reach_ok = recent_reach >= args.target_reach_rate
                improved = recent_score > best_recent_score + args.min_score_improvement
                if improved:
                    best_recent_score = recent_score
                    train_state.stable_windows = 0
                    status = "improving"
                elif reward_stable and score_stable and reach_stable and reach_ok:
                    train_state.stable_windows += 1
                    status = "stable"
                else:
                    train_state.stable_windows = 0
                    status = "unstable"

                with convergence_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{episode}\t{prev_return:.3f}\t{recent_return:.3f}\t"
                        f"{prev_score:.3f}\t{recent_score:.3f}\t"
                        f"{prev_reach:.4f}\t{recent_reach:.4f}\t{recent_residual:.6f}\t"
                        f"{train_state.stable_windows}\t{status}\n"
                    )

                if train_state.stable_windows >= args.patience:
                    reason = {
                        "episode": episode,
                        "recent_return": float(recent_return),
                        "previous_return": float(prev_return),
                        "recent_score": float(recent_score),
                        "previous_score": float(prev_score),
                        "recent_reach": float(recent_reach),
                        "previous_reach": float(prev_reach),
                        "recent_residual": float(recent_residual),
                        "best_recent_score": float(best_recent_score),
                        "stable_windows": train_state.stable_windows,
                        "criteria": {
                            "reward_tolerance": args.reward_tolerance,
                            "score_tolerance": args.score_tolerance,
                            "reach_tolerance": args.reach_tolerance,
                            "target_reach_rate": args.target_reach_rate,
                            "min_score_improvement": args.min_score_improvement,
                            "window": args.window,
                            "patience": args.patience,
                        },
                    }
                    (run_dir / "converged.json").write_text(json.dumps(reason, indent=2), encoding="utf-8")
                    print("training converged: " + json.dumps(reason), flush=True)
                    break

        export_actor(actor, run_dir / "final_astar_residual_policy_weights.mat", args)
        export_actor(actor, args.codex_dir / args.export_name, args)
    finally:
        try:
            eng.astar_residual_td3_env_close(nargout=0)
        finally:
            eng.quit()


if __name__ == "__main__":
    main()
