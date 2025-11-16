# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA
# SPDX-License-Identifier: BSD-3-Clause
#
# 该脚本用于在 Isaac Gym Viewer 中可视化 legged_gym 地形。
# 可根据命令行参数调整并行环境数量、镜头跟随的机器人等。
#
# 用法示例：
#   python legged_gym/legged_gym/scripts/visualize_terrain.py --task go2
#   python legged_gym/legged_gym/scripts/visualize_terrain.py --task go2 --vis-num-envs 64 --follow-env 0
#   python legged_gym/legged_gym/scripts/visualize_terrain.py --task go2 --max-steps 2000

import argparse
import sys
import time

import isaacgym  # 确保在导入 torch 前初始化 isaac gym 依赖
from legged_gym.envs import *  # 注册任务，避免循环导入
import torch

from legged_gym.utils import get_args, task_registry


def _parse_visualization_args():
    """
    先解析与可视化相关的自定义参数，并将其从 sys.argv 中剥离，
    以便后续 get_args()（依赖 gymutil）能顺利解析剩余参数。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--vis-num-envs",
        type=int,
        default=None,
        help="可视化时创建的并行环境数量（覆盖配置 env.num_envs）",
    )
    parser.add_argument(
        "--follow-env",
        type=int,
        default=-1,
        help="镜头跟随的环境索引，-1 表示使用配置中的静态镜头",
    )
    parser.add_argument(
        "--enable-follow",
        action="store_true",
        help="若设置，则在每个仿真步自动更新镜头跟随；默认只在仿真开始时对准一次，方便手动自由移动视角。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="仿真步数（-1 表示无限循环）",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="每个仿真步骤后的延时（秒），用于慢速观测。",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=5.0,
        help="跟随镜头距机器人质心的距离（米）。",
    )
    vis_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return vis_args


def _configure_env_cfg(env_cfg, vis_args):
    if vis_args.vis_num_envs is not None:
        env_cfg.env.num_envs = vis_args.vis_num_envs
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.max_init_terrain_level = min(
            env_cfg.terrain.max_init_terrain_level, env_cfg.terrain.num_rows - 1
        )


def _maybe_follow_actor(env, follow_env, distance):
    if follow_env < 0 or follow_env >= env.num_envs:
        return
    root_states = env.root_states
    base_pos = root_states[follow_env, :3]
    # 让镜头位于机器人后上方，朝向机器人前方。
    camera_offset = torch.tensor([ -distance, 0.0, distance * 0.5 ], device=env.device)
    camera_target_offset = torch.tensor([ 0.0, 0.0, 0.5 ], device=env.device)
    cam_pos = (base_pos + camera_offset).cpu().numpy()
    cam_target = (base_pos + camera_target_offset).cpu().numpy()
    env.set_camera(cam_pos, cam_target)


def main():
    vis_args = _parse_visualization_args()
    args = get_args()
    args.headless = False

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_env_cfg(env_cfg, vis_args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    print(
        f"[terrain-viewer] task={args.task}, num_envs={env.num_envs}, follow_env={vis_args.follow_env}, "
        f"max_steps={'inf' if vis_args.max_steps < 0 else vis_args.max_steps}"
    )

    step = 0
    while vis_args.max_steps < 0 or step < vis_args.max_steps:
        if vis_args.follow_env >= 0:
            if step == 0 or vis_args.enable_follow:
                _maybe_follow_actor(env, vis_args.follow_env, vis_args.camera_distance)

        obs, _, _, _, _, _, _ = env.step(actions)
        step += 1

        if vis_args.step_delay > 0.0:
            time.sleep(vis_args.step_delay)

    print("[terrain-viewer] Finished simulation loop.")


if __name__ == "__main__":
    main()
