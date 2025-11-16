# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import traceback
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

try:
    import wandb
except ImportError:
    wandb = None


def _init_wandb_run(args, headless: bool):
    """Create a wandb run in offline mode if the package is available."""
    if wandb is None:
        print("[wandb] 未安装 wandb，跳过离线日志记录。")
        return None

    os.environ.setdefault("WANDB_MODE", "offline")
    project = os.environ.get("WANDB_PROJECT", "legged_gym")
    run_name = args.run_name or f"{args.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config = {
        "task": args.task,
        "experiment": args.experiment_name,
        "run_name": run_name,
        "headless": headless,
        "num_envs": getattr(args, "num_envs", None),
        "rl_device": getattr(args, "rl_device", None),
    }
    config = {k: v for k, v in config.items() if v is not None}
    try:
        return wandb.init(project=project, name=run_name, reinit=True, config=config)
    except Exception as wandb_error:
        print(f"[wandb] 初始化失败（{wandb_error}），将继续训练但不记录 wandb。")
        return None


def _log_wandb_error(run, error: Exception, tb: str):
    """Record errors to wandb when training fails."""
    if run is None:
        return
    run.log(
        {
            "event": "train_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    run.summary["status"] = "error"
    run.summary["traceback"] = tb

def train(args, headless=True):
    args.headless = headless
    args.resume = False
    run = _init_wandb_run(args, headless)
    try:
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, wandb_run=run)
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
        if run is not None:
            run.summary["status"] = "success"
    except Exception as train_error:
        _log_wandb_error(run, train_error, traceback.format_exc())
        raise
    finally:
        if run is not None:
            run.finish()

if __name__ == '__main__':
    args = get_args()
    # headless=True: 训练时无“仿真的UI界面“，减少GPU占用，从而加快训练
    train(args, headless=True)
    # train(args, headless=False)
