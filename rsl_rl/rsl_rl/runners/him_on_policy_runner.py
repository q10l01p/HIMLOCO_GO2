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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, HIMPPO
from rsl_rl.modules import HIMActorCritic
from rsl_rl.env import VecEnv


class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 wandb_run=None):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # HIMActorCritic
        actor_critic: HIMActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_one_step_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # HIMPPO
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.wandb_run = wandb_run
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        should_record = (self.log_dir is not None) or (self.wandb_run is not None)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                    if should_record:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if should_record:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        eta = self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it'])
        use_tb = self.writer is not None
        wandb_logs = {} if self.wandb_run is not None else None

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if use_tb:
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                if wandb_logs is not None:
                    wandb_logs['Episode/' + key] = float(value.item())
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        if use_tb:
            self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
            self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
            self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
            self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
            self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
            self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
            self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
            self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if wandb_logs is not None:
            wandb_logs.update({
                'Loss/value_function': float(locs['mean_value_loss']),
                'Loss/surrogate': float(locs['mean_surrogate_loss']),
                'Loss/Estimation Loss': float(locs['mean_estimation_loss']),
                'Loss/Swap Loss': float(locs['mean_swap_loss']),
                'Loss/learning_rate': float(self.alg.learning_rate),
                'Policy/mean_noise_std': float(mean_std.item()),
                'Perf/total_fps': float(fps),
                'Perf/collection_time': float(locs['collection_time']),
                'Perf/learning_time': float(locs['learn_time']),
                'Perf/iteration_time': float(iteration_time),
            })

        mean_reward = statistics.mean(locs['rewbuffer']) if len(locs['rewbuffer']) > 0 else None
        mean_length = statistics.mean(locs['lenbuffer']) if len(locs['lenbuffer']) > 0 else None
        if use_tb and mean_reward is not None:
            self.writer.add_scalar('Train/mean_reward', mean_reward, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', mean_length, locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', mean_reward, self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', mean_length, self.tot_time)
        if wandb_logs is not None and mean_reward is not None:
            wandb_logs.update({
                'Train/mean_reward': float(mean_reward),
                'Train/mean_episode_length': float(mean_length),
                'Train/mean_reward/time': float(mean_reward),
                'Train/mean_episode_length/time': float(mean_length),
            })

        title = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if mean_reward is not None:
            log_string = (f"""{'#' * width}\n"""
                          f"""{title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {mean_reward:.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {mean_length:.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{title.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {eta:.1f}s\n""")
        if wandb_logs is not None:
            wandb_logs.update({
                'Time/total_timesteps': int(self.tot_timesteps),
                'Time/total_time_s': float(self.tot_time),
                'Time/eta_s': float(eta),
                'iteration': int(locs['it']),
            })
            clean_log = log_string.replace('\033[1m', '').replace('\033[0m', '')
            wandb_logs['Console/log'] = clean_log
            self.wandb_run.log(wandb_logs, step=locs['it'])
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        try:
            map_location = torch.device(self.device)
        except (TypeError, RuntimeError):
            map_location = self.device
        loaded_dict = torch.load(path, map_location=map_location)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
