# =============================================================================
# 功能简介：
#   本脚本作为 legged_gym 工程的“策略回放 / 可视化”入口，用于在仿真环境中加载
#   已训练好的策略（policy），并在给定速度指令下观察机器人运动表现，同时可选：
#   - 导出 TorchScript（JIT）模型，便于在 C++ 或部署环境中直接调用；
#   - 记录关节状态、接触力、奖励等曲线，用于调试与分析；
#   - 录制仿真图像序列，用于后期可视化或制作演示视频。
#
# 核心数据输入：
#   - 命令行参数（get_args()）：
#       * args.task：任务名称，对应具体机器人/场景（与训练时保持一致）；
#       * args.num_envs：测试用环境数量上限（不指定时自动限制为不超过 50）；
#       * 其他如 rl_device、sim_device 等，将透传给环境构造函数。
#   - 训练配置与环境配置：
#       * 通过 task_registry.get_cfgs(name=args.task) 获取与训练时一致的 env_cfg
#         与 train_cfg，确保回放策略时的物理与任务定义保持一致。
#
# 核心数据输出：
#   - 策略导出：
#       * 若 EXPORT_POLICY=True，则将 actor_critic 策略导出为 TorchScript（JIT），
#         保存在 logs/<experiment_name>/exported/policies 目录下。
#   - 日志与可视化：
#       * 通过 Logger 记录并绘制关键状态量的时序曲线（关节位置、速度、力矩、基座速度等）；
#       * 可选地将仿真画面保存为 PNG 序列，用于后续视频合成；
#       * 通过移动相机，实现对机器人在复杂地形上的跟踪观察。
# =============================================================================

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0):
    """
    功能描述：
        在 Isaac Gym 仿真环境中加载训练好的策略，并在给定期望速度命令下
        对机器人进行“回放式”测试与可视化：
          1. 基于任务名称加载训练时使用的环境与训练配置；
          2. 对环境配置进行适配性修改，使其更适合交互式测试与可视化；
          3. 从训练检查点恢复 PPO（或其他算法）策略，仅保留推理相关逻辑；
          4. 在多步仿真循环中执行策略、施加固定速度指令并记录状态/奖励；
          5. 可选导出策略为 JIT、录制图像以及移动相机。

    参数说明：
        args:
            - 类型：命令行解析得到的参数对象
            - 含义：指定任务名、设备信息、环境数量等；同时包含训练配置所需的
                    一些默认字段（如 experiment_name），以保证与训练阶段兼容。
        x_vel (float):
            - 物理含义：在世界坐标系 / 机器人局部坐标系下的期望前进线速度（m/s），
                        具体取决于任务定义中的 command 解释。
            - 用途：作为策略输入的一部分，引导机器人以指定速度向前行走。
        y_vel (float):
            - 物理含义：期望横向线速度（m/s），用于侧向移动或斜行。
        yaw_vel (float):
            - 物理含义：期望绕垂直轴的角速度（rad/s），用于转向控制。

    返回值说明：
        - 无显式返回值。该函数主要通过以下“外部效果”提供价值：
          * 在屏幕上实时展示机器人行为（若未 headless）；
          * 在日志目录下保存状态曲线图与奖励统计结果；
          * 可选地导出策略 JIT 模型与仿真图像帧。

    核心处理流程：
        1. 加载与任务绑定的 env_cfg 与 train_cfg：
             - 保证回放环境与训练环境在动力学、观测空间、奖励函数等方面一致；
             - 后续对 env_cfg 的修改仅针对“测试可视化友好性”，不改变任务本质。
        2. 覆盖环境配置以适应测试场景：
             - 限制环境数量为用户指定值或不超过 50，以降低可视化时的资源占用；
             - 开启地形 curriculum、调整地形网格尺寸，提高观测多样性但仍可视化；
             - 禁用噪声与 domain randomization，确保测试结果可复现、行为更稳定。
        3. 基于修改后的 env_cfg 构造仿真环境，并对所有环境实例统一设置速度命令；
        4. 将 train_cfg.runner.resume 设为 True，并调用 make_alg_runner 创建算法 Runner，
           然后通过 get_inference_policy 获取仅用于推理的 policy 函数；
           若 EXPORT_POLICY=True，则额外导出 TorchScript 策略模型。
        5. 创建 Logger，并设定：
             - robot_index：观测哪一个环境实例；
             - joint_index：观测哪一个关节；
             - stop_state_log：状态日志记录的步数上限；
             - stop_rew_log：奖励统计的步数上限。
        6. 在仿真循环中重复：
             - 使用当前观测 obs 经过 policy 生成动作 actions；
             - 重新写入 commands，确保机器人持续收到一致的速度指令；
             - 调用 env.step 推进仿真，并按阶段记录状态曲线与奖励统计；
             - 视需要录屏、移动相机和更新日志。
    """
    # 从任务注册表中获取“训练阶段”使用的环境与训练配置：
    # - env_cfg：环境物理与任务配置（如地形、噪声、随机化策略）；
    # - train_cfg：算法与训练流程配置（如学习率、迭代次数、日志路径等）。
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ======================
    # 1. 环境规模与可视化配置
    # ======================
    # 在回放/可视化阶段通常不需要像训练时那样使用上千个并行环境，
    # 因此这里通过 num_envs 做限制：
    if args.num_envs is not None:
        # 若用户通过命令行显式指定 num_envs，则优先遵循用户的测试需求，
        # 例如只看少量机器人以便更清晰地观察单个个体行为。
        env_cfg.env.num_envs = args.num_envs
    else:
        # 否则，将环境数量限制在 50 以内，以避免可视化时 GPU/CPU 压力过大，
        # 同时保持一定的并行环境数量以观察策略在不同初始条件下的泛化表现。
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

    # ================
    # 2. 地形与课程学习
    # ================
    # 对地形网格数量进行裁剪：
    # - 减少地形行列数可以降低内存与渲染开销；
    # - 仍保留足够多的地形块以测试机器人在不同难度上的表现。
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8

    # curriculum = True 表示仍然启用地形难度课程机制，
    # 使得机器人在可视化过程中也会在不同难度的地形上进行切换，便于评估鲁棒性。
    env_cfg.terrain.curriculum = True
    # 限制初始地形难度上限，避免一开始即在极端困难的地形上测试，
    # 这样更利于观察机器人在“可控难度”的场景下的行为。
    env_cfg.terrain.max_init_terrain_level = 9

    # =======================
    # 3. 关闭噪声与域随机化
    # =======================
    # 在测试/演示阶段关闭噪声，可以让策略行为更加平滑与可预测，
    # 减少视频中“随机抖动”的视觉效果，同时便于复现实验。
    env_cfg.noise.add_noise = False

    # 同样关闭一系列域随机化选项：
    env_cfg.domain_rand.randomize_friction = False   # 关闭摩擦系数随机化，保证地形物理一致；
    env_cfg.domain_rand.push_robots = False          # 关闭随机外力推扰，避免机器人被“突然撞击”；
    env_cfg.domain_rand.disturbance = False          # 关闭其他形式扰动，使回放环境更“干净”；
    env_cfg.domain_rand.randomize_payload_mass = False  # 关闭负载质量随机化，确保动力学参数固定。

    # 在回放阶段通过固定的 x/y/yaw 速度命令来控制机器人，
    # 因此通常无需使用 heading_command（例如通过目标朝向控制转向），这里显式关闭。
    env_cfg.commands.heading_command = False

    # 如需在平地上观察策略，可将 mesh_type 强制设为 'plane'，方便快速检查步态稳定性：
    # env_cfg.terrain.mesh_type = 'plane'

    # =====================
    # 4. 构造环境与下发命令
    # =====================
    # 使用经过修改的 env_cfg 构造环境：
    # - 通过显式传入 env_cfg，确保测试环境与训练配置兼容但又具备上面所做的调整。
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 将统一的速度命令写入所有并行环境：
    # - 这里直接写 env.commands 张量，从而绕过上层 command 采样逻辑，
    #   使机器人持续朝给定方向/速度行走，便于观测。
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    # 初始观测，用于第一次策略推理：
    obs = env.get_observations()

    # ==================
    # 5. 加载训练好的策略
    # ==================
    # 告诉 Runner 应该从已有训练结果中恢复（resume=True），
    # 这样 make_alg_runner 内部会加载最新或指定的检查点，而不是重新初始化策略。
    train_cfg.runner.resume = True

    # 在推理阶段关闭日志目录创建：
    # - 通过将 log_root=None 传入，使得推理时不再创建新的日志目录，
    #   以免与训练日志混淆；同时降低在仅做回放时的磁盘写入开销。
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None
    )

    # 从 Runner 中获取推理用 policy 函数：
    # - 该函数通常会将 obs 张量映射到动作空间（关节目标、力矩等）；
    # - get_inference_policy 会处理好 device/精度等细节，方便在测试时直接调用。
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ===========================
    # 6. （可选）导出策略为 JIT 模型
    # ===========================
    # 将策略导出为 TorchScript（JIT）后，可以在 C++ 或部署环境中直接加载并运行，
    # 避免 Python 依赖与动态图开销，非常适合部署到实时控制系统或嵌入式平台。
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            train_cfg.runner.experiment_name,
            'exported',
            'policies'
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # ====================
    # 7. 日志器与相机设置
    # ====================
    # Logger 内部通常会缓存时间序列数据并提供绘图/打印接口，这里传入 env.dt，
    # 方便后续将离散步数转换成物理时间（秒）。
    logger = Logger(env.dt)

    # 由于环境是并行的，这里选取其中一个机器人进行详细状态记录与可视化分析。
    robot_index = 0  # 用于记录状态和奖励的目标机器人索引
    joint_index = 1  # 关注的关节索引（例如某个膝关节），用于观察动作与反馈关系

    # 状态日志记录限定步数：
    # - 只记录前 100 个仿真步的状态，以避免图像过长难以阅读；
    # - 一般这些步数足以观察启动与进入稳态的过程。
    stop_state_log = 100

    # 奖励统计的记录窗口：
    # - 设置为 episode 长度 + 1，意味着大致在经历一整个 episode 后停止累积统计，
    #   便于比较单个 episode 的平均奖励与各项子奖励贡献。
    stop_rew_log = env.max_episode_length + 1

    # 用环境配置中的默认视角初始化相机位置与朝向，
    # 后续可根据 MOVE_CAMERA 标志平移相机来实现“跟随”效果。
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])  # 相机在 XY 平面上的平移速度（m/s）
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0  # 用于编号录制的图像帧

    # ==========================
    # 8. 仿真主循环：策略回放与日志记录
    # ==========================
    # 这里循环次数设为 10 个 episode 长度，意味着即使中途发生 reset，
    # 也可以连续观察多个 episode 的策略表现。
    for i in range(10 * int(env.max_episode_length)):

        # 使用当前观测通过策略网络推理得到动作：
        # - detach() 的作用是显式切断与计算图的连接，强调这里纯推理、无梯度更新，
        #   也避免误开启 autograd 带来的额外开销。
        actions = policy(obs.detach())

        # 每一步都重置 commands，保证机器人持续收到恒定速度指令：
        # - 这样可以抵消环境/策略内部可能对命令的修改，使测试场景更可控、可复现。
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel

        # 推进仿真一步：
        # - obs：新的观测；
        # - rews：即时奖励；
        # - dones：是否终止当前 episode；
        # - infos：额外信息（例如用于统计的 episode 级别奖励）。
        # 多返回值中的部分占位符（_）保持与训练接口兼容，但在回放中无需使用。
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        # ========= 8.1 可选：录制图像帧 =========
        if RECORD_FRAMES:
            # 每隔一步保存一帧图像，平衡图像数量与时间分辨率，避免文件数量过大。
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    'logs',
                    train_cfg.runner.experiment_name,
                    'exported',
                    'frames',
                    f"{img_idx}.png"
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

        # ========= 8.2 可选：移动相机 =========
        if MOVE_CAMERA:
            # 以固定速度在 XY 平面上平移相机，实现类似“扫视”或“跟随”的视觉效果，
            # 便于从不同视角检查机器人在复杂地形上的行为。
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # ========= 8.3 状态时序日志 =========
        if i < stop_state_log:
            # 记录关节目标位置、实际位置/速度/力矩以及基座速度和接触力：
            # - dof_pos_target：由动作经过缩放与偏移后得到的目标关节位置；
            # - dof_pos / dof_vel / dof_torque：反馈关节状态，用于评估控制效果；
            # - command_x/y/yaw：当前施加的高层速度指令；
            # - base_vel_*：机器人基座在世界坐标下的线/角速度；
            # - contact_forces_z：足端在 z 方向上的接触力，便于分析步态与冲击。
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale
                    + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            # 达到预定步数后触发一次性绘图：
            # - logger.plot_states() 通常会弹出或生成图像文件，以直观展示状态随时间变化；
            # - 只在阈值点调用一次，避免重复绘图带来的性能问题。
            logger.plot_states()

        # ========= 8.4 奖励统计与打印 =========
        if 0 < i < stop_rew_log:
            # infos["episode"] 通常在 episode 结束时包含累积奖励等信息：
            # - 通过该字段可以实现基于 episode 的统计，而不是逐步累加。
            if infos["episode"]:
                # env.reset_buf 中标记了哪些环境在当前 step 被重置，
                # 将其求和即可得到本步完成的 episode 数量。
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    # 记录当前 step 完成的 episode 的奖励信息，
                    # Logger 内部会完成平滑、聚合与打印的准备工作。
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            # 到达奖励记录时间上限后，打印汇总的奖励统计信息：
            # - 有助于快速评估策略在该测试场景下的整体表现（例如平均回报等）。
            logger.print_rewards()


if __name__ == '__main__':
    # 这里定义的全局开关会在 play(...) 中被读取，控制策略导出、录屏与相机移动行为：
    # - EXPORT_POLICY：是否在回放开始前导出 TorchScript 策略；
    # - RECORD_FRAMES：是否在仿真过程中定期抓取图像帧；
    # - MOVE_CAMERA：是否在仿真过程中移动相机，实现动态视角。
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # 解析命令行参数，使脚本既可以直接运行，也可以被其他模块导入并重用 play 函数。
    args = get_args()

    # 调用 play 以在给定速度命令（例如向前 1.0 m/s）下回放策略：
    # - 根据需要可调整 x_vel / y_vel / yaw_vel 以测试不同运动模式
    #   （直线行走、斜向行走、原地转向等）。
    play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0)
