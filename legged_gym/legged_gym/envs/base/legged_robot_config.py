# =============================================================================
# 功能简介：
#   本文件定义了“通用四足机器人环境”的基础配置 LeggedRobotCfg 及其对应的 PPO 训练配置
#   LeggedRobotCfgPPO。两者作为基类，被具体任务 / 具体机器人（如 GO2、ANYmal 等）的配置类
#   继承与覆盖，用于统一描述：
#     1）仿真环境的维度与物理参数；
#     2）地形与指令采样策略；
#     3）域随机化（domain randomization）规则；
#     4）奖励结构与归一化尺度；
#     5）PPO 算法与训练 Runner 的全局超参数。
#
# 核心数据输入：
#   - 下游环境/任务配置：
#       * 具体机器人配置类（如 GO2RoughCfg）会继承并覆盖本文件中的默认字段；
#       * 仿真实例构建时通过读取这些字段来确定观测维度、地形参数、控制模式等。
#
# 核心数据输出：
#   - LeggedRobotCfg：
#       * env / terrain / commands / init_state / control / asset / domain_rand /
#         rewards / normalization / noise / viewer / sim 等子类中的静态字段；
#   - LeggedRobotCfgPPO：
#       * policy / algorithm / runner 中的 PPO 超参数，用于构建策略网络与训练循环。
# =============================================================================

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    """
    设计目的：
        为“腿足机器人环境”提供一个通用的配置基类，涵盖从物理仿真到 RL 训练所需的
        各类参数。具体机器人仅需通过继承本类并覆盖少量字段，即可快速定义新的任务。

    设计思想：
        - 采用“嵌套 class + 类变量”的结构，将不同功能域的配置（地形、命令、奖励等）
          进行分组，避免使用长字典或大量全局变量；
        - 配置是“静态描述”，不直接参与运行逻辑，仅被环境构造函数和算法 Runner 读取；
        - 各字段的默认值设计成一个“合理工作”的基线，便于新任务在此基础上微调。
    """

    class env:
        """
        功能描述：
            环境级别的全局维度与时间长度配置，主要决定：
              - 并行环境数量；
              - 观测 / 特权观测 / 动作的维度；
              - episode 时间长度与超时信号的行为。

        使用场景：
            - 构建矢量化环境（vectorized env）时，RL 算法会据此决定批量大小；
            - 策略网络输入 / 输出层维度直接依赖这些字段。
        """
        num_envs = 8
        # 单步观测（one-step obs）的维度；对于有“历史堆叠”的情况，会基于此乘以堆叠窗口长度
        num_one_step_observations = 45
        # 最终传给策略的观测维度（这里默认堆叠 6 个时间步的观测）
        num_observations = num_one_step_observations * 6
        # 单步特权观测维度：
        #   额外包含：base_lin_vel（3）、external_forces（3）、scan_dots（187），
        #   仅在训练 critic 或使用 teacher-student 结构时可见。
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # additional: base_lin_vel, external_forces, scan_dots
        # 特权观测总维度（可配置为多个时间步，这里仅使用单步）
        num_privileged_obs = num_one_step_privileged_obs * 1  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        # 动作维度（关节数），策略输出的向量长度
        num_actions = 12
        # 环境之间的水平间距（仅对 plane/none 生效；heightfield/trimesh 由地形生成决定）
        env_spacing = 3.  # not used with heightfields/trimeshes 
        # 是否将超时信息发送给算法，便于区分“自然终止”和“因为时间到而重置”
        send_timeouts = True  # send time out information to the algorithm
        # 单个 episode 的时间长度（秒），用于将步数映射为时间
        episode_length_s = 20  # episode length in seconds

    class terrain:
        """
        功能描述：
            地形（terrain）配置，控制高度场 / 三角网格的分辨率、大小以及课程学习设置。

        关键设计点：
            - horizontal_scale / vertical_scale 决定高度场的空间分辨率和高度量化精度；
            - curriculum / max_init_terrain_level 用于难度递进训练；
            - terrain_proportions 控制不同地形类型在整张地图中的占比。
        """
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m] 栅格在 x/y 方向上的物理尺寸
        vertical_scale = 0.005  # [m] 高度量化步长
        border_size = 25  # [m] 地形边界缓冲区，用于防止机器人走出高度场
        curriculum = True  # 是否使用“课程学习”地形生成方式
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # 以下为只在 rough terrain 场景中使用的高度测量配置：
        measure_heights = True
        # 在机器人周围预采样一系列高度测量点（x, y），构成 height_measurements 观测
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # 若 selected = True，则使用 terrain_kwargs 中指明的唯一地形类型
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        # 课程学习起始难度等级（越大越难），后续会随训练进展进行调整
        max_init_terrain_level = 5  # starting curriculum state
        # 单个子地形（对应一个 env）的长宽（米）
        terrain_length = 8.
        terrain_width = 8.
        # 整张地形由 num_rows x num_cols 个子地形块组成
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # 不同地形类型（斜坡、楼梯等）的占比：
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # 当使用 trimesh 时，斜率超过该阈值的面会被“拉直”为竖直墙，避免出现非物理斜坡
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        """
        功能描述：
            任务级别速度 / 方向指令的采样与 curriculum 配置。命令向量作为高层目标，
            被策略网络用来生成对应的关节动作。

        典型命令格式（num_commands = 4）：
            [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
        """
        curriculum = True  # 是否对命令空间使用课程学习（逐步扩大命令范围）
        max_curriculum = 3.0  # curriculum 最大难度因子，用于放大命令范围
        # 命令维度（默认 4：前后速度、左右速度、yaw 角速度、目标 heading）
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # 命令重采样时间（秒）：每隔 resampling_time 重新采样一组命令
        resampling_time = 10.  # time before command are changed[s]
        # 若开启 heading_command，则不直接采样 ang_vel_yaw，而是根据 heading 误差计算
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            """
            功能描述：
                命令在采样时的取值范围（min, max），用于约束任务难度。
            """
            lin_vel_x = [-1.0, 1.0]  # min max [m/s] 前向 / 后退速度范围
            lin_vel_y = [-1.0, 1.0]   # min max [m/s] 侧向速度范围
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s] 偏航角速度范围
            heading = [-3.14, 3.14]  # 目标朝向范围（弧度）

    class init_state:
        """
        功能描述：
            机器人在环境 reset 时的初始状态，包括基座位姿、线/角速度以及默认关节角。
            这些参数决定了 episode 的起点，对稳定训练非常关键。
        """
        pos = [0.0, 0.0, 1.]  # x,y,z [m] 初始基座位置（离地高度 1m）
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat] 初始姿态（单位四元数 = 水平放置）
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s] 初始线速度
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s] 初始角速度
        # 默认关节角：当 action=0 时目标角度即为此，用于定义“中立姿态”。
        default_joint_angles = {  # target angles when action = 0.0
            "joint_a": 0.,
            "joint_b": 0.
        }

    class control:
        """
        功能描述：
            底层关节控制相关配置，包括控制模式（位置/速度/力矩）和 PD 增益等。

        设计思路：
            - 利用简单的 PD 控制器封装物理引擎的驱动接口；
            - 强化学习策略只需要输出“目标量”（如期望角度），由底层 PD 做平滑插值。
        """
        control_type = 'P'  # P: position, V: velocity, T: torques

        # PD Drive parameters:
        #   根据不同关节设置不同刚度/阻尼，反映机械设计和负载的差异。
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]

        # 动作缩放：目标角 = action_scale * action + defaultAngle
        # 通过限制 action_scale，保证策略输出在 [-1, 1] 时不会超出物理关节极限。
        action_scale = 0.5

        # decimation：控制频率降低系数。
        # 例如仿真步长为 dt，decimation=4 表示每 4 个仿真步才更新一次控制命令。
        decimation = 4

        # hip_reduction：髋关节减速比等可选参数（若与真实机器人齿轮比相关），
        # 默认 1.0 表示不做特别缩放。
        hip_reduction = 1.0

    class asset:
        """
        功能描述：
            机器人刚体资产（URDF / MJCF）及其物理属性的配置，
            控制仿真中刚体树结构与碰撞、阻尼等细节。
        """
        file = ""  # 机器人 URDF 文件路径，由具体机器人配置类覆盖
        name = "legged_robot"  # actor name，在 Isaac Gym 中创建 actor 时使用的名称前缀
        # 用于匹配足端 link 名称，以提取足部状态与接触力
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        # 列表中的 link 若与地面接触，将被视作“不良碰撞”，可在奖励中施加惩罚
        penalize_contacts_on = []
        # 列表中的 link 若与地面接触，则 episode 直接终止（例如 base 撞地）
        terminate_after_contacts_on = []
        # 是否禁用重力，多用于调试/特殊实验，正常训练一般为 False
        disable_gravity = False
        # 是否折叠固定关节，减少刚体数量以提高仿真效率
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        # 是否固定基座（如机械臂基座），对于行走机器人一般为 False
        fix_base_link = False  # fixe the base of the robot
        # 默认关节驱动模式（参考 GymDofDriveModeFlags）
        # 0: none, 1: pos tgt, 2: vel tgt, 3: effort
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # 自碰撞开关 bit 掩码：1 禁用自碰撞，0 启用。通常训练时关闭自碰撞以提升稳定性。
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # 将碰撞柱体替换为胶囊体，以提升仿真稳定性和效率
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        # 某些 .obj 模型坐标系为 y-up，需要翻转到 z-up
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        # 物理属性：密度、阻尼、速度上限等，影响接触动态与数值稳定性
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        """
        功能描述：
            域随机化（Domain Randomization）配置。通过在仿真中对质量、摩擦系数、
            电机强度等参数进行随机扰动，使得训练得到的策略在现实世界中更具鲁棒性。

        使用说明：
            - 每个 randomize_* 开关控制是否在 reset 时对相应属性采样随机值；
            - *_range 指定采样区间或缩放因子。
        """
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]  # 附加负载质量偏移（kg）

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]  # 质心偏移范围（m）

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]  # link 质量缩放范围

        randomize_friction = True
        friction_range = [0.2, 1.25]  # 接触摩擦系数范围

        randomize_restitution = False
        restitution_range = [0., 1.0]  # 弹性系数范围

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]  # 电机扭矩缩放

        randomize_kp = True
        kp_range = [0.9, 1.1]

        randomize_kd = True
        kd_range = [0.9, 1.1]

        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]  # 默认关节角随机缩放比例

        # 外力扰动：用于模拟推搡、风力等随机外力
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8  # 每隔若干秒施加一次扰动

        # 随机推机器人：模拟突然冲击，有助于提高抗推鲁棒性
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.

        # 延迟（delay）：可用于模拟感知/控制延时
        delay = True

    class rewards:
        """
        功能描述：
            奖励函数相关配置，包括各子项权重（scales）及若干全局阈值参数。

        设计原则：
            - “scales” 中的正值表示鼓励，负值表示惩罚；
            - 通过 only_positive_rewards / tracking_sigma 等参数调整训练稳定性。
        """

        class scales:
            """
            功能描述：
                各种子奖励项的线性权重，用于构成最终标量奖励：
                    r_total = sum_i ( scale_i * r_i )
            """
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.

        # 若为 True，将总奖励裁剪为非负数（负值视为 0），可减少早期“全负奖励”导致的训练困难
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 线/角速度 tracking 奖励中的高斯宽度参数 sigma：tracking_reward = exp(-error^2/sigma)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        # 超过软限制的关节位置/速度/扭矩会产生惩罚（但不直接终止 episode）
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        # 机身高度目标，用于 penalize/encourage 接近该高度
        base_height_target = 1.
        # 接触力上限（N），超出部分视为“硬碰撞”，产生惩罚
        max_contact_force = 100.  # forces above this value are penalized
        # 足部离地目标高度，用于 foot_clearance 奖励/惩罚
        clearance_height_target = 0.09

    class normalization:
        """
        功能描述：
            观测与动作的归一化配置。合理的归一化有助于稳定 RL 训练，
            降低不同量纲之间的尺度差异。
        """

        class obs_scales:
            """
            功能描述：
                对不同观测分量进行缩放（乘法因子），使其落入大致一致的数值范围。
            """
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        # 对观测/动作进行裁剪（clip），防止异常大值导致数值不稳定
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        """
        功能描述：
            观测噪声配置，通过向观测中注入高斯噪声提升策略对传感器误差的鲁棒性。
        """
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            """
            功能描述：
                各观测分量的噪声幅值基准，最终噪声 = N(0, noise_level * noise_scale)。
            """
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        """
        功能描述：
            可视化窗口的摄像机初始设置，在调试与演示时用于确定默认视角。
        """
        ref_env = 0  # 默认跟踪的环境索引
        pos = [10, 0, 6]  # [m] 摄像机位置
        lookat = [11., 5, 3.]  # [m] 摄像机注视点

    class sim:
        """
        功能描述：
            物理仿真器的全局配置（时间步长、重力、坐标系等）。
        """
        dt = 0.005  # 仿真步长（秒）
        substeps = 1  # 物理子步数量
        gravity = [0., 0., -9.81]  # [m/s^2] 重力加速度向量
        up_axis = 1  # 0 is y, 1 is z  （与 Isaac Gym 坐标系约定保持一致）

        class physx:
            """
            功能描述：
                NVIDIA PhysX 求解器相关参数，用于控制接触稳定性与性能。
            """
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            # 最大 GPU 接触对数量；并行环境数很大时需适当调高
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 接触收集策略：0 从不收集，1 仅最后一个子步，2 所有子步
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class LeggedRobotCfgPPO(BaseConfig):
    """
    设计目的：
        针对 LeggedRobotCfg 定义一组配套的 PPO 训练配置，涵盖：
          - 策略网络结构与初始化（policy）；
          - PPO 算法超参数（algorithm）；
          - 训练 Runner 级别的迭代/日志配置（runner）。

    使用场景：
        具体任务继承该类后，可覆盖少量字段（如 experiment_name、entropy_coef 等），
        即可在保持整体训练框架不变的前提下完成任务级别的调参。
    """
    # 随机种子，控制权重初始化与环境随机性，便于结果复现
    seed = 1
    # Runner 类名，用于从注册表中构造具体的训练执行器实现（如异步采样等）
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        """
        功能描述：
            策略网络 / 价值网络结构与初始化相关配置。
        """
        # 初始策略网络权重采样时的噪声尺度，用于鼓励早期探索
        init_noise_std = 1.0
        # Actor 网络隐藏层维度（多层 MLP）
        actor_hidden_dims = [512, 256, 128]
        # Critic 网络隐藏层维度（与 Actor 相同结构，便于共享经验规模）
        critic_hidden_dims = [512, 256, 128]
        # 激活函数类型
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # 以下为可选的循环网络配置，仅在使用 ActorCriticRecurrent 时启用：
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        """
        功能描述：
            PPO 算法层面的训练超参数配置。
        """
        # training params
        value_loss_coef = 1.0  # 价值函数损失在总损失中的权重
        use_clipped_value_loss = True  # 是否对 value loss 使用截断（clipped value）
        clip_param = 0.2  # PPO 目标中的剪切阈值 ε
        entropy_coef = 0.01  # 策略熵的权重，用于鼓励探索
        num_learning_epochs = 5  # 每次采样后进行的优化 epoch 数
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  #5.e-4  # 学习率
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99  # 折扣因子
        lam = 0.95  # GAE(λ) 中的 λ，平衡 bias/variance
        desired_kl = 0.01  # 目标 KL 散度，adaptive schedule 可参考该值调节学习率
        max_grad_norm = 1.  # 梯度裁剪上限，防止梯度爆炸

    class runner:
        """
        功能描述：
            训练 Runner 的流程控制参数，包括每轮采样步数、最大迭代次数以及日志 / 模型
            存储策略等。
        """
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'

        # 每个环境每次迭代采样的时间步数量，直接决定单次更新的数据量及采样开销
        num_steps_per_env = 100  # per iteration
        # 策略更新的最大迭代次数，即训练上限
        max_iterations = 200000  # number of policy updates

        # logging
        # 每隔多少次迭代检查一次是否需要保存模型（并不一定每次都保存）
        save_interval = 20  # check for potential saves every this many iterations
        experiment_name = 'test'  # 实验名称，用于日志分组
        run_name = ''  # 具体运行名称，留空时通常由脚本自动生成

        # load and resume
        # 是否从已有模型恢复训练（在脚本中通常会覆盖）
        resume = False
        # load_run = -1 表示加载最后一次 run（与实验名结合使用）
        load_run = -1  # -1 = last run
        # checkpoint = -1 表示加载该 run 下最后一个 checkpoint
        checkpoint = -1  # -1 = last saved model
        # resume_path 可显式指定 checkpoint 路径，优先级高于 load_run/checkpoint
        resume_path = None  # updated from load_run and chkpt