# =============================================================================
# 功能简介：
#   定义 Unitree GO2 四足机器人在 rough terrain（复杂地形）任务中的仿真与训练配置，
#   包含机器人初始姿态、控制方式、奖励结构以及 PPO 训练相关超参数。
#
# 核心数据输入：
#   - LeggedRobot 环境基类：
#       * 通过继承自 LeggedRobotCfg / LeggedRobotCfgPPO，仿真环境在构建时会读取
#         本文件中 GO2RoughCfg / GO2RoughCfgPPO 的字段，作为动力学参数与训练超参。
#
# 核心数据输出：
#   - GO2RoughCfg：
#       * init_state：机器人 reset 时的初始位姿与默认关节角；
#       * control：PD 控制器类型、增益、动作缩放等；
#       * asset：URDF 路径、接触几何与自碰撞设置；
#       * rewards：奖励目标与各项奖励权重，对应 RL 中的“任务定义”。
#   - GO2RoughCfgPPO：
#       * algorithm：PPO 算法层面的系数（如熵权重）；
#       * runner：训练运行名称与实验分组名，用于日志与模型存储。
# =============================================================================

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2RoughCfg(LeggedRobotCfg):
    """
    设计目的：
        描述 GO2 在“rough terrain”任务下的机器人与环境配置，是 LeggedRobot 通用配置的
        具体子类。通过只修改该配置而不改动环境/算法代码，可以方便地做不同机器人或
        任务场景的对比实验。

    核心职责：
        1. 指定机器人 reset 时的初始姿态和默认关节角，确保一开始就处于可行走姿势；
        2. 配置低层控制参数（PD 增益、动作缩放、控制频率）以匹配 GO2 的动力学尺度；
        3. 绑定 GO2 的 URDF 模型以及接触/自碰撞相关设置；
        4. 定义奖励函数的关键目标（速度跟踪、姿态稳定、能耗等）及其权重。

    典型应用场景：
        - 复杂地形（台阶、斜坡、障碍）上的行走/奔跑策略训练；
        - 对 GO2 进行 domain randomization 或控制律迁移前的基准配置。
    """

    class init_state(LeggedRobotCfg.init_state):
        """
        功能描述：
            设定机器人在每次环境 reset 时的初始位姿与默认关节角度，
            这些默认角也是“动作=0” 时的目标关节角（即策略输出零向量时的站立姿态）。

        设计考量：
            - 初始高度 0.42 m 使足端刚好接近地面，既避免“脚穿地”，也避免从过高处掉落；
            - 默认关节角对应一个略微弯曲的“站立/行走准备姿态”，
              便于策略从静止状态平滑过渡到行走。
        """
        # 机器人基座初始位置 [x, y, z]，保持水平放置在世界坐标原点附近
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]

        # 当动作向量为 0 时，各关节的目标角度（弯腿站立姿态）：
        # 该姿态通常来源于经验调节或通过 IK 求解的“舒适站立位”，
        # 目的是让机器人在无控制输入时也能静稳站立。
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # 髋关节：前腿略向外摆、后腿略向内摆，增加支撑多样性与稳定性
            'FL_hip_joint': 0.1,    # [rad]
            'RL_hip_joint': 0.1,    # [rad]
            'FR_hip_joint': -0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            # 大腿关节：前腿略小于后腿角度，使质心稍偏后，增强抗前倾能力
            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            # 小腿关节：较大的负角度使足端落在合适高度，形成“蹲姿”
            # 有利于吸收落地冲击并提供足够的可调节余量。
            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        """
        功能描述：
            定义低层控制接口的形式与参数，包括使用的控制类型（P/PD）、
            关节刚度和阻尼系数、动作缩放以及控制频率（decimation）。

        设计思路：
            - 使用简单的 P 控制（无显式速度反馈）降低策略学习难度，
              由策略直接输出目标关节角；
            - 较小的刚度与阻尼值使关节行为更“柔顺”，在 rough terrain 中
              能更好适应地形起伏；
            - 合理的 action_scale 和 decimation 在“策略表达能力”和“数值稳定性”
              之间做权衡。
        """
        # 仅使用位置型控制（P 控制），由策略直接生成关节角目标
        control_type = 'P'

        # 所有关节统一使用同一组刚度/阻尼参数：
        # - 刚度 20：足够让机器人保持姿态，又不会过于刚硬导致冲击；
        # - 阻尼 0.5：在 rough terrain 中提供适度阻尼，减少高频振荡。
        stiffness = {'joint': 20.0}
        damping = {'joint': 0.5}

        # 动作缩放系数：
        #   目标角 = action_scale * 动作 + default_angle
        # 通过限制 action_scale，可保证策略输出在 [-1, 1] 时关节偏移角度处于安全范围。
        action_scale = 0.25

        # decimation：每个策略时间步内，底层控制在仿真步长上更新的次数。
        # 例如：sim_dt = 1/400 s，decimation = 4 则策略频率为 100 Hz。
        # 较高的控制频率有利于稳定接触，但也增加计算开销。
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        """
        功能描述：
            指定 GO2 机器人模型及其接触相关配置，定义在仿真中的物理外观与碰撞属性。

        关键字段：
            - file：URDF 文件路径，用于加载机械结构与惯性参数；
            - name：环境内对象名称前缀，便于在调试/日志中识别；
            - foot_name：用于识别足端 link 的名称前缀，关联足部接触检测；
            - penalize_contacts_on / terminate_after_contacts_on：
                * 前者用于奖励中对“不合理碰撞”（如大腿撞地）进行惩罚；
                * 后者用于检测“严重碰撞”（如机身着地）并触发 episode 终止。
        """
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"

        # 用于在 URDF link 名称中匹配足端 link，决定哪些 link 参与足部接触逻辑
        foot_name = "foot"

        # 当这些部件与地面发生接触时，会触发碰撞惩罚（而非正常支撑力）
        # 例如：大腿/小腿撞地说明步态失衡或摆动过大。
        penalize_contacts_on = ["thigh", "calf"]

        # 一旦这些部件（如机身 base）触地，通常认为机器人已跌倒，episode 终止
        terminate_after_contacts_on = ["base"]

        # self_collisions：自碰撞开关的 bit 掩码（1 禁用自碰撞、0 启用，具体含义由 Isaac Gym 定义）
        # 在大多数 RL 训练中会禁用自碰撞以简化仿真，减少数值不稳定。
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        """
        功能描述：
            定义奖励函数的结构与标度，包括是否截断负奖励、跟踪误差宽度、
            软限制阈值以及各项奖励的权重（由内部 scales 子类给出）。

        设计目标：
            - 引导策略实现期望的线速度/角速度跟踪；
            - 惩罚不稳定姿态、垂直运动、横向摆动等不利于稳定行走的行为；
            - 约束关节速度、扭矩与功率，鼓励节能与平滑控制；
            - 控制脚步离地高度与碰撞行为，减少“拖脚”和“绊倒”。
        """

        # 是否将总奖励裁剪为非负：
        # - False：允许总奖励为负，有利于区分“差”和“非常差”的行为；
        # - True：可缓解早期训练阶段因强负奖励导致的梯度问题。
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)

        # 线速度/角速度 tracking reward 使用的高斯形状参数：
        #   tracking_reward = exp(- error^2 / sigma)
        # sigma 越大，对误差越“宽容”，梯度也越平缓。
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)

        # 软限制比例：超过该比例的关节位置/速度/扭矩会被施加惩罚，
        # 但不会像硬限制那样直接终止 episode，有助于训练出“接近极限但不过度”的行为。
        soft_dof_pos_limit = 1.0  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0

        # 期望的机身高度（m），用于鼓励机器人保持合理的重心高度。
        base_height_target = 0.30

        # 接触力上限（N），超过该阈值的接触力将产生惩罚，用于避免“硬碰撞”或踩踏过猛。
        max_contact_force = 100.0  # forces above this value are penalized

        # 足部抬离地面的目标高度（m），通常用于 foot_clearance 奖励：
        # - 对应负数是因为在某些实现中以“低于该高度”作为惩罚条件，
        #   具体意义取决于 env 中的奖励实现。
        clearance_height_target = -0.20

        class scales(LeggedRobotCfg.rewards.scales):
            """
            功能描述：
                定义各子奖励项在总奖励中的权重（线性系数），
                正值表示鼓励，负值表示惩罚。

            设计取舍：
                - tracking_* 系为正，强化速度跟踪；
                - lin_vel_z / ang_vel_xy / orientation 等姿态相关项为负，
                  抑制垂向剧烈运动和大幅摆动；
                - dof_acc / joint_power / action_rate / smoothness 等与能耗
                  和控制变化相关项为负，鼓励节能和动作平滑；
                - 其他如 collision / feet_stumble 等可根据需要开启，当前配置
                  将其权重设为 0 或极小值，便于先对主要行为进行收敛。
            """

            # 终止奖励：通常用于在 episode 提前结束时施加额外惩罚。
            # 这里设为 -0，等价于关闭该项（预留可调节空间）。
            termination = -0.0

            # 期望线速度跟踪的正奖励权重（主奖励之一）
            tracking_lin_vel = 1.0

            # 期望角速度（如 yaw 旋转）跟踪的正奖励（次于线速度）
            tracking_ang_vel = 0.5

            # 垂直速度惩罚：鼓励机器人尽量保持在水平面运动，避免上下跳动
            lin_vel_z = -2.0

            # 横向滚转/俯仰角速度惩罚：减少身体摇晃，提高乘坐舒适性与稳定性
            ang_vel_xy = -0.05

            # 姿态（orientation）惩罚：抑制大角度倾斜（roll/pitch 偏离）
            orientation = -0.2

            # 关节加速度惩罚：限制动作变化速率，避免高频控制导致数值不稳定
            dof_acc = -2.5e-7

            # 关节功率惩罚：鼓励节能，避免大扭矩与高速度同时出现
            joint_power = -2e-5

            # 与基座高度偏离相关的惩罚：鼓励保持在 base_height_target 附近
            base_height = -1.0

            # 足部抬升不足惩罚：脚离地太低可能导致绊倒或拖地
            foot_clearance = -0.01

            # 动作变化率惩罚：限制连续两个时间步之间的动作差值，鼓励控制平滑
            action_rate = -0.01

            # 轨迹平滑度惩罚：可以基于高阶差分定义，避免剧烈抖动
            smoothness = -0.01

            # feet_air_time：通常用于鼓励步伐节奏（例如足在空中的时间分布），
            # 当前设为 0，表示默认不启用。
            feet_air_time = 0.0

            # 一系列与碰撞/绊倒/站立相关的惩罚目前均设为 0 或 -0，
            # 方便后续根据需要逐项打开调节。
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0


class GO2RoughCfgPPO(LeggedRobotCfgPPO):
    """
    设计目的：
        为 GO2 rough terrain 任务定义基于 PPO 的训练超参数配置，
        与 GO2RoughCfg 搭配使用，完成“机器人配置 + 算法配置”的组合。

    核心职责：
        - 在 algorithm 子类中配置 PPO 算法级别的系数（如熵权重、clip 范围等）；
        - 在 runner 子类中设置日志/模型存储相关的标识（run_name、experiment_name）。

    使用方式：
        环境构建时会自动查找与 GO2RoughCfg 对应的 PPO 配置，
        通过统一接口传入算法 Runner。
    """

    class algorithm(LeggedRobotCfgPPO.algorithm):
        """
        功能描述：
            调整 PPO 算法的一些关键超参数，这里仅显式覆盖熵权重 entropy_coef，
            其余参数沿用基类默认值。

        设计考量：
            - 较小的熵权重（0.01）鼓励策略在 early stage 保持一定探索，
              但不会过度随机；当环境较难（rough terrain）时，适中的熵系数
              有助于避免陷入较差局部最优。
        """
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        """
        功能描述：
            定义训练 Runner 级别的运行标识，例如日志目录结构中的 experiment_name，
            以及每次运行可选的 run_name。

        字段说明：
            - run_name：单次运行名称，留空时通常由脚本自动根据时间戳等生成；
            - experiment_name：实验组名称，用于在日志与模型存储中对该任务进行分组。
        """
        # 留空 run_name，便于在训练脚本中自动生成更具区分度的名称（如加时间戳）
        run_name = ''

        # experiment_name 将出现在 logs/<experiment_name>/... 路径中，
        # 也常被用作 wandb 等日志系统中的 project/run 标签。
        experiment_name = 'rough_go2'
