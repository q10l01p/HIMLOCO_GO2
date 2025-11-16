# =============================================================================
# 功能简介：
#   本文件定义了用于生成机器人行走地形的 Terrain 类及两个辅助地形生成函数
#   （gap_terrain、pit_terrain）。该模块基于 Isaac Gym 自带的 terrain_utils，
#   根据配置自动生成随机 / 课程学习 / 指定类型的高度场，并为每个并行环境计算
#   对应的世界坐标原点与高度。
#
# 核心数据输入：
#   - LeggedRobotCfg.terrain：
#       * mesh_type：地形网格类型（none / plane / heightfield / trimesh 等）；
#       * terrain_length / terrain_width：单个环境在世界尺度下的长宽（米）；
#       * horizontal_scale / vertical_scale：栅格尺度与高度量化尺度；
#       * num_rows / num_cols：整个地形网格中子地形（sub-terrain）的行列数；
#       * terrain_proportions：不同地形类型所占比例，用于随机地形混合；
#       * curriculum / selected：控制使用课程难度分布还是选定地形类型；
#       * 其他 domain randomization / 地形细节参数（通过 terrain_kwargs 传入）。
#   - num_robots：
#       * 并行机器人 / 环境数量，用于后续 env_origins 等与环境实例一一对应。
#
# 核心数据输出：
#   - self.height_field_raw：
#       * 组合后的全局高度场（二维 int16 数组），包含所有子地形与边界；
#   - self.env_origins：
#       * 每个子地形对应的环境原点 (x, y, z)，用于在世界中摆放机器人；
#   - self.vertices, self.triangles（在 mesh_type == "trimesh" 时）：
#       * 由高度场转换得到的三角网格，用于渲染与碰撞。
# =============================================================================

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    """
    设计目的：
        封装地形生成与管理逻辑，为多环境并行仿真构建统一的全局高度场，并为每个环境
        分配对应的子地形与世界坐标原点。该类是“任务配置（LeggedRobotCfg.terrain）”
        与 isaacgym.terrain_utils 之间的适配层。

    核心职责：
        1. 根据配置 mesh_type / curriculum / selected / terrain_proportions 等决定地形生成模式；
        2. 按 num_rows × num_cols 的布局生成多个子地形，并拼接成一个大 heightfield；
        3. 为每个子地形计算 env_origin（放置机器人时的世界坐标起点）；
        4. 在需要时将 heightfield 转换为 trimesh 顶点与三角形索引。

    典型应用场景：
        - 强化学习训练中，为多个并行环境提供多样化 / 课程难度递增的地形；
        - 策略评估时，加载相同配置以保证训练–测试场景的一致性。
    """

    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        """
        功能描述：
            初始化 Terrain 对象，根据传入配置构建高度场内存结构，并按不同策略
            填充各子地形的高度值。对于 mesh_type 为 "none" 或 "plane" 的情况，
            为了效率直接返回，不生成复杂地形。

        参数说明：
            cfg (LeggedRobotCfg.terrain):
                - 地形相关配置，包含长宽、网格分辨率、行列数以及课程学习和随机参数。
            num_robots (int):
                - 并行机器人数量，用于后续根据机器人数量调整地形布局或分配策略。

        返回值说明：
            - 构造函数无返回值，但会初始化以下关键成员：
              * height_field_raw：全局高度场；
              * env_origins：每个环境的原点；
              * vertices / triangles（若为 trimesh）。
        """
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            # 对于“无地形”或“无限平面”两种简单场景，不需要生成 heightfield，
            # 由 Isaac Gym 内部简单配置即可完成；直接返回提升初始化速度。
            return

        # 单个环境在世界坐标中的占用尺寸（米），用于从子地形索引换算世界坐标。
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        # 将 terrain_proportions 转换成累积占比，方便后续用一个随机数直接落在区间里
        # 决定地形类型，避免多次随机判断。
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        # 整个地形由 num_rows × num_cols 个子地形组成：
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # env_origins[i, j] 存储第 (i, j) 个子地形的环境原点 (x, y, z)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # 将连续空间尺寸转换为高度图中的像素尺寸：
        # - horizontal_scale 决定每个像素对应多少米；
        # - 注意这里取 int，意味着会向下取整，有少量几何误差但可接受。
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # 为整个高度场添加边界区域（border），以避免机器人走出子地形时直接越界，
        # 同时给课程学习/复杂地形留出过渡缓冲区。
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 全局高度场初始化为 0（平地）；后续将逐块填充具体子地形高度值。
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # 根据配置选择不同的地形生成策略：
        # - curriculum=True：按行列渐进增加难度与改变类型，用于课程学习；
        # - selected=True：仅生成一个指定类型的地形，常用于 ablation 或对比实验；
        # - 否则：使用 randomized_terrain，在整体区域内随机混合多种地形类型。
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        # height_field_raw 既是内部生成的工作缓冲，也是对外暴露的高度样本。
        self.heightsamples = self.height_field_raw

        # 若使用 trimesh，则需将高度图转换为三角网格，以供渲染与碰撞使用。
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold
            )

    def randomized_terrain(self):
        """
        功能描述：
            为每个子地形随机采样地形类型与难度级别，并生成相应的子高度场。
            该模式主要用于训练阶段，增加环境多样性以提升策略泛化能力。

        核心逻辑：
            1. 遍历 num_sub_terrains，将线性索引 k 映射到二维网格 (i, j)；
            2. 对每个子地形：
               2.1 使用 [0, 1) 均匀分布的随机数 choice 决定地形类型；
               2.2 从预设难度集合 {0.5, 0.75, 0.9} 中随机抽取难度 difficulty；
               2.3 调用 make_terrain(choice, difficulty) 生成具体子地形；
               2.4 使用 add_terrain_to_map 将该子地形拼接到全局 heightfield 中。
        """
        for k in range(self.cfg.num_sub_terrains):
            # 将线性索引 k 映射为二维网格 (i, j)，对应 cfg.num_rows × cfg.num_cols。
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 这里局部变量名 choice 会遮蔽上方 from numpy.random import choice，
            # 但本函数中并未使用该 import，因此不影响行为，仅是命名上的“轻微陷阱”。
            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])

            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        """
        功能描述：
            以“课程学习（curriculum）”的方式生成地形网格：沿行方向渐进增加难度，
            沿列方向平滑改变地形类型，从而在一个大地图上构成由易到难、类型多样的
            训练场景。

        核心逻辑：
            1. 双重循环遍历列 j 与行 i；
            2. 使用 i / num_rows 将难度映射为 [0, 1) 的连续值；
            3. 使用 j / num_cols（加微小偏移 0.001）在不同 type 区间间插值；
            4. 为每个 (i, j) 调用 make_terrain(...) 然后 add_terrain_to_map(...)。
        """
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 难度沿行方向逐行递增：上方地形简单，下方地形更复杂。
                difficulty = i / self.cfg.num_rows
                # choice 沿列方向平滑分布，用于选择不同类型的地形；
                # 加 0.001 避免落在 0 边界上出现极端情况。
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        """
        功能描述：
            生成“单一指定类型”的地形网格。与 randomized / curriculum 不同，本模式将
            所有子地形都设置为相同类型，非常适合做 ablation 或某一特定地形上的详细调试。

        参数来源：
            - cfg.terrain_kwargs.type：字符串形式的地形函数名，例如 "pyramid_sloped_terrain"；
            - cfg.terrain_kwargs.terrain_kwargs：传给该函数的关键字参数字典。

        重要说明：
            - 本函数通过 eval(terrain_type) 动态获取地形生成函数，虽然灵活，
              但存在潜在安全与可维护性问题；后续若重构可考虑使用显式映射表。
            - 注意：当前实现中使用 self.vertical_scale / self.horizontal_scale，
              然而类未定义这两个属性，实际应从 cfg 中读取，这可能是一个隐藏 bug，
              在修改时需特别留意。
        """
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 这里 vertical_scale / horizontal_scale 来源于 self，而 Terrain 并未定义
            # 这两个属性，可能依赖于外部给实例动态添加属性，或是配置迁移时遗留问题。
            # 虽然不影响当前注释，但在维护时需要检查这一点。
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale
            )

            # 使用 eval 根据字符串获取地形生成函数，并传入 kwargs：
            # - 若 terrain_type 来自可信配置，则可以简化“类型→函数”的映射；
            # - 若配置可被外部任意修改，则需要注意 eval 的安全风险。
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        """
        功能描述：
            根据随机数 choice 与难度 difficulty 生成一个子地形（SubTerrain），
            并对不同类型地形的参数（坡度、台阶高度、障碍尺寸等）进行难度缩放。

        参数说明：
            choice (float):
                - 范围：[0, 1)，用于根据 proportions 决定地形类型：
                  * < proportions[0]：斜坡；
                  * < proportions[1]：斜坡 + 噪声；
                  * < proportions[3]：楼梯；
                  * < proportions[4]：离散障碍；
                  * < proportions[5]：“跳石”（stepping stones）；
                  * < proportions[6]：gap 地形；
                  * 否则：pit 地形。
            difficulty (float):
                - 范围：[0, 1]，用于调节坡度、台阶高度、障碍高度/间距等，
                  难度越大，机器人通过地形的挑战性越强。

        返回值说明：
            - terrain (terrain_utils.SubTerrain)：
                * 填充了高度场的子地形对象，后续会被 add_terrain_to_map 合并到全局。
        """
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )

        # 根据难度 difficulty 设计一组连续可调的几何参数：
        slope = difficulty * 0.4
        amplitude = 0.01 + 0.07 * difficulty
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty

        # 通过 choice 落在 proportions 分段区间中选择地形类型：
        if choice < self.proportions[0]:
            # 斜坡类地形：choice 的前半段使用下坡（负 slope），后半段使用上坡。
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            # 斜坡 + 随机噪声：同时存在整体坡度和局部起伏，提高策略鲁棒性。
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-amplitude,
                max_height=amplitude,
                step=0.005,
                downsampled_scale=0.2
            )
        elif choice < self.proportions[3]:
            # 楼梯地形：根据 choice 决定向上台阶还是向下台阶（负 step_height）。
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.30,
                step_height=step_height,
                platform_size=3.
            )
        elif choice < self.proportions[4]:
            # 离散障碍地形：在平台上随机放置多个矩形障碍块。
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.
            )
        elif choice < self.proportions[5]:
            # “跳石”地形：机器人需在高台之间跨越，考验其脚步精度与稳定性。
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.,
                platform_size=4.
            )
        elif choice < self.proportions[6]:
            # gap 地形：中间挖空形成沟壑，机器人需要跨越 gap。
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            # pit 地形：中央挖坑，机器人需要绕行或跨越坑洞。
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        """
        功能描述：
            将一个子地形的高度图插入到全局 height_field_raw 对应的子区域中，
            并计算该子地形对应的环境原点 env_origins[row, col]（x, y, z）。

        参数说明：
            terrain (terrain_utils.SubTerrain):
                - 已生成高度场的子地形。
            row (int), col (int):
                - 子地形在网格中的行列索引，用于确定其在全局高度图中的位置。

        核心逻辑：
            1. 根据 row / col 计算该子地形在全局 heightfield 中的起止像素坐标；
            2. 将子地形的 height_field_raw 复制到全局高度图对应子块；
            3. 以子地形中心附近 2×2 米区域的最大高度为基准，计算 env_origin_z；
            4. 将 (env_origin_x, env_origin_y, env_origin_z) 记录到 env_origins。
        """
        i = row
        j = col
        # 计算在全局高度图中的起止索引（注意带 border 偏移）：
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels

        # 将子地形高度图“贴”到全局 height_field_raw 对应的区域：
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # 计算当前子地形在世界坐标系下的 (x, y) 原点（中心点）：
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width

        # 为了避免机器人初始状态“脚底悬空”或“被埋入地面”，
        # 这里在子地形中心附近取一个 2×2 米的窗口，使用该窗口内的最大高度
        # 作为 env_origin_z，从而保证机器人被放置在略高于地面的安全高度。
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale

        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    """
    功能描述：
        在给定子地形中生成“gap（沟壑）”类型地形：在中央区域挖出一个深坑，
        仅在中间留下一块平台，使机器人需要跨越 gap 或沿平台边缘行走。

    参数说明：
        terrain (terrain_utils.SubTerrain):
            - 待修改高度场的子地形对象。
        gap_size (float):
            - gap 的宽度（米），将按 horizontal_scale 离散为像素数。
        platform_size (float):
            - 中央平台的半宽（米），对应机器人可安全落脚的区域大小。

    实现思路：
        1. 将 gap_size / platform_size 转换成像素尺度；
        2. 以高度图中心为基准，先在较大区域内赋值为 -1000（深坑）；
        3. 再在中间一个较小矩形区域赋值为 0（平台），形成“坑中平台”的几何结构。
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    # -1000 表示远低于正常地面高度的“深坑”，通常在碰撞几何中会被视作不可行走区域。
    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    # 将中心平台区域高度设为 0，使其成为可安全落脚的“桥面”。
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    """
    功能描述：
        在子地形中心生成一个“坑状（pit）”区域：在中心矩形内整体下挖指定深度，
        周围保持原始高度不变，从而构造出一个四周高、中间低的地形。

    参数说明：
        terrain (terrain_utils.SubTerrain):
            - 待修改高度场的子地形对象。
        depth (float):
            - 坑的深度（米），通过 vertical_scale 转换为高度图中的整数高度。
        platform_size (float):
            - 中心坑宽度的一半（米），坑区域大小约为 (2 * platform_size) × (2 * platform_size)。

    实现思路：
        1. 将 depth 按 vertical_scale 离散化为高度格；
        2. 将 platform_size 转换为像素数量，并以地图中心为基准计算矩形区域边界；
        3. 在该矩形中将高度设为 -depth，从而形成一个坑。
    """
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
