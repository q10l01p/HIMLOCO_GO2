# =============================================================================
# 功能简介：
#   该脚本是 legged_gym 工程中的统一训练入口脚本，用于：
#   1）根据命令行参数构建指定任务的 Isaac Gym 仿真环境；
#   2）通过 task_registry 创建对应的强化学习算法 Runner（例如 PPO）；
#   3）在本地启动训练主循环，并可选地以 wandb 离线模式记录训练日志。
#
# 核心数据输入：
#   - 命令行参数（get_args()）：
#       * args.task：任务名称，对应具体的机器人/场景配置；
#       * args.experiment_name：实验名称，用于区分不同超参组合或实验组；
#       * args.run_name：可选运行名称，不指定时脚本自动生成；
#       * 其他如 num_envs、rl_device 等，会透传给环境与算法配置。
#   - 仿真环境：
#       * 由 task_registry.make_env 基于 task 名称和 args 构造的 Isaac Gym 环境，
#         内部包括机器人模型、地形、随机扰动等。
#
# 核心数据输出：
#   - 训练产出（由算法 Runner 管理）：
#       * 策略网络权重与检查点文件（通常保存在 log_dir 目录中）；
#       * 训练过程中的标量指标（例如 reward、episode length、loss 值等）；
#   - wandb 日志（可选，离线模式）：
#       * 若安装了 wandb 且 log_dir 有效，则将超参配置与训练指标写入本地 wandb 缓存，
#         后续可通过 `wandb sync` 上传到云端进行可视化与对比分析。
# =============================================================================

import numpy as np
import os
import traceback
from datetime import datetime
from typing import Optional

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

try:
    import wandb
except ImportError:
    # 设计选择：在 wandb 不可用时将其视为可选依赖，
    # 通过将 wandb 设为 None，而不是直接抛出异常，
    # 使训练逻辑在无日志系统的情况下依然可以正常工作。
    wandb = None


def _init_wandb_run(args, headless: bool, log_dir: Optional[str]):
    """
    功能描述：
        在本地以“离线模式”初始化一个 wandb 训练 run，用于统一记录实验配置与训练指标。
        若 wandb 不可用或初始化失败，则优雅退化为不使用 wandb 的训练流程。

    参数说明：
        args:
            - 类型：命令行解析得到的参数对象（通常是 argparse.Namespace）
            - 含义：包含任务名称（task）、实验名称（experiment_name）、run_name 等元信息，
                    还可能包含 num_envs、rl_device 等系统配置。
        headless (bool):
            - 含义：当前训练是否以无界面仿真模式进行（headless 模式），
                    该信息写入 wandb config 用于实验复现。
        log_dir (Optional[str]):
            - 含义：本地日志目录，用于存放 wandb 的离线缓存文件。
            - 约束：若为 None，说明上层未配置日志目录，此时不启用 wandb。

    返回值说明：
        - 返回值类型：wandb.sdk.wandb_run.Run 或 None
        - 语义：
            * 成功初始化时，返回 wandb 的 run 对象；
            * 在 wandb 包缺失、log_dir 无效或初始化异常时，返回 None，
              保证训练主流程不因为监控系统出错而中断。

    核心处理流程：
        1. 先检查 wandb 包和日志目录是否可用，不满足条件则直接退出（返回 None）。
        2. 将 wandb 设置为离线模式（WANDB_MODE=offline），以减少对网络环境的依赖，
           同时确保所有日志首先缓存在本地。
        3. 准备 wandb 所需的项目名称、run 名称与 config 字典，将 task、实验名、设备等
           关键信息写入 config，用于后续分析和复现。
        4. 调用 wandb.init 创建 run；若过程中出现异常，则打印友好提示，
           并返回 None，使上层训练逻辑可以继续执行但不记录 wandb。
    """
    if wandb is None:
        print("[wandb] 未安装 wandb，跳过离线日志记录。")
        return None

    if log_dir is None:
        # 这里选择在日志目录缺失时不使用 wandb，而不是临时创建默认目录，
        # 主要是为了避免日志散落到难以管理的位置，强制用户显式配置 log_dir。
        print("[wandb] 未找到日志目录，跳过 wandb 初始化。")
        return None

    # 使用环境变量配置 wandb 的工作模式与输出目录，而不是在代码中硬编码路径，
    # 这样可以保持与 wandb CLI 工具行为一致，便于运维与部署。
    os.environ.setdefault("WANDB_MODE", "offline")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = log_dir

    project = os.environ.get("WANDB_PROJECT", "legged_gym")
    # 若用户未提供 run_name，则自动拼接 task + 时间戳，保证每次运行都有唯一标识，
    # 便于后续在 wandb 页面区分不同实验。
    run_name = args.run_name or f"{args.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 将与实验复现密切相关的核心信息写入 config。
    # 注意：通过 getattr 访问可选字段（如 num_envs、rl_device），增强兼容性。
    config = {
        "task": args.task,
        "experiment": args.experiment_name,
        "run_name": run_name,
        "headless": headless,
        "num_envs": getattr(args, "num_envs", None),
        "rl_device": getattr(args, "rl_device", None),
    }
    # 过滤掉值为 None 的字段，避免在 wandb 面板上出现噪声信息。
    config = {k: v for k, v in config.items() if v is not None}

    try:
        # reinit=True：允许在同一进程内多次调用 wandb.init（例如多次训练重启），
        # 避免“已有活动 run”之类的错误。
        return wandb.init(project=project, name=run_name, reinit=True, config=config, dir=log_dir)
    except Exception as wandb_error:
        # 设计原则：监控系统不应影响主业务（训练），因此这里吞掉异常并打印提示，
        # 而不是向上抛出导致训练逻辑中断。
        print(f"[wandb] 初始化失败（{wandb_error}），将继续训练但不记录 wandb。")
        return None


def _log_wandb_error(run, error: Exception, tb: str):
    """
    功能描述：
        在训练发生未捕获异常时，将错误信息记录到 wandb，以便后续排查问题。

    参数说明：
        run:
            - 类型：wandb.sdk.wandb_run.Run 或 None
            - 含义：当前训练对应的 wandb run 对象；为 None 时不执行任何记录操作。
        error (Exception):
            - 含义：捕获到的异常对象，用于提取异常类型和错误消息。
        tb (str):
            - 含义：通过 traceback.format_exc() 等方式生成的完整堆栈字符串。

    返回值说明：
        - 无显式返回值；该函数仅产生副作用（向 wandb 写入日志与摘要字段）。

    核心处理流程：
        1. 若 run 为 None，说明未启用 wandb，直接返回。
        2. 通过 run.log 记录一条结构化事件，包含错误类型与错误信息。
        3. 在 run.summary 中写入 status 和 traceback 字段，形成“实验元信息”，
           便于在 wandb 面板上快速筛选出失败的实验并查看详细堆栈。
    """
    if run is None:
        # 未启用 wandb 时无需做任何事情，保持函数为“无副作用空操作”。
        return

    # 使用结构化字段记录错误事件，而不是简单的字符串拼接，
    # 有利于后续在 wandb 中通过筛选器或脚本进行分析。
    run.log(
        {
            "event": "train_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    # 将最终状态与 traceback 写入 summary，相当于“实验级别”的元数据。
    run.summary["status"] = "error"
    run.summary["traceback"] = tb


def train(args, headless=True):
    """
    功能描述：
        构建指定任务的 Isaac Gym 仿真环境与强化学习算法 Runner，
        并在可选的 wandb 监控下启动训练主循环。

    参数说明：
        args:
            - 类型：命令行解析得到的参数对象
            - 含义：包含任务名（task）、实验名（experiment_name）、算法与环境相关配置等信息。
                    函数内部会覆盖其中部分运行时相关标志（如 headless、resume）。
        headless (bool):
            - 含义：是否以“无图形界面”的方式运行仿真。
            - 影响：
                * True：关闭仿真 UI，减少 GPU/CPU 占用，适用于批量训练或远程服务器；
                * False：打开仿真 UI，便于实验调试与可视化观察，但会占用更多资源。

    返回值说明：
        - 无显式返回值。训练产出的策略权重、日志文件以及可能的 wandb 记录
          将写入 log_dir 等外部存储位置。

    核心处理流程：
        1. 将 headless 标志写回 args 中，并显式禁用自动 resume，确保本次运行从头训练。
        2. 通过 task_registry.make_env 基于 task 名称和 args 构造仿真环境及其配置 env_cfg。
        3. 调用 task_registry.make_alg_runner 创建与该任务匹配的算法 Runner（典型为 PPO），
           并返回训练配置 train_cfg。
        4. 若 Runner 提供 log_dir，则尝试使用该目录初始化 wandb run，并将其挂载到 Runner 上，
           使训练过程中可以直接向 wandb 写入指标。
        5. 调用 ppo_runner.learn 进入训练主循环：
             - num_learning_iterations 由配置中的 max_iterations 决定；
             - init_at_random_ep_len=True 用于随机化 episode 起始长度，增加训练多样性。
        6. 在训练成功完成或发生异常时，分别更新 wandb run 的 summary 状态，
           并在 finally 中确保调用 run.finish 正确关闭 wandb 会话。
    """
    # 将运行时参数写回 args，保证下游组件（例如 task_registry）能够感知当前是否 headless，
    # 从而使用对应的仿真配置。
    args.headless = headless

    # 显式关闭“从上次中断位置自动恢复”的行为：
    # - 这样可以避免在用户忘记清理旧权重或日志时，误把当前实验当作上一次实验的继续；
    # - 若确实需要 resume，一般会在命令行显式设置，由其他脚本或逻辑处理。
    args.resume = False

    # 通过注册表创建环境与环境配置：
    # - 这种“按任务名索引”的方式使得训练入口对具体任务解耦，
    #   便于后续新增/修改任务时无需改动通用训练脚本。
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # 创建算法 Runner（例如 PPO）及其训练配置：
    # - env：用于采集数据和计算回报的仿真环境；
    # - train_cfg：包含最大迭代次数、优化器配置、学习率调度等训练超参。
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # 通过 Runner 提供的 log_dir 来统一管理所有与本次训练相关的文件（模型、标量日志等），
    # 并基于该目录初始化 wandb（若可用），使 wandb 文件也集中存放。
    run = _init_wandb_run(args, headless, getattr(ppo_runner, "log_dir", None))
    if run is not None:
        # 将 wandb run 挂载到 Runner 上，使 Runner 内部可以在训练过程中随时记录指标，
        # 避免到处传递 run 或在多个模块间耦合 wandb 依赖。
        ppo_runner.wandb_run = run

    try:
        # 进入强化学习训练主循环：
        # - num_learning_iterations：从训练配置中读取，通常由实验配置文件统一管理；
        # - init_at_random_ep_len=True：将 episode 起始位置随机化，
        #   可以缓解“所有轨迹都从同一初始状态开始”带来的过拟合或探索不足问题。
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True
        )

        if run is not None:
            # 在 wandb summary 中标记本次实验成功结束，
            # 便于在实验列表中快速过滤掉失败/中断的 run。
            run.summary["status"] = "success"
    except Exception as train_error:
        # 一旦训练过程中抛出异常，则在 wandb 中记录详细错误信息，
        # 同时重新抛出异常让上层（例如调度脚本）可以感知失败并采取相应措施。
        _log_wandb_error(run, train_error, traceback.format_exc())
        raise
    finally:
        # 无论训练成功与否，都要确保正确关闭 wandb run：
        # - 避免本地离线日志文件处于不一致状态；
        # - 防止进程退出时 wandb 仍在后台刷新导致资源泄漏。
        if run is not None:
            run.finish()


if __name__ == '__main__':
    # 统一从命令行解析训练参数，使脚本既可直接运行，又可被其他 Python 模块复用。
    args = get_args()

    # headless=True：训练时不启动 Isaac Gym 的 GUI 窗口，
    # 降低 GPU/CPU 占用，适合批量训练或在服务器上以 nohup / tmux 等方式长期运行。
    train(args, headless=True)

    # 若需要调试机器人行为或观察仿真细节，可以改为：
    # train(args, headless=False)
