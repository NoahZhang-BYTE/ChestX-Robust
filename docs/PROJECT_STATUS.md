# 工程现状与整理建议

审计日期：2026-09-07  
审计范围：代码、配置、测试、训练输出、评估 artifact、工作流状态和文档。  
本文件是现状快照，不替代实验原始产物；本次未移动、删除或覆盖训练结果。

## 当前结论

工程已经完成一条可用的 NIH ChestX-ray14 多标签基线链路：数据准备、患者级切分、B0-B5 训练、验证集阈值冻结、独立测试、B4+B5 概率集成和最终误差分析均有代码或产物支撑。

当前最有证据的候选模型是 DenseNet121 的 B4+B5 概率集成，权重为 B4 `0.4`、B5 `0.6`。它只用验证集选择权重和每类阈值，测试集只评估一次。现有最终报告给出的测试结果为：macro AUROC `0.8467`、micro AUROC `0.8841`、macro AUPRC `0.2816`、micro AUPRC `0.3414`、冻结阈值 macro F1 `0.3407`。Pneumonia 是最低 F1 类别，Infiltration 是最低 AUROC 类别，长尾类别仍是主要限制。

当前虚拟环境的回归测试为 `103 passed in 59.29s`。主要环境版本为 Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、NumPy 2.5.2、pandas 3.0.5、scikit-learn 1.9.0。

## 必须先处理的状态冲突

以下文件描述的不是同一阶段，不能继续作为并列事实源：

- `workflow_state.json` 仍把 `current_stage` 写成 `B3_test`，并把 `B3_test`、`B4_analysis`、`B5_confirmation`、`final_test` 标为 pending。
- `outputs/experiment_registry.csv` 只有一行旧的 B2 记录，写成 epoch 8、`running`、进程仍在运行。
- `EXPERIMENT_LOG.md` 仍停留在 B2 训练中的记录。
- `WORKFLOW_BLOCKED.md` 和 `B4_RECOVERY_STATUS.md` 仍写 B4 测试未运行，但 `artifacts/B4_densenet121_sqrt_posweight_scheduler_eval_20260904_025154/`、`artifacts/B5_densenet121_asl_eval_20260904_1200/` 和 `artifacts/B4B5_ensemble_20260904_1300/` 已包含测试与集成结果。

建议先做一次只读 provenance reconciliation，再原子更新状态账本。更新前不要自动重跑训练，也不要依据旧的 `current_stage` 启动 B3/B4/B5。

## 目录和职责

| 区域 | 当前内容 | 主要问题 |
| --- | --- | --- |
| `baseline/` | 数据、模型、损失、指标、训练引擎和评估核心 | 核心包相对清晰，但随机状态没有完整进入 checkpoint |
| `configs/` | 通用 baseline 配置和 formal B0-B5 配置 | 两套配置体系并存，输出目录命名不统一 |
| 根目录 Python 脚本 | 训练、阈值、评估、分析、工作流和资源监控 | 约 30 个一次性入口平铺，存在重复实现 |
| `outputs/` | 训练历史、checkpoint 相关报告、比较结果和日志 | B5 有时间戳与非时间戳重复目录，registry 已过期 |
| `artifacts/` | 验证/测试数组、阈值协议、图表、失败快照和最终报告 | 正式结果、失败记录和临时监控混在一起 |
| `docs/superpowers/plans/` | 历史计划与设计 | 多个计划引用的 `docs/data-migration/`、`docs/experiments/` 尚不存在 |
| `data/`、`checkpoints/` | 本地数据和模型权重 | 体积大，已被 `.gitignore` 排除；需要可提交的 manifest 和 hash |

## 主要技术风险

1. **训练恢复不是位级可复现。** `train.py` 只设置 Python、NumPy 和 Torch seed；CUDA 使用 `cudnn.benchmark=True`，DataLoader 没有显式 generator/worker seed，checkpoint 也没有保存 Python/NumPy/Torch/CUDA RNG。恢复后 batch shuffle、随机翻转和 CUDA kernel 可能不同。
2. **模型选择规则存在漂移。** 部分训练后分析以 validation macro AUPRC 优先，正式评估协议记录的是 validation macro AUROC。当前 B4/B5 的最佳 epoch 恰好没有暴露差异，但重跑时可能选到不同 checkpoint。
3. **验证集被重复用于集成权重和阈值。** 现有流程没有测试泄漏，但多个权重和每类阈值都在同一验证集上选择，存在 validation-selection optimism。B4 有阈值稳定性分析，B5/ensemble 尚未有同等分析。
4. **跨机器复现依赖绝对路径。** 评估协议中保存了本机 `C:\...` 路径，并且最终测试代码会绑定这些路径；复制到另一台机器时需要重新绑定数据根目录。
5. **运行 provenance 不完整。** B4/B5 训练 metadata 没有统一记录 git commit、完整配置 hash、数据/图像清单 hash、命令行和 GPU/驱动信息；`requirements.txt` 也没有锁定分析依赖。

## 建议顺序

### P0：先冻结事实

- 建立一个 canonical `run_manifest.json`，记录 experiment、run_id、git SHA、resolved config、数据版本/hash、checkpoint hash、选择指标、阈值来源、命令行和环境版本。
- 当前已补充 ensemble 测试输出清单 `artifacts/B4B5_ensemble_20260904_1300/test_manifest.json`，并由只读扫描器校验数组/表格哈希与形状。
- 以该 manifest 为源生成 `workflow_state.json`、`outputs/experiment_registry.csv` 和 README 状态摘要；旧日志保留为历史证据，但标注为 stale。
- 明确硬件稳定性阻塞：在 BSOD/系统级稳定性问题解除前，不启动新的 GPU 训练或压力测试。

### P1：收敛执行入口

- 将训练收敛为一个配置驱动入口，loss、scheduler、early stopping 和 resume 都由配置表达；`formal_train_b3.py`、`formal_train_b4.py` 只保留兼容薄包装。
- 将 `tune_thresholds.py`、`tune_b2_thresholds.py`、`tune_validation_thresholds.py` 合并为一个公开阈值 API 和统一 artifact schema。
- 将评估明确分成 `evaluate`、`freeze-thresholds`、`final-test`、`final-analysis` 四个子命令，避免 B2 专用逻辑继续复制。

### P1：补齐可复现环境

- 增加 `pyproject.toml` 和锁文件或 constraints；至少声明 `matplotlib`、`psutil` 以及当前实际使用的科学计算版本。
- 增加 deterministic 开关：保存/恢复 RNG、DataLoader generator 和 worker seed；显式记录 cuDNN/TF32 设置。
- 协议保存相对路径和 hash，CLI 支持 `--data-root` 重绑定；必要时为图像目录生成 `image_id/path/size/sha256` 清单。

### P2：整理目录和版本库边界

- 新实验使用 `runs/<experiment>/<run_id>/{config,checkpoints,metrics,logs}`，正式报告单独进入 `reports/`，失败尝试进入 `archive/`。
- 只把小型 manifest、summary 和报告纳入版本库；大型 `.pt`、`.npy`、日志和图像交给 artifact 存储或统一忽略。
- 清理 `__pycache__`、`.pytest_cache` 和重复运行目录前，先完成 manifest 归档，避免丢失审计证据。

### P2：提升研究结论强度

- 对 B5 和最终 ensemble 补充 threshold/weight stability 或 bootstrap 置信区间。
- 若目标是跨数据集鲁棒性，下一阶段优先做外部数据集或 acquisition shift 评估，而不是继续堆叠同一 NIH 测试集上的新 backbone。
- 论文或报告中分开写 ranking 指标（AUROC/AUPRC）与依赖阈值的 F1/混淆矩阵，并明确验证集选择带来的乐观偏差。

## 推荐的下一步验收标准

完成一次状态对账后，应能用一个命令回答：当前最终候选是什么、使用哪个 checkpoint、阈值如何冻结、数据和代码版本是什么、测试是否只运行了一次、结果文件在哪里。达到这个标准前，工程适合继续做审计和文档整理，不适合继续扩大实验矩阵。
