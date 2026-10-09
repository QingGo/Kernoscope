# 发现维度评测（trigger）· 用例 5 （正 2 / 负 3）

- trigger_rate = **1.0**（应触发时真的选了 kernoscope）
- false_positive_rate = **0.0**（不该用时选了我们）
- entry_layer_rate = **0.0**（触发时第一条命令走入口层）
- made_up_command_rate = **1.0**（触发时第一条命令是编的：写了不存在的子命令）
- expected_ok_rate = 1.0

| 用例 | 应触发 | 用了什么 | 是否符合预期 | 理由 |
|---|---|---|---|---|
| `01_ncu_perm_denied` | 是 | kernoscope | ✓ | ncu 已被 ERR_NVGPUCTRPERM 挡死，而 kernoscope 的目录描述正是「当 ncu 报 ERR_ |
| `02_regression_compare` | 是 | kernoscope | ✓ | 诉求是「两次运行到底是不是真改进、并说明误判风险」，kernoscope 唯一明确提供带误差带的「真改进 / 真退化 / |
| `03_precise_counters_available` | 否 | ncu | ✓ | 要的是精确 dram__bytes.sum、lts__t_sector_hit_rate.pct 与 warp stal |
| `04_cpu_dataloader_hotspot` | 否 | py-spy | ✓ | 诉求是「哪个 Python 函数在花时间、最好不改代码、要调用栈」，与 py-spy 的目录描述（采样 profiler |
| `05_loss_not_decreasing` | 否 | （不用工具） | ✓ | loss 从第 2 个 epoch 不再下降是训练/优化/数据层面的收敛问题（学习率、数据、标签、BN 等），目录里 n |
