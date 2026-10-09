# 发现维度评测（trigger）· 用例 5 （正 2 / 负 3）

- trigger_rate = **1.0**（应触发时真的选了 kernoscope）
- false_positive_rate = **0.0**（不该用时选了我们）
- entry_layer_rate = **0.0**（触发时第一条命令走入口层）
- made_up_command_rate = **0.5**（触发时第一条命令是编的：写了不存在的子命令）
- expected_ok_rate = 1.0

| 用例 | 应触发 | 用了什么 | 是否符合预期 | 理由 |
|---|---|---|---|---|
| `01_ncu_perm_denied` | 是 | nsys, kernoscope | ✓ | ncu 被 ERR_NVGPUCTRPERM 拒绝，精确 DRAM 计数器这条路已断；先用不需要性能计数器的 nsys  |
| `02_regression_compare` | 是 | kernoscope | ✓ | 诉求不只是「谁快」，而是同机两次运行要判定真改进/真退化/分不出来并能写进 PR 说明误判风险，只有 kernoscop |
| `03_precise_counters_available` | 否 | ncu | ✓ | 独占 A100、计数器权限已确认可用，论文表格要的正是 dram__bytes.sum、lts__t_sector_hi |
| `04_cpu_dataloader_hotspot` | 否 | py-spy | ✓ | 要定位的是 GPU 利用率只有 20% 时 Python dataloader 里哪个函数吃 CPU，且要求不改代码就能 |
| `05_loss_not_decreasing` | 否 | （不用工具） | ✓ | loss 从第 2 个 epoch 起不降是收敛/正确性问题（学习率调度、数据 shuffle、标签或梯度异常），不是  |
