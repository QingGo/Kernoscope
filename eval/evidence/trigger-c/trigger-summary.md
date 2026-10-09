# 发现维度评测（trigger）· 用例 3 （正 2 / 负 1）

- trigger_rate = **1.0**（应触发时真的选了 kernoscope）
- false_positive_rate = **0.0**（不该用时选了我们）
- entry_layer_rate = **1.0**（触发时第一条命令走入口层）
- made_up_command_rate = **0.0**（触发时第一条命令是编的：写了不存在的子命令）
- expected_ok_rate = 1.0

| 用例 | 应触发 | 用了什么 | 是否符合预期 | 理由 |
|---|---|---|---|---|
| `01_ncu_perm_denied` | 是 | kernoscope | ✓ | ncu 因 ERR_NVGPUCTRPERM 拿不到计数器，kernoscope 正是为这种受限/多租户 GPU 设计， |
| `02_regression_compare` | 是 | kernoscope | ✓ | 需求是判断同机两次运行是「真改进/真退化/分不出来」并说明误判风险，kernoscope 的 diff 判定正是做这件事 |
| `03_precise_counters_available` | 否 | ncu | ✓ | drv 策略已放开、ncu 实测可用，而论文表格要的精确 dram__bytes.sum / lts__t_sector |
