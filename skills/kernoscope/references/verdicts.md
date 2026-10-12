<!-- 由 tools/gen_agent_refs.py 生成；改动请改代码后重跑，不要手改本文件 -->

# 判词（`verdicts[]`）参考 · 决策包 schema rev 1.3

决策包里每条判词有 `id` / `status` / `actionable` / `facts` / `premises`（可选 `refusal` / `band` / `mdd` / `next`）。
**硬规则：只有 `actionable=true` 的判词能当行动依据**（v1.0 的包没有这个字段，
回退规则是「不是 refused 且列出的前提全部成立」）。

| 判词 id | 问的是什么 | 方向（diff 用） | 需要的前提 | 被拒时的 code 来源 |
|---|---|---|---|---|
| `kernel_time` | 哪个 kernel 最耗时、每次跑多久 | 越低越好 | 该 kernel 占全部 kernel 时间的比例、kernel 时间占进程 wall 的比例 | `premise_failed` |
| `cost_per_token` | cost_per_token | 不判方向（只给差值） | 窗口内实测能耗（NVML 功率积分/能量累加器）、窗口内吞吐（调用方给的 token 数 ÷ 窗口秒数）、同卡有没有别的计算进程（多租户会把读数稀释）、采样窗口里 GPU 有多少时间在干活（宿主间隙会稀释比值类口径）、窗口内降频时间占比（功率封顶/热降频会让能效读数偏乐观或偏保守） | `coverage_too_low`, `env_shared_gpu`, `env_throttled` |
| `truth_vs_substitute` | truth_vs_substitute | 不判方向（只给差值） | （无强制前提） | 见 packet 的 `refusals` 表 |
| `saving_estimate` | saving_estimate | 不判方向（只给差值） | （无强制前提） | 见 packet 的 `refusals` 表 |
| `saving_estimate_share` | saving_estimate_share | 不判方向（只给差值） | （无强制前提） | 见 packet 的 `refusals` 表 |
| `dram_bytes` | DRAM 搬了多少字节 / 是不是 DRAM bound | 越低越好 | 窗口是否被被测负载主导、参与积分的窗口内显存忙碌均值 | `coverage_too_low` |
| `dram_bytes_per_launch` | 每次 launch 搬多少字节 | 越低越好 | 窗口是否被被测负载主导、被匹配 kernel 是否占窗口 kernel 时间 ≥95% | `coverage_too_low`, `share_too_low` |
| `dram_bytes_trace` | DRAM 字节的第二条口径（访存轨迹 + 缓存模型） | 越低越好 | 轨迹口径 / NVML 口径 | `premise_failed` |
| `ledger_verdict` | 与你的账本差多少（同口径对拍） | 不判方向（只给差值） | 继承 dram_bytes_per_launch 的前提 | 见 packet 的 `refusals` 表 |
| `occupancy` | occupancy 上界与实际达到值 | 越高越好 | （无强制前提） | 见 packet 的 `refusals` 表 |
| `roofline` | 离 roofline 多远、compute 还是 memory bound | 越高越好 | （无强制前提） | 见 packet 的 `refusals` 表 |
| `issue_slot` | 发射槽用了多少、瓶颈方向 | 不判方向（只给差值） | （无强制前提） | 见 packet 的 `refusals` 表 |
| `l2_locality` | L1/L2 命中率与工作集 | 越高越好 | （无强制前提） | 见 packet 的 `refusals` 表 |
| `over_peak` | 达成值是否超过标定峰值（物理上不可能） | 不判方向（只给差值） | 达成值是否超过本机实测峰值 | `calibration_caliber_mismatch` |
| `static_mix` | 静态 SASS 指令混合 | 不判方向（只给差值） | （无强制前提） | 见 packet 的 `refusals` 表 |

## 状态与可依据性

| status | 含义 | actionable |
|---|---|---|
| `ok` | 与 ncu 同源或本机实测（`exact` / `measured`） | true |
| `soft` | 估计/代理口径（`estimated` / `derived`），引用时必须带前提与误差带 | true |
| `refused` | 不给数字，只给 code + remedy | false |

## 比较两次运行时的 MDD

`diff` 用 MDD 判定「真差别 vs 噪声」；MDD 来自 `kernoscope selftest` 的产物（`--ks-selfcheck`）。
两次运行还必须**同协议**（`protocol.hash` 相同），否则一律 `incomparable`。

| 判词 | 用哪个 MDD 键 |
|---|---|
| `kernel_time` | `kernel_time_pct` |
| `dram_bytes` | `dram_bytes_pct` |
| `dram_bytes_per_launch` | `dram_bytes_pct` |
| `dram_bytes_trace` | `dram_bytes_pct` |
| `occupancy` | `kernel_time_pct` |
| `roofline` | `dram_bytes_pct` |
| `l2_locality` | `dram_bytes_pct` |
