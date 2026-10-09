# 四个闭环配方

每个配方都是"从问题到结论"的完整路径，可以直接照抄命令。所有命令都遵循同一条纪律：
**先确认前提，再引用数字；判词不可依据就先修测量口径。**

---

## 配方 1：取证（这个 kernel 慢在哪）

```bash
kernoscope capabilities --brief                        # 先看这台机器能答什么
kernoscope --packet - --ks-window auto -- python bench_loop.py   # 一次测量
```

读决策包时按顺序看：

1. `verdicts[].actionable` —— 只有 `true` 的能当依据；
2. `verdicts[].premises[]` —— 结论成立的前提，以及实测值 vs 门槛；
3. `refusals[code].action` —— 被拒的怎么办、代价多少；
4. `cost` —— wall 时间、包大小、完整报告大小（决定要不要再往深读）。

被判词指出方向之后再动手：`roofline` 说 memory bound 就先查访问模式；说 compute bound
就用 `--ks-instr-hist` 拿发射槽上界与方向，而不是去调 occupancy。

---

## 配方 2：与账本对拍（我的模型说应该是 X 字节）

```bash
kernoscope --packet - --ks-calibration machine-profile.json --ks-window auto \
    --ks-ledger-mb <每次 launch 的 MB> -k regex:<kernel> -- python bench_loop.py
```

- 账本是**你**给的解析值；工具报同口径误差 `est__dram_vs_ledger_err_pct`。
- 口径两种：`--ks-ledger-mb`（per-launch，需要窗口被该 kernel 主导）与
  `--ks-ledger-total-mb`（整窗口）。**两者不混用**。
- per-launch 口径的两个前提：窗口覆盖率 ≥ 80%、被匹配 kernel 的内核时间份额 ≥ 95%。
  任一不成立 → 该判词 `refused`，且**由它推出的账本误差同样不可引用**（会一起被拒）。
- 代理口径不成立时可以换轨迹口径：`--ks-memtrace <trace>`（只测被追踪的 kernel）。

---

## 配方 3：拒绝处置（`n/a` 了怎么办）

```bash
python skills/kernoscope/scripts/remedy.sh share_too_low   # 只查这一个 code
```

顺序永远一样：**先解除前提，再要数字**。三种典型：

| code | 处置 |
|---|---|
| `no_nvml_samples` / `no_calibration` | 把设备级采样打开 / 跑一次本机标定（`selftest` 会给峰值与误差带） |
| `share_too_low` / `coverage_too_low` / `multi_kernel_window` | 缩小窗口（`-k`、`--ks-window-s`）、拉长稳态段，或换轨迹口径 |
| `counters_denied` | 与 ncu 同源的精确量不可得；容器内没有合法绕过，改用物理观测量口径 |

不要因为拿不到数字就跳过口径检查 —— 那正是把"测量问题"误判成"优化空间"的地方。

---

## 配方 4：回归对比（这次改动真的更快吗）

```bash
kernoscope selftest --out machine-selfcheck.json        # 一次，之后复用
kernoscope diff before.json after.json --selfcheck machine-selfcheck.json --json
```

判词只有四种：`improved` / `regressed` / `indistinguishable` / `incomparable`。

- `indistinguishable`：差值在 MDD 之内 —— **这是噪声，不要写进周报**；
- `incomparable`：协议不同（换过窗口/标定/kernel 选择）、缺 MDD、或一侧被拒；
- `direction=lower` 的判词（DRAM 字节、kernel 时长）变小才是 improved；
  方向不明确的判词（`ledger_verdict`、发射槽等）只给差值，工具不替你定价值方向。

MDD 是**测量出来的**且只对测出它的机器状态成立：换机器、时钟/温度状态变化、同机其他
租户负载变化之后都要重跑 `selftest`。
