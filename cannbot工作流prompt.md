# cannbot 工作流 Prompt —— mish 算子第三轮性能优化（round-3）

> 本文件由浏览器提交 Agent 维护，是交付给开发 Agent（cannbot）的任务指令。
> 本仓库（rib12316/s10kernel）是双方交接通道：开发 Agent 从本仓库读取本文件与基线代码，
> 完成优化后将新代码推回 `code/`，并把自己维护的《浏览器操作.md》更新推回，由浏览器 Agent 执行线上提交。

---

## 直接可用的 Prompt（复制给 cannbot）

```
请使用 ops-direct-invoke 工作流，对当前项目中的 mish 算子执行第三轮性能优化。
先分析本轮交接材料（基线代码、两轮榜单数据、差距分析）与运行环境
（Atlas A2 / ascend910b / CANN 社区版 9.1.0），再完成方案设计、代码实现、测试、
精度与性能验收，并输出平台提交版工程文件。

【当前状态（2026-10-31）】
- round1：70.50 分（动态重算值）｜round2：72.49 分，均第 20 名
- round2 可见点用时：4.43/4.62/4.55/4.80/4.52μs，TBest：3.84/4.03/4.13/4.19/3.91μs
- 已落地优化：单 tile 轻量路径（TBuf+显式事件）、UB 精确核算（替代 9/10 保守系数）、
  多核粒度均衡划分、8 运算计算链（1/w 以 2/(2w) 实现）
- 未达预期主因：同期 TBest 被刷新稀释收益（点3 4.25→4.13、点5 3.95→3.91）

【round3 目标（差距焦点）】
第 19 名「天天debug」可见点 4.12~4.39μs，比我方快 0.3~0.4μs —— 证明小 shape
固定开销仍有压缩空间。本轮以可见点（小 shape）为主攻方向：
1. tiling 数据下发与 kernel 启动链路：tiling 结构 3×int64（24B）能否压缩
   （字段宽度/数量，注意 GET_TILING_DATA_WITH_STRUCT 与 REGISTER_TILING_DEFAULT 配套）
2. 小张量执行路径进一步减负：轻量路径的初始化（InitBuffer 分配、事件 ID 获取）
   能否按"元素总数 vs 单 tile 容量"的通用量分支再精简
3. 点 2 回归分析：round2 点 2 由 4.54 退到 4.62μs，定位是哪项改动引入并修复
4. 大 shape 持续压缩：精确核算后 tile 是否已用满 UB；尾核粒度是否已均衡
5. 数值变体（窄 dtype tanh 原生路径）最后做，必须先过全量精度

【红线（违反 = 0 分）】
- 所有分支条件只能来自运行时通用量（元素总数、UB 容量、AIV 核数、dtype 位宽），
  严禁任何已知测试 shape/数值特化常量
- 精度阈值不放宽：fp32 atol 2⁻¹⁶/rtol 2⁻¹⁰、fp16 2⁻⁹/2⁻⁹、bf16 2⁻⁶/2⁻⁶；
  特殊值语义：mish(-inf)=NaN、mish(+inf)=+inf、NaN 传播、-0 保留
- tiling key 位宽必须保持 2（ASCENDC_TPL_UINT_DECL(schMode, 2, ...)，改回 1 会 Compile Error）

【基线代码】本仓库 code/ 目录（平台 op_host/op_kernel 结构，round2 已 Pass 72.49 分版本）：
  code/op_kernel/mish.h           kernel 类（262 行，含单 tile 轻量路径）
  code/op_kernel/mish.cpp         入口三 dtype 分派（35 行）
  code/op_kernel/mish_tiling_key.h 三模式声明，位宽 2（22 行）
  code/op_host/mish_tiling.cpp    tiling（156 行，精确核算+粒度均衡）
其余平台文件（mish_def.cpp / mish_infershape.cpp / mish_tiling_data.h / CMake / build.sh）
保持模板原样不碰。本地完整工程模板与直调版参考见开发机
/workspace/算子挑战赛/mish-handoff-extract/（round2 交接遗留）。

【验收标准】
- 精度：golden.py 全量黑盒回归（±Inf/NaN/-1.1924/次正规数/1 元素~67M 大 tensor）
- 构建：覆盖进平台模板后 bash build.sh --soc=ascend910b 产出 custom_opp_*.run
- 性能：msprof 中位计时，逐 case 与 round2 基线量化对比；小 shape 目标再降 ≥0.3μs
- 分数目标：≥76 分（追平第 19 名），冲刺 ≥79（第 17 名水平）

【交付物（推回本仓库）】
1. code/ 下 4 个文件（文件名与结构不变）
2. 优化报告.md：每项改动动机、本地性能数据、精度结论、round3 预期分数区间
3. 《浏览器操作.md》更新（如线上提交流程有新注意事项）
线上提交由浏览器 Agent 在 CANNJudge 在线编辑器完成，无需你操作。
```

---

## 两轮轨迹（供参考）

| 轮次 | 分数 | 排名 | 可见点用时(1-5) | 主要改动 |
|---|---|---|---|---|
| round1 | 70.50（动态重算） | 20 | 4.49/4.54/4.86/4.84/4.55 | 直调 kernel → 平台结构移植（首次 Pass） |
| round2 | 72.49 | 20 | 4.43/4.62/4.55/4.80/4.52 | 单 tile 轻量路径、UB 精确核算、多核粒度均衡、8 运算链 |

榜单动态重算机制：TBest 更新会重算全场分数，优化幅度跑赢大盘才能升名次。
