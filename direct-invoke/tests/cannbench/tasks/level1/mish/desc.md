# mish 算子 API 描述

<!-- workflow4 物化：冻结复用 /workspace/算子挑战赛/mish/tests/cannbench/tasks/level1/mish/desc.md
     @ sha256 aa1c1ef09068297a3f5831587d8897dbd82dcd7204355473730c43a40ab07d15。
     相对源文件的唯一内容变更（$WORK_DIR/0.1-黑盒测试设计.md 第 8.2 节定稿，事实性同步）：
     「支持范围」表维度数行 1~8 → 0~8（0 维标量按 1 元素，与空 tensor 语义区分），及
     「规则与约束」同一事实行同步；数学公式、特殊情况表、精度要求节未动。本注释为来源留痕。 -->

## 1. 算子简介

Mish 算子用于计算输入张量的 Mish 激活函数，是深度学习中常用的平滑激活函数（如 YOLOv4/5 检测网络），数学上是 `x * tanh(softplus(x))` 的复合，处处光滑、无界、非单调。

**主要应用场景**：
- 目标检测网络（YOLO 系列 Neck/Head）激活层
- 自编码器与生成模型的隐藏层激活
- 需要平滑、非单调激活特性的轻量网络

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致

## 2. 算子定义

### 数学公式

$$
y = x \cdot \tanh(\mathrm{softplus}(x)), \quad \mathrm{softplus}(x) = \ln(1 + e^x)
$$

**数值稳定要求**：`softplus(x)` 在大 `x` 时与 `x` 差别小于浮点精度（约 `x > 20` 起），必须使用数值稳定形式计算，避免 `e^x` 上溢：例如 `softplus(x) = max(x, 0) + ln(1 + e^{-|x|})`。

### 特殊情况

| 条件 | 输出 | 说明（golden 按 torch 2.10.0 CPU FP64 实测） |
|------|------|------|
| x = +Inf | +Inf | tanh(softplus(+Inf))=1，+Inf·1=+Inf |
| x = -Inf | **NaN** | softplus(-Inf)=0，tanh(0)=0，(-Inf)·0=NaN（IEEE）；**kernel 不得将 -Inf 分支为 y=x** |
| x = NaN | NaN | IEEE 传播 |
| x = ±0 | ±0 | x·tanh(sp)≈x·0.5x²→±0（按 IEEE 相等判定，不要求位级） |
| x → -∞（有限大负值） | → 0⁻ | softplus→0⁺，mish→x·0⁺→0⁻（次正规输出区） |
| \|x\| ≥ ~20 | y ≈ x | 线性区，tanh(softplus(x))≈1 |
| x = 88（float32 内部 exp 边界） | y ≈ x | 中间计算用稳定 softplus 时不依赖 e^x |

## 3. 接口规范

### 算子原型

```python
cann_bench.mish(Tensor x) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，任意维 ND 连续张量，无 attrs |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | Mish 激活结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致，无广播
- `x` 为任意维 ND 连续张量（0D ~ 8D，0 维标量按 1 元素处理），输出为连续张量
- 数值稳定：softplus 必须使用稳定形式，大 `x`（含 ±65504 / ±3.4e38 量级）时不得因 `e^x` 上溢产生错误结果，线性区 `y ≈ x`
- 特殊值行为（与 golden 一致）：NaN 按 IEEE 传播；**mish(-Inf) = NaN**；mish(+Inf) = +Inf；±0 按 IEEE 传播；比对按 NaN==NaN、Inf 同号匹配

### 支持范围

输入 tensor 各维度的支持范围：

| 维度 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 0 ~ 8 | 0 维标量按 1 元素处理（与空 tensor 0 元素语义区分）；cases.yaml 实测 0D ~ 8D |
| `x` 各维大小 | 1 ~ 8192 | cases.yaml 实测 1 ~ 8192（含 1D 长度 1000007） |
| `x` 元素总数 | 0 ~ 67M | 含空 tensor（0 元素）到 67,108,864（8192×8192） |
| `x` 元素值 | 全范围（含 ±Inf / NaN） | 特殊值语义见「特殊情况」与「规则与约束」 |
| 对齐性 | 32B 对齐与非对齐尾块均支持 | cases.yaml 含质数 shape / 奇数 numel |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。本 proto 不配置 `precision_thresholds` 覆写，评测器 `thresholds.py` 默认表生效；NaN 位置严格相等、双 Inf 同号匹配、单侧 Inf 按 dtype 最大有限值饱和替换后继续 MERE/MARE 判定，另含小值域/相消兜底口径（详见评测器 `compare.py`）。

## 5. 标准 Golden 代码

```python
import torch

def mish(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入张量的 Mish 激活函数

    y = x * tanh(softplus(x))，softplus(x) = ln(1 + e^x)

    Args:
        x: 输入张量（float16 / bfloat16 / float32）

    Returns:
        Mish 激活结果，shape/dtype 与输入一致
    """
    # 检测输入 dtype
    input_dtype = x.dtype

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    x_compute = x.to(compute_dtype)
    y = torch.nn.functional.mish(x_compute)

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return y.to(input_dtype)
    return y
```

说明：评测器默认 `golden_precision=fp64_cpu`，输入先被升为 float64 并在 CPU 上调用 golden，实际以 FP64 精度计算；checker 将结果截断回输出 dtype 后比对。上式的 dtype 自适应升精度写法用于脱离评测器直接调用时保持同等口径。

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.mish(x)  # shape/dtype 与 x 一致
```
