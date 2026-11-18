# 贝叶斯 MINSTED 原型

后续已新增连续圆周扫描版本，其 [验证方案](circular_bayesian_validation.md) 与 [200 组配对试验结果](circular_bayesian_results.md) 单独记录。本文介绍较早的离散点曝光原型。

## 实现范围

`simulators/BayesianMinstedSimulator.py` 将序贯贝叶斯推断和预期信息增益（EIG）实验设计用于当前高斯 MINSTED 发光模型。原有 `MinstedSimulator` 仍提供连续圆周扫描、逐光子反馈的基准。

该原型在每次曝光期间把 E-PSF 保持在一个选定位置和宽度，曝光结束后读取光子总数，包括零光子。它不是原圆周扫描协议的逐光子替换实现，也不是贝叶斯 MINFLUX 论文的原样复现。

共同的高斯公式集中在 `simulators/gaussian_epsf.py`；新旧两种模拟器使用相同的空间响应。假设分子静止，峰值亮度和背景率已知。

## 如何替代经验更新规则

| 原算法 | 新原型 |
|---|---|
| `C_i = C_(i-1) + alpha * (S_i-C_(i-1))` | 完整贝叶斯更新后，以后验均值作为估计；步长为新旧均值之差，方向为该差向量的方向。 |
| 由圆周运动决定下一次照明方向 | 在候选方向中选使下一次计数 EIG 最大的方向。 |
| 每个光子按 `gamma` 缩小扫描半径 | 联合选择下一次照明相对当前估计的偏移半径和方向，无固定收缩率。 |
| 每个光子按 `gamma` 缩小 PSF | 在设备允许的 FWHM 候选值中选择 EIG 最大者，允许重新放宽。 |
| 没有光子就不更新圆心 | 零光子曝光也改变后验和位置估计。 |

这里的“偏移半径”是一次点曝光位置到曝光前后验均值的距离，不是持续执行的圆周扫描半径。位置估计和下一次照明位置是不同的量，不应把两者的移动长度混为一谈。方向不确定的零位移也不应强行赋予角度。

## 观测模型与目标函数

对动作 `a=(s,d,tau)`，`s` 为照明位置，`d` 为高斯 E-PSF 的 FWHM，`tau` 为曝光时间：

\[
\Lambda_a(\mathbf x)=\tau\left[
\epsilon\exp\left(-4\ln2\frac{\|\mathbf x-\mathbf s\|^2}{d^2}\right)+\beta(\mathbf s)
\right],\qquad n\sim\mathrm{Pois}(\Lambda_a(\mathbf x)).
\]

坐标单位 nm，`tau` 单位 ms，`epsilon` 与 `beta` 单位 kHz，即 photons/ms。背景映射以扫描位置为输入，与原模拟器一致。

贝叶斯更新：

\[
P_{k+1}(\mathbf x)\propto P_k(\mathbf x)
e^{-\Lambda_a(\mathbf x)}\Lambda_a(\mathbf x)^n/n!.
\]

在二维网格上使用对数似然归一化；最终估计取后验均值，它在模型成立时最小化后验平方误差。也提供 `get_map_estimate()`，方便与使用 MAP 的实验比较。`get_uncertainty()` 为 `sqrt(trace(covariance)/2)`，属于后验不确定性，不等于真实误差或跨实验标准差。

动作评分采用互信息形式：

\[
\mathrm{EIG}(a)=\sum_{n=0}^{\infty}\sum_j
p_j p(n\mid\mathbf x_j,a)
\log\frac{p(n\mid\mathbf x_j,a)}{\sum_l p_l p(n\mid\mathbf x_l,a)}.
\]

优化器不接收真实分子位置或光子来源。真实位置仅供示例的合成探测器生成计数与计算误差。

候选动作曝光时间相同，因此当前目标是**每个固定时长曝光的信息量**，不是每个光子的信息量，也不是最小光损伤。不使用 MINFLUX 论文的 `eta=mu/E[I]` 归一化，因为这里明确固定亮度、背景率和曝光时间。若要改为固定预测光子预算，必须另外定义可调曝光时间或功率及其物理限制。

## 候选集合与数值近似

默认 FWHM 在 40–250 nm 间取 5 个几何间隔值；每个宽度的照明位置位于以后验均值为中心、半径为 `0/.25/.5/.75 * d` 的候选圆环上，非零半径各取 8 个方向，共 125 个候选动作。

这些比例是搜索范围和分辨率，不是每步必然执行的移动或收缩律。EIG 只在这个有限集合内取最大，不能声称连续空间或全部实验轨迹的全局最优。可以增加 `width_levels`、`n_angles` 或自定义 `radius_factors` 做候选分辨率检查。对多峰或严重偏斜后验，还应检查围绕均值产生候选点的覆盖情况。

泊松计数求和按所有网格点的最大期望计数控制尾部概率，默认 `poisson_tail=1e-10`，计入零光子。EIG 按批次计算以控制内存；大计数曝光的计算成本会上升。

默认初始网格为 65×65，覆盖初始均值各方向 ±5 个先验标准差，网格值表示概率质量。后验边缘质量足够小时，裁去其边缘并在线性插值后重新归一化；仅在某轴范围可缩小至少 20% 时重建网格。`grid_tail_mass=1e-8` 控制每次裁剪的边缘概率，`total_pruned_mass` 记录累计裁剪量。插值还会引入数值误差，因此需要检查网格分辨率。

算法不能恢复初始有限网格之外的真值，错误先验、亮度/背景失配、分子运动或闪烁都可能造成过度自信。不能只因后验很窄就宣布实验定位成功。

## 最小调用

```python
import numpy as np
from minsted.simulators import BayesianMinstedSimulator

tracker = BayesianMinstedSimulator(
    prior_mean=(-50., 40.),
    prior_sigma=60.,
    brightness=30.,
    bg_global=3.,
    d0=250.,
    d_min=40.,
    dwell_ms=.05,
)

# 实验运行中，用硬件在 action.position_nm、action.fwhm_nm 下曝光，
# 再把这段 action.dwell_ms 内的实际总检测计数交给 observe。
action = tracker.select_next_exposure()
tracker.observe(photon_count=0, action=action)
estimate = tracker.get_localization_estimate()
covariance = tracker.get_covariance()
```

`history` 记录每次曝光后的计数、累计光子、采集时间、实际选用宽度、偏移半径、估计更新向量和后验不确定性。观测动作必须与实际执行动作一致；不能先按一个位置采集，再用另一个动作计算似然。

## 运行对比示例

在项目根目录，使用含 NumPy、SciPy、Matplotlib 的 Python：

```powershell
& 'C:/Users/20291/miniconda3/envs/minsted/python.exe' examples/bayesian_minsted.py --no-plot
```

默认每次 120 个曝光、每次 .05 ms，总采集时间 6 ms，重复 1 次。可以增加重复次数：

```powershell
& 'C:/Users/20291/miniconda3/envs/minsted/python.exe' examples/bayesian_minsted.py --exposures 240 --repeats 10 --no-plot
```

默认输出：

- `examples/bayesian_minsted_comparison.png`：第一组实验的位置轨迹、按采集时间和检测光子计数绘制的真实误差、选用宽度、贝叶斯曝光偏移/估计步长和最终后验。
- `examples/bayesian_minsted_comparison.csv`：所有重复实验的轨迹记录。

`--output-dir` 可更改保存位置；已有同名输出会由新运行替换。命令行支持设置先验、初始估计、真实位置、亮度、背景、网格及候选数，详情见 `--help`。

对照使用相同的固定真值、初始估计、前向模型、峰值亮度、背景和采集时间，但动作协议及检测光子数不同。原算法按真正的逐事件反馈执行；贝叶斯版本按离散曝光反馈执行。所有重复样本都保留，没有按真值筛掉失败运行；示例分别报告等效单轴 RMSE 和中位径向误差除以 `sqrt(2)`。默认重复噪声而固定真值/初始估计，不能据此推断所有初始条件下的性能。

`compute` 单独报告 CPU 计算时间，不能与 `acquisition` 采集时间混淆。当前 Python EIG 计算并不满足微秒反馈要求，也未建模位置切换和功率调节延迟。单次示例及少量重复不能证明贝叶斯版本优于原算法，更不能证明四倍光子效率提升。

### 已执行的默认示例（2026-10-16）

种子 `20261016`，默认参数，1 次重复、6 ms 采集。原算法检测 94 光子，最终径向误差 0.783 nm；贝叶斯版本检测 85 光子，误差 3.428 nm，后验等效单轴标准差 3.259 nm，其中有 58 次零光子曝光。贝叶斯选用了 100、63.246、40 nm 三种宽度，序列中存在放宽和收缩。

此运行贝叶斯计算时间约 75.25 s，原算法约 .013 s（计算时间随机器和参数变化）。它说明实现能够运行并作出不同的自适应动作，但这一次原算法的真实误差更小。当前没有通过大规模、多初值、多先验及网格/动作分辨率收敛研究证明性能改善。

## 验证

使用内置 `unittest`，无需额外安装 pytest：

```powershell
& 'C:/Users/20291/miniconda3/envs/minsted/python.exe' -m unittest discover -s tests -v
```

覆盖新旧前向模型与 FWHM 一致性、零/正计数更新、EIG 与显式后验熵的数值一致性、无位置相关信号时零信息、高计数稳定性、候选动作最大值选择、自适应网格与 401×401 固定参考网格的比较、非法动作及不可能观测的处理，以及重置状态。

参考思想：[Bayesian MINFLUX Localization Microscopy](https://doi.org/10.1021/acsphotonics.5c02739)。当前实现针对高斯 MINSTED 重新计算似然与 EIG，不使用原文甜甜圈激发模型或其最优偏移查表。
