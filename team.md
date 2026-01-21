# 机器学习课程设计项目报告（同README.md）

本报告面向 “Airbnb / AirROI 数据分析与建模” 课程设计，聚焦 Tokyo 与 London 两个城市的市场级月度动态，完成入住率、平均成交价与收入的 3 个月滚动预测。

---

## 1. 项目基本信息

- 小组编号：超级棒的队名
- 课题名称：Tokyo & London 民宿市场月度动态预测与收益管理分析
- 选定城市及范围：Tokyo、London（市场级聚合 + 可选子市场/层级尝试）
- 选定时间范围：2024-11-01–2025-10-01（测试集为最后 3 个月）
- 所用数据表：Listings、Calendar Rates（项目实现未使用 Reviews）
- 项目 Github 仓库地址：https://github.com/ZJUT-CS/ml-course-project-2025-team

### 1.1 小组成员与角色分工

| 成员   | 姓名 | 学号         | GitHub 用户名 | 主要负责内容（简要） |
|--------|------|--------------|---------------|----------------------|
| 学生 1 | 曾辉 | 302023562016 | fenxin666     | 辅助建模、实验复现、参数调优、报告撰写 |
| 学生 2 | 杨飞 | 302023562055 | YF-1212       | 模型对比、结果分析、PPT 制作 |
| 学生 3 | 徐铖 | 302023562017 | Serein37      | 工程实现、统一入口、输出归档 |
| 学生 4 | 张凡 | 302023562014 | nagatoyuki11  | 资料搜集、可视化优化、结果校对 |

---

## 2. 问题背景与数据集说明

### 2.1 业务背景与研究问题

- 业务背景：短租市场受季节性、节假日、旅游热度与供给变化影响，入住率与收入波动明显。平台/房东需要对未来一段时间的市场走势有“提前量”判断，从而制定动态定价、促销、维护与投放策略。
- 核心问题：本项目为时间序列预测（回归任务）。从 Calendar 表构建市场级月度序列，预测未来 3 个月的市场走势；同时尝试层级与时空策略以评估细粒度建模的收益与稳定性。
- 目标变量定义（均在月度聚合后计算）：
  - `occupancy_rate`：入住率，衡量需求热度与供需匹配。
  - `average_price`：平均成交价（有预订天数为分母），衡量成交价水平。
  - `revenue`：市场总收入（按 listing 聚合后求和），衡量市场现金流强弱。
- 价值与应用：将 3 个月预测曲线作为运营决策输入，用于旺季溢价/控量、淡季促销、投放节奏、库存与维护排期等。

### 2.2 数据来源与筛选规则

- 数据来源：AirROI Data Portal（https://www.airroi.com/data-portal/），本项目使用仓库 `data/` 下的导出 CSV。
- 使用的数据表及其作用：
  - Listings：提供房源静态属性与地理/类别字段（如 `room_type`、`neighbourhood*` 等），用于子市场过滤与层级/时空建模的分组键。
  - Calendar：提供每月（数据中以月初日期表示）每个 listing 的预订天数、可售天数、收入与价格相关统计，用于构建预测序列。
- 样本量概览：
  - Tokyo：Listings 约 301 行；Calendar 3555 行（300 个唯一 `listing_id`），日期覆盖 2024-11-01–2025-10-01。
  - London：Listings 约 301 行；Calendar 3463 行（300 个唯一 `listing_id`），日期覆盖 2024-11-01–2025-10-01。
- 筛选与清洗（工程实现口径）：
  - 以 `listing_id` 连接 Listings 与 Calendar（用于层级/时空策略的分组；市场级预测可直接使用 Calendar 的聚合字段）。
  - 日期字段转换为月末 `month_end` 作为时间索引，确保月度序列对齐。
  - 数值列强制转为数值类型，对不可解析项置为缺失并在构建序列时处理。
- 市场级月度聚合口径（核心公式）：
  - 月总可售天数：`total_days = sum(reserved_days) + sum(vacant_days)`
  - 入住率：`occupancy_rate = sum(reserved_days) / total_days`
  - 平均成交价：`average_price = sum(revenue) / sum(reserved_days)`（若当月 `reserved_days=0` 则为缺失）
  - 收入：`revenue = sum(revenue)`
- 核心字段与用途：
  - Calendar：`date`、`listing_id`、`reserved_days`、`vacant_days`、`occupancy`、`revenue`、`rate_avg`、`booked_rate_avg`、`booking_lead_time_avg`、`length_of_stay_avg`。
  - Listings：`room_type`、`neighbourhood/neighborhood*`（不同数据源字段名可能不同，工程实现会自动在候选列中选择可用字段）。

### 2.3 预期目标与分析思路

- 预期输出：
  - Tokyo 与 London 的三项目标变量在 “最后 3 个月测试窗口” 上的预测对比曲线（真实值 vs 预测值）。
  - 模型对比指标表，并给出策略选择与业务建议。
  - 可复现的运行产物：预测表格、指标表、图表（PNG）以及运行清单（manifest）。
- 技术路径：
  - 数据清洗与月度聚合 → 描述性统计与相关性分析 → 多模型对比（基线 + 传统时序 + 轻量 ML）→ 指标评估（MAE/RMSE 等）→ 输出图表与业务解读。

---

## 3. 方法与模型设计

### 3.1 数据预处理与特征工程

- 时间索引统一：将 Calendar 的 `date` 转换为 “月末日期” 索引（month end），形成长度为 12 的月度序列。
- 缺失与异常处理（与实现一致的原则）：
  - `occupancy_rate` 在预测后会裁剪到 [0, 1]，避免出现不可解释的负值或大于 1 的值。
  - `average_price` 在无预订月份会产生缺失；市场级月度预测会在训练前丢弃目标列缺失的月份（周/日频率下会通过插值生成更密序列，但本报告实验均为月度）。
- 特征工程（用于轻量 ML 与时空策略）：
  - Lag 特征：对目标序列构造固定长度滞后向量（例如 `lags=6`），用于 Ridge 回归做对照实验，并用于“滞后特征系数排序”
  - 时空（ST-GNN）策略：在节点级别构造邻接矩阵（支持 correlation/knn/threshold/geo），并用图传播后的 lag 特征进行预测；当 PyTorch 不可用时，回退到 “图 + lag + Ridge” 的近似实现。

关键实现片段（与代码一致，便于复现对照）：

- 月末索引与层级底层聚合（month_end + 指标构造）：源码见 [data_loader.py](src/core/data_loader.py#L858-L888)

```python
cal[date_col] = pd.to_datetime(cal[date_col], errors="coerce")
cal = cal.dropna(subset=[date_col])
cal["month_end"] = cal[date_col].dt.to_period("M").dt.to_timestamp("M")

group = merged.groupby(["month_end", level1_col, level2_col], as_index=False)
agg = group.agg(
    reserved_days=("reserved_days", "sum"),
    vacant_days=("vacant_days", "sum"),
    revenue=("revenue", "sum"),
)
agg["total_days"] = agg["reserved_days"] + agg["vacant_days"]
agg["occupancy_rate"] = np.where(
    agg["total_days"] > 0, agg["reserved_days"] / agg["total_days"], np.nan
)
agg["average_price"] = np.where(
    agg["reserved_days"] > 0, agg["revenue"] / agg["reserved_days"], np.nan
)
```

- 市场级月度序列的回退聚合（由 daily availability 推 occupancy_rate/average_price）：源码见 [data_loader.py](src/core/data_loader.py#L789-L802)

```python
df = cal.rename(columns={date_col: "date", avail_col: "available", price_col: "price"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")
avs = df["available"].astype(str).str.lower()
occ = np.where(
    avs.isin(["f", "false", "0"]), 1, np.where(avs.isin(["t", "true", "1"]), 0, np.nan)
)
df["is_occupied"] = occ
df = df.dropna(subset=["date", "price", "is_occupied"]).set_index("date")
monthly = df.resample("ME").agg({"is_occupied": "mean", "price": "mean"})
monthly = monthly.rename(columns={"is_occupied": "occupancy_rate", "price": "average_price"})
```

- 目标值裁剪（保证 occupancy_rate 合法）：源码见 [forecast.py](src/pipelines/forecast.py#L51-L67)

```python
def clip_target(target: str, series: pd.Series) -> pd.Series:
    if target == "occupancy_rate":
        return series.clip(lower=0, upper=1)
    return series
```

### 3.2 模型选择与设计

本项目在 **“数据点较少（月度序列）、、需要同时覆盖 market/hierarchical/spatial 三种策略”** 的约束下，选择了“基线 + 传统时序 + 轻量 ML + 结构化策略（层级/时空）”的组合：

- 可比性：所有模型输出统一对齐到同一测试窗口（最后 3 个月），并用同一套指标（MAE/RMSE/MAPE/sMAPE/MASE 等）评估；预测会严格按时间切分避免泄漏。
- 可训练性：当数据长度不足以支撑季节项/复杂结构时，模型会自动降级（关闭季节项或退化为朴素预测），保证每个城市/目标都能稳定完成训练与落盘。
- 可解释性：除传统时序模型外，加入滞后特征的 Ridge 作为轻量对照，用于解释“短期预测主要依赖最近几期”的现象（可通过系数排序观察贡献）。
- 可扩展性：层级（HTS）与时空（ST-GNN）策略与 market 共享同一数据加载与运行上下文（RunContext），输出路径、表格与图表统一归档，便于在 GitHub 上直接复现与对照。

- Seasonal Naive（季节性朴素基线）：用 “季节周期前的观测值” 作为预测；当训练长度不足以支撑季节周期时，退化为使用最近一期值作为预测。
- ETS（指数平滑）：用误差/趋势/季节的状态空间建模刻画平滑趋势；当训练长度不足以支撑季节组件时，自动关闭季节项以保证可训练，作为对 “平滑趋势 + 可能的季节性” 的稳健对照。
- SARIMA（季节性 ARIMA）：差分 + AR/MA + 季节项，适合在短序列下表达“惯性 + 季节性”结构；实现支持可选自动调参，但本次实验为默认参数与稳定性优先，并在需要时输出预测区间用于不确定性对照。
- Ridge（滞后特征 + Ridge 回归）：将最近若干期的滞后向量作为特征进行回归，提供可解释的系数权重；主要用于解释与对照，而非追求最优精度。
- HTS（层级时间序列 + MinT 协调）：以 `level1 × level2`（如行政区/社区 × 房型等）构建 bottom-level 序列；对每条底层序列先训练基础预测（默认 SARIMA），再用 MinT 做一致性协调，得到从底层到总量层级一致的预测结果（sum-consistent）。
- ST-GNN（时空图模型/回退版本）：以 `level1`（行政区等）作为节点构建空间图（支持 geo/correlation 等邻接构建）；将图结构与时序滞后特征联合用于预测。当节点很少、分组列缺失或深度学习依赖不可用时，回退为 “图 + lag + Ridge” 的近似实现，空间收益会显著下降。

关键实现片段（与代码一致，便于复现对照）：

- 统一预测引擎的策略分发（market/hierarchical/spatial）：源码见 [ForecastEngine.run](src/pipelines/forecast.py#L470-L496)

```python
def run(self) -> ForecastResult:
    strategy = self.config.strategy
    if strategy == ForecastStrategy.MARKET:
        return self._run_market()
    elif strategy == ForecastStrategy.HIERARCHICAL:
        return self._run_hierarchical()
    elif strategy == ForecastStrategy.SPATIAL:
        return self._run_spatial()
    else:
        raise ValueError(f"未知策略: {strategy}")
```

- Seasonal Naive 的“季节不足自动退化”（短序列下回退到 last value）：源码见 [SeasonalNaiveModel.predict](src/models/time_series_model.py#L398-L420)

```python
for i in range(int(steps)):
    idx = n - sp + i
    if sp > 0 and idx >= 0 and idx < n:
        preds.append(y[idx])
    else:
        preds.append(y[-1])
```

- 层级预测的 MinT 协调（sum-consistent）：源码见 [_mint_reconcile](src/pipelines/forecast.py#L1427-L1441)

```python
W_inv = np.diag(1.0 / W_diag)
middle = S.T @ W_inv @ S
middle_inv = np.linalg.pinv(middle)
P = S @ middle_inv @ S.T @ W_inv
return P @ yhat_base
```

- ST-GNN 的可用性检测与回退（无 torch 时自动降级到 legacy ridge）：源码见 [ForecastEngine._run_spatial](src/pipelines/forecast.py#L1107-L1140)

```python
try:
    from ..models.spatial_gnn import STGNNForecaster
    if not STGNNForecaster.is_available():
        raise ImportError("PyTorch 未安装")
    forecaster = STGNNForecaster(...)
    forecaster.fit(X_train, A_norm)
    pred_array = forecaster.predict(test_len)
except Exception as e:
    model_tag = "stgnn_legacy"
    pred_array = _run_spatial_legacy_ridge(
        X_train=X_train, A_norm=A_norm, test_len=test_len, lags=lookback, alpha=alpha
    )
```

- legacy 版本的“图传播 + 滞后特征”（A_norm @ X[t-k]）：源码见 [_make_lag_features](src/pipelines/forecast.py#L1551-L1565)

```python
for t in range(lags, T):
    feats = []
    for k in range(1, lags + 1):
        feats.append(A_norm @ X[t - k])
    rows.append(np.concatenate(feats, axis=0))
    ys.append(X[t])
```

### 3.3 超参数设置与训练细节

- 训练/测试划分：按时间切分，测试集固定为最后 3 个月，训练集为剩余历史月份；严格避免随机打乱以规避时间泄漏。
- 关键参数：
  - `test_months=3`（所有预测曲线统一为 3 个月窗口）。
  - `seasonal_period=12`（月度按年季节性；当训练长度不足时会自动调整为可行周期）。
  - SARIMA：默认 `order=(1,1,1)`、`seasonal_order=(1,1,1,12)`；未启用自动调参。
  - ETS：默认 `trend=add`、`seasonal=add`；若 `len(series) < 2*seasonal_period` 则禁用季节项。
  - LagRidge（对照分析）：默认 `lags=6`、`alpha=1.0`。
  - ST-GNN：回看窗口 `lookback=3`，图构建方法 `correlation`；若 PyTorch 不可用，使用 legacy ridge 近似实现。
- 输出与可复现性：
  - 每次运行使用 `output-layout run` 将结果归档到 `results/outputs/<run_id>/`，并写入 `run_manifest.json` 记录本次运行的模块输出情况。

---

## 4. 实验设计与结果分析

### 4.1 数据集划分与评估指标

- 划分方式：Time-based Split（最后 3 个月为测试集）。
  - 训练集：2024-11 至 2025-07（共 9 个点）
  - 测试集：2025-08 至 2025-10（共 3 个点）
- 评价指标：
  - MAE、RMSE、MSE：衡量预测误差幅度。
  - MAPE、sMAPE：相对误差（对量纲不同的目标更便于对比）。
  - MASE：相对朴素模型的缩放误差（越小越好，<1 表示优于朴素基准）。
  - Coverage / Interval Width：当输出预测区间时，用覆盖率与区间宽度辅助判断不确定性估计质量。

### 4.2 结果展示

 “市场级预测（market）、层级预测（hierarchical）、时空预测（spatial）” 的模型对比表，以及相应策略的误差对照。

#### 4.2.1 Tokyo：`occupancy_rate`

市场级模型对比（测试集 3 个月）：

| model | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 0.2148 | 0.2198 | 55.0036 | 42.7990 | 3.0855 |
| ets | 0.0308 | 0.0315 | 7.9382 | 7.7273 | 0.4430 |
| sarima | 0.0277 | 0.0309 | 7.2200 | 6.9063 | 0.3980 |

策略对照：

| strategy | model | MAE | RMSE |
|---|---|---:|---:|
| market | sarima（最优） | 0.0277 | 0.0309 |
| hierarchical | mint_reconciled | 0.8525 | 0.9050 |
| spatial | stgnn_legacy | 0.2022 | 0.2069 |

关键图表与文件：

- market（market_forecast）：
  ![Tokyo occupancy_rate market forecast](results/outputs/20260111_014451_b1b37e/market_forecast/figures/forecast_occupancy_rate.png)
  图注：market（SARIMA）在测试窗口内能较好跟随真实值变化，误差较小。
  - 指标表：[metrics.csv](results/outputs/20260111_014451_b1b37e/market_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260111_014451_b1b37e/market_forecast/tables/forecast.csv)
  - 区间明细：[forecast_with_intervals.csv](results/outputs/20260111_014451_b1b37e/market_forecast/tables/forecast_with_intervals.csv)
- hierarchical（hierarchical_forecast）：
  ![Tokyo occupancy_rate hierarchical forecast](results/outputs/20260111_014451_b1b37e/hierarchical_forecast/figures/forecast_occupancy_rate.png)
  图注：hierarchical（MinT）在小样本分组下偏离较大，预测稳定性不足。
  - 指标表：[metrics.csv](results/outputs/20260111_014451_b1b37e/hierarchical_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260111_014451_b1b37e/hierarchical_forecast/tables/forecast.csv)
- spatial（spatial_forecast）：
  ![Tokyo occupancy_rate spatial forecast](results/outputs/20260111_014451_b1b37e/spatial_forecast/figures/forecast_occupancy_rate.png)
  图注：spatial（ST-GNN legacy）能捕捉方向但偏差仍明显，表现介于两者之间。
  - 指标表：[metrics.csv](results/outputs/20260111_014451_b1b37e/spatial_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260111_014451_b1b37e/spatial_forecast/tables/forecast.csv)

#### 4.2.2 Tokyo：`average_price`

市场级模型对比（测试集 3 个月）：

| model | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 4.0168 | 4.3386 | 2.9613 | 2.9547 | 0.4428 |
| ets | 4.9814 | 5.3864 | 3.5905 | 3.6005 | 0.5491 |
| sarima | 5.5515 | 6.0685 | 4.0360 | 4.0095 | 0.6120 |

策略对照：

| strategy | model | MAE | RMSE |
|---|---|---:|---:|
| market | seasonal_naive（最优） | 4.0168 | 4.3386 |
| hierarchical | mint_reconciled | 599.7470 | 765.9348 |
| spatial | stgnn_legacy | 4.9105 | 6.7821 |

关键图表与文件：

- market（market_forecast）：
  ![Tokyo average_price market forecast](results/outputs/20260116_222347_59229d/market_forecast/figures/forecast_average_price.png)
  图注：market（Seasonal Naive）在测试窗口内较稳定，能贴近真实水平。
  - 指标表：[metrics.csv](results/outputs/20260116_222347_59229d/market_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222347_59229d/market_forecast/tables/forecast.csv)
  - 区间明细：[forecast_with_intervals.csv](results/outputs/20260116_222347_59229d/market_forecast/tables/forecast_with_intervals.csv)
- hierarchical（hierarchical_forecast）：
  ![Tokyo average_price hierarchical forecast](results/outputs/20260116_222347_59229d/hierarchical_forecast/figures/forecast_average_price.png)
  图注：hierarchical 在价格目标上误差被显著放大，反映底层样本稀疏导致的不稳定。
  - 指标表：[metrics.csv](results/outputs/20260116_222347_59229d/hierarchical_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222347_59229d/hierarchical_forecast/tables/forecast.csv)
- spatial（spatial_forecast）：
  ![Tokyo average_price spatial forecast](results/outputs/20260116_222347_59229d/spatial_forecast/figures/forecast_average_price.png)
  图注：spatial 与真实同量级但存在偏差，未体现出明显优势。
  - 指标表：[metrics.csv](results/outputs/20260116_222347_59229d/spatial_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222347_59229d/spatial_forecast/tables/forecast.csv)

#### 4.2.3 Tokyo：`revenue`

市场级模型对比（测试集 3 个月）：

| model | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 264465.0 | 285471.1 | 55.5686 | 42.1227 | 2.0369 |
| ets | 54087.7 | 54098.2 | 11.4219 | 11.1365 | 0.4166 |
| sarima | 51530.1 | 62088.6 | 11.4209 | 10.5399 | 0.3969 |

策略对照：

| strategy | model | MAE | RMSE |
|---|---|---:|---:|
| market | sarima（最优） | 51530.1 | 62088.6 |
| hierarchical | mint_reconciled | 162522800.0 | 261651100.0 |
| spatial | stgnn_legacy | 300261.9 | 309432.2 |

关键图表与文件：

- market（market_forecast）：
  ![Tokyo revenue market forecast](results/outputs/20260116_222413_0e8ced/market_forecast/figures/forecast_revenue.png)
  图注：market（SARIMA）能较好捕捉收入变化的方向与幅度。
  - 指标表：[metrics.csv](results/outputs/20260116_222413_0e8ced/market_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222413_0e8ced/market_forecast/tables/forecast.csv)
  - 区间明细：[forecast_with_intervals.csv](results/outputs/20260116_222413_0e8ced/market_forecast/tables/forecast_with_intervals.csv)
- hierarchical（hierarchical_forecast）：
  ![Tokyo revenue hierarchical forecast](results/outputs/20260116_222413_0e8ced/hierarchical_forecast/figures/forecast_revenue.png)
  图注：hierarchical 在收入目标上出现数量级偏差，体现层级拆分下训练不稳定被放大。
  - 指标表：[metrics.csv](results/outputs/20260116_222413_0e8ced/hierarchical_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222413_0e8ced/hierarchical_forecast/tables/forecast.csv)
- spatial（spatial_forecast）：
  ![Tokyo revenue spatial forecast](results/outputs/20260116_222413_0e8ced/spatial_forecast/figures/forecast_revenue.png)
  图注：spatial 能跟随近期趋势但整体误差较大，表现弱于 market。
  - 指标表：[metrics.csv](results/outputs/20260116_222413_0e8ced/spatial_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222413_0e8ced/spatial_forecast/tables/forecast.csv)

#### 4.2.4 London：`occupancy_rate`

市场级模型对比（测试集 3 个月）：

| model | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 0.0973 | 0.1052 | 29.1952 | 29.7049 | 0.9573 |
| ets | 0.2112 | 0.2217 | 67.2371 | 48.1482 | 2.0782 |
| sarima | 0.1826 | 0.1908 | 57.9822 | 43.2583 | 1.7968 |

策略对照：

| strategy | model | MAE | RMSE |
|---|---|---:|---:|
| market | seasonal_naive（最优） | 0.0973 | 0.1052 |
| hierarchical | mint_reconciled | 0.2264 | 0.2428 |
| spatial | stgnn_legacy | 0.1438 | 0.1482 |

关键图表与文件：

- market（market_forecast）：
  ![London occupancy_rate market forecast](results/outputs/20260116_222436_c4a85f/market_forecast/figures/forecast_occupancy_rate.png)
  图注：market（Seasonal Naive）在测试窗口内较稳健，体现“近期形态延续”的特征。
  - 指标表：[metrics.csv](results/outputs/20260116_222436_c4a85f/market_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222436_c4a85f/market_forecast/tables/forecast.csv)
  - 区间明细：[forecast_with_intervals.csv](results/outputs/20260116_222436_c4a85f/market_forecast/tables/forecast_with_intervals.csv)
- hierarchical（hierarchical_forecast）：
  ![London occupancy_rate hierarchical forecast](results/outputs/20260116_222436_c4a85f/hierarchical_forecast/figures/forecast_occupancy_rate.png)
  图注：hierarchical 偏差更大，说明分组后样本不足导致稳定性下降。
  - 指标表：[metrics.csv](results/outputs/20260116_222436_c4a85f/hierarchical_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222436_c4a85f/hierarchical_forecast/tables/forecast.csv)
- spatial（spatial_forecast）：
  ![London occupancy_rate spatial forecast](results/outputs/20260116_222436_c4a85f/spatial_forecast/figures/forecast_occupancy_rate.png)
  图注：spatial 相比 market 无明显提升，预测误差仍较明显。
  - 指标表：[metrics.csv](results/outputs/20260116_222436_c4a85f/spatial_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222436_c4a85f/spatial_forecast/tables/forecast.csv)

#### 4.2.5 London：`average_price`

市场级模型对比（测试集 3 个月）：

| model | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 10.2835 | 14.1288 | 6.0209 | 6.3741 | 0.7405 |
| ets | 17.5576 | 18.8991 | 10.8299 | 10.1815 | 1.2642 |
| sarima | 15.3398 | 17.0697 | 9.4982 | 8.9641 | 1.1045 |

策略对照：

| strategy | model | MAE | RMSE |
|---|---|---:|---:|
| market | seasonal_naive（最优） | 10.2835 | 14.1288 |
| hierarchical | mint_reconciled | 85.1000 | 96.6963 |
| spatial | stgnn_legacy | 3.0454 | 3.5638 |

关键图表与文件：

- market（market_forecast）：
  ![London average_price market forecast](results/outputs/20260116_222458_0d3207/market_forecast/figures/forecast_average_price.png)
  图注：market（Seasonal Naive）偏差较大，难以反映短期价格波动。
  - 指标表：[metrics.csv](results/outputs/20260116_222458_0d3207/market_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222458_0d3207/market_forecast/tables/forecast.csv)
  - 区间明细：[forecast_with_intervals.csv](results/outputs/20260116_222458_0d3207/market_forecast/tables/forecast_with_intervals.csv)
- hierarchical（hierarchical_forecast）：
  ![London average_price hierarchical forecast](results/outputs/20260116_222458_0d3207/hierarchical_forecast/figures/forecast_average_price.png)
  图注：hierarchical 整体偏离真实水平，显示层级拆分在该目标上收益有限。
  - 指标表：[metrics.csv](results/outputs/20260116_222458_0d3207/hierarchical_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222458_0d3207/hierarchical_forecast/tables/forecast.csv)
- spatial（spatial_forecast）：
  ![London average_price spatial forecast](results/outputs/20260116_222458_0d3207/spatial_forecast/figures/forecast_average_price.png)
  图注：spatial（ST-GNN legacy）更贴近真实，体现一定的区域联动信息。
  - 指标表：[metrics.csv](results/outputs/20260116_222458_0d3207/spatial_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222458_0d3207/spatial_forecast/tables/forecast.csv)

#### 4.2.6 London：`revenue`

市场级模型对比（测试集 3 个月）：

| model | MAE | RMSE | MAPE | sMAPE | MASE |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 161715.0 | 189322.0 | 32.8476 | 34.8884 | 1.0066 |
| ets | 389912.3 | 405863.0 | 88.2773 | 58.5345 | 2.4270 |
| sarima | 343143.4 | 356079.1 | 77.5137 | 53.6415 | 2.1359 |

策略对照：

| strategy | model | MAE | RMSE |
|---|---|---:|---:|
| market | seasonal_naive（最优） | 161715.0 | 189322.0 |
| hierarchical | mint_reconciled | 548247.8 | 576826.8 |
| spatial | stgnn_legacy | 391085.4 | 407136.4 |

关键图表与文件：

- market（market_forecast）：
  ![London revenue market forecast](results/outputs/20260116_222520_3896ca/market_forecast/figures/forecast_revenue.png)
  图注：market（Seasonal Naive）在测试窗口内更稳健，能延续近期收入水平。
  - 指标表：[metrics.csv](results/outputs/20260116_222520_3896ca/market_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222520_3896ca/market_forecast/tables/forecast.csv)
  - 区间明细：[forecast_with_intervals.csv](results/outputs/20260116_222520_3896ca/market_forecast/tables/forecast_with_intervals.csv)
- hierarchical（hierarchical_forecast）：
  ![London revenue hierarchical forecast](results/outputs/20260116_222520_3896ca/hierarchical_forecast/figures/forecast_revenue.png)
  图注：hierarchical 误差更大，外推不稳定，体现样本不足的限制。
  - 指标表：[metrics.csv](results/outputs/20260116_222520_3896ca/hierarchical_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222520_3896ca/hierarchical_forecast/tables/forecast.csv)
- spatial（spatial_forecast）：
  ![London revenue spatial forecast](results/outputs/20260116_222520_3896ca/spatial_forecast/figures/forecast_revenue.png)
  图注：spatial 在收入目标上未体现优势，预测偏差明显。
  - 指标表：[metrics.csv](results/outputs/20260116_222520_3896ca/spatial_forecast/tables/metrics.csv)
  - 预测明细：[forecast.csv](results/outputs/20260116_222520_3896ca/spatial_forecast/tables/forecast.csv)

#### 4.2.7 滞后特征重要性（Ridge 系数排序，lags=6）

为满足“重要特征排序/可解释性”的展示要求，并进一步解释 “短期预测主要由最近若干期驱动”，我们在训练集（9 个点）上构造 `lags=6` 的滞后向量，训练 Ridge 回归并按系数绝对值排序（数值为系数，正负表示方向）。

- Tokyo：

| target | top lags by \|coef\|（从高到低） |
|---|---|
| occupancy_rate | lag_1=0.010553, lag_3=-0.00847203, lag_6=0.00649954, lag_4=-0.00514644, lag_5=0.00418318, lag_2=0.00088672 |
| average_price | lag_5=0.197046, lag_4=-0.126415, lag_3=-0.0848449, lag_2=0.0590616, lag_1=0.0309975, lag_6=-0.00454513 |
| revenue | lag_3=-0.371495, lag_5=0.253575, lag_1=0.246668, lag_6=-0.0603365, lag_4=0.0296728, lag_2=-0.0211126 |

- London：

| target | top lags by \|coef\|（从高到低） |
|---|---|
| occupancy_rate | lag_5=-0.0151737, lag_3=0.00821407, lag_1=0.0065422, lag_2=0.0053621, lag_6=-0.00519072, lag_4=0.000150205 |
| average_price | lag_4=-0.278809, lag_6=0.132764, lag_2=0.0587793, lag_3=-0.0415987, lag_5=0.0323498, lag_1=0.0279802 |
| revenue | lag_5=-0.344394, lag_3=0.131063, lag_4=-0.1269, lag_2=0.115299, lag_1=0.0976727, lag_6=0.021049 |

### 4.3 结果分析与业务解读

#### 4.3.1 数据现象

- Tokyo（月度 12 点，范围 2024-11-30–2025-10-31）：
  - `occupancy_rate`：均值 0.5242，标准差 0.0991，最小 0.3745，最大 0.6652。
  - `average_price`：均值 137.39，标准差 6.32，最小 127.65，最大 146.25。
  - `revenue`：均值 647003.4，标准差 140810.1，最小 440174，最大 852455。
  - 相关性（Pearson）：`occupancy_rate` 与 `revenue` 高相关（≈0.947），说明市场收入主要由需求强弱驱动；`average_price` 与 `occupancy_rate` 相关较弱（≈0.161），价格对需求的即时联动在该时间窗口不明显。
- London（月度 12 点，范围 2024-11-30–2025-10-31）：
  - `occupancy_rate`：均值 0.3738，标准差 0.0834，最小 0.2606，最大 0.5219。
  - `average_price`：均值 174.01，标准差 17.06，最小 150.79，最大 203.32。
  - `revenue`：均值 577842.8，标准差 167451.4，最小 327998，最大 830787。
  - 相关性（Pearson）：`occupancy_rate` 与 `revenue` 高相关（≈0.939），`average_price` 与 `revenue` 也高度相关（≈0.819）；同时 `occupancy_rate` 与 `average_price` 呈中等相关（≈0.601），说明 London 的短期收入波动同时受到“需求强弱 + 成交价水平”共同驱动。

#### 4.3.2 模型表现差异的原因解释

- 小样本约束：每个城市的月度序列长度为 12，其中训练仅 9 点，模型复杂度稍高（含趋势/季节）时容易出现不稳定或退化，导致部分指标（如 R²）在 3 点测试集上出现极端值。
- 时空/层级策略的收益取决于分组粒度与数据充分性：
  - 层级预测需要足够多的底层序列与足够长的训练期；在本数据快照下底层序列非常短，导致整体误差偏大。
  - 时空策略在节点极少或缺少有效空间结构时会退化为近似的 lag 回归；在 London 的 `average_price` 上表现较好，说明该目标在短期内具有更强的可线性预测性。
- 按 “策略 × 目标变量” 拆解三种策略的预测效果（均为测试集 3 个月的 MAE/RMSE，对应 4.2 节的策略对照表）：
  - `occupancy_rate`：
    - Tokyo：market（SARIMA，MAE=0.0277）显著优于 spatial（MAE=0.2022）与 hierarchical（MAE=0.8525），说明在总量级别需求序列上，“传统时序 + 季节项/差分” 比分组后的结构化方法更稳健。
    - London：market（Seasonal Naive，MAE=0.0973）优于 spatial（MAE=0.1438）与 hierarchical（MAE=0.2264），更符合“测试期与训练末期相近、最近观测足够好”的特征。
  - `average_price`：
    - Tokyo：market（Seasonal Naive，MAE=4.0168）略优于 spatial（MAE=4.9105），hierarchical（MAE=599.7470）远差，体现出底层分组后样本稀疏/缺失会显著放大价格误差。
    - London：spatial（ST-GNN legacy，MAE=3.0454）优于 market（Seasonal Naive，MAE=10.2835）与 hierarchical（MAE=85.1000），说明该城市在短期价格上存在更强的“区域联动/共振”，简单的市场级朴素模型难以捕捉。
  - `revenue`：
    - Tokyo：market（SARIMA，MAE=51530.1）最佳，spatial（MAE=300261.9）次之，hierarchical（MAE=162522800.0）最差；在 “总量收入强相关于总量入住” 的设定下，层级拆分会把短序列进一步切碎，导致不稳定被放大。
    - London：market（Seasonal Naive，MAE=161715.0）仍最优，spatial（MAE=391085.4）与 hierarchical（MAE=548247.8）误差更大，说明该快照下收入更像“近期水平的延续”，结构化策略在小样本下难以体现收益。
- 模型是否出现过拟合/欠拟合？如何判断：
  - 判断口径（适用于本项目的小样本时序场景）：对比 “训练期拟合误差 vs 测试期误差” 的差距、观察预测曲线在测试窗口的偏差形态（系统性高估/低估还是随机波动）、以及复杂模型是否出现不稳定迹象（例如参数退化、预测区间异常宽、对末端点过度贴合但外推偏离）。
  - 本实验观察：London 上 ETS/SARIMA 在测试窗口明显劣于朴素基线，更像是 “复杂度相对样本不足带来的高方差/过拟合或结构错配”；而 Tokyo 的 `occupancy_rate`/`revenue` 上 SARIMA/ETS 在测试集误差更低，未体现出明显的过拟合迹象。层级策略在若干目标上出现极端误差，主要来自 bottom-level 序列过短与缺失导致的训练不稳定，这类现象更接近 “有效样本不足引发的方差爆炸”，而非传统意义的“欠拟合”。

#### 4.3.3 面向业务的建议

- 将模型结果和业务问题联系起来（基于本项目三类目标 `occupancy_rate`/`average_price`/`revenue` 的预测与相关性分析）：
  - 典型画像（结合 Listings 的 `room_type`、`neighbourhood*` 等字段做落地分组；价格区间使用当城历史 `average_price` 的分位数划分）：
    - 高收益型：`revenue` 高且由 `occupancy_rate` 驱动（两城 `occupancy_rate` 与 `revenue` 均高相关），通常表现为较高入住率与稳定的预订天数；适合做旺季溢价与库存管理。
    - 溢价型：`average_price` 高、但 `occupancy_rate` 不一定最高；在 London 中价格与收入相关更强，说明“高价仍能成交”的子市场更容易转化为收入优势，关键在于位置与供需环境。
    - 均衡型：价格与入住率都处于中位，波动相对小；适合作为“稳态现金流”组合，重点优化转化与评价，避免无效降价。
  - 可操作建议（按城市与目标拆解）：
    - Tokyo（以 market 策略为主）：在 `occupancy_rate` 与 `revenue` 上 market（SARIMA/ETS）显著优于层级/时空策略，建议以市场级预测作为主线：当未来 1–3 个月预测入住率上行时，分阶段上调价格与最短入住天数并减少折扣；当预测下行时，优先采用限时折扣、长住优惠与可取消政策改善转化，而不是全量大幅降价。
    - London（“需求延续 + 分区定价”结合）：`occupancy_rate` 与 `revenue` 上朴素基线在测试窗口更稳健，可用于快速部署与保底决策；同时 `average_price` 上 spatial 明显更优，建议在定价侧引入分区信号：以 `neighbourhood*` 为粒度，针对“预测价格上行/下行”的区域差异化调整价格与投放，避免用单一市场均价覆盖所有区域。
    - 运营监控与复盘：将每次运行输出的 `anomaly_months.csv` 作为重点复盘清单，优先核对异常月份的数据质量与外部事件（节假日/大型活动），并把这些月份作为后续引入外生变量或做情景预测（保守/基准/乐观）的切入点。

---

## 5. 结论与不足

### 5.1 主要结论

- 在 Tokyo 的 `occupancy_rate` 与 `revenue` 任务上，SARIMA/ETS 这类传统时序模型在 3 个月测试窗口内明显优于朴素基线，适合作为主力预测模型。
- 在 London 的本数据快照上，市场级朴素基线对三项目标的短期预测表现较稳健，说明 “最近一期/类季节性” 模式足以覆盖测试窗口，是更高性价比的部署方案。
- `occupancy_rate` 与 `revenue` 在两城均表现出强相关，市场需求强弱是收入波动的主导因素；London 的价格与收入相关性更强，价格策略对收入弹性更明显。

### 5.2 不足与改进方向

- 数据层面不足：
  - 月度样本点仅 12 个，训练点 9 个，难以稳定支撑复杂模型与严格回测；建议扩展更长时间跨度或使用更高频数据（周/日）并引入外生特征。
  - Reviews 未纳入：缺少口碑/评分变化对需求与价格的解释变量，限制了策略建议的可解释性。
- 方法层面不足：
  - 当前主要为单变量预测；可扩展为多变量（SARIMAX/VAR）并显式引入节假日、天气、汇率、重大活动等外生变量。
  - 层级/时空策略在本数据快照上受限于分组粒度与样本长度；后续应增强节点定义（更细的区域划分）并保证每个节点足够历史长度。
- 工程层面改进：
  - 增加更系统的 walk-forward 回测（多折）、指标置信区间与模型稳定性诊断，避免 3 点测试导致的指标波动被误读为显著差异。

---

## 6. 小组协作与个人收获

### 6.1 分工与协作情况

- 分工方式：按 “数据处理/建模/工程/写作展示” 四个角色拆分任务，统一在仓库内以模块化代码组织实现。
- 协作流程：使用 Git 进行管理，通过分支开发与合并保持主干稳定；
- 可复现保障：统一 CLI 入口将参数（城市、目标、策略、测试窗口）显式化，保证不同成员可在同一命令下复现实验与图表。

### 6.2 个人收获

- 学生 1（曾辉）：掌握了从原始日历数据构建市场级月度序列的工程流程，理解了时间序列场景下按时间切分避免信息泄漏的重要性。对比了朴素基线、ETS、SARIMA 与轻量 ML 方法，能够根据小样本条件解释模型稳定性差异。也进一步理解了预测结果如何映射到收益管理的运营决策。
- 学生 2（杨飞）：系统整理了不同模型在三类目标变量上的误差表现，熟悉了 MAE/RMSE/MAPE/MASE 等指标在小样本条件下的解读方式。通过对 Tokyo 与 London 的对比，能够从业务角度解释“需求驱动收入”与“价格弹性差异”。完成了可视化与展示材料的结构化输出。
- 学生 3（徐铖）：完成统一 CLI 入口与输出归档设计，将 market/hierarchical/spatial 三种策略整合到同一预测引擎，保障实验参数可控且结果可追溯。对运行上下文、输出路径管理与结果表格/图表落盘机制有了系统理解。通过多次复现实验提升了对工程稳定性与可维护性的认识。
- 学生 4（张凡）：补充了相关方法资料调研与业务背景梳理，协助完善图表展示与报告表达，使结论更贴近收益管理场景。对比不同城市的数据特征与相关性结构，形成了更具可操作性的建议。参与结果校对与异常情况排查，增强了对数据质量与指标解释的敏感度。

---

## 7. 附录
###  目录结构

```
ml-course-project-2025-team/
├── cli.py                 # 统一 CLI 入口
├── config.default.yaml    # 默认配置文件
├── requirements/          # 依赖列表（base.txt, full.txt）
│
├── src/                   # 核心源码
│   ├── core/              # 核心模块
│   │   ├── interfaces.py  # 数据契约与接口定义
│   │   ├── data_loader.py # 数据加载与预处理
│   │   ├── metrics.py     # 评估指标计算
│   │   ├── anomaly.py     # 🆕 异常检测模块
│   │   └── config.py      # 配置管理
│   │
│   ├── models/            # 模型实现
│   │   ├── time_series_model.py  # 核心时序模型
│   │   ├── extended.py    # 扩展模型 (Prophet/LightGBM)
│   │   ├── deep_learning.py  # 🆕 深度学习模型
│   │   └── registry.py    # 模型注册表
│   │
│   ├── pipelines/         # 预测流水线
│   │   ├── forecast.py    # 统一预测引擎
│   │   └── automl.py      # 🆕 AutoML 模块
│   │
│   ├── viz/               # 可视化模块
│   │   └── plotly_charts.py    # 图表生成
│
├── web/                   # Web 应用 (模块化)
│   ├── app.py             # Streamlit 主入口
│   ├── config.py          # 侧边栏配置
│   └── tabs/              # 功能标签页模块
│
├── data/                  # 数据目录（包含 outputs/ 输出目录）
├── tests/                 # 测试用例
├── reports/                  # 文档
└── notebooks/             # Jupyter Notebooks
```
- 主要代码结构说明：
  - `cli.py`：统一 CLI 入口（forecast/batch/web/legacy）。
  - `src/pipelines/forecast.py`：统一预测引擎，整合 market/hierarchical/spatial 策略与指标计算。
  - `src/core/data_loader.py`：数据加载与月度/层级序列构建（market 与 bottom-level）。
  - `src/models/time_series_model.py`：SeasonalNaive/ETS/SARIMA/LagRidge 等模型封装与统一接口。
  - `web/app.py`：Streamlit 可视化界面入口。
###  系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Airbnb 市场分析                    │
├─────────────────────────────────────────────────────────────────┤
│  Presentation Layer (Streamlit)                                 │
│  ┌─────────┬──────────┬───────────┬──────────┬────────────┐    │
│  │  EDA    │ Forecast │ Diagnose  │ AutoML   │  Spatial   │    │
│  └─────────┴──────────┴───────────┴──────────┴────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Pipeline Layer (ForecastEngine)                                │
│  ┌──────────────┬─────────────────┬──────────────────────┐     │
│  │    MARKET    │  HIERARCHICAL   │      SPATIAL         │     │
│  │  单序列预测   │  层级协调预测    │   时空图网络预测     │     │
│  └──────────────┴─────────────────┴──────────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  Model Layer (统一 BaseForecaster 接口)                         │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Statistical │ Machine Learning │    Deep Learning      │    │
│  │ ────────── │ ─────────────── │ ──────────────────────│    │
│  │ • Naive    │ • Ridge          │ • LSTM                │    │
│  │ • ETS      │ • LightGBM       │ • Transformer         │    │
│  │ • SARIMA   │ • XGBoost        │ • ST-GNN              │    │
│  │ • Prophet  │                  │                       │    │
│  └────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ DataLoader → 时间序列聚合 → 特征工程 → 数据分片          │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```
```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Streamlit Web App  │  CLI 命令行  │  Jupyter Notebook  ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      业务逻辑层                              │
│  ┌──────────────┬──────────────┬──────────────────────────┐│
│  │ 预测引擎     │ AutoML      │ 异常检测                  ││
│  │ ForecastEngine│ AutoMLEngine │ AnomalyDetector          ││
│  └──────────────┴──────────────┴──────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                      模型层                                  │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────┐│
│  │ SARIMA   │ ETS      │ Prophet  │ LightGBM │ LSTM      ││
│  │ 统计模型 │ 指数平滑 │ FB 预测  │ 梯度提升 │ 深度学习  ││
│  └──────────┴──────────┴──────────┴──────────┴───────────┘│
├─────────────────────────────────────────────────────────────┤
│                      数据层                                  │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  DataLoader  │  数据清洗  │  特征工程  │  缓存管理      ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```
- 打开前端界面:
```powershell
& $py cli.py web
# 或
& $py -m streamlit run web/app.py --server.port 8501
```
**问题**: 运行时提示模块未找到

**解决**:
```powershell
# 重新安装依赖
& $py -m pip install -r requirements/full.txt

# 使用 venv 解释器运行
& $py cli.py web
```
### 最后，详细项目说明可以参考：\reports\final\项目说明文档.md
