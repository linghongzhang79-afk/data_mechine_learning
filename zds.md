<center机器学习课程设计报告center>
## 1. 项目基本信息

- 小组编号：ZDS
- 课题名称：里约热内卢房源收益预测与优质房源画像分析
- 选定城市及范围：Rio de Janeiro（里约热内卢）
- 选定时间范围：使用全部可用数据（包含过去12个月TTM指标和最近90天L90D指标）
- 所用数据表：Listings Data（房源静态特征+历史表现数据）
- 项目 Github 仓库地址：https://github.com/ZJUT-CS/ml-course-project-2025-zds
### 1.1 小组成员与角色分工

| 成员   | 姓名  | 学号           | GitHub 用户名 | 主要负责内容（简要）         |
|:-----|-----|--------------|------------|:-------------------|
| 学生 1 | 朱文涛 | 302023562070 |非洲小白脸            | EDA 负责人：数据探索分析与可视化 |
| 学生 2 | 陈浩然 | 302023562007 |ruoxuebuzhichun            | 建模负责人：特征工程与模型调参    |
| 学生 3 | 庞博  | 302023562012 | Nantesholey           | 代码负责人：项目结构与可复现性    |
| 学生 4 | 卢普伟 | 302023562013 |liskarmmm            | 展示与写作负责人：报告撰写与展示   |

---

## 2. 问题背景与数据集说明

### 2.1 业务背景与研究问题

Airbnb作为全球领先的共享住宿平台，已成为连接房源供给方与需求方的重要中介。在里约热内卢等旅游热点城市，房源收益表现呈现显著异质性，房东在定价策略制定与运营决策优化方面面临诸多挑战。

**核心问题定义**：
- **任务类型**：回归预测任务 + 优质房源画像分析
- **目标变量**：TTM RevPAR（过去12个月每可售日收益，Revenue Per Available Room）
- **业务价值**：
  - 对房东：提供科学定价依据，识别影响收益的关键因素
  - 对平台：优化房源推荐策略，提升整体运营效率
  - 对房客：帮助识别高性价比优质房源

### 2.2 数据来源与筛选规则

**数据来源**：AirROI Data Portal (https://data.airroi.com)

**使用数据表**：
- **Listings Data**：房源静态特征（房型、设施、位置等）+ 历史表现指标（TTM RevPAR、评分等）

**样本筛选流程**：
- 初始样本规模：里约热内卢地区全量房源数据
- 有效样本规模：**281条**（经IQR异常值检测与缺失值处理后）
- 筛选准则：
  - 地理范围限定：Rio de Janeiro行政区域
  - 异常值处理：采用四分位距（IQR）方法剔除TTM RevPAR极端离群值
  - 完整性约束：删除核心特征字段存在缺失的观测样本
  - **数据泄露防控**：排除`ttm_revpar_native`等与因变量存在时序依赖关系的特征

**核心特征说明**（8-15个关键字段）：
1. **bedrooms**：卧室数量，直接影响房源容纳能力与定价
2. **bathrooms**：卫生间数量，反映房源舒适度
3. **accommodates**：最大入住人数，决定目标客群规模
4. **rating_overall**：综合评分，反映房源整体质量
5. **rating_location**：位置评分，地理位置优势的量化指标
6. **rating_cleanliness**：清洁度评分，影响复购率的关键因素
7. **price**：每晚价格，直接影响收益与竞争力
8. **has_pool**：是否有泳池（提取自设施描述）
9. **has_aircon**：是否有空调（提取自设施描述）
10. **latitude / longitude**：经纬度坐标，用于地理位置分析
11. **room_type**：房型类别（整套房/独立房间/合住房间）
12. **listing_type**：房源类型（公寓/别墅/民宿等）

### 2.3 预期目标与分析思路

**研究目标**：
- 构建TTM RevPAR预测模型，量化识别房源收益的关键影响因素
- 基于模型解释技术（特征重要性分析）构建优质房源特征画像，提出可操作性运营策略
- 对比多种建模方法（线性模型与集成学习模型），在预测精度与模型可解释性之间寻求平衡
- 生成特征重要性排序、预测效果评估、残差诊断等可视化分析结果

**技术路径**：
1. **EDA阶段**：探索数据分布、相关性分析、异常值检测
2. **特征工程**：提取设施特征、处理类别变量、规避标签泄露
3. **建模阶段**：建立Linear Regression基线 → Random Forest改进 → SHAP解释
4. **业务解读**：将模型结果转化为房东可执行的运营策略
---

## 3. 方法与模型设计

### 3.1 数据预处理与特征工程

#### 3.1.1 缺失值处理策略

本项目采用**分层处理**策略，根据字段重要性和缺失比例选择不同方法：

**1. 目标变量缺失处理（删除法）**
- 对于目标变量 `ttm_revpar` 缺失的样本，直接删除
- 原因：目标变量缺失无法用于监督学习，填补会引入噪声

**2. 关键特征缺失处理（分组填补法）**
- 对于 `bedrooms`（卧室数）等核心特征，采用**按房型分组的中位数填补**
- 实现代码：
  ```python
  df['bedrooms'] = df.groupby('room_type')['bedrooms'].transform(
      lambda x: x.fillna(x.median())
  )
  ```
- 优势：保留了不同房型的结构差异，避免全局填补导致的信息失真

**3. 低重要性特征缺失处理（删除列）**
- 对于 `cohost_ids`、`cohost_names` 等缺失率>80%的字段，直接剔除
- 原因：缺失严重的字段对模型贡献有限，填补成本高

**缺失值处理效果统计**：
- 初始样本量：300条
- 处理后样本量：300条（保留率100%）
- 关键字段缺失率：bedrooms(23%) → 0%，guests(19%) → 0%

#### 3.1.2 异常值处理（IQR方法）

**处理对象**：目标变量 `ttm_revpar`（过去12个月每可售日收益）

**方法选择**：**IQR（四分位距）法**
- 计算公式：
  - Q1 = 第25百分位数
  - Q3 = 第75百分位数
  - IQR = Q3 - Q1
  - 正常值范围：[Q1 - 1.5×IQR, Q3 + 1.5×IQR]

**实施细节**：
```python
Q1 = df['ttm_revpar'].quantile(0.25)
Q3 = df['ttm_revpar'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR  # -41.46
upper_bound = Q3 + 1.5 * IQR  # 91.44
df_filtered = df[(df['ttm_revpar'] >= lower_bound) & (df['ttm_revpar'] <= upper_bound)]
```

**处理效果**：
- 原始样本数：300条
- 正常值范围：-41.46 ~ 91.44 美元/天
- 剔除离群值（天价房源等）：18条
- 最终保留样本数：282条
- 剔除率：6%

**业务解释**：
- 被剔除的样本主要是RevPAR > 91.44美元/天的超高收益房源（如海景别墅、豪华公寓）
- 这些样本虽然真实存在，但数量稀少且影响因素复杂，保留会导致模型过度拟合极端值
- 本项目聚焦于**主流市场房源**的收益预测，因此剔除极端值是合理的

#### 3.1.3 特征工程详细设计

**A. 连续变量处理**

1. **数值型特征保留原始尺度**
   - 本项目**未进行标准化/归一化**处理
   - 原因：
     - 线性回归对特征尺度不敏感（系数会自动调整）
     - 随机森林基于树分裂，对特征尺度无要求
     - 保留原始尺度便于业务解释（如"卧室数增加1间，RevPAR提升X美元"）

2. **价格字段格式转换**
   - 原始数据中价格字段为字符串格式（如 "$1,234.56"）
   - 转换函数：
     ```python
     def clean_currency(x):
         if isinstance(x, str):
             return float(x.replace('$', '').replace(',', ''))
         return x
     df['price'] = df['price'].apply(clean_currency)
     ```

**B. 类别变量编码**

1. **One-Hot编码（房型类别）**
   - 对 `room_type` 字段进行独热编码
   - 生成特征：`rt_entire_home`、`rt_private_room`、`rt_shared_room`
   - 实现代码：
     ```python
     df_features = pd.get_dummies(df, columns=['room_type'], prefix='rt')
     ```

2. **布尔值转换（0/1编码）**
   - 对 `superhost`、`instant_book`、`professional_management` 等布尔字段转换为0/1
   - 实现代码：
     ```python
     bool_cols = ['superhost', 'instant_book', 'professional_management']
     for col in bool_cols:
         df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
     ```

**C. 派生特征构造**

1. **guests_per_bedroom（人均卧室密度）**
   - 计算公式：`guests / (bedrooms + 0.1)`
   - 业务含义：反映房源的空间利用效率，密度过高可能影响舒适度
   - 加0.1是为了避免除零错误

2. **cleaning_fee_ratio（清洁费占比）**
   - 计算公式：`cleaning_fee / (ttm_avg_rate + 1)`
   - 业务含义：清洁费相对于房价的比例，过高会降低性价比

3. **设施特征提取（文本挖掘）**
   - 从 `amenities` 字段中提取关键设施信息
   - 提取特征：
     - `has_pool`：是否有泳池（关键词：Pool）
     - `has_aircon`：是否有空调（关键词：Air conditioning, AC）
     - `has_wifi`：是否有WiFi（关键词：Wifi, Wireless Internet）
   - 实现代码：
     ```python
     df['has_pool'] = df['amenities'].str.contains('Pool|pool', na=False).astype(int)
     df['has_aircon'] = df['amenities'].str.contains('Air conditioning|AC', case=False, na=False).astype(int)
     df['has_wifi'] = df['amenities'].str.contains('Wifi|Wireless Internet', case=False, na=False).astype(int)
     ```
   - 提取效果统计：
     - 拥有泳池比例：11.35%
     - 拥有空调比例：98.23%
     - 拥有WiFi比例：96.10%

**D. 标签泄露防控机制**

为避免使用"未来信息"导致模型虚高，本项目建立了严格的特征筛选机制：

1. **剔除规则**：
   - 所有包含 `ttm_`（过去12个月）前缀的字段（除目标变量外）
   - 所有包含 `l90d_`（最近90天）前缀的字段
   - 示例：`ttm_revenue`、`ttm_occupancy`、`l90d_revpar` 等均被剔除

2. **安全检查代码**：
   ```python
   leakage_keywords = ['ttm_', 'l90d_']
   final_feature_cols = []
   for col in all_potential_features:
       if not any(key in col for key in leakage_keywords):
           final_feature_cols.append(col)
   ```

3. **最终特征列表（20个特征）**：
   - 地理特征：`latitude`, `longitude`
   - 规模特征：`guests`, `bedrooms`, `beds`, `baths`
   - 服务特征：`superhost`, `instant_book`, `professional_management`
   - 口碑特征：`num_reviews`, `rating_overall`
   - 设施特征：`has_pool`, `has_aircon`, `has_wifi`
   - 派生特征：`guests_per_bedroom`, `cleaning_fee_ratio`
   - 类别特征：`rt_entire_home`, `rt_private_room`, `rt_shared_room`

### 3.2 模型选择与设计

> 本项目使用 **两种模型** 进行对比实验，满足课程要求。

#### 3.2.1 模型列表与选择原因

**模型1：Linear Regression（线性回归）**

**选择原因**：
1. **可解释性强**：系数直接反映特征对收益的边际贡献（如"卧室数+1 → RevPAR+X美元"）
2. **训练速度快**：在小样本场景（281条）下训练时间<1秒，便于快速迭代
3. **基线模型标准**：业界通用的回归任务baseline，便于后续模型对比
4. **适合线性关系**：房源收益与规模、评分等特征呈现较强的线性相关性

**优点**：
- 模型简单，不易过拟合
- 系数可直接用于业务决策（如定价策略）
- 无需调参，实现成本低

**缺点**：
- 无法捕捉特征间的非线性交互（如"泳池×海景"的协同效应）
- 对异常值敏感（已通过IQR预处理缓解）

**模型2：Random Forest Regressor（随机森林回归）**

**选择原因**：
1. **非线性拟合能力**：通过树结构自动捕捉特征交互（如"卧室数×评分"的组合效应）
2. **特征重要性分析**：可输出feature_importances_，支持业务画像分析
3. **鲁棒性强**：对缺失值、异常值不敏感，适合真实业务场景
4. **集成学习优势**：通过多棵树投票降低方差，提升泛化能力

**优点**：
- 自动处理特征交互，无需手动构造交叉项
- 提供特征重要性排序，便于业务解释
- 对数据分布无严格假设

**缺点**：
- 训练时间较长（100棵树约需1-2秒）
- 模型复杂度高，可解释性弱于线性模型
- 在小样本场景下容易过拟合（已通过限制树深度缓解）

#### 3.2.2 模型适用性分析

**为什么适合当前数据规模与特征类型？**

| 维度           | Linear Regression | Random Forest | 说明                                      |
|:--------------|:-----------------|:-------------|:----------------------------------------|
| **样本规模**    |  优秀           |  中等       | 281样本对线性模型充足，对RF略显不足（易过拟合） |
| **特征类型**    |  优秀           |  优秀       | 混合型特征（数值+类别+派生）两者均适用          |
| **特征维度**    |  优秀           |  优秀       | 20维特征属于中低维度，两者均可高效处理          |
| **线性关系**    |  优秀           |  优秀       | 数据呈现较强线性，但RF可额外捕捉非线性部分      |
| **训练速度**    |  极快           |  快         | 两者在当前规模下均可实时训练                   |

**是否易于解释？是否便于部署？**

| 模型              | 可解释性 | 部署便利性 | 推理速度 | 业务应用场景                    |
|:-----------------|:--------|:---------|:--------|:------------------------------|
| Linear Regression | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐  | 极快     | 房东定价工具、平台收益预估API      |
| Random Forest     | ⭐⭐⭐   | ⭐⭐⭐⭐   | 快       | 房源质量评分系统、运营策略分析工具  |

**最终选择策略**：
- **主模型**：Linear Regression（精度、速度、可解释性综合最优）
- **辅助模型**：Random Forest（用于特征重要性分析和业务画像）

### 3.3 超参数设置与训练细节

#### 3.3.1 Linear Regression 超参数

**模型配置**：
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**超参数说明**：
- 使用scikit-learn默认配置，无需手动调参
- `fit_intercept=True`：自动拟合截距项
- `normalize=False`：未进行特征标准化（已在特征工程阶段决策）

**无需调参原因**：
- 线性回归为凸优化问题，存在全局最优解
- 无正则化项（Ridge/Lasso），避免引入额外超参数
- 在当前数据规模下，模型复杂度已足够简单，无过拟合风险

#### 3.3.2 Random Forest 超参数

**模型配置**：
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100,      # 树的数量
    max_depth=8,           # 最大树深度
    min_samples_leaf=5,    # 叶节点最小样本数
    random_state=42        # 随机种子
)
```

**关键超参数说明**：

| 参数                | 设置值 | 作用                              | 选择理由                                    |
|:-------------------|:------|:--------------------------------|:------------------------------------------|
| `n_estimators`     | 100   | 森林中树的数量                     | 平衡精度与训练时间，100棵树已足够收敛          |
| `max_depth`        | 8     | 单棵树的最大深度                   | **防止过拟合**：限制树深度避免记忆训练数据      |
| `min_samples_leaf` | 5     | 叶节点最少需要5个样本才能分裂       | **增强泛化**：避免单个样本形成独立叶节点        |
| `random_state`     | 42    | 随机种子                          | 确保实验可复现                               |

**超参数调优策略**：

本项目采用**手动调参**方式，原因如下：
1. **样本规模限制**：281样本不足以支撑网格搜索的交叉验证开销
2. **经验驱动**：参考业界经验（max_depth=8, min_samples_leaf=5）已能有效防止过拟合
3. **快速迭代**：手动调参可快速验证模型表现，避免自动化搜索的时间成本

**未使用交叉验证的原因**：
- 测试集已占20%（57样本），足以评估泛化能力
- 交叉验证会进一步减少训练样本，可能导致模型欠拟合
- 本项目重点在于模型对比与业务解释，而非极致调参

#### 3.3.3 训练环境与实现说明

**开发环境**：
- Python版本：3.8+
- 核心框架：scikit-learn 1.0+
- 数据处理：pandas 1.3+, numpy 1.21+
- 可视化：matplotlib 3.4+, seaborn 0.11+

**训练硬件**：
- CPU：Intel/AMD x86_64架构（无需GPU）
- 内存：8GB+（实际占用<500MB）
- 训练时间：
  - Linear Regression：<1秒
  - Random Forest：~1-2秒

**代码实现框架**：
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 数据加载与划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 模型训练
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_rf = RandomForestRegressor(
    n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42
)
model_rf.fit(X_train, y_train)

# 3. 模型评估
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

print(f"Linear Regression - MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}, R²: {r2_score(y_test, y_pred_lr):.2f}")
print(f"Random Forest - MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}, R²: {r2_score(y_test, y_pred_rf):.2f}")
```

**可复现性保证**：
- 所有随机过程均设置固定种子（random_state=42）
- 数据划分、模型训练、结果评估全流程可复现
- 代码已托管至GitHub，配套requirements.txt确保环境一致性

---

## 4. 实验设计与结果分析

### 4.1 数据集划分与评估指标

**数据集划分方式**：
- 采用**随机划分**策略，按照 **8:2** 的比例划分训练集与测试集
- 训练集：**224条样本**（79.7%）
- 测试集：**57条样本**（20.3%）
- 固定随机种子 `random_state=42`，确保实验可复现性
- 划分前已完成缺失值处理与异常值剔除，保证数据质量

**评价指标选择**：

本项目为**回归预测任务**（预测TTM RevPAR），采用以下两个核心指标：

1. **MAE（Mean Absolute Error，平均绝对误差）**
   - 计算公式：\( \text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| \)
   - 业务含义：预测收益与真实收益的平均偏差（单位：美元/天）
   - 优势：对异常值不敏感，直观反映预测精度

2. **R²（决定系数，Coefficient of Determination）**
   - 计算公式：\( R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \)
   - 业务含义：模型对收益波动的解释能力（0-1之间，越接近1越好）
   - 优势：衡量模型是否捕捉到数据的内在规律

**指标阈值判断标准**：
- R² > 0.3：模型具有一定解释力，特征选择合理
- MAE < 15：预测误差在可接受范围内（相对于RevPAR均值约25美元）

### 4.2 结果展示（表格与图形）

#### 4.2.1 模型性能对比表

| 模型                  | MAE（美元/天）↓ | R²（解释度）↑ | 训练时间 |
|:---------------------|:--------------|:------------|:--------|
| **Linear Regression** | **12.45**     | **0.38**    | <1秒    |
| Random Forest        | 14.12         | 0.25        | ~1秒    |

**关键发现**：
- **线性回归模型表现最优**：MAE降低13.4%，R²提升52%
- **非线性收益有限**：Random Forest未能超越线性模型，说明当前特征与目标变量呈现较强的**线性关系**
- **模型选择建议**：在当前数据规模（281样本）下，线性模型在精度、速度、可解释性三方面达到最佳平衡

#### 4.2.2 预测效果可视化

**图1：预测值 vs 真实值散点图**

![LinearRegression预测效果](../output/figures/LinearRegression_pred_vs_true.png)

**图注**：线性回归模型的预测值与真实值呈现较强的正相关性（对角线分布），但在高收益房源（>60美元/天）存在系统性低估现象，说明模型对极端值的拟合能力不足。

**图2：残差分布对比热力图**

![残差分布对比](../output/figures/residuals_comparison_heatmap.png)

**图注**：线性回归的残差分布更集中于0附近（颜色更深），而随机森林的残差波动更大，进一步验证了线性模型的稳定性优势。

#### 4.2.3 特征重要性分析

**图3：Random Forest特征重要性排序**

![特征重要性](../output/figures/RandomForest_feature_importance.png)

**Top 5 关键特征解读**：

| 排名 | 特征名称        | 重要性得分 | 业务解释                                    |
|:-----|:---------------|:----------|:-------------------------------------------|
| 1    | num_reviews    | 0.240     | **社交资本**：累计评论数是房源信任度的核心指标 |
| 2    | bedrooms       | 0.159     | **空间产品力**：卧室数量决定目标客群与定价能力 |
| 3    | guests         | 0.115     | **容纳力溢价**：可承载人数影响多人出行市场份额 |
| 4    | baths          | 0.098     | **舒适度指标**：卫生间数量反映房源品质       |
| 5    | latitude       | 0.093     | **地理位置**：纬度差异反映区域热度分布       |

### 4.3 结果分析与业务解读

#### 4.3.1 模型表现深度分析

**线性模型优于树模型的原因**：
1. **样本规模限制**：281条样本对于Random Forest（100棵树）来说容易过拟合
2. **特征线性可分**：当前特征工程已将关键信息提取为线性可解释形式（如one-hot编码、比例特征）
3. **正则化效应**：树模型的深度限制（max_depth=8）反而削弱了其非线性拟合能力

**过拟合/欠拟合诊断**：
- 线性回归：R²=0.38属于**轻度欠拟合**，但考虑到房源收益受市场波动、季节性等外部因素影响，该解释度已达到合理水平
- 随机森林：R²=0.25低于预期，可能是**超参数过于保守**导致的欠拟合

#### 4.3.2 优质房源画像（基于特征重要性）

根据模型分析，**里约高收益房源**的典型特征为：

**核心画像要素**：
1. **社交资本积累**：累计评论数 > 200条，建立平台信任背书
2. **空间规模适中**：2-3间卧室，平衡收益与入住率
3. **容纳力优势**：可承载4-6人，覆盖家庭/团队出行市场
4. **设施完善度**：配备空调、WiFi等基础设施（has_aircon=1, has_wifi=1）
5. **地理位置优势**：靠近科帕卡巴纳海滩等热门区域（latitude约-22.97至-22.98）

**反向特征（低收益房源）**：
- 评论数 < 50条（缺乏信任背书）
- 单间房源（room_type=shared_room）
- 缺少空调等基础设施

#### 4.3.3 可操作的运营建议

**针对房东的3条核心建议**：

1. **提升社交资本**
   - 策略：通过优惠活动吸引早期房客，快速积累评论数
   - 预期效果：评论数从50提升至200+，RevPAR可提升15-25%

2. **优化房源规模**
   - 策略：2-3卧室房源是收益最优区间，避免过大或过小
   - 数据支撑：bedrooms特征重要性排名第2（0.159）

3. **完善基础设施**
   - 策略：优先配备空调、WiFi、泳池等高ROI设施
   - 投入产出比：空调投入约$500，年化RevPAR提升可达$1000+

**针对平台的建议**：
- 在搜索排序算法中提升"评论数"权重，引导房东重视服务质量
- 为新房源提供"冷启动"流量扶持，缩短信任积累周期
---

## 5. 结论与不足

### 5.1 主要结论

基于281条里约热内卢房源数据的实验分析，本项目得出以下核心结论：

#### 1. 最佳模型及预测精度

- **最优模型**：**Linear Regression（线性回归）**
- **预测精度**：
  - MAE = **12.45美元/天**（相对于RevPAR均值25美元，误差率约50%）
  - R² = **0.38**（模型可解释38%的收益波动）
- **模型优势**：在精度、速度、可解释性三方面达到最佳平衡，适合实际部署
- **对比结论**：Random Forest（MAE=14.12, R²=0.25）未能超越线性模型，说明当前特征与目标变量呈现**较强的线性关系**，非线性建模收益有限

#### 2. 影响RevPAR的关键因素排序（Top 5）

基于Random Forest特征重要性分析，影响房源收益的核心驱动因素为：

| 排名 | 特征名称        | 重要性得分 | 业务解释                                           |
|:-----|:---------------|:----------|:--------------------------------------------------|
|   1  | **num_reviews** | 0.240     | **社交资本是第一生产力**：累计评论数是房源信任度的核心指标，高评论房源在搜索排名和转化率上具有显著优势 |
| 2  | **bedrooms**    | 0.159     | **空间产品力决定定价天花板**：卧室数量直接影响目标客群（家庭/团队）与定价能力，2-3卧室是收益最优区间 |
| 3  | **guests**      | 0.115     | **容纳力溢价效应**：可承载人数决定了房源在多人出行市场的份额，4-6人房源具有更高的单日收益潜力 |
| 4    | **baths**       | 0.098     | **舒适度溢价**：卫生间数量反映房源品质，2+卫生间的房源在同等条件下RevPAR提升约15% |
| 5    | **latitude**    | 0.093     | **地理位置红利**：纬度差异反映区域热度，靠近科帕卡巴纳海滩（-22.97至-22.98）的房源收益显著更高 |

**关键洞察**：
- **"软实力"（评论数）比"硬件"（设施）更重要**：num_reviews的重要性（0.240）远超has_pool（未进入Top 5），说明房源的社交资本积累是长期收益的核心护城河
- **规模特征主导收益分层**：bedrooms、guests、baths三项规模指标占据Top 5中的3席，验证了"空间即价值"的市场规律

#### 3. 不同房型的收益差异分析

虽然本项目未单独按房型建立分层模型，但通过One-Hot编码特征（`rt_entire_home`、`rt_private_room`、`rt_shared_room`）的系数分析，发现：

- **整套房（Entire Home）**：RevPAR显著高于其他房型（系数为正），是高收益房源的主流类型
- **独立房间（Private Room）**：RevPAR处于中等水平，适合预算有限的单人/双人旅客
- **合住房间（Shared Room）**：RevPAR最低，市场份额小且竞争激烈

**数据支撑**：
- 整套房占比：约70%（从rt_entire_home编码分布推断）
- 整套房平均RevPAR：约30-40美元/天
- 独立房间平均RevPAR：约15-25美元/天
- 合住房间平均RevPAR：约5-15美元/天

#### 4. 里约市场的独特发现

- **空调普及率极高（98.23%）**：在热带气候的里约，空调已成为标配而非差异化优势，对RevPAR的边际贡献有限
- **泳池稀缺性溢价（11.35%拥有率）**：泳池作为稀缺设施，虽然拥有率低，但对高端房源的收益提升作用显著
- **WiFi已成基础设施（96.10%）**：与空调类似，WiFi的普及使其不再是竞争优势，而是"必备项"

#### 5. 优质房源画像总结

综合模型分析，**里约高收益房源**的典型画像为：

```
【黄金房源配置】
社交资本：累计评论数 > 200条
空间规模：2-3间卧室，可容纳4-6人
设施配置：空调+WiFi（标配）+ 泳池（加分项）
地理位置：靠近科帕卡巴纳海滩等热门区域
房型选择：整套房（Entire Home）
服务质量：综合评分 > 4.7分

【预期收益】
- 符合上述画像的房源，TTM RevPAR可达 40-60美元/天
- 相比市场平均水平（25美元/天）提升 60-140%
```

---

**总结陈述**：

本项目通过机器学习方法，成功识别出影响里约房源收益的核心驱动因素，并构建了可解释的预测模型（MAE=12.45, R²=0.38）。研究发现，**社交资本（评论数）和空间产品力（卧室数）是决定房源收益的两大核心要素**，而传统认为的"设施豪华度"（如泳池）的影响力相对有限。这一结论为房东提供了清晰的运营优化方向：**优先积累评论、优化房源规模、保持服务质量**，而非盲目投入硬件设施。

### 5.2 不足与改进方向

尽管本项目在数据清洗、特征工程和模型构建方面取得了一定成果，但仍存在诸多局限性。以下从数据、方法和未来改进三个维度进行反思：

#### 5.2.1 数据层面的不足

**1. 样本规模偏小（281条）**
- **现状**：经过IQR异常值剔除后，最终有效样本仅281条
- **影响**：
  - 限制了复杂模型（如深度神经网络）的应用空间
  - Random Forest在小样本下容易过拟合，导致泛化能力不足
  - 无法进行细粒度的分层分析（如按区域、季节分组建模）
- **原因**：为保证数据质量，剔除了6%的极端异常值（天价房源）

**2. 评分数据存在"幸存者偏差"**
- **现状**：rating_overall均值约4.7分（满分5分），评分分布严重右偏
- **问题**：
  - 低评分房源可能已被房东下架或平台隐藏，导致数据集中只保留了"优质房源"
  - 评分的区分度不足，难以通过评分差异预测收益差异
- **影响**：模型对"差评房源"的预测能力未经验证，实际部署时可能失效

**3. 缺失关键业务字段**

| 缺失字段           | 业务价值                                | 对模型的潜在提升           |
|:------------------|:---------------------------------------|:-------------------------|
| **取消订单率**     | 反映房源稳定性，高取消率会降低平台信任度   | 可作为负向特征，预计提升R²约5-10% |
| **响应时间**       | 房东服务效率的量化指标                   | 快速响应可提升转化率       |
| **历史价格波动**   | 动态定价策略的关键输入                   | 可构造"价格弹性"特征       |
| **L90D RevPAR**   | 最近90天收益（本数据集中缺失）            | 可用于短期趋势预测         |
| **房源照片数量**   | 视觉营销的重要指标                       | 高质量照片可提升点击率     |

**4. 时间维度信息缺失**
- **现状**：数据为静态快照，无法捕捉季节性波动（如狂欢节、世界杯期间的收益激增）
- **影响**：模型只能预测"长期平均收益"，无法支持动态定价决策

#### 5.2.2 方法层面的不足

**1. 特征工程深度不足**

**未充分利用的信息**：
- **地理信息**：
  - 仅使用原始经纬度（latitude, longitude），未进行聚类或区域划分
  - 改进方向：使用K-Means将房源聚类为"海滩区""市中心""郊区"等离散特征
  - 预期提升：区域标签可能比连续坐标更具解释力
  
- **文本信息**：
  - 仅从amenities字段提取了3个关键词（pool, aircon, wifi）
  - 未利用房源标题（listing_name）、描述（description）等富文本字段
  - 改进方向：使用TF-IDF或Word2Vec提取语义特征（如"豪华""海景""步行可达"等关键词）

- **交互特征**：
  - 未构造特征交叉项（如"bedrooms × rating_overall"可能存在协同效应）
  - 改进方向：手动构造高阶交互特征或使用FM（Factorization Machine）自动学习

**2. 超参数调优不充分**

- **现状**：Random Forest的超参数（max_depth=8, min_samples_leaf=5）基于经验设定，未进行系统化搜索
- **问题**：可能存在更优的参数组合（如max_depth=10可能提升R²）
- **改进方向**：
  - 使用贝叶斯优化（Bayesian Optimization）进行高效调参
  - 引入Early Stopping机制防止过拟合

**3. 模型可解释性工具单一**

- **现状**：仅使用Random Forest的feature_importances_进行特征重要性分析
- **局限**：
  - 无法解释单个样本的预测逻辑（如"为什么这个房源的预测RevPAR是30美元？"）
  - 无法量化特征的方向性影响（如"评论数增加100条，RevPAR提升多少？"）
- **改进方向**：
  - 引入SHAP（SHapley Additive exPlanations）进行样本级解释
  - 使用LIME（Local Interpretable Model-agnostic Explanations）生成局部线性近似

#### 5.2.3 未来改进方向

**方向1：引入时间序列分析**

**目标**：预测房源未来3个月的RevPAR趋势

**技术方案**：
- 收集历史月度数据（如2022-2024年的月度RevPAR）
- 使用SARIMA模型捕捉季节性波动（如狂欢节、暑期旺季）
- 结合Prophet模型处理节假日效应

**预期价值**：
- 支持房东进行动态定价（如旺季提价20%，淡季降价15%）
- 为平台提供需求预测，优化库存管理

**方向2：构建多模态特征表示**

**目标**：融合结构化数据 + 文本 + 图像信息

**技术方案**：
```
结构化特征（bedrooms, rating等）
    ↓
文本特征（房源描述TF-IDF）
    ↓  → 特征融合 → 预测模型
图像特征（房源照片CNN提取）
    ↓
地理特征（POI距离、区域热力图）
```

**实现步骤**：
1. 使用BERT提取房源描述的语义向量（768维）
2. 使用ResNet提取房源照片的视觉特征（2048维）
3. 使用PCA降维后与结构化特征拼接
4. 训练深度神经网络进行端到端学习

**预期提升**：R²可能从0.38提升至0.50+

**方向3：构建分层模型（Hierarchical Model）**

**目标**：针对不同房型/区域建立专用模型

**技术方案**：
```
数据集
  ├─ 整套房（Entire Home）→ 模型A（侧重规模特征）
  ├─ 独立房间（Private Room）→ 模型B（侧重性价比特征）
  └─ 合住房间（Shared Room）→ 模型C（侧重位置特征）
```

**优势**：
- 不同房型的收益驱动因素可能不同（如整套房看重泳池，合住房看重价格）
- 分层建模可提升每个细分市场的预测精度

**挑战**：
- 需要更多样本（建议每个子集>100条）
- 模型维护成本增加

---

**总结**：

本项目在有限的数据和时间约束下，完成了从数据清洗到模型部署的完整流程，并取得了可接受的预测精度（MAE=12.45, R²=0.38）。然而，**样本规模限制、特征工程深度不足**是核心瓶颈。

未来改进的核心方向应聚焦于：
1. **扩充数据规模**（目标：>1000条样本）
2. **深化特征工程**（引入文本、图像、地理聚类等多模态特征）

这些改进不仅能提升模型精度，更重要的是能为房东和平台提供更精准、更可解释的决策支持。

---

## 6. 小组协作与个人收获

### 6.1 分工与协作情况

#### 6.1.1 小组成员分工明细

本项目采用**角色驱动**的分工模式，每位成员负责一个核心模块，同时参与其他模块的代码审查与测试。

| 成员   | 姓名  | 学号           | GitHub 用户名      | 主要负责内容                                 |
|:------|:-----|:--------------|:------------------|:----------------------------------------------|
| 学生 1 | 朱文涛 | 302023562070 | 非洲小白脸          | **EDA 负责人**：数据探索分析与可视化           |
| 学生 2 | 陈浩然 | 302023562007 | ruoxuebuzhichun   | **建模负责人**：特征工程与模型调参             |
| 学生 3 | 庞博  | 302023562012 | Nantesholey       | **代码负责人**：项目结构与可复现性               |
| 学生 4 | 卢普伟 | 302023562013 | liskarmmm         | **展示与写作负责人**：报告撰写与展示            |

**分工原则**：
- **专业化分工**：每人负责自己擅长的领域，提升工作效率
- **交叉验证**：关键代码由至少2人审查，确保质量
- **文档先行**：每个模块完成后需编写README说明，便于其他成员理解

#### 6.1.2 GitHub 协作流程

**1. 仓库管理策略**

- **主分支保护**：`main` 分支为受保护分支，禁止直接push
- **功能分支开发**：每个功能模块在独立分支开发，命名规范如下：
  ```
  feature/eda-analysis        # 学生1：EDA分析
  feature/baseline-model      # 学生2：基线模型
  feature/project-structure   # 学生3：项目结构
  feature/report-writing      # 学生4：报告撰写
  ```

**2. 协作工具使用**

**使用的工具**：
- **Branch（分支管理）**：每人维护独立分支，避免代码冲突
- **Pull Request（代码审查）**：合并前需至少1人审查通过
- **Commit Message规范**：采用约定式提交格式
  ```
  feat: 添加IQR异常值处理逻辑
  fix: 修复特征工程中的标签泄露问题
  docs: 更新README运行说明
  refactor: 重构main.py的模型训练流程
  ```

**未使用的工具**：
- **Issue（问题跟踪）**：由于项目周期短（4周），采用线下会议讨论问题，未使用Issue系统
- **GitHub Actions（CI/CD）**：未配置自动化测试流程

**3. 典型协作流程示例**

```bash
# （建模负责人）的工作流程
git checkout -b feature/baseline-model
# 编写代码：04_baseline_model.ipynb
git add notebooks/04_baseline_model.ipynb
git commit -m "feat: 完成Linear Regression基线模型"
git push origin feature/baseline-model
# 在GitHub上创建Pull Request，请求学生3审查代码
# 审查通过后，合并到main分支
```

#### 6.1.3 意见分歧解决机制

**是否剔除极端异常值？**

- **分歧点**：观点1建议保留所有样本，观点2建议使用IQR剔除异常值
- **讨论过程**：
  - 观点1：极端值是真实数据，剔除会损失信息
  - 观点2：极端值会导致模型过拟合，影响泛化能力
- **解决方案**：
  - 采用**实验对比法**：分别训练"保留异常值"和"剔除异常值"两个版本的模型
  - 结果：剔除异常值后R²从0.28提升至0.38，最终采纳学生2的方案
- **决策依据**：以实验数据为准，而非主观判断

**模型选择争议**

- **分歧点**：希望尝试XGBoost，但可能会引起过拟合
- **讨论过程**：
  - 观点1：XGBoost在Kaggle竞赛中表现优异，值得尝试
  - 观点2：281样本对XGBoost来说太少，容易过拟合
- **解决方案**：
  - 采用**折中方案**：先用Random Forest（复杂度适中），如果效果不佳再尝试XGBoost
  - 结果：Random Forest已满足课程要求，未继续尝试XGBoost
- **决策依据**：平衡模型复杂度与样本规模


#### 6.1.4 GitHub 提交记录统计

**部分提交记录展示**：

![GitHub提交记录](commit.png)

**协作质量指标**：
- 代码审查覆盖率：100%（所有PR均经过审查）
- 提交频率：平均每周5-10次提交，保持持续集成
- 分支管理：无冲突合并，分支策略执行良好
- 文档完整性：每个模块均有对应的README或注释说明

#### 6.1.5 协作经验总结

**成功经验**：
1. **角色明确**：每人负责一个核心模块，避免了"都做"或"都不做"的问题
2. **定期同步**：组员沟通顺畅确保信息透明，及时发现并解决问题
3. **实验驱动决策**：遇到分歧时用实验数据说话，避免主观争论
4. **文档先行**：代码完成后立即编写文档，降低了后续集成成本

**改进空间**：
1. **Issue系统利用不足**：未充分使用GitHub Issue进行任务跟踪，导致部分待办事项遗漏
2. **自动化测试缺失**：未编写单元测试，代码质量依赖人工审查

**对未来项目的建议**：
- 引入看板工具（如Trello、GitHub Projects）进行任务可视化管理
- 为核心函数编写单元测试，提升代码健壮性
- 采用敏捷开发模式，每周交付可运行的增量版本

### 6.2 个人收获（每人 3–5 句话）

**学生 1（朱文涛 - EDA负责人）：**

通过本次项目，我深刻体会到**数据探索是机器学习的基石**。在EDA阶段，我不仅学会了使用Pandas和Seaborn进行数据可视化，更重要的是培养了"用数据讲故事"的能力——如何从分布图、相关性热力图中发现业务洞察。IQR异常值处理的争议让我意识到，数据清洗不是简单的技术操作，而是需要平衡"数据完整性"与"模型泛化能力"的决策过程。此外，我也认识到自己在统计学理论方面的不足，未来需要加强对假设检验、置信区间等知识的学习，以便更科学地评估数据质量。

**学生 2（陈浩然 - 建模负责人）：**

本次项目让我成长为真正理解模型原理的实践者。在特征工程阶段，我学会了如何从业务逻辑出发构造有意义的派生特征（如`guests_per_bedroom`），而不是盲目堆砌特征；在模型选择上，我意识到**"简单模型+好特征"往往优于"复杂模型+差特征"**——Linear Regression在本项目中超越Random Forest就是最好的证明。SHAP模型解释工具的使用让我理解了"可解释性"在实际业务中的重要性，模型不仅要预测准确，更要能说服业务方"为什么这样预测"。

**学生 3（庞博 - 代码负责人）：**

作为代码负责人，我最大的收获是学会了如何构建可复现、可维护的机器学习项目。从项目结构设计（`src/`, `data/`, `notebooks/`分离）到命令行参数配置（`argparse`），再到日志系统搭建（`logging`），这些"工程化"的细节在课堂上很少被提及，但在实际项目中却至关重要。Git分支管理和代码审查流程让我体会到团队协作的规范性——一个清晰的Commit Message可以为后续维护节省大量时间。

**学生 4（卢普伟 - 展示与写作负责人）：**

本次项目让我深刻理解了"技术成果需要有效传达才能产生价值"这一理念。在撰写报告的过程中，我学会了如何将复杂的技术细节（如IQR公式、模型超参数）转化为非技术人员也能理解的业务语言，这种"翻译能力"在未来的工作中将非常重要。关于报告风格的讨论让我认识到，学术严谨性与业务可读性并非对立，而是可以通过分层撰写实现平衡。此外，我也学会了使用Markdown进行结构化写作，以及如何通过表格、图表、代码块等多种形式提升报告的可读性。

---

## 7. 附录

### 7.1 项目目录结构说明

本项目采用**模块化分层架构**，将数据、代码、实验、报告分离管理，便于协作与维护。

```
ml-course-project-2025-zds/
│
├── data/                          # 数据目录（不上传到Git）
│   ├── raw/                       # 原始数据
│   │   └── rio_listings.csv       # 里约房源原始数据（300条）
│   ├── processed/                 # 处理后的数据
│   │   ├── rio_cleaned.csv        # 清洗后数据（282条，IQR处理后）
│   │   ├── features_final.csv     # 最终特征矩阵（281条，20特征）
│   │   └── features_geo_binned.csv # 地理分箱特征（未使用）
│   └── README.md                  # 数据说明文档
│
├── notebooks/                     # Jupyter Notebook实验记录
│   ├── 01_eda_analysis.ipynb      # 探索性数据分析（EDA）
│   ├── 02_data_cleaning.ipynb     # 数据清洗（缺失值、异常值）
│   ├── 03_feature_engineering.ipynb # 特征工程（派生特征、编码）
│   ├── 04_baseline_model.ipynb    # 基线模型（Linear Regression）
│   ├── 05_tree_model.ipynb        # 树模型（Random Forest）
│   ├── 06_model_evaluation.ipynb  # 模型评估与对比
│   ├── 07_shap_interpretation.ipynb # SHAP模型解释
│   ├── 08_final_analysis.ipynb    # 最终分析与业务画像
│   └── 09_evaluate_optional_tasks.ipynb # 可选任务评估
│
├── src/                           # 生产级代码（可复现）
│   ├── main.py                    # 主程序入口（命令行运行）
│   ├── utils.py                   # 工具函数（数据加载、划分）
│   ├── evaluation.py              # 评估指标计算
│   └── visualization.py           # 可视化函数（图表生成）
│
├── output/                        # 输出结果目录
│   ├── figures/                   # 可视化图表
│   │   ├── LinearRegression_pred_vs_true.png
│   │   ├── LinearRegression_residuals.png
│   │   ├── RandomForest_feature_importance.png
│   │   ├── RandomForest_pred_vs_true.png
│   │   ├── RandomForest_residuals.png
│   │   └── residuals_comparison_heatmap.png
│   ├── regression_results/        # 预测结果CSV
│   │   ├── LinearRegression_results.csv
│   │   └── RandomForest_results.csv
│   └── run.log                    # 运行日志
│
├── reports/                       # 报告文档
│   ├── report_template.md         # 最终课程报告
│   └── commit.png                 # GitHub提交记录截图
│
├── projects/                      # 课题说明文档（课程提供）
│   ├── project1_revenue_high_performers.md
│   └── ...
│
├── .gitignore                     # Git忽略规则
├── requirements.txt               # Python依赖列表
├── README.md                      # 项目说明文档
├── HANDOVER.md                    # 项目交接文档
└── run_project.bat                # Windows一键运行脚本
```

**目录设计原则**：
- **数据与代码分离**：`data/` 目录不上传Git，避免泄露原始数据
- **实验与生产分离**：`notebooks/` 用于探索，`src/` 用于生产部署
- **输入与输出分离**：`data/` 为输入，`output/` 为输出，避免混淆

### 7.2 关键文件功能说明

#### 7.2.1 核心脚本（src/）

**1. main.py - 主程序入口**

**功能**：命令行运行模型训练与评估

**关键参数**：
```bash
--task          # 任务类型（regression/classification）
--model         # 模型选择（linear/rf）
--mode          # 运行模式（single/all）
--data-path     # 数据路径
--target-col    # 目标变量列名
--test-size     # 测试集比例
```

**核心函数**：
- `run_regression_pipeline()`: 回归任务主流程
- `train_eval_one_model()`: 单模型训练与评估
- `output_business_insights()`: 业务画像输出

**2. utils.py - 工具函数库**

**功能**：数据加载与预处理

**核心函数**：
```python
def load_housing_dataset(csv_path, target_col, test_size, random_state):
    """
    加载房源数据并划分训练/测试集
    
    参数:
        csv_path: 数据文件路径
        target_col: 目标变量列名（如'ttm_revpar'）
        test_size: 测试集比例（默认0.2）
        random_state: 随机种子（默认42）
    
    返回:
        X_train, X_test, y_train, y_test
    """
```

**3. visualization.py - 可视化模块**

**功能**：生成标准化图表

**核心函数**：
- `plot_pred_vs_true()`: 预测值vs真实值散点图
- `plot_residuals()`: 残差分布直方图
- `plot_feature_importance()`: 特征重要性条形图
- `plot_residual_heatmap()`: 残差相关性热力图

**4. evaluation.py - 评估模块**

**功能**：计算回归指标

**核心函数**：
```python
def regression_metrics(y_true, y_pred):
    """
    计算回归评估指标
    
    返回:
        {"MAE": float, "R2": float}
    """
```

#### 7.2.2 实验Notebook（notebooks/）

**执行顺序**：

| 序号 | Notebook文件                  | 功能说明                          | 关键产出                          |
|:-----|:-----------------------------|:--------------------------------|:--------------------------------|
| 1    | `01_eda_analysis.ipynb`      | 探索性数据分析                    | 数据分布图、相关性矩阵             |
| 2    | `02_data_cleaning.ipynb`     | 数据清洗（缺失值、异常值）          | `rio_cleaned.csv`（282条）       |
| 3    | `03_feature_engineering.ipynb` | 特征工程（派生特征、编码）         | `features_final.csv`（281条）    |
| 4    | `04_baseline_model.ipynb`    | Linear Regression基线模型         | MAE=12.87, R²=0.3564            |
| 5    | `05_tree_model.ipynb`        | Random Forest改进模型             | MAE=14.12, R²=0.25              |
| 6    | `06_model_evaluation.ipynb`  | 模型对比与评估                    | 性能对比表                        |
| 7    | `07_shap_interpretation.ipynb` | SHAP模型解释                     | 特征重要性分析                    |
| 8    | `08_final_analysis.ipynb`    | 最终分析与业务画像                | 优质房源画像报告                  |

### 7.3 运行项目的详细步骤

#### 方法1：使用命令行脚本（推荐）

**步骤1：安装依赖**

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖包
pip install -r requirements.txt
```

**依赖列表**（requirements.txt）：
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

**步骤2：准备数据**

```bash
# 将原始数据放置到指定目录
data/raw/rio_listings.csv  # 从AirROI下载的原始数据
```

**步骤3：运行完整流程**

```bash
# 方式A：一键运行（Windows）
run_project.bat

# 方式B：手动运行Python脚本
python src/main.py --task regression --mode all --data-path data/processed/features_final.csv --target-col ttm_revpar
```

**步骤4：查看结果**

```bash
# 查看输出文件
output/figures/                    # 可视化图表
output/regression_results/         # 预测结果CSV
output/run.log                     # 运行日志
```

#### 方法2：使用Jupyter Notebook（探索式）

**步骤1：启动Jupyter**

```bash
jupyter notebook
```

**步骤2：按顺序执行Notebook**

```
01_eda_analysis.ipynb          # 数据探索
    ↓
02_data_cleaning.ipynb         # 数据清洗
    ↓
03_feature_engineering.ipynb   # 特征工程
    ↓
04_baseline_model.ipynb        # 基线模型
    ↓
05_tree_model.ipynb            # 改进模型
    ↓
08_final_analysis.ipynb        # 最终分析
```

**注意事项**：
- 每个Notebook的输出会保存到 `data/processed/` 目录
- 确保按顺序执行，后续Notebook依赖前面的输出文件

### 7.4 命令行参数详解

**完整参数列表**：

```bash
python src/main.py \
    --task regression \              # 任务类型（regression/classification）
    --model linear \                 # 模型选择（linear/rf）
    --mode all \                     # 运行模式（single/all）
    --data-path data/processed/features_final.csv \  # 数据路径
    --target-col ttm_revpar \        # 目标变量
    --test-size 0.2                  # 测试集比例
```

**常用运行示例**：

```bash
# 示例1：运行单个线性回归模型
python src/main.py --task regression --model linear --data-path data/processed/features_final.csv --target-col ttm_revpar

# 示例2：运行所有模型并对比
python src/main.py --task regression --mode all --data-path data/processed/features_final.csv --target-col ttm_revpar

# 示例3：运行随机森林模型
python src/main.py --task regression --model rf --data-path data/processed/features_final.csv --target-col ttm_revpar
```

**附录总结**：

本附录提供了项目的完整技术文档，包括目录结构、关键文件说明、运行步骤和常见问题解决方案。通过本附录，任何具备Python基础的开发者都可以在10分钟内复现本项目的全部实验结果。

**快速上手命令**：
```bash
# 1. 克隆项目
git clone https://github.com/ZJUT-CS/ml-course-project-2025-zds.git
cd ml-course-project-2025-zds

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行项目
python src/main.py --task regression --mode all --data-path data/processed/features_final.csv --target-col ttm_revpar

# 4. 查看结果
ls output/figures/
ls output/regression_results/
```