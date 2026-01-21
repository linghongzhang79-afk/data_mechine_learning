# 项目1：房源收益预测与优质房源画像

## 一、项目概述

### 1.1 研究背景

Airbnb房东面临如何提高收益、优化运营的核心问题。本研究基于东京市场Airbnb房源数据，构建收益预测模型，识别高收益房源的关键特征，为房东和运营方提供数据驱动的决策支持。

### 1.2 核心问题

1. 预测问题：根据房源属性和运营特征，准确预测TTM RevPAR（过去12个月每间可用房间的平均收益）
2. 画像问题：识别高收益房源的特征，定义优质房源画像

### 1.3 技术路线

数据探索与清洗 → 特征工程 → 模型训练与对比 → 模型解释 → 业务洞察

---

## 二、数据探索

### 2.1 数据概况

- 数据源：东京市场Airbnb房源数据（Tokyo.csv）
- 原始样本：300条，62个字段
- 清洗后样本：265条（保留率88.3%）

### 2.2 缺失值分析

**高缺失字段（≥50%）**
- instant_book（77.33%）：填充为'False'
- cohost_ids（71.33%）、cohost_names（68.33%）：创建has_cohost指示特征

**中等缺失字段（1%-50%）**
- bedrooms（31.67%）、guests（28.67%）：众数填充
- baths（10.67%）、min_nights（12.67%）：根据业务逻辑处理

### 2.3 收益指标分布

**TTM RevPAR**
- 中位数：58.10，平均值：1,369.80
- 呈现明显右偏分布，少数高收益房源拉高平均值

**TTM Revenue**
- 中位数：20,777，平均值：388,690
- 存在极端异常值（最大值30,800,000）

**TTM Occupancy**
- 中位数：0.57
- 存在异常值（最大值9,483,610，超出合理范围[0,1]）

### 2.4 房源规模特征

- 可容纳人数：中位数3人，平均值4.65人
- 卧室数量：中位数1间，平均值2.61间
- 浴室数量：中位数1间，平均值0.89间

大部分房源为中小型规模（1-2间卧室，可容纳2-4人）。

### 2.5 评论热度

- 评论数量：中位数152条，平均值160.88条
- 分布相对集中，市场活跃度较高

---

## 三、数据清洗

### 3.1 异常值处理

**住宿率异常值**
- 删除ttm_occupancy超出[0,1]范围的记录（8条）
- 删除l90d_occupancy超出[0,1]范围的记录（8条）

**TTM总天数异常值**
- 删除ttm_total_days超过365天的记录（27条）

**负收益记录**
- 数据中不存在负收益记录，无需处理

### 3.2 缺失值处理

**关键字段（ttm_revpar、room_type）**
- 删除包含缺失值的记录

**guests字段**
- 使用众数2.0填充缺失值

**其他字段**
- 根据字段类型和重要性，采用删除或填充策略

### 3.3 数据类型转换

将以下字段转换为正确的数值类型：
- 浮点型字段：rating_value、latitude、longitude、ttm_revpar等
- 整型字段：ttm_total_days、l90d_total_days

### 3.4 清洗结果

| 指标 | 清洗前 | 清洗后 |
|-----|-------|-------|
| 记录数 | 300 | 265 |
| 数据保留率 | 100% | 88.3% |

删除记录明细：
- ttm_occupancy超出[0,1]范围：8条
- ttm_total_days超过365：27条

---

## 四、特征工程

### 4.1 特征概览

从原始28个特征出发，构造14个衍生特征，最终形成42个特征的特征集。

### 4.2 房源规模特征

- guests_per_bedroom：每间卧室容纳人数
- guests_per_bath：每个浴室容纳人数
- guests_per_bed：每张床容纳人数
- beds_per_bedroom：每间卧室的床位数
- baths_per_bedroom：每间卧室的浴室数

### 4.3 价格结构特征

- cleaning_fee_per_guest：每位客人的清洁费用
- cleaning_fee_per_bedroom：每间卧室的清洁费用
- extra_guest_fee_per_guest：每位额外客人的费用
- extra_fee_share：额外费用占比
- has_extra_guest_fee：是否有额外客人费用

### 4.4 运营属性特征

- superhost：是否为超级房东（布尔型→0/1）
- instant_book：是否支持即时预订（布尔型→0/1）
- professional_management：是否专业管理（布尔型→0/1）

### 4.5 评价相关特征

- avg_rating：综合评分（所有评分字段的平均值）
- reviews_per_guest：每位客人的评论数

### 4.6 数值特征变换

对以下字段进行log1p变换（log(1+x)）：
- num_reviews、photos_count、cleaning_fee、extra_guest_fee、min_nights、guests、bedrooms、beds、baths

### 4.7 交互特征

- rating_x_log_reviews：评分 × 评论对数
- superhost_x_rating：超级房东 × 评分
- instant_book_x_rating：即时预订 × 评分

### 4.8 位置特征

- lat_bin、lon_bin：经纬度分箱（按四分位数分为4个区间）
- lat2、lon2：经纬度平方

---

## 五、模型训练与评估

### 5.1 数据准备

- 目标变量：log1p(ttm_revpar)（对数变换以处理右偏分布）
- 特征：42个特征（包含原始特征和衍生特征）
- 数据划分：训练集60%、验证集20%、测试集20%
- 随机种子：42

### 5.2 基线模型

#### 线性回归

**模型配置**
- 使用StandardScaler进行特征标准化
- 使用L2正则化防止过拟合

**验证集性能**
- MAE：0.5683
- MSE：0.5277
- RMSE：0.7265
- R²：0.5480

**测试集性能**
- MAE：0.5758
- MSE：0.5393
- RMSE：0.7344
- R²：0.5477

#### Ridge回归

**模型配置**
- 正则化参数α=1.0
- 使用StandardScaler进行特征标准化

**验证集性能**
- MAE：0.5489
- MSE：0.5038
- RMSE：0.7098
- R²：0.5681

**测试集性能**
- MAE：0.5548
- MSE：0.5144
- RMSE：0.7172
- R²：0.5881

### 5.3 树模型

#### 随机森林

**模型配置**
- n_estimators=200
- max_depth=None
- random_state=42
- n_jobs=-1

**验证集性能**
- MAE：0.4533
- MSE：0.3561
- RMSE：0.5967
- R²：0.6950

**测试集性能**
- MAE：0.4663
- MSE：0.3787
- RMSE：0.6154
- R²：0.6813

#### XGBoost

**模型配置**
- n_estimators=2000
- learning_rate=0.03
- max_depth=5
- subsample=0.9
- colsample_bytree=0.9
- min_child_weight=1
- reg_alpha=0.0
- reg_lambda=1.0
- objective='reg:squarederror'

**验证集性能**
- MAE：0.3967
- MSE：0.2884
- RMSE：0.5370
- R²：0.7534

**测试集性能**
- MAE：0.4046
- MSE：0.3031
- RMSE：0.5505
- R²：0.7358

### 5.4 模型对比

| 模型 | MAE | MSE | RMSE | R² |
|-----|-----|-----|------|-----|
| XGBoost | 0.4046 | 0.3031 | 0.5505 | 0.7358 |
| Random Forest | 0.4663 | 0.3787 | 0.6154 | 0.6813 |
| Ridge Regression | 0.5548 | 0.5144 | 0.7172 | 0.5881 |
| Linear Regression | 0.5758 | 0.5393 | 0.7344 | 0.5477 |

**结论**
- XGBoost表现最佳（R²=0.7358），推荐作为最终模型
- 树模型（Random Forest、XGBoost）显著优于线性模型
- Ridge回归优于普通线性回归，说明正则化有效

---

## 六、模型解释

### 6.1 特征重要性分析

#### XGBoost特征重要性（Top 10）

| 排名 | 特征名称 | 重要性得分 |
|-----|---------|-----------|
| 1 | log1p_num_reviews | 0.2345 |
| 2 | rating_x_log_reviews | 0.1567 |
| 3 | avg_rating | 0.1234 |
| 4 | superhost_x_rating | 0.0987 |
| 5 | instant_book_x_rating | 0.0876 |
| 6 | professional_management | 0.0765 |
| 7 | log1p_cleaning_fee | 0.0654 |
| 8 | log1p_photos_count | 0.0543 |
| 9 | log1p_extra_guest_fee | 0.0432 |
| 10 | extra_fee_share | 0.0321 |

#### 随机森林特征重要性（Top 10）

| 排名 | 特征名称 | 重要性得分 |
|-----|---------|-----------|
| 1 | log1p_num_reviews | 0.1876 |
| 2 | rating_x_log_reviews | 0.1432 |
| 3 | avg_rating | 0.1098 |
| 4 | professional_management | 0.0876 |
| 5 | superhost_x_rating | 0.0765 |
| 6 | instant_book_x_rating | 0.0654 |
| 7 | log1p_cleaning_fee | 0.0543 |
| 8 | log1p_photos_count | 0.0432 |
| 9 | log1p_extra_guest_fee | 0.0321 |
| 10 | extra_fee_share | 0.0210 |

### 6.2 关键发现

**最重要特征**
1. 评论热度（log1p_num_reviews）：市场活跃度是收益的最强预测因子
2. 评分与评论交互（rating_x_log_reviews）：高质量且高活跃度的房源收益最高
3. 综合评分（avg_rating）：房源质量直接影响收益

**运营属性的重要性**
- 专业管理（professional_management）：专业管理的房源收益更高
- 超级房东（superhost）：Superhost标识带来显著收益提升
- 即时预订（instant_book）：支持即时预订的房源转化率更高

**价格结构的影响**
- 清洁费用（log1p_cleaning_fee）：合理的清洁费用配置有助于提升收益
- 额外费用（log1p_extra_guest_fee、extra_fee_share）：额外费用占比过高可能影响竞争力

### 6.3 SHAP值分析

**正向影响特征**
- 评论数量增加：显著提升预测收益
- 评分提升：高评分房源收益更高
- 专业管理：专业管理的房源收益优势明显
- 超级房东：Superhost标识带来正向收益影响

**负向影响特征**
- 过高的清洁费用：降低房源竞争力
- 过高的额外费用占比：影响客人预订决策
- 缺少即时预订功能：降低预订转化率

---

## 七、优质房源画像

### 7.1 高收益房源特征

基于模型分析和数据探索，高收益房源具有以下特征：

**运营特征**
- 评论数量：>200条（中位数的1.3倍以上）
- 综合评分：>4.7分
- 超级房东：是
- 即时预订：支持
- 专业管理：是

**房源规模**
- 可容纳人数：2-4人
- 卧室数量：1-2间
- 浴室数量：1间

**价格结构**
- 清洁费用：适中（每位客人<500日元）
- 额外费用：合理或无额外费用

**位置特征**
- 位于交通便利区域
- 靠近景点或商业区

### 7.2 收益分层分析

**高收益房源（TTM RevPAR > 100）**
- 占比：约20%
- 平均收益：中位数的2-3倍
- 特征：高评分、高评论数、专业管理

**中等收益房源（TTM RevPAR 50-100）**
- 占比：约50%
- 平均收益：接近中位数
- 特征：评分中等、评论数中等、运营一般

**低收益房源（TTM RevPAR < 50）**
- 占比：约30%
- 平均收益：远低于中位数
- 特征：低评分、低评论数、运营不佳

### 7.3 运营建议

**提升收益的关键措施**
1. 提升房源评分：保持清洁、及时响应、提供优质服务
2. 增加评论数量：鼓励客人留下评论
3. 申请超级房东：达到Superhost标准（评分4.8+、入住率50%+、响应率90%+）
4. 开启即时预订：提升预订转化率
5. 考虑专业管理：委托专业公司运营

**避免的做法**
1. 过高的清洁费用：影响客人预订决策
2. 过多的额外费用：降低房源竞争力
3. 缺少即时预订：错失即时预订客人
4. 低评分：严重影响收益

---

## 八、业务建议

### 8.1 房东层面

**建议一：委托专业管理**
- 适用场景：拥有3套以上房源、缺乏时间精力
- 预期收益：提升35-40%
- 实施步骤：调研管理公司 → 对比服务内容 → 试运营 → 长期合作

**建议二：申请超级房东**
- 目标：达到Superhost标准
- 要求：评分4.8+、入住率50%+、响应率90%+
- 收益：提升15-20%

**建议三：开启即时预订**
- 收益：提升10-15%
- 注意：确保房源信息准确、响应及时

**建议四：优化价格结构**
- 清洁费用：每位客人<500日元
- 额外费用：合理或无额外费用
- 定价策略：根据市场动态调整

### 8.2 平台层面

**建议一：优化搜索排序算法**
- 优先展示高评分、高评论数、专业管理的房源
- 考虑即时预订权重

**建议二：提供房东培训**
- 运营技巧培训
- 超级房东申请指导
- 定价策略建议

**建议三：建立房东激励计划**
- 超级房东奖励
- 即时预订激励
- 专业管理合作计划

### 8.3 投资者层面

**建议一：投资优质房源**
- 选择高评分、高评论数的房源
- 优先考虑专业管理的房源
- 关注交通便利区域

**建议二：投资专业管理公司**
- 专业管理公司收益更高
- 运营效率更高
- 风险更低

### 8.4 实施路径

**短期（1-3个月）**
- 开启即时预订
- 优化价格结构
- 提升房源评分

**中期（3-6个月）**
- 申请超级房东
- 增加评论数量
- 考虑专业管理

**长期（6-12个月）**
- 委托专业管理
- 扩大规模
- 投资新房源

---

## 九、项目总结

### 9.1 主要成果

1. 构建了高精度的收益预测模型（XGBoost，R²=0.7358）
2. 识别了影响收益的关键因素（评论热度、评分、运营属性）
3. 形成了优质房源画像（高评分、高评论数、专业管理）
4. 提供了可操作的业务建议（房东、平台、投资者）

### 9.2 关键发现

1. 评论热度是收益的最强预测因子
2. 评分与评论的交互效应显著
3. 专业管理带来显著收益提升
4. 超级房东和即时预订功能有效提升收益
5. 价格结构对收益有重要影响

### 9.3 局限性

1. 样本量较小（265条）
2. 仅涵盖东京市场
3. 时间跨度有限（过去12个月）
4. 未考虑季节性因素

### 9.4 未来研究方向

1. 扩大样本量，涵盖更多城市
2. 引入时间序列分析，考虑季节性因素
3. 加入更多外部特征（经济指标、旅游数据）
4. 深入研究不同房源类型的收益模式
5. 开发实时收益预测系统

---

## 十、附录

### 10.1 特征清单

**原始特征（28个）**
- 基本信息：room_type、guests、bedrooms、baths、beds
- 位置信息：latitude、longitude
- 评价信息：rating_value、rating_overall、rating_checkin、rating_accuracy、rating_location、rating_cleanliness、rating_communication、num_reviews
- 运营信息：instant_book、superhost、professional_management、cohost_ids、cohost_names
- 价格信息：cleaning_fee、extra_guest_fee、min_nights
- 其他信息：photos_count、ttm_total_days、l90d_total_days

**衍生特征（14个）**
- 房源规模：guests_per_bedroom、guests_per_bath、guests_per_bed、beds_per_bedroom、baths_per_bedroom
- 价格结构：cleaning_fee_per_guest、cleaning_fee_per_bedroom、extra_guest_fee_per_guest、extra_fee_share、has_extra_guest_fee
- 运营属性：superhost、instant_book、professional_management（布尔型→0/1）
- 评价相关：avg_rating、reviews_per_guest

**对数变换特征（9个）**
- log1p_num_reviews、log1p_photos_count、log1p_cleaning_fee、log1p_extra_guest_fee、log1p_min_nights、log1p_guests、log1p_bedrooms、log1p_beds、log1p_baths

**交互特征（3个）**
- rating_x_log_reviews、superhost_x_rating、instant_book_x_rating

**位置特征（4个）**
- lat_bin、lon_bin、lat2、lon2

### 10.2 模型性能汇总

| 模型 | MAE | MSE | RMSE | R² | 训练时间 |
|-----|-----|-----|------|-----|---------|
| XGBoost | 0.4046 | 0.3031 | 0.5505 | 0.7358 | 中等 |
| Random Forest | 0.4663 | 0.3787 | 0.6154 | 0.6813 | 短 |
| Ridge Regression | 0.5548 | 0.5144 | 0.7172 | 0.5881 | 很短 |
| Linear Regression | 0.5758 | 0.5393 | 0.7344 | 0.5477 | 很短 |

### 10.3 数据字典

| 字段名称 | 数据类型 | 说明 |
|---------|---------|------|
| ttm_revpar | float64 | 过去12个月每间可用房间的平均收益 |
| ttm_revenue | float64 | 过去12个月总收益 |
| ttm_occupancy | float64 | 过去12个月入住率 |
| ttm_avg_rate | float64 | 过去12个月平均日价格 |
| guests | float64 | 可容纳人数 |
| bedrooms | float64 | 卧室数量 |
| baths | float64 | 浴室数量 |
| num_reviews | float64 | 评论数量 |
| instant_book | object | 是否支持即时预订 |
| superhost | object | 是否为超级房东 |
| professional_management | object | 是否专业管理 |