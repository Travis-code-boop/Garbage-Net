# 垃圾分类建模实验记录

本项目已经完成写报告前的全部建模工作：数据整理、特征提取、单特征基线、特征融合广筛、强模型精调、LightGBM 消融与类别不平衡处理、stacking 集成，以及最终 test 集评价。

当前最终结论：

```text
最终模型：final_stack_small_plus_lgbm_anchor
最终 test accuracy：0.8238
最终 test macro-F1：0.8097
最终 test weighted-F1：0.8228
最终 test 少数类 recall 均值：0.7420
```

最终模型是一个 stacking 集成：5 个小模型 base learner 加 1 个 LightGBM 强基线 anchor，再用 Logistic Regression 作为 meta learner。

## 1. 实验协议

数据固定划分为 train / validation / test = 70% / 15% / 15%。

| Split | 样本数 |
|---|---:|
| train | 8581 |
| validation | 1839 |
| test | 1839 |

类别分布：

| 类别 | 总数 | train | validation | test |
|---|---:|---:|---:|---:|
| battery | 756 | 530 | 113 | 113 |
| biological | 699 | 489 | 105 | 105 |
| cardboard | 1411 | 987 | 212 | 212 |
| clothes | 1892 | 1324 | 284 | 284 |
| glass | 1736 | 1216 | 260 | 260 |
| metal | 930 | 650 | 140 | 140 |
| paper | 1336 | 936 | 200 | 200 |
| plastic | 1597 | 1117 | 240 | 240 |
| shoes | 1449 | 1015 | 217 | 217 |
| trash | 453 | 317 | 68 | 68 |

评价协议：

- 主要指标：macro-F1。
- 辅助指标：accuracy、macro precision、macro recall、weighted-F1。
- 少数类重点观察：`trash`、`battery`、`biological` 的 recall 均值。
- validation 用于模型选择、调参、消融和 stacking 方案选择。
- test 在最终模型封存后只使用一次，用于最终评价和错误分析。
- 非线性核 SVM 不纳入本项目后续研究；已有 Stanford / TrashNet 相关研究做过 SIFT/BoF + RBF-SVM，本项目后续 SVM 均采用线性核路线。

运行环境：

```bash
/Users/wanzhaoning/.local/bin/python3.12
```

本地依赖在 `vendor/python`。不要直接用系统默认 `python3`，它可能指向 Python 3.13 并导致 numpy / sklearn 扩展不匹配。

## 2. 特征工程

图像统一使用 `standardized_256` 版本。已经提取并缓存以下手工特征：

| 编号 | 特征 | 维度 | 用途与说明 |
|---|---|---:|---|
| F0 | 灰度像素 | 1024 | 像素基线，只作参照 |
| F1 | Color | 1039 | BGR/HSV 直方图、颜色统计，少数类 recall 很有帮助 |
| F2 | Texture | 30 | GLCM + LBP，低维稳定 |
| F3 | HOG | 1764 | 梯度方向特征，单特征最强 |
| F4 | Shape | 12 | 形状/几何/Hu 矩，单独弱但全量中略有正贡献 |
| F5 | GIST | 64 | 全局结构，低维互补 |
| F6 | SIFT+ORB BoF | 256 | SIFT 128 词 + ORB 128 词，局部词袋，提升最终模型和少数类表现 |

关键实现选择：

- Color 特征中 skewness / kurtosis 不做人为截断，只做零方差保护。
- Shape 特征最初最大连通域实现太慢，改为基于完整前景 mask 的快速形体近似，保持 12 维。
- GIST 由慢速纯 Python 卷积改为 `sliding_window_view + tensordot`。
- BoF 使用 SIFT 和 ORB 两套词典，各 128 词，再拼接为 256 维。

## 3. 阶段二：单特征基线

先用 7 个单特征组和 4 类传统模型做固定参数基线，共 28 组实验。

固定参数：

| 模型 | 参数 |
|---|---|
| Logistic Regression | StandardScaler + `LogisticRegression(C=1, solver=lbfgs, max_iter=1000)` |
| KNN | StandardScaler + `KNeighborsClassifier(n_neighbors=5, weights=distance, metric=euclidean)` |
| Linear SVM | StandardScaler + `SGDClassifier(loss=hinge, alpha=0.0001, max_iter=1000)` |
| Decision Tree | `DecisionTreeClassifier(random_state=42)` |

固定参数阶段前列结果：

| 排名 | 特征 | 模型 | Val Macro-F1 | 少数类 recall 均值 |
|---:|---|---|---:|---:|
| 1 | F3 HOG | KNN | 0.5163 | 0.4344 |
| 2 | F5 GIST | KNN | 0.5037 | 0.4984 |
| 3 | F2 Texture | Logistic Regression | 0.4956 | 0.5555 |
| 4 | F2 Texture | KNN | 0.4942 | 0.5174 |
| 5 | F1 Color | KNN | 0.4816 | 0.5470 |

固定参数结果说明 HOG 是最强单特征，但 Color、Texture、GIST 在少数类和互补性上有价值。

### 3.1 单特征补充调参

没有对全部组合做穷举搜索，而是根据固定参数结果筛选 14 个候选再调参。

筛选原则：

- 保留固定参数 macro-F1 前列组合。
- 保留少数类 recall 贡献明显的组合。
- 额外保留 BoF+KNN 作为局部词袋互补性检查。
- 舍弃明显弱且耗时不划算的组合。
- 高维 KNN 使用 train 内部 holdout/subsample，低维组合使用 3-fold CV。
- SVM 搜索空间曾因耗时过长被收窄为线性 hinge + l2，只调核心参数。

单特征调参后的最佳结果：

| 特征 | 最佳模型与参数 | Val Macro-F1 | 少数类 recall 均值 |
|---|---|---:|---:|
| F3 HOG | Normalizer + KNN, k=9, manhattan, distance | 0.5612 | 0.5208 |
| F2 Texture | RobustScaler + KNN, k=11, euclidean, distance | 0.5024 | 0.5081 |
| F1 Color | MinMaxScaler + KNN, k=11, cosine, distance | 0.4980 | 0.5620 |
| F5 GIST | StandardScaler + KNN, k=3, manhattan, distance | 0.4942 | 0.4969 |
| F6 BoF | RobustScaler + KNN, k=31, cosine, distance | 0.4324 | 0.4957 |
| F0 Gray | RobustScaler + KNN, k=1, manhattan, uniform | 0.4317 | 0.3524 |

阶段二结论：

- `F3_hog` 是融合起点。
- `F1_color` 对少数类最重要，尤其 biological。
- `F2_texture` 低维稳定，必须进入融合。
- `F5_gist` 是低维全局结构补充。
- `F6_bof` 单独一般，但对 battery / biological 有帮助，应在融合中验证。
- `F4_shape` 单特征弱，但后续可作为低成本消融对象。

## 4. 阶段三：自适应特征融合与模型广筛

阶段三从 HOG 出发，根据阶段二结果逐步加入 Color、Texture、GIST、BoF、Shape，同时保留 C10/C11 参考组合。总共完成：

```text
14 个特征组合 × 7 个模型 = 98 组实验
```

广筛模型与核心参数：

| 模型 | 核心设置 |
|---|---|
| Logistic Regression | StandardScaler，`C=0.3`，`class_weight=balanced` |
| KNN | Normalizer，`n_neighbors=9`，manhattan，distance |
| Linear SVM | StandardScaler，`SGDClassifier(loss=hinge, alpha=0.0003, average=True, class_weight=balanced)` |
| Decision Tree | `max_depth=16`，`max_features=0.6`，`min_samples_leaf=10`，`class_weight=balanced` |
| Random Forest | `n_estimators=260`，`max_features=sqrt`，`min_samples_leaf=2`，`class_weight=balanced_subsample` |
| XGBoost | `n_estimators=360`，`max_depth=5`，`learning_rate=0.055`，`subsample=0.85`，`colsample_bytree=0.75`，`reg_lambda=2.0`，balanced sample weight |
| LightGBM | `n_estimators=460`，`num_leaves=31`，`learning_rate=0.05`，`subsample=0.85`，`colsample_bytree=0.75`，`reg_lambda=1.0`，balanced sample weight |

阶段三最佳结果：

| 排名 | 特征组合 | 模型 | Val Macro-F1 | 少数类 recall 均值 |
|---:|---|---|---:|---:|
| 1 | F1+F2+F4+F5+F3+F6 | LightGBM | 0.8096 | 0.8158 |
| 2 | F3+F1+F2+F5+F6 | LightGBM | 0.8071 | 0.8109 |
| 3 | F1+F2+F4+F5+F3 | LightGBM | 0.8020 | 0.7961 |
| 4 | F3+F1+F2 | LightGBM | 0.8012 | 0.8006 |
| 5 | F3+F1+F2+F5+F6 | XGBoost | 0.7980 | 0.8140 |

按模型族的最佳表现：

| 模型族 | 最佳特征组合 | Val Macro-F1 | 少数类 recall 均值 |
|---|---|---:|---:|
| LightGBM | C11 全量手工特征 | 0.8096 | 0.8158 |
| XGBoost | F3+F1+F2+F5+F6 | 0.7980 | 0.8140 |
| Random Forest | F1+F2+F5 | 0.7049 | 0.7604 |
| Linear SVM | C11 全量手工特征 | 0.6885 | 0.7008 |
| Logistic Regression | F3+F1+F2+F5+F6 | 0.6688 | 0.7118 |
| KNN | F3+F5 | 0.5787 | 0.5739 |
| Decision Tree | C10 no BoF | 0.4705 | 0.5864 |

阶段三结论：

- LightGBM 明显领先，是主模型。
- XGBoost 次强，但整体不如 LightGBM。
- `F1_color` 是 HOG 后最关键增益；`F3+F1+F2` 已达到 0.8012。
- BoF 对最终模型有实质贡献，尤其对少数类 recall。
- Shape 单独弱，但加入全量组合后仍有小幅正贡献。
- 低维组合 `F1+F2+F5` 只有 1133 维，macro-F1 0.7845，少数类 recall 0.8104，可作为解释性对照。

## 5. 阶段四：强模型精调

第一轮精调只针对阶段三短名单中的 LightGBM / XGBoost 强候选。搜索只在 train 内部 holdout 上进行；validation 只用于最终统一评价。

搜索设置：

- LightGBM：每个候选保留阶段三基准参数，再随机采样 5 组参数。
- XGBoost：每个候选保留阶段三基准参数，再随机采样 3 组参数。
- train 内部搜索使用最多 4200 个 inner-train 样本和 1400 个 inner-validation 样本。
- XGBoost trial 内使用 early stopping；最终 full-train refit 固定 best iteration。

第一轮精调结果：

| 候选 | 模型 | Val Macro-F1 | 少数类 recall 均值 | 结论 |
|---|---|---:|---:|---|
| C11 full | LightGBM tuned | 0.8036 | 0.8272 | 少数类提高，但 macro-F1 低于阶段三 |
| no Shape | LightGBM tuned | 0.8000 | 0.8302 | 本轮少数类 recall 最高 |
| C11 full | XGBoost tuned | 0.7996 | 0.7795 | XGB macro 小幅改善，但少数类下降 |
| no Shape | XGBoost tuned | 0.7959 | 0.7827 | 不优于 LightGBM |

结论：第一轮随机精调没有刷新阶段三最高 macro-F1。balanced sample weight 倾向提高少数类 recall，但会牺牲总体 macro-F1。

## 6. 阶段四：LightGBM 消融与类别不平衡处理

此阶段固定阶段三 LightGBM 参数：

```text
n_estimators=460
num_leaves=31
learning_rate=0.05
subsample=0.85
colsample_bytree=0.75
reg_lambda=1.0
```

比较三种 sample weight：

- none：不加权。
- balanced：sklearn balanced sample weight。
- sqrt_balanced：balanced 权重开方后再归一化，作为更温和的不平衡处理。

消融结果：

| 消融组合 | 权重 | Val Macro-F1 | 少数类 recall 均值 | 相对 C11 balanced |
|---|---|---:|---:|---:|
| C11 full | balanced | 0.8096 | 0.8158 | 0.0000 |
| no Shape | balanced | 0.8071 | 0.8109 | -0.0025 |
| no GIST | balanced | 0.8054 | 0.8221 | -0.0043 |
| no BoF | balanced | 0.8020 | 0.7961 | -0.0077 |
| F3+F1+F2 | balanced | 0.8012 | 0.8006 | -0.0085 |
| F1+F2+F5 | balanced | 0.7845 | 0.8104 | -0.0252 |

不平衡处理结果：

| 组合 | 权重 | Val Macro-F1 | 少数类 recall 均值 |
|---|---|---:|---:|
| C11 full | sqrt_balanced | 0.8131 | 0.8158 |
| C11 full | balanced | 0.8096 | 0.8158 |
| C11 full | none | 0.8086 | 0.7888 |
| no Shape | sqrt_balanced | 0.8109 | 0.8035 |
| no Shape | balanced | 0.8071 | 0.8109 |

结论：

- `sqrt_balanced` 在 C11 full 上把 macro-F1 提升到 0.8131，同时保持少数类 recall。
- BoF 是最重要的附加特征之一；去掉 BoF 后 macro-F1 和少数类 recall 都明显下降。
- Shape 贡献小但为正；去掉 Shape 只下降约 0.0025。
- GIST 对 macro-F1 有帮助，但 no-GIST 的少数类 recall 更高。

## 7. 阶段四：stacking 集成

在单模型 LightGBM 已经接近瓶颈后，构建基于小模型和强基线 anchor 的 stacking。训练 meta learner 时使用 train split 的 5-fold OOF 预测；validation 只用于最终评价。

Base learner 设计：

| Base learner | 特征 | 输出 | 参数 |
|---|---|---|---|
| `lr_c11_balanced` | C11 full | proba | `LogisticRegression(C=0.8, max_iter=700, class_weight=balanced)` |
| `sgd_svm_hct_balanced` | F3+F1+F2 | decision | `SGDClassifier(loss=hinge, alpha=8e-5, max_iter=3000, tol=0.001, class_weight=balanced)` |
| `rf_lowdim_balanced` | F1+F2+F5 | proba | `RandomForest(n_estimators=240, max_features=sqrt, min_samples_leaf=2, class_weight=balanced_subsample)` |
| `knn_compact_gist_texture_shape` | F5+F2+F4 | proba | `KNN(k=11, weights=distance, p=2)` |
| `lgbm_small_hct_sqrt` | F3+F1+F2 | proba | `LGBM(220 trees, lr=0.06, num_leaves=31, sqrt_balanced)` |
| `lgbm_anchor_c11_sqrt` | C11 full | proba | `LGBM(460 trees, lr=0.05, num_leaves=31, sqrt_balanced)` |

Meta learner：

```text
LogisticRegression(C=0.1, weight_mode=none)
```

Validation 上的 stacking 结果：

| Ensemble | Val Macro-F1 | 少数类 recall 均值 | 相对单模型 0.8131 |
|---|---:|---:|---:|
| small + LGBM anchor stacking | 0.8178 | 0.8151 | +0.0047 |
| small-only stacking | 0.7981 | 0.8129 | -0.0151 |
| probability average + anchor | 0.8009 | 0.7988 | -0.0123 |
| small probability average | 0.7686 | 0.7817 | -0.0445 |

结论：

- 纯小模型 stacking 不能超过强 LightGBM，但保留了较好的少数类 recall。
- 简单概率平均明显弱于 stacking。
- 加入 LGBM anchor 后，stacking 成为 validation 最优，macro-F1 从 0.8131 提升到 0.8178。

## 8. 最终 test 评价

最终模型已经在 validation 阶段封存为：

```text
final_stack_small_plus_lgbm_anchor
```

最终评价流程：

1. 合并 train + validation，作为最终训练数据，共 10420 张图像。
2. 在 train+validation 上重新做 5-fold OOF，训练固定 stacking meta learner。
3. 每个 base learner 再用完整 train+validation 重训。
4. 对 test 集预测一次并输出最终指标、per-class 指标、混淆矩阵和错误分析。

最终 test 上的 ensemble 对比：

| 模型 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | 少数类 recall 均值 |
|---|---:|---:|---:|---:|
| final stacking + anchor | 0.8238 | 0.8097 | 0.8228 | 0.7420 |
| final small-only stacking | 0.8173 | 0.8036 | 0.8170 | 0.7486 |
| probability average + anchor | 0.8173 | 0.8029 | 0.8159 | 0.7444 |
| small probability average | 0.7868 | 0.7687 | 0.7851 | 0.7147 |

最终 test 上的 base learner 对比：

| Base learner | Test Macro-F1 | Test Accuracy | 少数类 recall 均值 |
|---|---:|---:|---:|
| LGBM C11 anchor | 0.7950 | 0.8097 | 0.7231 |
| LGBM HOG+Color+Texture | 0.7891 | 0.7988 | 0.7410 |
| RandomForest lowdim | 0.6955 | 0.7036 | 0.6739 |
| Logistic Regression C11 | 0.6574 | 0.6770 | 0.6221 |
| Linear SVM | 0.6148 | 0.6362 | 0.5660 |
| KNN compact | 0.5444 | 0.5628 | 0.5198 |

最终 stacking 的 per-class test 指标：

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
| battery | 0.8230 | 0.8230 | 0.8230 |
| biological | 0.8317 | 0.8000 | 0.8155 |
| cardboard | 0.8502 | 0.8302 | 0.8401 |
| clothes | 0.9085 | 0.9437 | 0.9257 |
| glass | 0.8141 | 0.8423 | 0.8280 |
| metal | 0.7333 | 0.7857 | 0.7586 |
| paper | 0.8079 | 0.8200 | 0.8139 |
| plastic | 0.7686 | 0.7333 | 0.7505 |
| shoes | 0.8214 | 0.8479 | 0.8345 |
| trash | 0.8542 | 0.6029 | 0.7069 |

最终错误分析：

| 真实类别 | 预测类别 | 错误数 |
|---|---|---:|
| plastic | glass | 24 |
| glass | plastic | 15 |
| cardboard | plastic | 13 |
| glass | shoes | 9 |
| paper | shoes | 9 |
| plastic | cardboard | 8 |
| plastic | paper | 8 |
| plastic | metal | 8 |
| shoes | clothes | 8 |
| shoes | paper | 8 |
| trash | metal | 8 |
| battery | glass | 7 |

最终 test 结论：

- stacking 在 test 上依然优于单模型 LightGBM anchor：0.8097 vs 0.7950。
- validation 到 test 有正常回落：validation stacking macro-F1 0.8178，test macro-F1 0.8097。
- 小模型本身不强，但可以通过 stacking 提供互补信息。
- 少数类中 battery 和 biological 表现较好，trash recall 只有 0.6029，是最终模型主要短板。
- 最常见错误是 plastic / glass 互相混淆，其次是 cardboard / plastic、paper / shoes、trash / metal。

## 9. 报告素材

报告中建议直接呈现以下图表：

- 最终模型对比图：`final_model_comparison.svg`
- 最终 stacking 混淆矩阵：`final_stack_confusion_matrix.svg`
- 最终 stacking 各类别 F1/Recall：`final_stack_per_class_performance.svg`

报告中建议呈现的核心表格：

1. 数据集 split 与类别分布。
2. F0-F6 特征说明。
3. 阶段二单特征调参最佳结果。
4. 阶段三融合广筛 Top 5。
5. LightGBM 消融和不平衡处理结果。
6. stacking validation 结果。
7. 最终 test 对比结果。
8. 最终 per-class 指标和主要错误对。

报告中的叙事主线可以这样写：

1. 先用固定参数单特征基线了解特征强弱。
2. 再对有前景的单特征模型做轻量调参，避免全组合穷举。
3. 根据调参结果自适应设计融合顺序，而不是直接套用固定 C7-C11。
4. 通过 98 组阶段三实验确认 LightGBM 是主模型，C11 全量特征最强。
5. 阶段四发现盲目随机调参不一定提升，反而 sample weight 和特征消融更有解释价值。
6. `sqrt_balanced` 是最合适的不平衡处理，BoF 有实质贡献，Shape 贡献小但正。
7. 最后用 stacking 学习小模型与强 LightGBM 的互补性，得到最终最佳 test macro-F1。

## 10. 当前状态

已经完成：

- 数据整理与固定 split。
- F0-F6 特征提取。
- 阶段二固定参数单特征基线。
- 阶段二候选筛选与轻量调参。
- 阶段三自适应特征融合和 7 模型广筛。
- 阶段四 LightGBM / XGBoost 第一轮精调。
- 阶段四 LightGBM 消融与不平衡处理。
- 阶段四 stacking 集成。
- 最终 train+validation 重训和 test 集评价。
- 最终混淆矩阵、per-class 指标、错误分析和报告图。

下一步不建议继续围绕 validation 或 test 做模型试错。现在应该进入报告撰写与 notebook 整理阶段。
