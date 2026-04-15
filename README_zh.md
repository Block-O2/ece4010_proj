# 基于收容所入所数据预测宠物是否会被快速领养

这是一个面向机器学习课程项目的 baseline 仓库，目标是基于 Austin Animal Center 的公开数据，预测一只进入收容所的狗或猫是否会在入所后 30 天内被领养。

英文说明见 [README.md](/Users/hankli/Desktop/coding/ece4010_proj/README.md)。

## 任务定义

主任务：

- 仅保留 `Dog` 和 `Cat`
- 使用可复现的时间顺序规则配对 intake 和 outcome 记录
- 定义 `fast_adoption_30d = 1` 当且仅当：
  - `Outcome Type == "Adoption"`
  - `outcome_datetime - intake_datetime <= 30 days`
- 其余情况记为 `0`

备用任务：

- `adoption = 1` 如果 `Outcome Type == "Adoption"`
- 否则 `adoption = 0`

仓库中保留这两个标签，但默认训练目标是 `fast_adoption_30d`。

## 数据来源

主要数据来自 Austin Animal Center 的两张公开表：

1. Austin Animal Center Intakes
2. Austin Animal Center Outcomes

默认下载地址写在 [src/data_loading.py](/Users/hankli/Desktop/coding/ece4010_proj/src/data_loading.py) 中。

## 原始数据放置方式

代码会优先尝试从官方公开 URL 下载 CSV。如果下载失败，就从本地 `data/raw/` 目录读取。

请将原始 CSV 放到 [data/raw](/Users/hankli/Desktop/coding/ece4010_proj/data/raw)：

- `data/raw/austin_animal_center_intakes.csv`
- `data/raw/austin_animal_center_outcomes.csv`

也兼容以下文件名：

- `intakes.csv`
- `outcomes.csv`
- `Austin_Animal_Center_Intakes.csv`
- `Austin_Animal_Center_Outcomes.csv`
- `Austin_Animal_Center_Intakes__10_01_2013_to_05_05_2025_.csv`
- `Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv`

不依赖 Kaggle token。

## 目录结构

```text
ece4010_proj/
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   `-- eda.ipynb
|-- outputs/
|   |-- figures/
|   |-- metrics/
|   |-- models/
|   `-- results_summary.md
|-- src/
|   |-- data_loading.py
|   |-- dataset_builder.py
|   |-- evaluate.py
|   |-- feature_engineering.py
|   |-- modeling.py
|   |-- preprocessing.py
|   |-- train.py
|   `-- utils.py
|-- matching_strategy.md
|-- proposal_outline.md
|-- slides_outline.md
|-- README.md
`-- requirements.txt
```

## 特征设计

模型输入严格限制为 intake 时刻可获得的信息，避免数据泄漏。

当前输入特征包括：

- `Animal Type`
- `Breed`
- `Color`
- `Sex upon Intake`
- `Age upon Intake`
- `Intake Type`
- `Intake Condition`
- 从 intake datetime 派生的时间特征：
  - `month`
  - `weekday`
  - `season`
  - `year`

说明：

- outcome 侧字段只用于构造标签和做分析，不进入模型输入
- 年龄会统一转换为天数
- `breed` 和 `color` 这类高基数字段会做低频合并，低频类别归入 `Other`

## 模型设置

- `logreg`：Logistic Regression，基础 baseline
- `rf`：Random Forest，主模型
- `xgb`：可选 XGBoost，仅在额外扩展时使用

## 运行方式

推荐先创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

构建数据集：

```bash
python src/dataset_builder.py
```

训练单个模型：

```bash
python src/train.py --model logreg
python src/train.py --model rf
```

运行完整评估：

```bash
python src/evaluate.py
```

如果不想自动下载：

```bash
python src/evaluate.py --no-download
```

切换到备用标签：

```bash
python src/evaluate.py --target adoption
```

使用 time-based split：

```bash
python src/evaluate.py --split-method time
```

低资源试运行：

```bash
python src/evaluate.py --no-download --max-rows 50000
```

Windows 下建议命令：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.check_data --no-download
python -m src.evaluate --no-download --max-rows 50000
```

## 已验证运行记录

本仓库已于 2026-04-15 在 Windows 环境完成完整验证。

已完成的实际工作：

- 在 Windows 上成功克隆仓库
- 创建本地 `.venv` 并安装依赖
- 自动下载 Austin Animal Center 官方 CSV
- 生成 `data/processed/paired_dataset.csv` 和 `data/processed/modeling_dataset.csv`
- 生成 `data/processed/dataset_report.json` 和 `data/processed/data_audit.md`
- 完成主任务 baseline 的全量评估
- 额外完成 `adoption` 备用任务实验和 `time-based split` 对照实验
- 保存最新 metrics、混淆矩阵、特征重要性图和模型文件

兼容性修复：

- `RareCategoryGrouper` 已实现 `get_feature_names_out`，解决了新版 scikit-learn 下随机森林特征重要性导出报错的问题
- 去掉了 Logistic Regression 中无效的 `n_jobs` 参数，避免新版 scikit-learn warning

## 最新主结果

默认任务 `fast_adoption_30d`，使用 stratified split 的全量结果：

- 数据行数：`163932`
- `fast_adoption_30d` 正类比例：`0.3294`
- `adoption` 正类比例：`0.5036`
- intake / outcome 配对后的 outcome 匹配率：`0.99`
- 时间错配率：`0.0043`

测试集结果：

- `logreg`：Accuracy `0.6463`，F1-score `0.5828`，ROC-AUC `0.7313`
- `rf`：Accuracy `0.7289`，F1-score `0.6406`，ROC-AUC `0.8149`

最新随机森林最重要的特征：

- `age_upon_intake_days`
- `intake_year`
- `color`
- `sex_upon_intake`
- `intake_type`

## 额外对照实验

为了让课程展示更完整，仓库还补做了两组对照实验：

1. 备用标签 `adoption`，使用 stratified split
2. 主任务 `fast_adoption_30d`，使用 time-based split

结果如下：

- `adoption` + `logreg`：Accuracy `0.6648`，F1-score `0.6892`，ROC-AUC `0.7270`
- `adoption` + `rf`：Accuracy `0.7156`，F1-score `0.7268`，ROC-AUC `0.7961`
- `fast_adoption_30d` + time split + `logreg`：Accuracy `0.6165`，F1-score `0.5498`，ROC-AUC `0.6724`
- `fast_adoption_30d` + time split + `rf`：Accuracy `0.6983`，F1-score `0.5773`，ROC-AUC `0.7654`

这些结果说明：

- `adoption` 比 `fast_adoption_30d` 更容易预测
- Random Forest 在所有测试设置下都优于 Logistic Regression
- 使用时间切分后表现下降，说明这个对照实验更接近真实“预测未来数据”的场景

## 输出内容

处理后的数据：

- [data/processed/paired_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/paired_dataset.csv)
- [data/processed/modeling_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/modeling_dataset.csv)
- [data/processed/dataset_report.json](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/dataset_report.json)
- `data/processed/data_audit.md`

训练与评估输出：

- `outputs/models/*.joblib`
- `outputs/metrics/*_metrics.json`
- `outputs/metrics/model_comparison.json`
- `outputs/metrics/rf_feature_importance.csv`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/figures/rf_feature_importance.png`
- [outputs/results_summary.md](/Users/hankli/Desktop/coding/ece4010_proj/outputs/results_summary.md)
- [outputs/extended_experiments.md](/Users/hankli/Desktop/coding/ece4010_proj/outputs/extended_experiments.md)

## 配对规则

由于同一个 `Animal ID` 可能在 intake 和 outcome 表中重复出现，仓库采用一个简单、可解释、可复现的规则：

- 先按 `Animal ID` 和时间排序
- 对每个动物分别给 intake 和 outcome 记录编号 `0, 1, 2, ...`
- 将同一个 `Animal ID` 下第 `k` 条 intake 与第 `k` 条 outcome 配对

这个策略的目的：

- 避免 many-to-many merge 导致的重复配对
- 保持规则简单、稳定、易于汇报
- 让项目重心放在课程项目级别的 ML pipeline，而不是复杂事件重建

更详细说明见 [matching_strategy.md](/Users/hankli/Desktop/coding/ece4010_proj/matching_strategy.md)。

## 当前限制

- 官方开放数据接口偶尔会阻止脚本下载，因此代码仍然支持手动放置 CSV
- 重复 `Animal ID` 的处理是启发式 chronological pairing，不是严格事件重建
- `breed` 和 `color` 等高基数字段做了低频合并
- 该仓库是课程 baseline，不是生产级收容所决策系统

## 已完成工作

- 完成项目结构、核心 Python 模块、EDA notebook 骨架、proposal/slides 文档骨架
- 完成数据下载与手动 CSV 回退机制
- 完成列名标准化、年龄处理、时间解析、标签构造和特征工程
- 完成 `fast_adoption_30d` 主任务 baseline
- 完成 `adoption` fallback 对照实验
- 完成 time-based split 对照实验
- 完成 Windows 本地全流程验证
- 完成 README 和结果记录同步更新

## 下一步建议

- 从现有图和指标中选出最适合展示的 2 到 3 张图放进 slides
- 决定 `adoption` 是作为 backup task 还是作为 comparison experiment 展示
- 如果课程还需要更多分析，可以在 notebook 里补一些 EDA 图
- 如果后续还想继续提升，可再尝试 XGBoost 或参数调优
