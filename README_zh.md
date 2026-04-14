# 基于收容所入所数据预测宠物是否会被快速领养

这是一个面向机器学习课程项目的基线仓库，目标是基于 Austin Animal Center 的公开数据，预测一只进入收容所的狗或猫是否会在入所后 30 天内被领养。

对应英文说明见 [README.md](/Users/hankli/Desktop/coding/ece4010_proj/README.md)。

## 项目目标

本项目重点不是追求最先进模型，而是完成一个：

- 数据公开且容易获取
- 任务定义清晰
- 方法直观
- 技术路线可解释
- 可运行、可展示、适合课程项目 proposal 和后续实现的机器学习基线

## 任务定义

主任务：

- 仅保留 `Dog` 和 `Cat`
- 将 intake 和 outcome 记录进行可解释、可复现的配对
- 定义标签 `fast_adoption_30d = 1` 当且仅当：
  - `Outcome Type == "Adoption"`
  - 且 `outcome_datetime - intake_datetime <= 30 days`
- 其余情况记为 `0`

备用任务：

- `adoption = 1` 如果 `Outcome Type == "Adoption"`
- 否则 `adoption = 0`

仓库中同时保留这两个标签，但默认训练目标是 `fast_adoption_30d`。

## 数据来源

主数据源固定为 Austin Animal Center 的两张公开表：

1. Austin Animal Center Intakes
2. Austin Animal Center Outcomes

默认官方下载地址写在 [src/data_loading.py](/Users/hankli/Desktop/coding/ece4010_proj/src/data_loading.py) 中。

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
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── results_summary.md
├── src/
│   ├── data_loading.py
│   ├── dataset_builder.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── matching_strategy.md
├── proposal_outline.md
├── slides_outline.md
├── README.md
├── README_zh.md
└── requirements.txt
```

## 特征设计

模型输入严格限制为 intake 时刻可获得的信息，避免数据泄漏。

当前使用的输入特征包括：

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
- 年龄会被统一转换为天数
- `breed`、`color` 这类高基数字段会做低频合并，低频类别归入 `Other`

## 模型设置

按你的固定要求实现：

- `logreg`：Logistic Regression，基础 baseline
- `rf`：Random Forest，主模型
- `xgb`：可选 XGBoost，仅在环境和进度允许时启用

## 配对规则

由于同一个 `Animal ID` 可能在 intake 和 outcome 表中重复出现，仓库采用一个简单且可解释的规则：

- 先按 `Animal ID` 和时间排序
- 对每个动物分别给 intake 和 outcome 记录编号 `0, 1, 2, ...`
- 将同一个 `Animal ID` 下第 `k` 条 intake 与第 `k` 条 outcome 配对

这个策略的目的是：

- 避免 many-to-many merge 导致的重复配对
- 保持规则简单、稳定、易于汇报
- 让整个项目重点仍然放在课程项目级别的 ML pipeline，而不是复杂事件重建

更详细的说明见 [matching_strategy.md](/Users/hankli/Desktop/coding/ece4010_proj/matching_strategy.md)。

## 环境安装

推荐使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行方式

### 1. 构建配对后的数据集

```bash
python src/dataset_builder.py
```

如果你已经手动把 CSV 放到了 `data/raw/`，并且不想自动下载：

```bash
python src/dataset_builder.py --no-download
```

### 2. 训练 Logistic Regression

```bash
python src/train.py --model logreg
```

### 3. 训练 Random Forest

```bash
python src/train.py --model rf
```

### 4. 一次性跑完整 baseline 评估

```bash
python src/evaluate.py
```

### 5. 切换到 fallback 标签

```bash
python src/evaluate.py --target adoption
```

### 6. 使用 time-based split

```bash
python src/evaluate.py --split-method time
```

### 7. 在配置较弱的电脑上先跑轻量版本

```bash
python src/evaluate.py --no-download --max-rows 50000
```

## 输出内容

处理后的数据：

- [data/processed/paired_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/paired_dataset.csv)
- [data/processed/modeling_dataset.csv](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/modeling_dataset.csv)
- [data/processed/dataset_report.json](/Users/hankli/Desktop/coding/ece4010_proj/data/processed/dataset_report.json)
- `data/processed/data_audit.md`

训练和评估输出：

- `outputs/models/*.joblib`
- `outputs/metrics/*_metrics.json`
- `outputs/metrics/model_comparison.json`
- `outputs/metrics/rf_feature_importance.csv`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/figures/rf_feature_importance.png`
- [outputs/results_summary.md](/Users/hankli/Desktop/coding/ece4010_proj/outputs/results_summary.md)

## 当前状态

目前仓库已经完成：

- 数据下载与本地回退读取模块
- 列名标准化与字段映射
- intake/outcome 简单配对逻辑
- 标签构造
- 特征工程
- Logistic Regression / Random Forest 训练与评估脚本
- EDA notebook 骨架
- README 与 proposal/slides 文档骨架
- Austin 原始数据的真实读入验证
- mixed-format 时间字段解析修复
- 数据审计脚本与 `data_audit.md`
- 低资源运行选项 `--max-rows`

## 当前限制

- Austin 官方开放数据接口有时会阻止脚本下载，出现 `403 Forbidden`，因此代码默认支持手动放置 CSV
- 重复 `Animal ID` 的配对使用的是简单启发式规则，不是复杂事件重构算法
- 当前 baseline 主要服务于课程项目，强调可复现和可讲清楚，而不是追求复杂建模
- 对全量数据直接跑 sklearn baseline 时，部分笔记本可能会因为内存或散热压力导致 Python 进程被系统杀掉；建议先用 `--max-rows` 跑轻量版本

## 已完成工作

- 完成项目目录结构、核心 Python 模块、EDA notebook 骨架和 proposal/slides 文档骨架。
- 完成官方 URL 下载 + 本地手动 CSV 回退读取。
- 完成列名标准化、年龄处理、低频类别合并和 intake 时间特征提取。
- 完成 intake/outcome 的 chronological pairing 逻辑，并保留主任务与 fallback 标签。
- 在真实 Austin Animal Center 数据上验证了主任务标签可以稳定构造。
- 修复了 outcomes 表混合时间格式导致的大量时间解析失败问题。
- 增加了 `check_data.py` 与 `data_audit.md`，可直接查看列名、重复 ID、配对情况和标签分布。
- 增加了复用 `data/processed/modeling_dataset.csv` 和 `--max-rows` 的低资源运行路径。

## 下一步工作

- 在另一台机器上优先运行：
  - `python src/evaluate.py --no-download --max-rows 50000`
- 如果能稳定跑完，再逐步增大 `--max-rows`，最后尝试全量数据。
- 检查 `outputs/metrics/`、`outputs/figures/` 和 `outputs/results_summary.md` 是否都生成完整。
- 根据真实 baseline 结果补齐结果总结，并决定是否需要保留 fallback 任务作为备选展示。

## Fallback 说明

默认仍然优先使用 `fast_adoption_30d`。

只有当后续真实数据检查发现：

- 配对逻辑导致过多异常
- 时间字段质量不稳定
- 30 天标签明显不可靠

才建议切换到 `adoption` 作为 fallback 任务。即便如此，也应在文档和汇报中保留主任务设计。
