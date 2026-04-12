# Proposal Outline

## Problem Statement

Predict whether a dog or cat entering the Austin Animal Center will be adopted within 30 days, using only information available at intake time.

## Why It Matters

- Faster adoption is a concrete shelter outcome that is easy to explain.
- Early prediction can help shelters understand which animals may need additional promotion or intervention.
- The task is a clean binary classification problem that fits a course-level machine learning project.

## Dataset

- Austin Animal Center Intakes
- Austin Animal Center Outcomes
- Publicly available open-data records from the City of Austin
- Focus on dogs and cats only
- Label built by pairing intake and outcome records and checking whether the outcome is `Adoption` within 30 days

## Method

- Build a reproducible intake-outcome pairing rule based on chronological order within each `Animal ID`
- Clean and standardize intake-time fields
- Convert age strings into numeric values
- Engineer calendar features from intake datetime
- Collapse rare breed and color categories into `Other`
- Train:
  - Logistic Regression baseline
  - Random Forest main model
  - Optional XGBoost if implementation remains lightweight

## Expected Results

- A working baseline that predicts `fast_adoption_30d`
- Random Forest is expected to outperform Logistic Regression because breed, intake condition, age, and time effects may interact non-linearly
- Final deliverables include metrics, confusion matrices, and random forest feature importance

## Feasibility

- Data is public and well-known
- The pipeline uses standard Python ML tools: pandas and scikit-learn
- The task avoids heavy computation, deep learning, and custom annotation
- The output is easy to present with interpretable plots and simple baselines

## Risks And Backup Plan

- Risk: repeated `Animal ID`s may make 30-day label construction noisy
- Mitigation: use a documented chronological pairing rule and report inspection statistics
- Backup plan: switch target to overall `adoption` while preserving the same features and modeling pipeline
