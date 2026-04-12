# Slides Outline

## Slide 1: Title

- Predicting Fast Pet Adoption from Shelter Intake Data
- Course project proposal
- Team / course / date
- Austin Animal Center as the case study

Suggested visual:

- Shelter photo or simple title graphic

## Slide 2: Problem Motivation

- Animal shelters need to understand which animals are likely to be adopted quickly
- Early prediction can support outreach and resource planning
- We focus on a clear binary outcome: adoption within 30 days

Suggested visual:

- One simple flow chart from intake to outcome

## Slide 3: Prediction Task

- Input: information available at intake time only
- Output: `fast_adoption_30d`
- Keep only dogs and cats
- Avoid data leakage from outcome-only fields

Suggested visual:

- Input/output pipeline diagram

## Slide 4: Dataset

- Austin Animal Center Intakes
- Austin Animal Center Outcomes
- Public open-data source
- Covers multiple years and large sample size

Suggested visual:

- Dataset table summary or screenshot of source page

## Slide 5: Label Construction

- Match intake and outcome records by `Animal ID`
- Repeated IDs handled by chronological pairing
- `fast_adoption_30d = 1` only if outcome is adoption within 30 days
- Fallback label: overall adoption if pairing proves unstable

Suggested visual:

- Small timeline example of intake and outcome pairing

## Slide 6: Features

- Animal Type
- Breed
- Color
- Sex upon Intake
- Age upon Intake
- Intake Type and Intake Condition
- Intake month / weekday / season / year

Suggested visual:

- Feature list graphic or grouped table

## Slide 7: Methods

- Logistic Regression baseline
- Random Forest main model
- Optional XGBoost if time allows
- Stratified train/validation/test split

Suggested visual:

- Simple model comparison table

## Slide 8: Evaluation Plan

- Accuracy
- F1-score
- ROC-AUC
- Confusion matrix
- Random forest feature importance

Suggested visual:

- Metric dashboard mockup

## Slide 9: Expected Outcomes

- A strong baseline, not SOTA
- Clear explanation of which intake factors correlate with faster adoption
- Reproducible code and presentation-friendly visuals

Suggested visual:

- Example feature importance chart placeholder

## Slide 10: Feasibility, Risks, And Backup

- Feasible with public data and standard ML tools
- Main risk is repeated-record pairing
- Backup task is overall adoption prediction
- Timeline: data cleaning, modeling, evaluation, presentation

Suggested visual:

- Risk/backup table or project timeline
