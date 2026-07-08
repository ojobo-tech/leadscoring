# AI Lead Prioritization and Engagement Scoring Platform
https://leadbiz.streamlit.app/
A modular AI decision-support product that helps SMB sales teams focus on the best opportunities first, understand why a lead was ranked a certain way, and act faster through a simple dashboard or decision workspace.

## Project overview

This was built as a lead prioritization and engagement scoring system for SMB sales workflows. The goal is not just to predict whether a lead converts, but to help sales teams identify which leads deserve attention first, why those leads are higher priority, and what action should be taken next.

The solution combines:
- data harmonization and feature engineering,
- a business-friendly prioritization scorecard,
- a benchmark predictive model,
- explainable outputs,
- and a client-ready workflow framing.

## What this project does

- ingests lead and CRM-style data
- cleans and prepares the data
- builds lead prioritization and engagement scoring features
- trains and compares predictive and ranking models
- validates ranking quality on a holdout split
- explains scores using business-friendly logic
- produces a scorecard that can be shown in a dashboard or decision workspace

## Target user

SMB sales teams, including:
- sales representatives
- sales managers
- business owners
- revenue leaders who want a clearer prioritization process

## Business problem

Sales teams often spend time on low-value leads and miss the best opportunities because prioritization is manual, inconsistent, or based on incomplete context. This project is designed to help the team focus on the leads most likely to deserve immediate attention.

## Final product framing

The final product is a **lead prioritization and engagement scoring system**.

It is intended to answer:
- Which leads should be contacted first?
- Which leads are warm, active, or urgent?
- Why is this lead ranked above another one?
- What action should the sales team take next?

## Repository structure

- `data/` raw and processed datasets
- `docs/` project brief, notes, validation summary, and documentation
- `notebooks/` exploration and analysis
- `src/` reusable Python code
- `models/` saved model artifacts
- `reports/` EDA, validation, and model comparison outputs
- `app/` dashboard or interface code

## Current implementation

The current solution includes:

### 1. Data pipeline
The project uses modular scripts to build the data layer step by step:
- raw lead table
- harmonized master table
- model-ready table
- engineered scorecard table

### 2. Scorecard layer
A business-facing scorecard was built to produce:
- `priority_score`
- `priority_percentile`
- `priority_bucket`
- `recommended_action`
- `engagement_score`
- `urgency_score`
- `activity_strength_score`

### 3. Benchmark model
A Logistic Regression benchmark was trained on the same underlying signal features to validate the scorecard and compare ranking quality.

### 4. Validation approach
The final validation focused on:
- top-10% lift
- top-20% lift
- capture rate
- segment stability by `status`, `lead_source`, and `owner_role`

## Final validation result

The holdout validation showed that both the scorecard and Logistic Regression produce useful prioritization lift, with Logistic Regression slightly outperforming the scorecard overall on top-slice ranking. The Logistic Regression benchmark produced a top-10% lift of **1.0794**, while the scorecard produced a top-10% lift of **1.0591**. The scorecard remains valuable because it is highly explainable and business-friendly. :contentReference[oaicite:0]{index=0}

The validation also showed that performance is not identical across segments. The scorecard performed better in some groups, while Logistic Regression was stronger in others, which is useful context for sales operations and future tuning. :contentReference[oaicite:1]{index=1}

## Why this matters

This project is valuable because it turns messy CRM-style data into a repeatable decision framework. Instead of asking the sales team to guess which leads matter most, it gives them:
- a prioritized lead list,
- a clear reason for the ranking,
- and a recommended next action.

## Success criteria

The project is successful if it:
- ranks leads better than random ordering,
- provides a clear and explainable priority output,
- supports a realistic SMB sales workflow,
- and can be understood by business users without technical translation.

## Current state

The current version is ready for:
- dashboard design,
- explanation layer design,
- presentation packaging,
- and client-facing workflow mockups.

## Data note

The data used in this project is synthetic, open, and redistributable, making it suitable for open-source replication and experimentation.

## Next steps

- build the dashboard interface
- add explanation cards and reason codes
- create a lead review workflow
- package the validation results into a client presentation
- prepare a simple commercialization story
