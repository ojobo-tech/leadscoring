# README.md

# AI Lead Scoring Decision-Support Platform

A modular AI decision-support product that helps SMB sales teams prioritize leads, understand why a lead was scored a certain way, and act faster on the best opportunities.

## What this project does

* ingests lead and CRM-style data
* cleans and prepares the data
* trains a lead scoring model
* explains predictions using XAI
* presents results in a simple dashboard or decision workspace

## Target user

SMB sales teams, including sales representatives, sales managers, and business owners.

## First module

Lead scoring

## Project goal

Build a client-ready AI product that turns business data into predictions, explanations, and actionable sales insights.

## Repository structure

* `data/` raw and processed datasets
* `docs/` project brief, notes, and documentation
* `notebooks/` exploration and analysis
* `src/` reusable Python code
* `models/` saved model artifacts
* `app/` dashboard or interface code

## Next steps

1. data acquisition
2. data cleaning and feature preparation
3. model training and comparison
4. explainability layer
5. dashboard and deployment

---

# .gitignore

**pycache**/
*.pyc
.ipynb_checkpoints/
.env
venv/
.venv/
models/
data/raw/
data/processed/
.DS_Store

---

# requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
xgboost
shap
streamlit

---

# docs/project_brief.md

## Project Title

AI Lead Scoring Decision-Support Platform

## Business Problem

SMB sales teams often waste time on low-quality leads and miss the best opportunities because lead prioritization is manual and inconsistent.

## Data

This is an open source data, it is synthetic and redistributable

## Target User

SMB sales representatives, sales managers, and business owners who need to prioritize leads more effectively.

## Scope

The first version of the product will focus on lead scoring only. It will include data ingestion, data cleaning, predictive modeling, explainability, and a simple dashboard or decision workspace. The product supports a modular ingestion framework: is designed for multiple ingestion methods.

## Success Criteria

The project is successful if it can rank leads by likelihood to convert, explain why each lead received its score, present the results clearly, and support a realistic SMB sales workflow.

## Business Impact

Overall, this solution means the company can turn scattered data into repeatable business decisions.

The real impact is usually:
- more revenue because sales teams focus on the best leads first,
- less wasted time because low-value leads or risky customers are flagged early,
- better forecasting because the company can see patterns before they become problems,
- more consistent decisions because the team is using the same scoring logic instead of guesswork,
- better visibility because managers can see what is happening across leads, accounts, and activity,
- scalability because the same platform can later support churn, purchase likelihood, and demand forecasting.