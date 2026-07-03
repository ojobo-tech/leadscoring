# docs/data_quality_notes.md

# Data Quality Notes

## 1. General Notes

The dataset is synthetic CRM-style data and is structured for analytics and model prototyping.

## 2. Expected Quality Checks

Before modeling, the following checks should be performed:

* missing values
* duplicate IDs
* invalid dates
* inconsistent boolean strings
* mismatched foreign keys
* empty optional columns
* outliers in numeric fields
* class imbalance in the conversion label

## 3. Table-Specific Risks

### lead.csv

* `is_converted` may be imbalanced
* converted ID fields may be empty for non-converted leads
* lead source values may need normalization

### accounts.csv

* `industry` and `billing_*` fields may have inconsistent labels
* `annual_revenue` and `employee_count` should be checked for unrealistic values

### contact.csv

* duplicate people across accounts may exist
* `is_primary` values should be standardized
* email and phone may contain synthetic formatting differences

### event.csv

* start and end datetimes must be parsed carefully
* time zone or datetime format issues may appear
* some records may have missing `who_id` or `what_id`

### opportunity.csv

* `stage_name` and `probability` should be logically aligned
* `amount` should be checked for outliers
* `lead_source_id` may be empty if not created from a lead

### order.csv

* `effective_date` and `created_date` should be checked for consistency
* `total_amount` should be validated against related line items where possible

### order_items.csv

* `line_total` should be checked against quantity and unit price
* discount values should be standardized

### tasks.csv

* `is_closed` should be standardized
* `what_id` may be empty
* task status and priority values may need normalization

### user.csv

* `manager_id` may be empty for top-level roles
* `role` values should be standardized
* `is_active` should be converted to a consistent boolean format

## 4. Data Preparation Rules

* standardize date formats
* convert boolean-like strings to true/false
* deduplicate by record ID
* normalize category values
* create a cleaned processed dataset for modeling

## 5. Lead Scoring Specific Notes

* `lead.csv.is_converted` is the main target variable
* likely useful features include lead source, owner, created date, account context, activity counts, and engagement history
* downstream tables may help enrich the model but should be joined carefully to avoid leakage

## 6. Next Step

After documenting these files, the next task is to load the raw data into `data/raw/`, inspect actual values, and build the first processed dataset in `data/processed/`.
