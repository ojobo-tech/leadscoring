# Data Harmonization Plan

## Goal

Standardize overlapping field names across the raw CRM-style tables so the processed layer is clear, consistent, and safe for modeling.

## Harmonization Rules

* Keep raw files unchanged in `data/raw/`
* Rename overlapping fields in the processed layer with table-specific prefixes
* Preserve IDs as join keys, but make their meaning explicit in the processed dataset
* Standardize dates, booleans, and category labels
* Use one canonical name for each concept in the processed layer

## Canonical Naming Scheme

### 1. ID fields

Raw tables often use `id`, but in the processed layer the meaning should be explicit.

| Raw Field          | Processed Name | Meaning                      |
| ------------------ | -------------- | ---------------------------- |
| lead.csv.id        | lead_id        | Unique lead record id        |
| accounts.csv.id    | account_id     | Unique account record id     |
| contact.csv.id     | contact_id     | Unique contact record id     |
| event.csv.id       | event_id       | Unique event record id       |
| opportunity.csv.id | opportunity_id | Unique opportunity record id |
| order.csv.id       | order_id       | Unique order record id       |
| order_items.csv.id | order_item_id  | Unique order item record id  |
| tasks.csv.id       | task_id        | Unique task record id        |
| user.csv.id        | user_id        | Unique user record id        |

### 2. Owner fields

Use table-specific names so ownership is always clear.

| Raw Field                | Processed Name       | Meaning                  |
| ------------------------ | -------------------- | ------------------------ |
| lead.csv.owner_id        | lead_owner_id        | Owner of the lead        |
| accounts.csv.owner_id    | account_owner_id     | Owner of the account     |
| contact.csv.owner_id     | contact_owner_id     | Owner of the contact     |
| event.csv.owner_id       | event_owner_id       | Owner of the event       |
| opportunity.csv.owner_id | opportunity_owner_id | Owner of the opportunity |
| order.csv.owner_id       | order_owner_id       | Owner of the order       |
| tasks.csv.owner_id       | task_owner_id        | Owner of the task        |

### 3. Date fields

Dates should stay clear and table-specific.

| Raw Field                          | Processed Name           | Meaning                                |
| ---------------------------------- | ------------------------ | -------------------------------------- |
| lead.csv.created_date              | lead_created_date        | Lead creation date                     |
| accounts.csv.created_date          | account_created_date     | Account creation date                  |
| contact.csv.created_date           | contact_created_date     | Contact creation date                  |
| event.csv.created_date             | event_created_date       | Event record creation date             |
| event.csv.start_datetime           | event_start_datetime     | Event start timestamp                  |
| event.csv.end_datetime             | event_end_datetime       | Event end timestamp                    |
| opportunity.csv.created_date       | opportunity_created_date | Opportunity creation date              |
| opportunity.csv.close_date         | opportunity_close_date   | Opportunity expected/actual close date |
| order.csv.created_date             | order_created_date       | Order creation date                    |
| order.csv.effective_date           | order_effective_date     | Effective order date                   |
| order_items.csv.service_start_date | service_start_date       | Service/subscription start date        |
| tasks.csv.created_date             | task_created_date        | Task creation date                     |
| tasks.csv.activity_date            | task_activity_date       | Due/activity date                      |
| user.csv.hire_date                 | user_hire_date           | User hire date                         |

### 4. Source and status fields

These often repeat across tables and should be table-specific in the processed layer.

| Raw Field                   | Processed Name          | Meaning                                    |
| --------------------------- | ----------------------- | ------------------------------------------ |
| lead.csv.lead_source        | lead_source             | Lead origin source                         |
| contact.csv.lead_source     | contact_lead_source     | Source attributed to contact               |
| opportunity.csv.lead_source | opportunity_lead_source | Channel or marketing label for opportunity |
| lead.csv.status             | lead_status             | Lead lifecycle status                      |
| opportunity.csv.stage_name  | opportunity_stage_name  | Opportunity pipeline stage                 |
| order.csv.status            | order_status            | Order status                               |
| tasks.csv.status            | task_status             | Task status                                |
| tasks.csv.priority          | task_priority           | Task priority                              |
| user.csv.role               | user_role               | Simplified organizational role             |
| user.csv.department         | user_department         | Department name                            |

### 5. Person and company fields

These are shared concepts but should be renamed clearly in the processed layer.

| Raw Field              | Processed Name     | Meaning              |
| ---------------------- | ------------------ | -------------------- |
| lead.csv.first_name    | lead_first_name    | Lead first name      |
| lead.csv.last_name     | lead_last_name     | Lead last name       |
| lead.csv.company       | lead_company       | Lead company name    |
| contact.csv.first_name | contact_first_name | Contact first name   |
| contact.csv.last_name  | contact_last_name  | Contact last name    |
| contact.csv.email      | contact_email      | Contact email        |
| contact.csv.phone      | contact_phone      | Contact phone        |
| accounts.csv.name      | account_name       | Account/company name |
| user.csv.name          | user_name          | User full name       |
| user.csv.first_name    | user_first_name    | User first name      |
| user.csv.last_name     | user_last_name     | User last name       |

### 6. Booleans

Standardize booleans to true/false in the processed layer.

| Raw Field                  | Processed Name         | Meaning                    |
| -------------------------- | ---------------------- | -------------------------- |
| lead.csv.is_converted      | lead_is_converted      | Whether lead converted     |
| accounts.csv.is_active     | account_is_active      | Whether account is active  |
| contact.csv.is_primary     | contact_is_primary     | Whether contact is primary |
| event.csv.is_all_day_event | event_is_all_day_event | Whether event is all-day   |
| tasks.csv.is_closed        | task_is_closed         | Whether task is closed     |
| user.csv.is_active         | user_is_active         | Whether user is active     |

### 7. Numeric fields

Keep these as numeric business measures.

| Raw Field                      | Processed Name          | Meaning                   |
| ------------------------------ | ----------------------- | ------------------------- |
| accounts.csv.annual_revenue    | annual_revenue          | Annual revenue            |
| accounts.csv.employee_count    | employee_count          | Employee count            |
| opportunity.csv.amount         | opportunity_amount      | Deal amount               |
| opportunity.csv.probability    | opportunity_probability | Win probability           |
| order.csv.total_amount         | order_total_amount      | Header-level order amount |
| order.csv.contract_term_months | contract_term_months    | Contract length           |
| order_items.csv.quantity       | quantity                | Number of units           |
| order_items.csv.unit_price     | unit_price              | Unit price                |
| order_items.csv.discount_pct   | discount_pct            | Discount percent          |
| order_items.csv.line_total     | line_total              | Extended line amount      |

## Recommended Processed Tables

For modeling, create these processed outputs:

* `lead_master.csv` → main table for lead scoring
* `account_summary.csv` → optional account-level enrichment
* `activity_summary.csv` → aggregated task/event engagement
* `opportunity_summary.csv` → downstream sales context

## Lead Master Recommended Base Columns

The first processed model table should start from `lead.csv` and add:

* `lead_id`
* `lead_created_date`
* `lead_source`
* `lead_status`
* `lead_owner_id`
* `lead_is_converted`
* joined account and contact context
* aggregated task/event activity features
* user/owner context

