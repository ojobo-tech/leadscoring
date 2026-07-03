# docs/data_dictionary.md

# Data Dictionary

This document defines the tables and fields used in the lead scoring project.

---

## 1. lead.csv

Primary table for lead scoring.

| Column                   | Data Type            | Description                                                      |
| ------------------------ | -------------------- | ---------------------------------------------------------------- |
| id                       | String (ID)          | 18-character Lead record id (00Q prefix).                        |
| first_name               | String               | Lead’s first name.                                               |
| last_name                | String               | Lead’s last name.                                                |
| company                  | String               | Company or account name for the lead.                            |
| email                    | String               | Email address (synthetic).                                       |
| phone                    | String               | Phone number (synthetic).                                        |
| status                   | String               | Lead status such as Open, Working, Nurturing, Qualified.         |
| lead_source              | String               | Source of the lead such as Web, Outbound, Trade Show.            |
| owner_id                 | String (ID)          | User id of the lead owner.                                       |
| created_date             | Date                 | Record creation date in YYYY-MM-DD format.                       |
| is_converted             | Boolean (as string)  | Whether the lead was converted in the generated scenario.        |
| converted_account_id     | String (ID) or empty | Account id when converted. Empty if not converted.               |
| converted_contact_id     | String (ID) or empty | Contact id when a matching contact was created. Empty otherwise. |
| converted_opportunity_id | String (ID) or empty | Opportunity id when created from this lead. Empty when none.     |

---

## 2. accounts.csv

Company-level context table.

| Column          | Data Type           | Description                                             |
| --------------- | ------------------- | ------------------------------------------------------- |
| id              | String (ID)         | 18-character Account record id (001 prefix).            |
| name            | String              | Account or company name.                                |
| account_number  | String              | Human-readable account number.                          |
| owner_id        | String (ID)         | User id of the account owner.                           |
| type            | String              | Account classification: Customer, Prospect, or Partner. |
| industry        | String              | Industry category such as Technology or Healthcare.     |
| annual_revenue  | Integer             | Annual revenue, synthetic whole number.                 |
| employee_count  | Integer             | Approximate headcount.                                  |
| billing_city    | String              | Billing address city.                                   |
| billing_state   | String              | Billing address state.                                  |
| billing_country | String              | Billing address country.                                |
| created_date    | Date                | Record creation date in YYYY-MM-DD.                     |
| is_active       | Boolean (as string) | Whether the account is considered active.               |

---

## 3. contact.csv

Person-level context table linked to accounts.

| Column       | Data Type           | Description                                                     |
| ------------ | ------------------- | --------------------------------------------------------------- |
| id           | String (ID)         | 18-character Contact record id (003 prefix).                    |
| account_id   | String (ID)         | Parent Account id.                                              |
| owner_id     | String (ID)         | User id of the contact owner.                                   |
| first_name   | String              | Contact’s first name.                                           |
| last_name    | String              | Contact’s last name.                                            |
| email        | String              | Email address (synthetic).                                      |
| phone        | String              | Phone number (synthetic).                                       |
| title        | String              | Job title such as Director or Manager.                          |
| department   | String              | Department such as Sales, Finance, or IT.                       |
| lead_source  | String              | Attributed lead source, aligned when created from a lead.       |
| created_date | Date                | Record creation date in YYYY-MM-DD.                             |
| is_primary   | Boolean (as string) | Whether this contact is marked primary for the account context. |

---

## 4. event.csv

Sales activity and meeting history.

| Column           | Data Type           | Description                                                            |
| ---------------- | ------------------- | ---------------------------------------------------------------------- |
| id               | String (ID)         | 18-character Event record id (00U prefix).                             |
| subject          | String              | Event title.                                                           |
| type             | String              | Event type such as Meeting, Demo, Onsite Visit, QBR, Executive Review. |
| owner_id         | String (ID)         | User id of the event owner.                                            |
| who_id           | String (ID)         | Related Contact id.                                                    |
| what_id          | String (ID)         | Related Account id or Opportunity id.                                  |
| location         | String              | Where the event takes place such as Zoom, Onsite, Phone.               |
| start_datetime   | DateTime            | Event start, ISO 8601 local-style string.                              |
| end_datetime     | DateTime            | Event end, ISO 8601 local-style string.                                |
| is_all_day_event | Boolean (as string) | All-day flag; synthetic data uses false.                               |
| created_date     | Date                | Record creation date in YYYY-MM-DD.                                    |

---

## 5. opportunity.csv

Sales pipeline and outcome table.

| Column         | Data Type            | Description                                                                               |
| -------------- | -------------------- | ----------------------------------------------------------------------------------------- |
| id             | String (ID)          | 18-character Opportunity record id (006 prefix).                                          |
| name           | String               | Opportunity name.                                                                         |
| account_id     | String (ID)          | Related Account id.                                                                       |
| owner_id       | String (ID)          | User id of the opportunity owner.                                                         |
| stage_name     | String               | Stage such as Prospecting, Qualification, Proposal, Negotiation, Closed Won, Closed Lost. |
| amount         | Integer              | Deal amount, synthetic whole number.                                                      |
| probability    | Integer              | Win probability from 0–100, aligned to stage.                                             |
| close_date     | Date                 | Expected or actual close date in YYYY-MM-DD.                                              |
| lead_source    | String               | Marketing or channel label; not always equal to originating lead source.                  |
| lead_source_id | String (ID) or empty | Lead id when the opportunity was created from a converted lead.                           |
| created_date   | Date                 | Record creation date in YYYY-MM-DD.                                                       |

---

## 6. order.csv

Order-level revenue and contract table.

| Column               | Data Type   | Description                                                  |
| -------------------- | ----------- | ------------------------------------------------------------ |
| id                   | String (ID) | 18-character Order record id (801 prefix).                   |
| order_number         | String      | Human-readable order number.                                 |
| account_id           | String (ID) | Bill-to or sold-to Account id.                               |
| opportunity_id       | String (ID) | Source Opportunity id.                                       |
| owner_id             | String (ID) | User id of the order owner.                                  |
| status               | String      | Order status such as Draft, Activated, Fulfilled, Cancelled. |
| effective_date       | Date        | Date the order is effective.                                 |
| total_amount         | Integer     | Header-level total, synthetic whole number.                  |
| contract_term_months | Integer     | Contract length in months.                                   |
| created_date         | Date        | Record creation date in YYYY-MM-DD.                          |

---

## 7. order_items.csv

Line-item details for orders.

| Column             | Data Type   | Description                                       |
| ------------------ | ----------- | ------------------------------------------------- |
| id                 | String (ID) | 18-character OrderItem record id (802 prefix).    |
| order_id           | String (ID) | Parent Order id.                                  |
| product_code       | String      | SKU or product code.                              |
| product_name       | String      | Product display name.                             |
| quantity           | Integer     | Number of units.                                  |
| unit_price         | Integer     | Unit price, synthetic whole number.               |
| discount_pct       | Integer     | Discount percentage such as 0, 5, 10, 15, 20.     |
| line_total         | Integer     | Extended line amount in the file.                 |
| service_start_date | Date        | Service or subscription start date in YYYY-MM-DD. |

---

## 8. tasks.csv

Sales activity and follow-up table.

| Column        | Data Type            | Description                                                         |
| ------------- | -------------------- | ------------------------------------------------------------------- |
| id            | String (ID)          | 18-character Task record id (00T prefix).                           |
| subject       | String               | Short description of the task.                                      |
| type          | String               | Task type such as Email, Call, Follow-up, Discovery, Qualification. |
| status        | String               | Task status such as Not Started, In Progress, Completed, Deferred.  |
| priority      | String               | Priority such as High, Normal, Low.                                 |
| owner_id      | String (ID)          | User id of the task owner.                                          |
| who_id        | String (ID)          | Related person: Lead id or Contact id.                              |
| what_id       | String (ID) or empty | Related non-person record: Account id or Opportunity id.            |
| activity_date | Date                 | Due or activity date in YYYY-MM-DD.                                 |
| is_closed     | Boolean (as string)  | Whether the task is closed.                                         |
| created_date  | Date                 | Record creation date in YYYY-MM-DD.                                 |

---

## 9. user.csv

Sales organization and ownership table.

| Column     | Data Type            | Description                                              |
| ---------- | -------------------- | -------------------------------------------------------- |
| id         | String (ID)          | 18-character User record id (005 prefix).                |
| first_name | String               | User’s first name.                                       |
| last_name  | String               | User’s last name.                                        |
| name       | String               | Full display name.                                       |
| title      | String               | Job title such as Director of Sales or Sales Manager.    |
| role       | String               | Simplified org role: Director, Manager, BDR, SDR, or AE. |
| manager_id | String (ID) or empty | 18-character User id of the direct manager.              |
| department | String               | Department such as Sales.                                |
| email      | String               | Work email address.                                      |
| is_active  | Boolean (as string)  | Whether the user is active.                              |
| hire_date  | Date                 | Hire date in YYYY-MM-DD.                                 |

---

## Project Notes

* `lead.csv` is the main table for the first version of the model.
* `accounts.csv`, `contact.csv`, `tasks.csv`, `event.csv`, and `user.csv` provide context and engagement signals.
* `opportunity.csv`, `order.csv`, and `order_items.csv` provide downstream business outcome and revenue context.
* The data is synthetic and suitable for open-source replication and prototyping.
