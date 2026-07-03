# docs/schema.md

# Schema Overview

The project uses a CRM-style relational structure centered on leads, accounts, contacts, activities, opportunities, and orders.

## Core Relationship Map

* `lead.csv` is the primary starting table for lead scoring.
* `contact.csv.account_id` links contacts to accounts.
* `event.csv.who_id` links events to people, and `event.csv.what_id` links events to accounts or opportunities.
* `tasks.csv.who_id` links tasks to leads or contacts.
* `tasks.csv.what_id` links tasks to accounts or opportunities.
* `opportunity.csv.account_id` links opportunities to accounts.
* `opportunity.csv.lead_source_id` links an opportunity back to the originating lead when applicable.
* `order.csv.account_id` links orders to accounts.
* `order.csv.opportunity_id` links orders to opportunities.
* `order_items.csv.order_id` links line items to orders.
* `user.csv.id` links to owner_id fields across the system.

## Primary Use in the Project

The first product version will use `lead.csv` as the main scoring table and enrich it with related activity and context from:

* `accounts.csv`
* `contact.csv`
* `tasks.csv`
* `event.csv`
* `user.csv`

## Likely Join Logic

* Join `lead.csv.owner_id` to `user.csv.id`
* Join `lead.csv.converted_account_id` to `accounts.csv.id`
* Join `lead.csv.converted_contact_id` to `contact.csv.id`
* Join `tasks.csv.who_id` and `event.csv.who_id` to lead/contact records where possible
* Join `tasks.csv.what_id` and `event.csv.what_id` to account or opportunity records where applicable
* Join `opportunity.csv.account_id` to `accounts.csv.id`
* Join `order.csv.opportunity_id` to `opportunity.csv.id`
* Join `order_items.csv.order_id` to `order.csv.id`

## Modeling View

For lead scoring, the model-ready dataset will likely be built from:

* one row per lead
* target label from `lead.csv.is_converted`
* feature enrichment from account, contact, activity, and user tables
