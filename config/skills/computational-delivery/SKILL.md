---
name: computational-delivery
description: Deliver auditable results from existing tabular or sequence inputs without repeated long-output reads.
---

Use actual calculations and literal input fields. Inspect sheet names, headers and a small sample before selecting data. If inputs contain a ZIP, read its original members with zipfile; never execute supplied notebooks or scripts. Keep joins, inclusion rules, reference calls, missing values and counting units explicit. Do not silently replace an underspecified scientific definition with a guessed rule.

For each substantive calculation, write outputs/analysis_result.json using exactly the supported top-level keys: summary (object), optionally metrics (object) and limitations (array of strings). Keep summary compact (aim below 1500 characters), including the requested result, unit, numerator/denominator or group counts where relevant, and the actual selection and data basis. Save long tables to separate output files. Print only a short status, not full input tables or long subject lists. The existing tool returns summary directly; use it when it contains the requested result, and read artifacts only to resolve a specific missing detail.

Describe only computations that successfully ran. Distinguish calculations using a supplied derived table from independent recomputation on original data. A copied input path or inspected header does not establish that all rows were checked. If a verification or recomputation failed, retain that limitation rather than claiming it was performed. Once the required result and scope are established, deliver the requested final JSON with answer, method and limitations.
