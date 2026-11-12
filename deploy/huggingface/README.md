---
title: QueueIQ
colorFrom: gray
colorTo: green
sdk: streamlit
sdk_version: 1.67.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Withdrawal and completion outlook for US grid queue projects
---

# QueueIQ

For a proposed power project on the day it joins a US interconnection queue (PJM, MISO, CAISO or SPP),
this dashboard shows the model's chance of withdrawing or reaching commercial operation month by month,
with a range built from the model's own past errors, a what-if comparison and an explanation of each
prediction.

The probabilities run high on recent projects and withdrawal is close to unpredictable from entry-day
information. Read "How to read this and limitations" in the app before using any number.

Code, evaluation and full documentation: https://github.com/sheershh/QueueIQ

**Data:** Queued Up Data File. Copyright (c) 2026, The Regents of the University of California,
through Lawrence Berkeley National Laboratory & GridTracker. Licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: https://emp.lbl.gov/queues.
This Space contains a model trained on that data and a trimmed table derived from it (filtered to four
regions, cleaned, and reduced to the columns the model needs). LBNL and GridTracker do not endorse this
project.

**Code:** MIT License.
