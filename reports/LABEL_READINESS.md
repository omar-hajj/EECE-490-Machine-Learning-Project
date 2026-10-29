# Current label readiness

Recomputed from current interval annotations and existing feature-window bounds. Class presence is necessary but does not establish label quality or adequate coverage.

| Partition | Windows | Normal | Fault | Unknown | Both classes present? |
| --- | ---: | ---: | ---: | ---: | --- |
| test | 1927 | 0 | 0 | 1927 | No |
| train | 6632 | 1403 | 0 | 5229 | No |
| validation | 2095 | 0 | 0 | 2095 | No |

No positive labels are invented by this check. Unknown windows must not be treated as normal. Use LABEL_REVIEW_WORKSHEET.md to record independent evidence. Regenerate feature tables after any accepted annotation changes.
