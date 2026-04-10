# Vector Troubleshooting

## Symptom: wrong numerical result

Checks:
- verify dtype for compute path and accumulation path
- verify tail handling on block boundaries
- verify vcast round mode when converting types

## Symptom: compile-time op mismatch

Checks:
- confirm API signature for v-prefix call
- confirm src and dst shapes are compatible
- confirm reduce dims and reduce mode are valid

## Symptom: performance regression

Checks:
- reduce redundant copy steps
- avoid unnecessary cast pairs
- profile block sizes and balance kernel launch granularity
