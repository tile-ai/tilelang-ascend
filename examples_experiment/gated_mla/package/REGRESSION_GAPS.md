# Coverage and remaining acceptance

Production remains the selected r1.7 code. Implemented regression coverage,
recorded hardware evidence and the additional local cold-time target are distinct;
see [VALIDATION](VALIDATION.md) for the exact result boundaries.

## Gaps closed in the submitted source

- `test_prefill128_public_profile_core` adds mandatory default-PTO FP16/BF16
  B1/H96/physical Tq=Tk128 cases. It checks actual BN128 factory/kernel identity,
  original Q/Z/Y FP64 references and readonly inputs. Selector-only tests are
  not treated as device proof.
- The original Tq127/128/129/Tk129 extended combinations remain. In particular,
  Tq128/Tk129 is a distinct N64/KV-tail case, not a duplicate of the128/128 core.
- `test_value_r3_range_mixed_heads` adds four default-PTO stage cases at the
  existing B1/H96/T17 geometry: FP16/BF16 × separate/packed. Large and subnormal
  head groups, finite output, independent FP64/per-group budgets, repeats and
  readonly checks protect R3's overflow and low-part resolution fixes.
- The repository manifest now registers nine executable package-suite anchors
  plus the preserved L0 suite. It is active wiring, not the removed draft file.

The corrected final core run passed all 145 tests and all cache/compiler audits
without skips. The first packed-default registry failure is preserved in the
validation history. Full cold wall was 187.6994s, exceeding the wrapper's extra
strict local 120s target, not a hosted per-operator timeout;
hosted CI and reviewer approval remain separate outstanding checks.

The final registered extended selection also passed all 22 cases with zero skips
and clean audits (172.8109s whole process); this covers the preserved four L0
extensions and all 18 package extensions.

## Recorded installed-package evidence

- 8 direct/simulator combinations passed with actual BN64 decode/BN128 public
  prefill and matching production/native/cache/generated-kernel identities.
- 18 original extended cases passed with zero skips; their612.625s suite duration
  is a functional full run, not the later strict local cold-time measurement.
- The four frozen single-layer jobs cover12 candidate before/after-W_O FP64 phases,
  all passing with equal original external inputs, readonly data and exact caches.
  FP16 passes all original comparisons. BF16 retains `diagnostic_only_failed`
  and exit1 for official-HF differences. This is neither a claim that BF16 HF
  comparisons passed nor whole-model/checkpoint acceptance.

## Remaining boundaries

The final cold core exceeds the additional local 120s target. The repository
template's alternative core/daily marker split is implemented, with 145 core and
22 low_priority cases and their actual times recorded; no tests or markers were
changed to reclassify this result. Hosted CI and review must be recorded
separately. Source pytest imports the checkout; installed-wheel
checks retain module paths and source hashes. Optional model source remains an
external SHA-pinned input. Broader shapes, hardware, training and general cache
integration are not established by these finite tests. Official comparators
outside the precision budget remain unqualified and receive no performance ratio.
