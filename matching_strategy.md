# Matching Strategy for Intake and Outcome Records

## Goal

Construct a practical label for:

- `fast_adoption_30d = 1` if an animal is adopted within 30 days of intake

The challenge is that `Animal ID` can appear more than once in both the intake table and the outcome table.

## Why A Simple Rule Is Needed

If we merge the two datasets directly on `Animal ID`, repeated IDs can create a many-to-many join. That would produce duplicate combinations and make the target label unreliable.

For a machine learning course project, the matching rule should be:

- simple
- explainable
- reproducible

## Rule Used In This Repository

For each `Animal ID`:

1. Sort intake records by `intake_datetime`
2. Sort outcome records by `outcome_datetime`
3. Assign `pairing_sequence = 0, 1, 2, ...` within each animal
4. Match the `k`-th intake with the `k`-th outcome

This means the earliest intake is paired with the earliest outcome, the second intake with the second outcome, and so on.

## Why This Rule Is Reasonable

- It avoids many-to-many duplication.
- It is easy to implement with `groupby(...).cumcount()`.
- It is easy to describe in a proposal presentation.
- It keeps the project focused on baseline ML rather than complex event alignment.

## Handling Imperfect Cases

If a matched outcome is earlier than the intake timestamp, the code marks that pair as a temporal mismatch and drops the outcome side from the label construction for that row.

If an intake has no matched outcome, that row remains in the dataset and receives:

- `adoption = 0`
- `fast_adoption_30d = 0`

## When To Use The Fallback Label

The default remains `fast_adoption_30d`.

If the inspection report later shows too many temporal mismatches or obviously unstable pairings, the repository already supports the fallback target:

- `adoption = 1` if `Outcome Type == "Adoption"`
- `adoption = 0` otherwise

This fallback is implemented without changing the rest of the pipeline.
