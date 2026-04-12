# Raw Data Notes

Place the official Austin Animal Center CSV files in this folder if automatic download is blocked:

- `austin_animal_center_intakes.csv`
- `austin_animal_center_outcomes.csv`

The loader validates that these files are real CSVs. If a download endpoint returns an HTML error page such as `403 Forbidden`, the code will reject that file and ask for manual replacement.
