# Raw Data

Place the original, untouched dataset here:

```
ai4i2020.csv
```

Source: [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

**Rule:** Files in this folder are never edited in place. All cleaning, feature
engineering, and transformation happens in code (see `notebooks/02_Preprocessing.ipynb`
and `src/preprocessing.py`) and the *output* is written to `data/processed/`.
This keeps the pipeline reproducible — anyone who clones the repo can regenerate
`data/processed/` from `data/raw/` at any time.
