# Data Directory

Put your private raw input files here when you run the project locally.

Recommended layout:

```text
data/
  train.csv              # BERT masked-language-model training data
  eval.csv               # validation / inference text data
  panel.csv              # optional base panel data for final merging
  policy_pdfs/           # source policy PDFs for keyword extraction
```

Generated phrase tables, vector files, similarity indicators, and merged panel
outputs should be written under `outputs/`, not `data/`.

This directory is ignored by Git by default, because the research data is
usually private or not public.
