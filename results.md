# Results Summary

## Stage 1 — Email Boundary Detection

| Approach          | Precision | Recall | F1    |
|-------------------|-----------|--------|-------|
| Regex             | 0.926     | 0.922  | 0.924 |
| BERT (line only)  | 0.370     | 0.943  | 0.530 |
| BERT (line+filter)| 0.966     | 0.901  | 0.932 |
| GPT-5.5 (vision)  | 0.971     | 0.971  | 0.971 |

## Stage 2 — Field Extraction

| Approach         | Exact Macro F1 | ANLS  |
|------------------|----------------|-------|
| Regex            | 0.881          | 0.957 |
| BERT             | 0.312          | 0.571 |
| GPT-5.5 (vision) | 0.752          | 0.943 |

## Stage 2 Per-Field Results

| Field      | Regex F1 | Regex ANLS | BERT F1 | BERT ANLS | GPT-5.5 F1 | GPT-5.5 ANLS |
|------------|----------|------------|---------|-----------|------------|--------------|
| FROM       | 0.991    | 0.992      | 0.616   | 0.673     | 0.976      | 0.974        |
| TO         | 0.908    | 0.936      | 0.490   | 0.388     | 0.923      | 0.942        |
| CC         | 0.876    | 0.935      | 0.000   | 0.530     | 0.381      | 0.791        |
| DATE       | 0.977    | 0.974      | 0.769   | 0.937     | 0.971      | 0.992        |
| SUBJECT    | 0.909    | 0.968      | 0.000   | 0.000     | 0.906      | 0.990        |
| ATTACHMENT | 0.625    | 0.957      | 0.000   | 0.897     | 0.353      | 0.967        |
