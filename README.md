# Code and Data Package

This folder contains the code and annotated data used for ILE (Indirect Linguistic Encoding) detection experiments.

## Folder Structure

```text
code and data/
|- Code/
|  `- Detection.py
`- Data/
   |- Dataset.csv
   `- GroundTruth.csv
```

## Quick Dataset Summary

- Total posts: `2000`
- `ILE found (yes/no) = yes`: `895`
- `ILE found (yes/no) = no`: `1105`

## Data Files and Features

### 1) `Data/Dataset.csv`

This is the input file for inference/evaluation.

| Column | Type | Description |
|---|---|---|
| `ID` | string | Unique post identifier (used for exact join with `GroundTruth.csv`). |
| `Content` | string | Raw post text/caption. |

Notes:
- `ID` values are unique and should be treated as the primary key.
- `Content` can include hashtags, mentions, emojis, and multi-line text.

### 2) `Data/GroundTruth.csv`

This is the gold annotation file.

| Column | Type | Description |
|---|---|---|
| `ID` | string | Unique post identifier (must match `Dataset.csv`). |
| `Content` | string | Post text (included for reference). |
| `ILE found (yes/no)` | categorical (`yes`/`no`) | Document-level binary label for whether ILE exists in the post. |
| `ILE evidence` | list-like string | Minimal contiguous span(s) in text that encode indirect meaning. |
| `ILE meaning` | list-like string | Decoded intended meaning for each evidence span. |
| `Mechanism Class` | list-like string | Top-level taxonomy class for each span (for example `C1_...`, `C2_...`). |
| `Mechanism Category` | list-like string | Fine-grained sub-mechanism category for each span. |

Interpretation rules:
- When `ILE found (yes/no) = no`, list fields are typically empty (`[]`).
- When `ILE found (yes/no) = yes`, list fields can contain one or multiple entries.
- Entries are aligned by index across list fields (`ILE evidence[i]` corresponds to `ILE meaning[i]`, `Mechanism Class[i]`, and `Mechanism Category[i]`).
- Some posts contain multiple encoded spans and/or multiple mechanisms.

Important format note:
- List columns are serialized as text and may appear in Python-style list format (for example with single quotes), not strict JSON in every row.

## Code: `Code/Detection.py`

`Detection.py` benchmarks prompt variants using the OpenAI Responses API and writes prediction/metric artifacts.

Main behavior:
- Reads dataset and ground truth CSVs.
- Matches rows strictly by `ID`.
- Infers the text column by containing `content`.
- Infers the label column by containing `ILE found`.
- Runs model inference with a strict JSON schema output format.
- Saves predictions and evaluation metrics per run.



## Reproducibility Notes

- Default model in script: `gpt-5.4` (change with `--model` or `OPENAI_MODEL`).
- Prompt variations are defined in `INLINE_PROMPT_VARIATIONS` at the top of `Detection.py`.
- `temperature` defaults to `0.0` for deterministic behavior.
