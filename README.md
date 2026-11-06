# FALCON

Data and code for the paper *FALCON: Skill-Based Document-Level Machine Translation Evaluation with an LLM Judge* (Ahrii Kim and Seong-heum Kim, submitted to *Mathematics*).

FALCON evaluates each source segment in three steps. It labels the span of context the segment depends on, selects the three translation skills (out of nine, grouped under Field, Tenor, and Mode) that the segment demands, and scores each translation on those skills from 1 to 5. All three steps were run with `gpt-4.1-mini-2025-04-14` on the English source side of the WMT24 General MT test set and on the system outputs for English–German, English–Spanish, and English–Icelandic.

## Contents

```
data/
  source_profile.csv              Step 1 and 2 output for the 997 English source segments
  judgments/en-de.jsonl.gz        Step 3 output: skill scores and judge feedback, 26 systems
  judgments/en-es.jsonl.gz        23 systems
  judgments/en-is.jsonl.gz        20 systems
  system_info.csv                 System type, architecture, and context strategy
  context_config/                 En-Es context configuration study (Section 5.4)
  human_eval/                     Human validation of the judge (Section 4)
code/
  module/                         Prompts and scripts used to run the judge
  scripts/                        Commands for the three steps
analysis/
  reproduce.py                    Recomputes the main statistics of the paper
```

### `data/source_profile.csv`

One row per source segment. `docId` and `segId` identify the segment, `context` is the context label from Step 1, and `skills` is the list of three skills from Step 2 (empty for segments labeled `Sentence-level knowledge`, which are excluded from scoring).

### `data/judgments/*.jsonl.gz`

One JSON object per segment and system, for the 666 segments that require document-level context.

| Field | Description |
|---|---|
| `pair` | Language pair |
| `docId`, `segId`, `doc`, `domain` | Segment identifiers and WMT24 domain |
| `system` | WMT24 system |
| `context` | Context label of the segment |
| `skills` | The three profiled skills |
| `scores` | Judge scores (1 to 5), in the order of `skills` |
| `feedback` | Written feedback of the judge |
| `source`, `prev_src`, `prev_tgt`, `translation` | Source segment, the preceding source segments and their references given as context, and the system translation |

The segment score is the mean of `scores`, and the system score is the mean of the segment scores over all segments of a system. In a small number of rows the score parsed from the judge output lies outside 1 to 5 or could not be parsed. These rows are kept as produced.

### `data/context_config/`

`en-es_configurations.csv` contains the 2,526 En-Es segment–system pairs that have both a FALCON score and an ESA score. Each row holds the WMT24 ESA and MQM scores (MQM as error points, so lower is better) and the judge scores under five configurations.

| Column | Context supplied to the judge |
|---|---|
| `score_base` | Two preceding source segments and their references |
| `score_window4` | Four preceding source segments and their references |
| `score_no_source` | References of the two preceding segments only |
| `score_system_output_target` | Two preceding source segments and the system output for them |
| `score_current_reference` | Base configuration plus the reference of the current segment |

`feedback.jsonl.gz` holds the judge feedback for the four alternative configurations, keyed by `row`.

### `data/human_eval/`

`segments.csv` lists the 230 En-Es segments rated by the experts, with the source, context, translation, ESA and MQM scores, and the judge scores. `ratings.csv` has one row per segment and skill (690 rows). `item` refers to the row in `segments.csv`, `judge_score` is the score of the judge, and `W1` to `W3` are the scores of the three experts. The experts are anonymized, and no personal information about them is included.

## Reproducing the statistics

```
pip install -r requirements.txt
python analysis/reproduce.py
```

The script prints the overall score of every system, the agreement between the judge and the experts, and the segment-level correlation with ESA under each context configuration.

## Running the judge

Copy `code/module/my_keys.example.py` to `code/module/my_keys.py` and set `OPENAI_API_KEY`. The three steps are run with

```
code/scripts/run_context.sh <input.csv> <output.csv>
code/scripts/run_skill.sh   <input.csv> <output.csv>
code/scripts/run_review.sh  <input.csv> <output.csv>
```

The prompts are defined in `contextCreator.py`, `skillCreator.py`, and `judge.py`, and the skill definitions and rubrics in `skillset.json`. Outputs of a new run will not match the released judgments exactly, because the model output varies between runs.

## Sources and licenses

The source texts, references, system outputs, and ESA and MQM scores are from the WMT24 General Machine Translation Shared Task (Kocmi et al., 2024) and are subject to its terms of use. The judgments, profiles, and human ratings created in this work are released under CC BY 4.0, and the code under the MIT License.

## Citation

```
@article{kim2026falcon,
  title   = {FALCON: Skill-Based Document-Level Machine Translation Evaluation with an LLM Judge},
  author  = {Kim, Ahrii and Kim, Seong-heum},
  journal = {Mathematics},
  year    = {2026},
  note    = {Under review}
}
```
