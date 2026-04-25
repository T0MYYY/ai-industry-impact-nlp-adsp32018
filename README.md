# AI Industry Impact Analysis via NLP — UChicago ADSP 32018 Final Project

**Course:** ADSP 32018 — Next-Gen NLP: Transformers, LLMs and Agentic AI in Practice (Winter 2025, University of Chicago)  
**Author:** Chiyang Chen  
**Dataset:** ~200K news articles on AI, ML, and data science

---

## Project Overview

An end-to-end NLP pipeline that mines ~200K tech news articles to answer a core question: **which industries will be most impacted by AI, how (positively or negatively), and through what mechanisms?**

Motivated by the 2023 Goldman Sachs report estimating ~25% of US/Europe tasks are automatable by AI, and validated by Facebook Research's Moravec's Paradox findings — that AI disrupts cognitive/office work far more than physical/sensorimotor tasks.

**Research Questions**

1. Which industries and companies are most likely to be impacted by AI over the next several years?
2. How will they be impacted — positively, negatively, or ambiguously — and through what means (automation, augmentation, cost reduction, workflow redesign)?
3. What factors make AI adoption successful or unsuccessful?

---

## Pipeline Overview

```
~200K News Articles (Parquet)
           │
           ▼
  01  EDA & Data Profiling
           │
           ▼
  02A  Sample & Label Blocks ──────► 02C  Sample & Label Sentences
           │                                       │
           ▼                                       ▼
  02B  Block-level AI Relevance          02D  Sentence-level Classifier
       Classifier → Filter Corpus             → Rebuild Clean Docs
           │                                       │
           └──────────────┬────────────────────────┘
                          ▼
             AI-Relevant Article Corpus
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    03  Topic        04  Entity      05A/B/C  Sentiment
    Modeling        Extraction        Analysis
  (BERTopic)    (GLiNER + LLM)   (custom-trained model)
          │               │               │
          └───────────────┼───────────────┘
                          ▼
               06  Presentation Assets
                  (industry dashboards,
                  sentiment over time,
                  entity impact maps)
```

---

## Notebooks

| Notebook | Stage | Description |
|---|---|---|
| `01_data_ingestion_eda.ipynb` | EDA | Load parquet corpus; profile shape, fields, missing values, time distribution, and duplicates |
| `02A_block_sample_and_label.ipynb` | Filtering | Sample article text blocks; manually label for AI-relevance |
| `02B_block_train_predict.ipynb` | Filtering | Train block-level classifier; threshold selection; predict over full corpus to isolate AI-relevant blocks |
| `02C_sentence_sample_and_label.ipynb` | Filtering | Sample and label at sentence granularity for finer-grained filtering |
| `02D_sentence_train_predict_and_rebuild.ipynb` | Filtering | Train sentence classifier; predict; reconstruct cleaned AI-relevant document corpus |
| `03_topic_modeling.ipynb` | Analysis | BERTopic on filtered corpus; identify industry themes and AI application clusters |
| `04_entity_extraction.ipynb` | Analysis | GLiNER NER to extract organizations and technologies; LLM-based canonical name cleaning |
| `05A_sentiment_dataset_creation.ipynb` | Sentiment | Construct labeled sentiment training dataset from news content |
| `05B_sentiment_model_training.ipynb` | Sentiment | Train custom sentiment classifier (fine-tuned; no pre-labeled HuggingFace models used) |
| `05C_sentiment_inference_and_aggregation.ipynb` | Sentiment | Run inference at scale; aggregate by topic, entity, and time |
| `06_presentation_assets.ipynb` | Output | Final visualizations: industry impact rankings, sentiment trends over time, entity-level breakdowns |

---

## Tech Stack

| Component | Tools |
|---|---|
| Data handling | `pandas`, `pyarrow` |
| Article filtering | Custom block & sentence classifiers (`scikit-learn`) |
| Topic modeling | `BERTopic`, `sentence-transformers`, `UMAP`, `HDBSCAN` |
| Named entity recognition | `GLiNER`, LLM API (canonical cleaning) |
| Sentiment analysis | Custom fine-tuned model (trained from labeled data) |
| Visualization | `matplotlib`, `plotly` |

---

## Key Findings

See the presentation slides and `06_presentation_assets.ipynb` for the full analysis. High-level results cover:
- Industries with highest AI exposure (legal, finance, healthcare, office automation)
- Company-level sentiment breakdown (who is positioned positively vs. at risk)
- Temporal trends in AI adoption sentiment (2020–2023)
- Technologies driving impact (LLMs, automation tools, robotics)

---

## How to Run

```bash
jupyter notebook 01_data_ingestion_eda.ipynb
```

Due to `ipywidgets` compatibility, some interactive outputs may not render in GitHub's notebook viewer. Clone the repo and run locally, or open in [nbviewer](https://nbviewer.org/).
