# NLP News AI Detection Pipeline

An end-to-end NLP pipeline that detects AI-generated content blocks in a large news corpus, then performs downstream analysis — topic modeling, named entity extraction, and sentiment analysis — on the identified AI content.

**Course:** NLP Final Project | UC Davis  
**Author:** Chiyang Chen (Tom)

---

## Pipeline Overview

```
News Corpus (Parquet)
        │
        ▼
01  EDA & Data Ingestion
        │
        ▼
02A  Block Sampling & Labeling ──► 02C  Sentence Sampling & Labeling
        │                                        │
        ▼                                        ▼
02B  Block Classifier (train → predict)  02D  Sentence Classifier → Rebuild Docs
        │                                        │
        └──────────────┬──────────────────────────┘
                       ▼
              AI-Positive Block Corpus
               ┌────────┼────────┐
               ▼        ▼        ▼
              03        04      05A/B/C
         Topic       Entity   Sentiment
        Modeling   Extraction  Analysis
               │        │        │
               └────────┼────────┘
                        ▼
                06  Presentation Assets
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_ingestion_eda.ipynb` | Load news parquet dataset; inspect shape, fields, missing values, deduplication, and time distribution |
| `02A_block_sample_and_label.ipynb` | Sample text blocks from articles and create a labeled dataset for AI content detection |
| `02B_block_train_predict.ipynb` | Train a block-level AI classifier; select operating threshold; predict over the full corpus |
| `02C_sentence_sample_and_label.ipynb` | Sample and label at sentence granularity for finer-grained detection |
| `02D_sentence_train_predict_and_rebuild.ipynb` | Train sentence-level classifier; predict; reconstruct cleaned document corpus |
| `03_topic_modeling.ipynb` | BERTopic topic modeling on sentence-cleaned AI blocks; cluster and label emergent themes |
| `04_entity_extraction.ipynb` | GLiNER-based NER with LLM canonical cleaning to extract and normalize entities from AI content |
| `05A_sentiment_dataset_creation.ipynb` | Build a labeled sentiment dataset from the AI-positive corpus |
| `05B_sentiment_model_training.ipynb` | Fine-tune / train a sentiment classifier |
| `05C_sentiment_inference_and_aggregation.ipynb` | Run inference at scale; aggregate sentiment signals across topics and entities |
| `06_presentation_assets.ipynb` | Generate final visualizations and presentation-ready figures |

---

## Tech Stack

| Component | Library |
|---|---|
| Data handling | `pandas`, `pyarrow` |
| Text classification | `scikit-learn`, custom block/sentence classifiers |
| Topic modeling | `BERTopic`, `sentence-transformers`, `UMAP`, `HDBSCAN` |
| Named entity recognition | `GLiNER`, LLM-based canonical cleaning |
| Sentiment analysis | Custom fine-tuned model |
| Visualization | `matplotlib`, `plotly` |

---

## Data

The input dataset is a large news corpus stored in Parquet format (sourced from Google Cloud Storage). Due to size, raw data is not included in this repository. Set the dataset path in the configuration block of `01_data_ingestion_eda.ipynb` before running.

---

## Note on Previews

Due to `ipywidgets` compatibility, interactive outputs may not render in GitHub's notebook viewer. Clone the repo and run locally, or open in [nbviewer](https://nbviewer.org/).
