# TigSumm
This repository contains the datasets, preprocessing pipelines, model implementations, training scripts, and evaluation utilities for the paper:  TigSumm: A Cross-Lingual Framework for Sentiment-Aware Text Summarization in Low-Resource Tigrigna with Large Language Models, Accepted at Applied Intelligence (Springer Nature). 
The goal of TigSumm is to enable sentiment-aware abstractive summarization for low-resource Tigrigna through cross-lingual transfer learning, affective modeling, and parameter-efficient fine-tuning of large language models.

TigSumm/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   │   ├── en/
│   │   ├── am/
│   │   └── tir/
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   └── README.md
│
├── preprocessing/
│   ├── normalize_unicode.py
│   ├── sentencepiece_train.py
│   ├── sentiment_lexicon_mapping.py
│   └── build_multilingual_dataset.py
│
├── models/
│   ├── tigsumm_model.py
│   ├── sentiment_fusion.py
│   └── lora_config.py
│
├── training/
│   ├── train_tigsumm.py
│   ├── train_baselines.py
│   └── config.yaml
│
├── evaluation/
│   ├── rouge_eval.py
│   ├── bertscore_eval.py
│   ├── sentiment_metrics.py
│   └── qualitative_analysis.py
│
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── download_models.sh
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
│
└── appendix/
    ├── dataset_schema.tex
    ├── annotation_guidelines.tex
    └── inter_annotator_agreement.tex
