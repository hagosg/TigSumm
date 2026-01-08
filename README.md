# TigSumm
This repository contains the datasets, preprocessing pipelines, model implementations, training scripts, and evaluation utilities for the paper:  TigSumm: A Cross-Lingual Framework for Sentiment-Aware Text Summarization in Low-Resource Tigrigna with Large Language Models, submited to Expert Systems With Applications. 
The goal of TigSumm is to enable sentiment-aware abstractive summarization for low-resource Tigrigna through cross-lingual transfer learning, affective modeling, and parameter-efficient fine-tuning of large language models.

🧠 Overview of  the Model Architecture

TigSumm is a scalable cross-lingual summarization framework that unifies multilingual pretrained backbones, sentiment-guided adaptation, and parameter-efficient fine-tuning to produce sentiment-aware summaries in low-resource Tigrigna. The approach exploits cross-lingual transfer from high-resource languages (English, Amharic) while preserving semantic fidelity and emotional coherence.

Key features include:

- Multilingual backbones (mBART-50, mT5, LLaMA-2)

- Hybrid Sentiment Fusion Module (additive / gated / concatenation)

- Low-Rank Adaptation (LoRA) for efficient fine-tuning

- Balanced evaluation with ROUGE-L, BERTScore, SPR, and ECI

- Fully reproducible training, evaluation, and analysis pipelines

  
🔤 Data Preprocessing & Tokenization

python preprocessing/sentencepiece_train.py \
  -- input data/raw \
  -- vocab_size 32000 \
  -- model_prefix spm_tigsumm
  
  python preprocessing/build_multilingual_dataset.py

### Native Tigrigna Data 

**Dataset Splits and Statistics**

| Split | Documents | Summaries | Positive | Neutral | Negative |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 22,400 | 22,400 | 8,960 | 7,392 | 6,048 |
| **Val** | 2,800 | 2,800 | 1,140 | 980 | 680 |
| **Test** | 2,800 | 2,800 | 1,120 | 1,000 | 680 |

*Note:* Distribution of documents across splits with sentiment labels. Training set represents 80% of the total dataset; validation and test sets each represent 10%.
  
### Cross-Lingual Data Sources for Build Multilingual Dataset
The selected three cross-lingual high-resource languages(English, Amharic, and Arabic), chosen for typological or cultural proximity are availiable on the following:
1.	https://github.com/ybai-nlp/MCLAS
2.	https://github.com/google-deepmind/rc-data
3.	https://github.com/Ethanscuter/gigaword 
5.	https://github.com/IsraelAbebe/An-Amharic-News-Text-classification-Dataset
   


## Usage

* Istallation

conda env create -f environment.yml
conda activate tigsumm

OR 
```bash
git clone https://github.com/yourusername/tigsumm.git
cd tigsumm
pip install -e .
```

🚀 Training

python training/train_tigsumm.py \
  --model TigSumm \
  --fusion hybrid \
  --lora_rank 8 \
  --epochs 5



📊 Evaluation


- ROUGE-L

- BERTScore

- Sentiment Preservation Rate (SPR)

- Emotional Consistency Index (ECI)

python test_tigsumm.py



### Configuration



| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | 16 | Effective after gradient accumulation |
| **Learning Rate** | 2 × 10⁻⁵ | AdamW optimizer |
| **Epochs** | 8 | Early stopping with patience 2 |
| **Max Input Length** | 512 tokens | Summarization truncation |
| **Sentiment Weight (β)** | 0.4 | Balancing polarity regularization |
| **Fusion λ** | 0.7 | Encoder–sentiment tradeoff |
| **Hardware** | 4 × A6000 GPUs | FP16 precision training |
| **Training Time** | 5–14 h | Depending on model |

*Note:* Configuration for TigSumm model training across different architectures. All experiments used linear learning rate warm-up over 10% of training steps.



