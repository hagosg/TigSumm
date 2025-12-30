# TigSumm
This repository contains the datasets, preprocessing pipelines, model implementations, training scripts, and evaluation utilities for the paper:  TigSumm: A Cross-Lingual Framework for Sentiment-Aware Text Summarization in Low-Resource Tigrigna with Large Language Models, Accepted at Applied Intelligence (Springer Nature). 
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
  
## Cross-Lingual Data Sources for Build Multilingual Dataset:
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


ROUGE-L

BERTScore

Sentiment Preservation Rate (SPR)

Emotional Consistency Index (ECI)

python test_tigsumm.py



### Configuration

| Parameter   | Default                 | Description                                     |
| ----------- | ----------------------- | ----------------------------------------------- |
| model_name  | facebook/mbart-large-50 | Base model backbone                             |
| fusion_type | add                     | Fusion of sentiment and encoder representations |
| use_lora    | True                    | Enables PEFT-LoRA fine-tuning                   |
| lr          | 3e-5                    | Learning rate                                   |
| num_epochs  | 3                       | Training epochs                                 |



📚 Citation
@article{Gebremeskel2026TigSumm,
  title={TigSumm: A Cross-Lingual Framework for Sentiment-Aware Text Summarization in Low-Resource Tigrigna with Large Language Models},
  author={ Hagos Gebremedhin Gebremeskel, et. al.},
  journal={Applied Intelligence},
  year={2026}
}

📬 Contact

Hagos Gebremedhin Gebremeskel
Beijing Institute of Technology
📧 hagosg81@bit.edu.cn
