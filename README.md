# TigSumm
This repository contains the datasets, preprocessing pipelines, model implementations, training scripts, and evaluation utilities for the paper:  TigSumm: A Cross-Lingual Framework for Sentiment-Aware Text Summarization in Low-Resource Tigrigna with Large Language Models, Accepted at Applied Intelligence (Springer Nature). 
The goal of TigSumm is to enable sentiment-aware abstractive summarization for low-resource Tigrigna through cross-lingual transfer learning, affective modeling, and parameter-efficient fine-tuning of large language models.


🔤 Preprocessing & Tokenizatio
python preprocessing/sentencepiece_train.py \
  --input data/raw \
  --vocab_size 32000 \
  --model_prefix spm_tigsumm
* Build multilingual dataset:
python preprocessing/build_multilingual_dataset.py
🧠 Model Architecture

TigSumm integrates:

Multilingual backbone: mBART-50, mT5, or LLaMA-2

Sentiment prediction head

Sentiment embedding fusion (additive / concat / gated)

LoRA adaptation for parameter-efficient tuning

## Usage

* Istallation

conda env create -f environment.yml
conda activate tigsumm

OR 
```bash
git clone https://github.com/yourusername/tigsumm.git
cd tigsumm
pip install -e .



🚀 Training

python training/train_tigsumm.py \
  --model llama2 \
  --fusion gated \
  --lora_rank 8 \
  --epochs 5



📊 Evaluation


ROUGE-L

BERTScore

Sentiment Preservation Rate (SPR)

Emotional Consistency Index (ECI)

python test_tigsumm.py



* Configuration

| Parameter   | Default                 | Description                                     |
| ----------- | ----------------------- | ----------------------------------------------- |
| model_name  | facebook/mbart-large-50 | Base model backbone                             |
| fusion_type | add                     | Fusion of sentiment and encoder representations |
| use_lora    | True                    | Enables PEFT-LoRA fine-tuning                   |
| lr          | 3e-5                    | Learning rate                                   |
| num_epochs  | 3                       | Training epochs                                 |



📚 Citation
@article{Gebremeskel2025TigSumm,
  title={TigSumm: A Cross-Lingual Framework for Sentiment-Aware Text Summarization in Low-Resource Tigrigna with Large Language Models},
  author={ Hagos Gebremedhin Gebremeskel, et. al.},
  journal={Knowledge-Based Systems},
  year={2025}
}
