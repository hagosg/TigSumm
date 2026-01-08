#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TigSumm: An Intelligent Cross-Lingual Framework for Sentiment-Aware Abstractive Text Summarization

Author: Hagos et al. (2025)
"""

import argparse
import torch
import torch.nn as nn
from transformers import Trainer
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

from utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train TigSumm Framework")

    parser.add_argument(
        "--TigSumm",
        type=str,
        required=True,
        help="Pretrained model (e.g., google/mt5-base, facebook/mbart-large-50, meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--TigSumm/Raw_Data",
        type=str,
        required=True,
        help="Path containing TigSumm_Data.csv",
    )
    parser.add_argument(
        "--TigSumm/output",
        type=str,
        default="checkpoints/tigsumm",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable parameter-efficient LoRA fine-tuning",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset from {args.TigSumm/Raw_Data}/TigSumm_Data.csv")

    dataset = load_dataset(
        "csv",
        data_files=f"{args.TigSumm/Raw_Data}/TigSumm_Data.csv"
    )

    # Handle predefined or automatic split
    if "split" in dataset["train"].column_names:
        train_dataset = dataset["train"].filter(lambda x: x["split"] == "train")
        eval_dataset = dataset["train"].filter(lambda x: x["split"] == "test")
    else:
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    # Determine model type
    model_name = args.TigSumm.lower()
    is_seq2seq = any(k in model_name for k in ["t5", "bart"])

    tokenizer = AutoTokenizer.from_pretrained(
        args.TigSumm,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(batch):
        inputs = batch["text"]
        targets = batch["summary"]

        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=128,
                truncation=True,
                padding="max_length",
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        preprocess,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # Load model
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.TigSumm)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.TigSumm)

    # Apply LoRA
    if args.use_lora:
        print("Applying LoRA for parameter-efficient fine-tuning")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    model.to(device)

    training_args = TrainingArguments(
        TigSumm/output=args.TigSumm/output,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir=f"{args.TigSumm/output}/logs",
        logging_steps=50,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(args.TigSumm/output)
    tokenizer.save_pretrained(args.TigSumm/output)

    print(f"Training complete. Model saved to {args.TigSumm/output}")

'''
 Sentiment-Aware Seq2Seq Wrapper
 We introduce a multi-task learning formulation that jointly optimizes abstractive summarization 
 and sentiment classification, improving emotional fidelity without sacrificing semantic accuracy.
'''
class TigSummMultiTaskModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels=3):
        super().__init__()
        self.base_model = base_model
        self.sentiment_head = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        sentiment_labels=None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        # Encoder CLS-style pooling
        encoder_hidden = outputs.encoder_last_hidden_state
        pooled = encoder_hidden.mean(dim=1)

        sentiment_logits = self.sentiment_head(pooled)

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "sentiment_logits": sentiment_logits,
        }
  '''
# Custom Trainer with Joint Loss
'''

class TigSummTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        sentiment_labels = inputs.pop("sentiment")

        outputs = model(**inputs, labels=labels)

        summarization_loss = outputs["loss"]
        sentiment_loss = F.cross_entropy(
            outputs["sentiment_logits"],
            sentiment_labels,
        )

        lambda_sent = 0.3
        total_loss = summarization_loss + lambda_sent * sentiment_loss

        return (total_loss, outputs) if return_outputs else total_loss

 # Preprocessing Update (Minimal)

def preprocess(batch):
    model_inputs = tokenizer(
        batch["text"],
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["sentiment"] = batch["sentiment"]

    return model_inputs


if __name__ == "__main__":
    main()
