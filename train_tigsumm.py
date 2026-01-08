#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TigSumm: An Intelligent Cross-Lingual Framework Sentiment-Aware Summarization Training Script
Author: Hagos et al. (2025)
"""

import argparse
import torch
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

    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Pretrained model checkpoint (e.g., 'google/mt5-base' or 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--target_lang", type=str, default="ti",
                        help="Target language code (default: Tigrigna)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tigsumm")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--sentiment_guidance", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = load_dataset("json", data_files={
        "train": f"{args.dataset_path}/train.json",
        "test": f"{args.dataset_path}/test.json"
    })

    model_type = "seq2seq" if "t5" in args.model_name_or_path.lower() or "bart" in args.model_name_or_path.lower() else "decoder"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    # Tokenization
    def preprocess(batch):
        inputs = batch["text"]
        targets = batch["summary"]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        labels = tokenizer(targets, max_length=128, truncation=True).input_ids
        model_inputs["labels"] = labels
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True)

    # Model loading
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
print(f"Loading dataset from {args.dataset_path} ...")

# Expecting TigSumm_Data.csv
dataset = load_dataset(
    "csv",
    data_files=f"{args.dataset_path}/TigSumm_Data.csv"
)

# If a split column exists, use it
if "split" in dataset["train"].column_names:
    train_dataset = dataset["train"].filter(lambda x: x["split"] == "train")
    test_dataset = dataset["train"].filter(lambda x: x["split"] == "test")
else:
    # Otherwise perform an automatic split
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # LoRA adapter
    if args.use_lora:
        print("Applying LoRA adapter for parameter-efficient fine-tuning...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM" if model_type == "decoder" else "SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)

    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved at {args.output_dir}")


if __name__ == "__main__":
    main()
