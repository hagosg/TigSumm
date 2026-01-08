#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TigSumm: An Intelligent Cross-Lingual Framework for Sentiment-Aware Text Summarization in Low-Resource Tigrigna

Author: Hagos et al. (2026)
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

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Pretrained model (e.g., google/mt5-base, facebook/mbart-large-50, meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path containing TigSumm_Data.csv",
    )
    parser.add_argument(
        "--output_dir",
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

    print(f"Loading dataset from {args.dataset_path}/TigSumm_Data.csv")

    dataset = load_dataset(
        "csv",
        data_files=f"{args.dataset_path}/TigSumm_Data.csv"
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
    model_name = args.model_name_or_path.lower()
    is_seq2seq = any(k in model_name for k in ["t5", "bart"])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
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
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

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
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
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

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
