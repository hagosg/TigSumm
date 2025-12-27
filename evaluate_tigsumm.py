#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TigSumm Evaluation Script
Evaluates trained model using ROUGE-L, BERTScore, and Sentiment Preservation Rate
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from utils import compute_metrics
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TigSumm Model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    except:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    data = load_dataset("json", data_files={"test": args.test_file})["test"]

    inputs = data["text"]
    labels = data["summary"]

    preds = []
    for inp in inputs:
        inputs_enc = tokenizer(inp, return_tensors="pt", truncation=True, max_length=512).to(device)
        output = model.generate(**inputs_enc, max_new_tokens=128)
        preds.append(tokenizer.decode(output[0], skip_special_tokens=True))

    metrics = compute_metrics(preds, labels)
    print("\n==== TigSumm Evaluation Results ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
