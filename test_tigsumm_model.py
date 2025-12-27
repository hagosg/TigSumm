# tests/test_tigsumm_model.py
import os
import tempfile
import torch
from tigsumm_model import build_tigsumm_from_pretrained, TigSummConfig, count_parameters

def test_build_and_forward_tiny_mbart():
    model_name = "sshleifer/tiny-mbart"
    cfg = TigSummConfig(model_name_or_path=model_name, use_lora=False, device="cpu")
    model = build_tigsumm_from_pretrained(model_name_or_path=model_name, use_lora=False, device="cpu")
    tokenizer = model.tokenizer

    texts = ["This is a short test sentence.", "Another test input for unit testing."]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

    # create dummy labels (shifted input)
    labels = enc["input_ids"].clone()
    sentiments = torch.tensor([0, 1], dtype=torch.long)

    model.eval()
    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels, sentiment_labels=sentiments)
    assert "loss" in outputs
    assert outputs["loss"].item() >= 0

    # Test save/load in a temp dir
    tmpdir = tempfile.mkdtemp()
    model.save_pretrained(tmpdir)
    # Load back
    new_model = build_tigsumm_from_pretrained(model_name_or_path=tmpdir, use_lora=False, device="cpu")
    assert isinstance(new_model, object)
    # Parameter count sanity
    counts = count_parameters(model)
    assert counts["total"] > 0
    assert counts["trainable"] >= 0

def test_generate_short():
    model_name = "sshleifer/tiny-mbart"
    model = build_tigsumm_from_pretrained(model_name_or_path=model_name, use_lora=False, device="cpu")
    text = ["The weather is sunny today and people are happy."]
    out = model.generate(text, max_new_tokens=20)
    assert isinstance(out, list)
    assert len(out) == 1
