"""
TigSumm: A Cross-Lingual Framework for Sentiment-Aware Text Summarization
in Low-Resource Tigrigna with Large Language Models
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None
    print("[Warning] PEFT not installed — using full fine-tuning.")

class SentimentHead(nn.Module):
    def __init__(self, hidden_size, num_labels=3):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states):
        logits = self.fc(hidden_states[:, 0, :])
        probs = self.softmax(logits)
        return probs


class SentimentEmbedding(nn.Module):
    def __init__(self, hidden_size, num_labels=3):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, hidden_size)

    def forward(self, sentiment_ids):
        return self.embedding(sentiment_ids)


class SentimentFusionLayer(nn.Module):
    def __init__(self, hidden_size, fusion_type="add"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "gated":
            self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states, sentiment_emb):
        if self.fusion_type == "add":
            return hidden_states + sentiment_emb.unsqueeze(1)
        elif self.fusion_type == "concat":
            return torch.cat((hidden_states, sentiment_emb.unsqueeze(1)), dim=-1)
        elif self.fusion_type == "gated":
            gate_val = torch.sigmoid(self.gate(torch.cat([hidden_states, sentiment_emb.unsqueeze(1)], dim=-1)))
            return hidden_states * gate_val + sentiment_emb.unsqueeze(1) * (1 - gate_val)
        else:
            raise ValueError("Invalid fusion type")


class TigSummModel(nn.Module):
    def __init__(self, model_name, model_type="seq2seq", use_lora=True, fusion_type="add"):
        super().__init__()
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_type == "seq2seq":
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.backbone = AutoModelForCausalLM.from_pretrained(model_name)

        self.hidden_size = self.backbone.config.hidden_size
        self.sentiment_head = SentimentHead(self.hidden_size)
        self.sentiment_emb = SentimentEmbedding(self.hidden_size)
        self.fusion = SentimentFusionLayer(self.hidden_size, fusion_type=fusion_type)

        # LoRA integration
        if use_lora and get_peft_model is not None:
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
            self.backbone = get_peft_model(self.backbone, lora_config)

    def forward(self, input_ids, attention_mask, labels=None, sentiment_ids=None):
        if sentiment_ids is not None:
            sentiment_emb = self.sentiment_emb(sentiment_ids)
        else:
            with torch.no_grad():
                sentiment_emb = self.sentiment_emb(torch.tensor([1]).to(input_ids.device))  # neutral

        outputs = self.backbone.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        fused_hidden = self.fusion(outputs.last_hidden_state, sentiment_emb)

        gen_out = self.backbone.model.decoder(inputs_embeds=fused_hidden, labels=labels)
        sentiment_logits = self.sentiment_head(outputs.last_hidden_state)
        return gen_out.loss, sentiment_logits

    def generate(self, input_text, sentiment_label=None, max_length=128):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        sentiment_id = torch.tensor([sentiment_label]) if sentiment_label is not None else torch.tensor([1])
        sentiment_emb = self.sentiment_emb(sentiment_id)
        outputs = self.backbone.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def num_trainable_parameters(self):
        total, trainable = sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "ratio": trainable / total}


def build_tigsumm_from_pretrained(model_name="facebook/mbart-large-50", model_type="seq2seq", use_lora=True):
    return TigSummModel(model_name, model_type=model_type, use_lora=use_lora)
