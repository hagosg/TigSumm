import torch
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tigsumm_model import build_tigsumm_from_pretrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_tigsumm_from_pretrained("facebook/mbart-large-50", use_lora=True).to(device)
print(model.num_trainable_parameters())

dataset = load_dataset("json", data_files={"train": "data/train.json", "val": "data/val.json"})

optimizer = AdamW(model.parameters(), lr=3e-5)
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in dataset["train"]:
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)
        sentiment_ids = torch.tensor(batch["sentiment_id"]).to(device)

        loss, _ = model(input_ids, attention_mask, labels=labels, sentiment_ids=sentiment_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} done")

model.save_pretrained("checkpoints/tigsumm_final")
