import torch
from tigsumm_model import build_tigsumm_from_pretrained

model = build_tigsumm_from_pretrained("facebook/mbart-large-50", use_lora=False)
text = "እቲ መንነት ዝተኸፈለ መጽናዕቲ ኣዝዩ ብርቱዕ እዩ።"
summary = model.generate(text, sentiment_label=2)
print(summary)
