from evaluate import load

rouge = load("rouge")
bertscore = load("bertscore")


def compute_metrics(preds, refs):
    rouge_result = rouge.compute(predictions=preds, references=refs)
    bert_result = bertscore.compute(predictions=preds, references=refs, lang="en")

    metrics = {
        "ROUGE-L": rouge_result["rougeL"],
        "BERTScore": sum(bert_result["f1"]) / len(bert_result["f1"]),
        "SPR": _estimate_sentiment_preservation(preds, refs)
    }
    return metrics


def _estimate_sentiment_preservation(preds, refs):
    from textblob import TextBlob
    same = 0
    for p, r in zip(preds, refs):
        if (TextBlob(p).sentiment.polarity * TextBlob(r).sentiment.polarity) > 0:
            same += 1
    return 100 * same / len(preds)
