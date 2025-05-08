
import torch
import pandas as pd
from qafacteval.model import QAFactEval
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration,
    BertTokenizer, BertForMaskedLM
)
from sklearn.metrics.pairwise import cosine_similarity
import evaluate

# === Load Models ===
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

rouge = evaluate.load("rouge")
qae_model = QAFactEval(model_path="./qafacteval_model/", device="cuda" if torch.cuda.is_available() else "cpu")

# === Helper Functions ===
def get_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs).logits
    return outputs.mean(dim=1).numpy()

def extract_meta_features(source, summary):
    factuality_score = qae_model.score([source], [summary])[0]
    rouge_scores = rouge.compute(predictions=[summary], references=[source])
    compression = len(summary.split()) / (len(source.split()) + 1e-6)
    similarity = cosine_similarity(get_embedding(source), get_embedding(summary)).flatten()[0]
    return [
        rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL'],
        factuality_score, compression, similarity
    ]

def get_summary_and_logits(model, tokenizer, source):
    inputs = tokenizer(source, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(inputs["input_ids"], output_scores=True, return_dict_in_generate=True)
    summary_ids = output.sequences
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"]).logits
    return summary_text, logits

def build_fwls_datapoint(source, target):
    summaries, logits, features = [], [], []

    s1, l1 = get_summary_and_logits(bart_model, bart_tokenizer, source)
    s2, l2 = get_summary_and_logits(pegasus_model, pegasus_tokenizer, source)
    s3, l3 = s1, l1
    s4, l4 = s2, l2

    for s in [s1, s2, s3, s4]:
        features.append(extract_meta_features(source, s))

    summaries = [s1, s2, s3, s4]
    logits = [l1, l2, l3, l4]

    features_tensor = torch.tensor(features).float()
    target_tensor = bart_tokenizer(target, return_tensors="pt", max_length=128, truncation=True)["input_ids"].squeeze(0)
    return features_tensor, target_tensor, logits

if __name__ == "__main__":
    source = "The Mars rover Perseverance has been collecting rock samples on Mars."
    target = "Perseverance collects samples to study life signs."

    f, t, l = build_fwls_datapoint(source, target)
    print("Meta-features:", f.shape)
    print("Target token IDs:", t.shape)
    print("Logits shapes:", [li.shape for li in l])
