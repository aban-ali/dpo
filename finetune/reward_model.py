from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

rm_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
tokenizer = AutoTokenizer.from_pretrained(rm_name)
model = AutoModelForSequenceClassification.from_pretrained(rm_name).cuda()

def score_pair(question, answer):
    input_text = f"Question: {question}\nAnswer: {answer}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        scores = model(**inputs).logits
    return scores.item()
