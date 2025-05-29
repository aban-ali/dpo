from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
qa_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

with open("data/input_text.txt", "r") as f:
    content = f.read()

prompt = f"""Based on this content, generate a question and two different answers:
{content[:1000]}  # You can chunk if needed
Format:
Q: ...
A1: ...
A2: ...
"""

outputs = qa_gen(prompt, max_new_tokens=300)[0]['generated_text']
with open("data/qa_pairs.txt", "w") as f:
    f.write(outputs)
