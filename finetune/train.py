from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("json", data_files="data/dpo_dataset.json", split="train")

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=2,
    learning_rate=5e-6,
    logging_dir="./logs",
    output_dir="./dpo-model",
    save_total_limit=1,
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

trainer.train()