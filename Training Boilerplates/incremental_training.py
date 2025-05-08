# Install required libraries
!pip install transformers datasets evaluate rouge_score

# === Import Libraries ===
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
import evaluate

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# === Settings ===
# Ask user for base model choice
model_choice = input("Enter base model (bart or pegasus): ").strip().lower()
if model_choice == "bart":
    BASE_MODEL = "facebook/bart-large-cnn"
elif model_choice == "pegasus":
    BASE_MODEL = "google/pegasus-xsum"
else:
    raise ValueError("Invalid model choice. Choose either 'bart' or 'pegasus'.")

# Ask user for CSV path
CSV_PATH = input("Enter full path to your dataset CSV file: ").strip()

print(f"\n Using model: {BASE_MODEL}")
print(f"Using dataset: {CSV_PATH}")

CHUNK_SIZE = 5000                        # Number of samples per training round
EPOCHS = 1                               # You can increase if needed
NUM_CHUNKS = 10                          # e.g., 50,000 / 5,000

# === Load and Preprocess Entire Dataset ===
df = pd.read_csv(CSV_PATH).dropna(subset=['article', 'highlights']).reset_index(drop=True)

# === Initialize Tokenizer ===
tokenizer = BartTokenizer.from_pretrained(BASE_MODEL)

# === Define Preprocessing Function ===
def preprocess_data(examples):
    inputs = tokenizer(
        examples["article"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    targets = tokenizer(
        examples["highlights"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

# === Define Metric Computation ===
rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

# === Loop Over Dataset in Chunks ===
for chunk_index in range(NUM_CHUNKS):
    print(f"\n🔁 Training Chunk {chunk_index + 1}/{NUM_CHUNKS}...")

    # === Slice 5,000 rows ===
    start = chunk_index * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk_df = df.iloc[start:end].reset_index(drop=True)
    
    # Skip if no data
    if chunk_df.empty:
        print(f"No data found in chunk {chunk_index + 1}. Skipping.")
        continue

    # === Convert to Dataset and Tokenize ===
    chunk_dataset = Dataset.from_pandas(chunk_df)
    tokenized_chunk = chunk_dataset.map(preprocess_data, batched=True)

    # === Load Model (m0, m1, ...) ===
    model_checkpoint = BASE_MODEL if chunk_index == 0 else f"./model_m{chunk_index}"
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

    # === Move to GPU if available ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir=f"./model_m{chunk_index + 1}",       # Save m1, m2, ...
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="no",                       # Skip eval here
        logging_dir="./logs",
        save_total_limit=1,
        report_to="none"
    )

    # === Initialize Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_chunk,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        compute_metrics=compute_metrics
    )

    # === Train and Save the Model ===
    trainer.train()
    trainer.save_model(f"./model_m{chunk_index + 1}")
    tokenizer.save_pretrained(f"./model_m{chunk_index + 1}")

    print(f"Saved model_m{chunk_index + 1}\n")

print("🎯 Training complete for all chunks.")
