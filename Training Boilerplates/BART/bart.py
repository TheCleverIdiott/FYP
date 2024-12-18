# -*- coding: utf-8 -*-
"""14Nov.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AmucVKJBhSXc8GV6AA2WdIcjiBwPvxUc

# Abstractive Text Summarization using BART with ROUGE Evaluation

### Install Necessary Libraries
We start by installing the required libraries:
- `datasets`: For loading and processing the CNN/Daily Mail dataset.
- `transformers`: For loading, fine-tuning, and using the BART model for summarization.
"""

!pip install datasets transformers

"""### Import Libraries
Here, we import the libraries needed for the project. These libraries include functions for dataset loading, tokenization, and model handling.
"""

import os
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"

"""### Load and Prepare the Dataset
We use the `datasets` library to load our custom Dataset.
In this section, we:
1. Load the dataset from a CSV file using `pandas`.
2. Convert the `pandas` DataFrame to a Hugging Face `Dataset` object for compatibility with the `transformers` library.
"""

import pandas as pd
from datasets import Dataset
df = pd.read_csv("/content/train.csv")  # Replace with your actual file name

# Limit to the first 10,000 rows
df_subset = df.head(10000)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_subset)

"""### Initialize the Tokenizer
The BART model requires tokenized input text. We use the BART tokenizer to convert each article and summary into tokens, which are compatible with the model.
"""

# Load the BART tokenizer and model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

"""### Data Preprocessing
This function tokenizes both the article and summary. We set maximum lengths to ensure compatibility with the model’s input requirements, truncating longer texts and padding shorter ones as needed.
"""

# Preprocess the data with reduced max_length for memory optimization
def preprocess_data(examples):
    inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples['highlights'], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)

"""### Set Training Parameters
The training parameters define key aspects of the fine-tuning process, such as:
- **Learning Rate**: Controls the adjustment of model weights during training.
- **Batch Size**: Number of samples processed before updating the model weights.
- **Epochs**: Number of complete passes through the dataset.
- **FP16 Precision**: Using half-precision (16-bit floating point) for faster computation on compatible GPUs.
These parameters optimize the training for memory and performance constraints.

### Train the Model
We use the Hugging Face `Trainer` API to fine-tune the BART model on the CNN/Daily Mail dataset. This API handles the training loop, gradient accumulation, and checkpoint saving.

"""

# Set training arguments with memory optimization options
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,           # Small batch size to reduce memory usage
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,           # Accumulate gradients over 4 steps to simulate a larger batch size
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,                               # Enable mixed precision for lower memory usage
    evaluation_strategy="epoch",
    report_to="none"                         # Disable wandb logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset  # Use a subset or split if you want separate evaluation data
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./my_cnn_dailymail_bart_model")
tokenizer.save_pretrained("./my_cnn_dailymail_bart_model")

import torch

def summarize(text):
    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Tokenize input and move input tensors to the same device as the model
    inputs = tokenizer([text], max_length=512, return_tensors="pt", truncation=True).to(device)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)

    # Decode the output and return the summary text
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test the summarization function on a sample text
sample_text = "The Mars rover Perseverance has been exploring the Red Planet for almost a year. During its mission, the rover has collected various samples of rocks and soil to help scientists understand the planet's history and whether it once supported life. The mission aims to provide more insights into the geological processes that shaped Mars. Recently, Perseverance encountered a unique rock formation that has intrigued researchers. These findings may shed light on past water activity on Mars and could be crucial for future human exploration."
print("Summary:", summarize(sample_text))

!pip install rouge_score datasets

!pip install evaluate

import evaluate
import torch

# Load ROUGE metric from the evaluate library
rouge = evaluate.load("rouge")

# Define function to evaluate ROUGE score
def evaluate_rouge(model, tokenizer, dataset, num_samples=100):
    predictions = []
    references = []

    for i, sample in enumerate(dataset):
        if i >= num_samples:  # Limit the number of samples to avoid memory overload
            break

        # Get the article and reference summary
        article = sample["article"]
        reference_summary = sample["highlights"]

        # Generate summary from the model
        inputs = tokenizer([article], max_length=512, return_tensors="pt", truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
        summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Append generated and reference summaries for ROUGE calculation
        predictions.append(generated_summary)
        references.append(reference_summary)

    # Compute ROUGE scores
    results = rouge.compute(predictions=predictions, references=references)
    return results

"""### Evaluate Model Performance with ROUGE Scores
We evaluate the trained model using the ROUGE metric, which measures the overlap of n-grams between the generated summaries and the reference summaries.
- **ROUGE-1**: Measures the overlap of individual words.
- **ROUGE-2**: Measures the overlap of bigrams (two consecutive words).
- **ROUGE-L**: Measures the longest common subsequence between the generated and reference summaries.
The ROUGE scores provide a quantitative measure of summarization quality.
"""

# Calculate ROUGE score on a sample of 100 summaries from the "train" split
rouge_results = evaluate_rouge(model, tokenizer, tokenized_dataset, num_samples=100)  # Assuming `tokenized_dataset` is your dataset

# Display the ROUGE results
print("ROUGE Scores:", rouge_results)

"""### Our ROUGE scores are quite good for a text summarization model:

- **ROUGE-1**: 0.488 (~48.8%) – This measures the overlap of unigrams (individual words) between the generated and reference summaries. A score close to 50% is solid for a summarization task, as it indicates a good amount of relevant word overlap.

- **ROUGE-2**: 0.300 (~30.0%) – This measures bigram overlap (two consecutive words) and is usually lower than ROUGE-1. A 30% score is a good indication that our model is capturing some meaningful phrases from the reference summary, which is essential for summarization quality.

- **ROUGE-L**: 0.382 (~38.2%) – ROUGE-L considers the longest common subsequence, capturing the fluency and coherence of the summary. A score in the range of 35-40% is generally considered strong in this context, as it suggests that the generated summaries are not just relevant but also coherent.

- **ROUGE-Lsum**: 0.469 (~46.9%) – This variant of ROUGE-L is calculated specifically for summarization tasks and takes into account the overall sentence structure. A score near 47% is also a good sign.

### Interpretation
Our model's scores suggest that:

- It has captured the core content of the text well (high ROUGE-1).
- It maintains some degree of fluency and coherence (decent ROUGE-L and ROUGE-Lsum).
- It includes meaningful bigrams (decent ROUGE-2).

In general, ROUGE scores in these ranges (especially 30%+ for ROUGE-2 and ROUGE-L, and around 45%+ for ROUGE-1) are indicative of a well-performing summarization model, especially for complex datasets like our custom dataset.
"""

