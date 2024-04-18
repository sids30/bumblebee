# Import required libraries
from transformers import RagTokenizer, RagTokenForGeneration
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments
import pandas as pd

# Load scraped data
with open("durham_college_data.txt", "r") as file:
    scraped_data = file.read()

# Preprocess data
documents = scraped_data.split('\n\n')  # Split documents by empty lines

# Create question-answer pairs
qa_pairs = []
for doc in documents:
    lines = doc.strip().split('\n')
    if len(lines) > 1:  # Skip single line sections
        question = lines[0].replace(':', '')  # Extract question from section title
        answer = '\n'.join(lines[1:])  # Use the rest as the answer
        qa_pairs.append({'question': question, 'answer': answer})

# Convert to DataFrame
df = pd.DataFrame(qa_pairs)

# Save QA pairs to a CSV file
df.to_csv("qa_pairs.csv", index=False)

# Load dataset from CSV
dataset = Dataset.from_csv("qa_pairs.csv")

# Load the RagTokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

# Load the RagTokenForGeneration
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

# Define TrainingArguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    output_dir="./models",
    logging_dir="./logs",
    num_train_epochs=3,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train RAG model
trainer.train()

# Save trained model
trainer.save_model("./models/rag-trained")

# Evaluate the model on a test dataset
test_dataset = load_dataset('csv', data_files={'test': 'qa_pairs.csv'}, trust_remote_code=True)['test']

# Define function to compute exact match
def compute_exact_match(predictions, references):
    exact_match = 0
    for pred, ref in zip(predictions, references):
        if pred.strip() == ref.strip():
            exact_match += 1
    return exact_match / len(predictions)

# Evaluate model
predictions = trainer.predict(test_dataset)
exact_match = compute_exact_match(predictions.predictions, test_dataset['answer'])
print(f"Exact Match: {exact_match}")

# Define function to compute BLEU score
def compute_bleu(predictions, references):
    total_bleu = 0
    for pred, ref in zip(predictions, references):
        total_bleu += sentence_bleu([ref.strip().split()], pred.strip().split())
    return total_bleu / len(predictions)

# Compute BLEU score
bleu_score = compute_bleu(predictions.predictions, test_dataset['answer'])
print(f"BLEU Score: {bleu_score}")
