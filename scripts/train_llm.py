import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import pandas as pd

# Check if the dataset file exists
cache_dir='./new_cache'  # Change to a different directory

dataset_path = 'dataset/preprocessed_data.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at: {dataset_path}. Please make sure the file exists.")

try:
    # Load the dataset with error handling
    dataset = load_dataset(
        'csv',
        data_files=dataset_path,
        split='train',
        cache_dir='./cache'  # Local cache directory
    )

    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Tokenize the dataset
    def tokenize_function(examples):
        # Make sure your CSV has a 'text' column
        return tokenizer(
            examples["text"] if "text" in examples else examples[examples.keys()[0]],
            padding="max_length",
            truncation=True,
            max_length=512  # Adjust this based on your needs
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names  # Remove original columns after tokenization
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=4,  # Reduced batch size to prevent OOM errors
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    print("Saving model...")
    trainer.save_model("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Training completed successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nDebugging information:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset path exists: {os.path.exists(dataset_path)}")