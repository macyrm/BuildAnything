import os
import torch
import numpy as np
import evaluate
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, IntervalStrategy
from datasets import load_from_disk, ClassLabel
from sklearn.utils import compute_class_weight

PROCESSED_DATA_DIR="./processed_data"
BASE_MODEL="distilbert-base-uncased"
MODEL_SAVE_PATH = os.environ.get("MODEL_PATH", "./models/alzheimer_classifier")

def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    references = np.argmax(labels, axis=-1)
    f1_result = metric.compute(predictions=predictions, references=references, average="weighted")
    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=references)
    return {"accuracy": accuracy["accuracy"], "f1": f1_result["f1"]}

def filter_invalid_labels(example):
    """Filters out examples where the 'labels' value cannot be converted to an integer."""
    try:
        int(example['labels'])
        return True 
    except ValueError:
        return False

def one_hot_encode_labels(example, num_labels=2):
    """Converts the integer label into a one-hot vector of floats."""
    one_hot = [0.0] * num_labels
    # The 'labels' value is an integer index (0 or 1)
    index = int(example['labels'])
    one_hot[index] = 1.0
    example['labels'] = one_hot
    return example

def train_and_save_model():
    try:
        print(f"Loading tokenized datasets from {PROCESSED_DATA_DIR}...")
        tokenized_train_dataset = load_from_disk(os.path.join(PROCESSED_DATA_DIR, "train_dataset"))
        tokenized_test_dataset = load_from_disk(os.path.join(PROCESSED_DATA_DIR, "test_dataset"))
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print(e)
        return
    print("Checking for and removing rows with missing labels (NaN)...")
    # num_labels = 2

    new_features = tokenized_train_dataset.features.copy()
    new_features["labels"] = ClassLabel(num_classes=2, names=["Class 0", "Class 1"])
    
    tokenized_train_dataset = tokenized_train_dataset.cast(new_features)
    tokenized_test_dataset = tokenized_test_dataset.cast(new_features)
    
    print("Filtering invalid labels from training dataset...")
    initial_train_count = len(tokenized_train_dataset)
    initial_test_count = len(tokenized_test_dataset)

    tokenized_train_dataset = tokenized_train_dataset.filter(filter_invalid_labels)
    tokenized_test_dataset = tokenized_test_dataset.filter(filter_invalid_labels)

    print(f"Removed {initial_train_count - len(tokenized_train_dataset)} invalid rows from training dataset.")
    print(f"Removed {initial_test_count - len(tokenized_test_dataset)} invalid rows from test dataset.")

    print("Calculating class weights to address data imbalance...")
    train_labels = [int(label) for label in tokenized_train_dataset['labels']]
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    weights = torch.tensor(class_weights, dtype=torch.float32).to('cpu')
    print(f"Calculated Class Weights (Code 0, Code 1): {class_weights}")
    
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            
            target_labels = torch.argmax(labels, dim=1) 
            
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), target_labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
        
    tokenized_train_dataset = tokenized_train_dataset.map(one_hot_encode_labels, batched=False)
    tokenized_test_dataset = tokenized_test_dataset.map(one_hot_encode_labels, batched=False)
    
    columns_to_remove = ['Diagnosis', 'token_type_ids']
    train_cols_to_remove = [col for col in columns_to_remove if col in tokenized_train_dataset.column_names]
    test_cols_to_remove = [col for col in columns_to_remove if col in tokenized_test_dataset.column_names]
    
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(train_cols_to_remove)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(test_cols_to_remove)
    
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    
    tokenized_train_dataset.set_format(type="torch", columns=columns_to_keep)
    tokenized_test_dataset.set_format(type="torch", columns=columns_to_keep)
    
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        # eval_strategy="epoch",
        # save_strategy="epoch",
        # eval_strategy=IntervalStrategy.EPOCH,
        # save_strategy=IntervalStrategy.EPOCH,
        # logging_strategy=IntervalStrategy.EPOCH,
        #eval_strategy="epoch", # Change to "steps"
        #eval_steps=500, # Add a step value to evaluate every 500 steps
        #save_strategy="epoch",
        #save_steps=500, # Add a step value to save every 500 steps
        gradient_accumulation_steps=4,
        dataloader_num_workers=0,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        warmup_steps=5,
        weight_decay=0.001,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
    )
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    print("Starting model training...")
    trainer.train()
    print("Training completed. Saving the model...")
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model and tokenizer saved to {MODEL_SAVE_PATH}.")
    print("Model training and saving process completed successfully.")

if __name__ == "__main__":
    train_and_save_model()