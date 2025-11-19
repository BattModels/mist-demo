"""
Helper functions for optimizing fine-tuning hyperparameters.
"""
import optuna 
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from .regression_model import RegressionModel

def objective(trial, encoder, tokenizer, train_dataset, val_dataset, device):
    """
    Optuna objective function for Bayesian optimization.
    """
    # Suggest hyperparameters
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Create model with suggested hyperparameters
    model = RegressionModel(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        dropout=dropout,
        num_hidden_layers=num_hidden_layers
    ).to(device)
    
    training_args = TrainingArguments(
        output_dir=f"./optuna_trial_{trial.number}",
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        logging_steps=100,
        disable_tqdm=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results['eval_loss']


def tune_hyperparameters(encoder, tokenizer, train_dataset, val_dataset, 
                         device, n_trials=20):
    
    # Create smaller datasets for faster tuning
    small_train = train_dataset.select(range(min(5000, len(train_dataset))))
    small_val = val_dataset.select(range(min(1000, len(val_dataset))))
    
    print(f"Running Bayesian optimization with {n_trials} trials...")
    print(f"Train size: {len(small_train)}, Val size: {len(small_val)}\n")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, encoder, tokenizer, small_train, small_val, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"Optimization complete!")
    print(f"Best trial:")
    print(f"Value (eval_loss): {study.best_trial.value:.4f}")
    print(f"Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.params, study