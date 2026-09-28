from __future__ import annotations
import os, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ML.training.config import TextTrainConfig
from ML.training.dataset import SlopDataset
from ML.training.metrics import compute_metrics, save_metrics


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_clean() -> pd.DataFrame:
    path = Path("ML/data/processed/combined_clean.parquet")
    if not path.exists():
        sys.exit("Run train_baseline.py first to produce combined_clean.parquet")
    return pd.read_parquet(path)


def main() -> None:
    cfg = TextTrainConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = load_clean()
    y = df[cfg.target_column]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        df["clean"], y, test_size=cfg.val_size + cfg.test_size,
        stratify=y, random_state=cfg.seed,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=cfg.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2,
    )

    train_ds = SlopDataset(X_tr, y_tr, tokenizer, cfg.max_length)
    val_ds   = SlopDataset(X_val, y_val, tokenizer, cfg.max_length)
    test_ds  = SlopDataset(X_te, y_te, tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=["mlflow"],
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,          # <- было tokenizer=tokenizer (deprecated)
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

   

    mlflow.set_experiment("ai-slop-text-detector")
    with mlflow.start_run():
        mlflow.log_params(cfg.__dict__)
        trainer.train()
        test_metrics = trainer.evaluate(test_ds)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        save_metrics(test_metrics, Path(cfg.output_dir) / "test_metrics.json")
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
