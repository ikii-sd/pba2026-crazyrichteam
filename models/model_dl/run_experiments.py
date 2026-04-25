"""
run_experiments.py
==================
Main experiment runner — trains multiple BiLSTM configurations,
evaluates them on the test set, and selects the best model.

Usage (from repo root):
    python models/model_dl/run_experiments.py

Or (from models/model_dl/):
    python run_experiments.py
"""

from __future__ import annotations

import csv
import os
import shutil
import sys

# ── path setup so we can import from models/ regardless of cwd ──────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # pba2026-crazyrichteam/
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import torch

from models.model_design import SimpleBiLSTM, TextCNNMultiOutput, count_parameters
from models.data_processor import process_data, save_label_encoders
from models.vocabulary_builder import Vocabulary
from models.config_simplified import (
    DATASET_PATH,
    TEXT_COLUMN,
    SENTIMENT_LABEL_COLUMN,
    EMOTION_LABEL_COLUMN,
    DEVICE,
    VOCAB_PATH,
    LABEL_ENCODERS_PATH,
    MODEL_DIR,
    ARTIFACT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
)
from models.model_dl.data_pipeline import create_dataloaders
from models.model_dl.training_pipeline import train_model, save_checkpoint
from models.model_dl.evaluation import (
    test_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)

# ── Override paths (config_simplified.py was designed for a deeper dir tree) ─
# SCRIPT_DIR = models/model_dl/  →  MODELS_DIR = models/  →  REPO_ROOT = pba2026-…/
_MODELS_DIR = os.path.dirname(SCRIPT_DIR)             # D:/pba2026-crazyrichteam/models
_REPO_ROOT   = os.path.dirname(_MODELS_DIR)           # D:/pba2026-crazyrichteam
_DATA_PATH   = os.path.join(_REPO_ROOT, "data", "clean", "cleaned_dataset.csv")
_MODEL_OUT   = os.path.join(_MODELS_DIR, "model_dl", "saved_models")
_ARTIFACT_OUT= os.path.join(_MODELS_DIR, "model_dl", "artifacts")
PLOT_DIR     = os.path.join(SCRIPT_DIR, "plots")
VOCAB_PATH   = os.path.join(_ARTIFACT_OUT, "vocab_simplified.json")
LABEL_ENC_PATH = os.path.join(_ARTIFACT_OUT, "label_encoders_simplified.json")

# Override config values
import pathlib
DATASET_PATH     = pathlib.Path(_DATA_PATH)
MODEL_DIR_STR    = _MODEL_OUT
ARTIFACT_DIR_STR = _ARTIFACT_OUT

for _d in [PLOT_DIR, MODEL_DIR_STR, ARTIFACT_DIR_STR]:
    os.makedirs(_d, exist_ok=True)


def build_model(
    experiment: dict,
    vocab_size: int,
    num_sentiment_classes: int,
    num_emotion_classes: int,
):
    model_type = experiment["model_type"]

    common_kwargs = {
        "vocab_size": vocab_size,
        "embedding_dim": experiment.get("embedding_dim", 128),
        "dropout": experiment["dropout"],
        "num_sentiment_classes": num_sentiment_classes,
        "num_emotion_classes": num_emotion_classes,
    }

    if model_type == "bilstm":
        return SimpleBiLSTM(
            hidden_dim=experiment["hidden_dim"],
            num_layers=experiment["num_layers"],
            **common_kwargs,
        )

    if model_type == "textcnn":
        return TextCNNMultiOutput(
            num_filters=experiment["num_filters"],
            kernel_sizes=tuple(experiment["kernel_sizes"]),
            **common_kwargs,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    bar = "=" * 70
    print(bar)
    print("🚀  EXPERIMENT RUNNER — BiLSTM Sentiment & Emotion Analysis")
    print(bar)
    print(f"  Device   : {DEVICE}")
    print(f"  Dataset  : {DATASET_PATH}")
    print(f"  Model dir: {MODEL_DIR_STR}")
    print(f"  Plot dir : {PLOT_DIR}")

    # ── STEP 1: Load & process data ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[STEP 1]  Loading and processing data ...")

    processed = process_data(
        csv_path=DATASET_PATH,
        text_column=TEXT_COLUMN,
        sentiment_column=SENTIMENT_LABEL_COLUMN,
        emotion_column=EMOTION_LABEL_COLUMN,
        random_state=RANDOM_SEED,
    )
    print(f"  Train: {len(processed.train_df)} | "
          f"Val: {len(processed.val_df)} | "
          f"Test: {len(processed.test_df)}")
    print(f"  Sentiment classes : {processed.sentiment_label2id}")
    print(f"  Emotion   classes : {processed.emotion_label2id}")

    # Save label encoders
    save_label_encoders(
        output_path=LABEL_ENC_PATH,
        sentiment_label2id=processed.sentiment_label2id,
        emotion_label2id=processed.emotion_label2id,
    )

    # ── STEP 2: Build vocabulary ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[STEP 2]  Building vocabulary …")

    all_texts = pd.concat(
        [processed.train_df, processed.val_df, processed.test_df]
    )["text"].tolist()

    vocab = Vocabulary()
    vocab.build_from_texts(all_texts)
    vocab.save(VOCAB_PATH)
    print(f"  Vocab size: {len(vocab):,}  →  saved to {VOCAB_PATH}")

    # ── STEP 3: DataLoaders ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[STEP 3]  Creating DataLoaders …")

    loaders = create_dataloaders(processed, vocab, batch_size=BATCH_SIZE)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    num_sent = len(processed.sentiment_label2id)
    num_emo = len(processed.emotion_label2id)
    vocab_size = len(vocab)

    # ── STEP 4: Experiment definitions ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[STEP 4]  Experiment configurations …")

    experiments = [
        {
            "name": "BiLSTM_Baseline",
            "model_type": "bilstm",
            "hidden_dim": 128,
            "num_layers": 1,
            "dropout": 0.3,
            "lr": 1e-3,
            "epochs": min(NUM_EPOCHS, 15),
        },
        {
            "name": "BiLSTM_Improved",
            "model_type": "bilstm",
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.4,
            "lr": 5e-4,
            "epochs": min(NUM_EPOCHS, 20),
        },
        {
            "name": "BiLSTM_Large",
            "model_type": "bilstm",
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.5,
            "lr": 3e-4,
            "epochs": min(NUM_EPOCHS, 25),
        },
        {
            "name": "TextCNN",
            "model_type": "textcnn",
            "num_filters": 128,
            "kernel_sizes": [3, 4, 5],
            "dropout": 0.4,
            "lr": 1e-3,
            "epochs": min(NUM_EPOCHS, 15),
        },
    ]

    for exp in experiments:
        if exp["model_type"] == "bilstm":
            print(f"  - {exp['name']}  type={exp['model_type']}  hidden={exp['hidden_dim']}  "
                  f"layers={exp['num_layers']}  dropout={exp['dropout']}  "
                  f"lr={exp['lr']}  max_epochs={exp['epochs']}")
        else:
            print(f"  - {exp['name']}  type={exp['model_type']}  filters={exp['num_filters']}  "
                  f"kernels={exp['kernel_sizes']}  dropout={exp['dropout']}  "
                  f"lr={exp['lr']}  max_epochs={exp['epochs']}")

    # ── STEP 5: Train & evaluate ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("[STEP 5]  Starting experiments …")

    results_csv_path = os.path.join(MODEL_DIR_STR, "experiment_results.csv")
    results: dict = {}

    csv_fields = [
        "experiment", "model_type", "sent_accuracy", "sent_f1_macro", "sent_f1_weighted",
        "emo_accuracy", "emo_f1_macro", "emo_f1_weighted",
        "training_time_min", "total_params", "model_path",
    ]

    with open(results_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for exp in experiments:
            print(f"\n{'─'*70}")
            print(f"  ▶  {exp['name']}")
            print(f"{'─'*70}")

            model = build_model(
                experiment=exp,
                vocab_size=vocab_size,
                num_sentiment_classes=num_sent,
                num_emotion_classes=num_emo,
            )
            total_params = count_parameters(model)
            print(f"  Parameters: {total_params:,}")

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=exp["epochs"],
                learning_rate=exp["lr"],
                patience=EARLY_STOPPING_PATIENCE,
                model_name=exp["name"],
                device=DEVICE,
            )

            # save checkpoint
            model_path = os.path.join(MODEL_DIR_STR, f"{exp['name']}.pt")
            save_checkpoint(
                model=model,
                vocab=vocab,
                sentiment_label2id=processed.sentiment_label2id,
                emotion_label2id=processed.emotion_label2id,
                metadata={
                    "experiment": exp["name"],
                    "model_type": exp["model_type"],
                    "embedding_dim": exp.get("embedding_dim", 128),
                    "dropout": exp["dropout"],
                    "learning_rate": exp["lr"],
                    "vocab_size": vocab_size,
                    "num_sentiment_classes": num_sent,
                    "num_emotion_classes": num_emo,
                    "total_params": total_params,
                    "training_time_min": round(history["total_time_minutes"], 2),
                    "hidden_dim": exp.get("hidden_dim"),
                    "num_layers": exp.get("num_layers"),
                    "num_filters": exp.get("num_filters"),
                    "kernel_sizes": exp.get("kernel_sizes"),
                },
                path=model_path,
            )

            # evaluate on test set
            sent_preds, sent_labels, emo_preds, emo_labels = test_model(
                model=model,
                dataloader=test_loader,
                device=DEVICE,
            )

            sent_metrics = compute_metrics(
                sent_labels, sent_preds,
                task_name=f"Sentiment — {exp['name']}"
            )
            emo_metrics = compute_metrics(
                emo_labels, emo_preds,
                task_name=f"Emotion — {exp['name']}"
            )

            results[exp["name"]] = {
                "sentiment": sent_metrics,
                "emotion": emo_metrics,
                "training_time": history["total_time_minutes"],
                "total_params": total_params,
                "model_path": model_path,
            }

            writer.writerow({
                "experiment":       exp["name"],
                "model_type":       exp["model_type"],
                "sent_accuracy":    f"{sent_metrics['accuracy']:.4f}",
                "sent_f1_macro":    f"{sent_metrics['f1_macro']:.4f}",
                "sent_f1_weighted": f"{sent_metrics['f1_weighted']:.4f}",
                "emo_accuracy":     f"{emo_metrics['accuracy']:.4f}",
                "emo_f1_macro":     f"{emo_metrics['f1_macro']:.4f}",
                "emo_f1_weighted":  f"{emo_metrics['f1_weighted']:.4f}",
                "training_time_min": f"{history['total_time_minutes']:.1f}",
                "total_params":     total_params,
                "model_path":       model_path,
            })
            csv_file.flush()  # write row immediately

            # ── plots ──────────────────────────────────────────────────────
            plot_training_curves(
                history,
                model_name=exp["name"],
                save_path=os.path.join(PLOT_DIR, f"training_curves_{exp['name']}.png"),
            )

            # label lists for confusion matrices
            sent_idx2name = {v: k for k, v in processed.sentiment_label2id.items()}
            emo_idx2name = {v: k for k, v in processed.emotion_label2id.items()}
            sent_label_names = [sent_idx2name[i] for i in sorted(sent_idx2name)]
            emo_label_names = [emo_idx2name[i] for i in sorted(emo_idx2name)]

            plot_confusion_matrix(
                sent_labels, sent_preds,
                labels=sent_label_names,
                task_name=f"Sentiment — {exp['name']}",
                save_path=os.path.join(PLOT_DIR, f"cm_sentiment_{exp['name']}.png"),
            )
            plot_confusion_matrix(
                emo_labels, emo_preds,
                labels=emo_label_names,
                task_name=f"Emotion — {exp['name']}",
                save_path=os.path.join(PLOT_DIR, f"cm_emotion_{exp['name']}.png"),
            )

    # ── STEP 6: Select and export best model ────────────────────────────────
    print(f"\n{'='*70}")
    print("📊  EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    best_name, best_info = max(
        results.items(),
        key=lambda x: x[1]["sentiment"]["f1_macro"],
    )

    print(f"\n  ✨ BEST MODEL : {best_name}")
    print(f"     Sentiment Accuracy  : {best_info['sentiment']['accuracy']:.4f}")
    print(f"     Sentiment F1 Macro  : {best_info['sentiment']['f1_macro']:.4f}")
    print(f"     Sentiment F1 Weight : {best_info['sentiment']['f1_weighted']:.4f}")
    print(f"     Emotion   F1 Macro  : {best_info['emotion']['f1_macro']:.4f}")
    print(f"     Parameters          : {best_info['total_params']:,}")
    print(f"     Training time       : {best_info['training_time']:.1f} min")
    print(f"     Model path          : {best_info['model_path']}")

    best_generic = os.path.join(MODEL_DIR_STR, "best_model.pt")
    shutil.copy(best_info["model_path"], best_generic)
    shutil.copy(
        best_info["model_path"].replace(".pt", "_metadata.json"),
        os.path.join(MODEL_DIR_STR, "best_model_metadata.json"),
    )
    print(f"\n  best_model.pt          -> {best_generic}")
    print(f"  best_model_metadata.json -> saved")

    print(f"\n{'='*70}")
    print("  ALL EXPERIMENTS COMPLETE!")
    print(f"  Results CSV: {results_csv_path}")
    print(f"  Plots      : {PLOT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
