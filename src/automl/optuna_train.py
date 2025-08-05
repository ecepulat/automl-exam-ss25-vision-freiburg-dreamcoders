

import torch
import numpy as np
from sklearn.metrics import classification_report
from optuna_hpo import QuickTrain

def final_train(dataset_name, best_architecture_params, best_hpo_params):
    print("\n🚀 Retraining final model with best parameters...")

    final_trainer = QuickTrain(
        dataset_name=dataset_name,
        model_name="final_best_model",
        architecture_params=best_architecture_params,  # 👈 from NAS
        min_samples_per_class=best_hpo_params["min_samples_per_class"],
        batch_size=best_hpo_params["batch_size"],
        epochs=30,  # Longer training
        learning_rate=best_hpo_params["lr"],
        optimizer_name=best_hpo_params["optimizer"],
        weight_decay=best_hpo_params["weight_decay"],
        track_metrics=True #enable final log 


    )

    final_trainer.full_train()
    final_trainer.evaluate_test()

