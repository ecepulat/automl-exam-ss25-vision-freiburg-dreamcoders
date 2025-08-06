

import torch
import numpy as np
from sklearn.metrics import classification_report
from optuna_hpo import QuickTrain

def final_train(dataset_name, best_architecture_params, best_hpo_params, test_mode=False)-> QuickTrain:
    print("\n🚀 Retraining final model with best parameters...")
    print(f"in the final train min_samples_per_class {best_hpo_params["min_samples_per_class"]}")
    final_trainer = QuickTrain(
        dataset_name=dataset_name,
        model_name="final_best_model",
        architecture_params=best_architecture_params,  # 👈 from NAS
        min_samples_per_class=best_hpo_params["min_samples_per_class"],
        batch_size=best_hpo_params["batch_size"],
        epochs=4,  # Longer training
        learning_rate=best_hpo_params["lr"],
        optimizer_name=best_hpo_params["optimizer"],
        weight_decay=best_hpo_params["weight_decay"],
        test_mode=test_mode


    )

    final_trainer.full_train()
    if not test_mode:
        final_trainer.evaluate_test()  # ✅ safe only in non-test mode

    return final_trainer


