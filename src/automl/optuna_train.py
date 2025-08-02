

import torch
import numpy as np
from sklearn.metrics import classification_report
from optuna_hpo import QuickTrain

def evaluate_val(self):
    print("🧪 Evaluating on validation set...")
    self.model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Validation Accuracy: {acc:.4f}")
    return acc

def evaluate_test(self):
    print("🧪 Evaluating on test set...")
    self.model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Test Accuracy: {acc:.4f}")

    report = classification_report(all_labels, all_preds, digits=4)


    with open(self.LOG_FILE, "a") as f:
        f.write(f"\n✅ Test Accuracy: {acc:.4f}\n")
        f.write("\n📄 Classification Report:\n")
        f.write(report + "\n")

    return acc

def final_train(dataset_name, best_architecture_params, best_hpo_params):
    print("\n🚀 Retraining final model with best parameters...")

    final_trainer = QuickTrain(
        dataset_name=dataset_name,
        model_name="final_best_model",
        architecture_params=best_architecture_params,  # 👈 from NAS
        min_samples_per_class=best_hpo_params["min_samples_per_class"],
        batch_size=best_hpo_params["batch_size"],
        epochs=30,  # Longer training
        learning_rate=best_hpo_params["learning_rate"],
        optimizer_name=best_hpo_params["optimizer"],
        weight_decay=best_hpo_params["weight_decay"],
        track_metrics=True #enable final log 


    )

    final_trainer.full_train()
    final_trainer.evaluate_test()

