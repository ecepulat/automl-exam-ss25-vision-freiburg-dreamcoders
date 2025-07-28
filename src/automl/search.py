import os
import json

def search(dataset_name: str):
    """
    Chooses a NAS/HPO method based on dataset resolution.
    - If resolution > 250x250 -> Optuna + Hyperband (Bayesian search)
    - Else -> DARTS (Differentiable NAS)
    """
    metadata_path = os.path.abspath(f"dataset_analysis_{dataset_name}.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run data_analyze first.")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    width, height = metadata.get("image_resolution", (0, 0))
    resolution_product = width * height
    print(f"[INFO] Resolution for '{dataset_name}': {width}x{height} → {resolution_product}")

    if resolution_product > 250 * 250:
        print("[DECISION] Using Optuna + Hyperband (multi-fidelity HPO)")
        from optuna_train_search import run_optuna_search
        run_optuna_search(dataset_name)
    else:
        print("[DECISION] Using DARTS search strategy")
        from darts_train_search import main as darts_search
        darts_search()
