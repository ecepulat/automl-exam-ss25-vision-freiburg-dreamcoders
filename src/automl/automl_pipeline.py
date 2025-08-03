import argparse
from pathlib import Path
from data_analyze import analyze_dataset
from optuna_hpo import run_hpo
from optuna_train_search import optuna_arch_search
from optuna_train import final_train  
from darts.darts_train_search import darts_arch_search
from darts.darts_hpo import run_darts_hpo
from darts.darts_train import full_train
def main():
    parser = argparse.ArgumentParser(description="AutoML pipeline entrypoint")

    parser.add_argument(
        '--datasetname', type=str, required=True,
        help='Name of the dataset folder inside the ./data directory'
    )

    args = parser.parse_args()
    dataset_name = args.datasetname

   
    # Analyze the dataset to get resolution
    metadata = analyze_dataset(dataset_name)

    width, height = metadata.get("image_resolution", (0, 0))
    resolution_product = width * height

    print(f"[INFO] Resolution for '{dataset_name}': {width}x{height} → {resolution_product}")

    if resolution_product > 250 * 250:
        print("[DECISION] Using Optuna + Hyperband (multi-fidelity HPO)")
        best_architecture_params = optuna_arch_search(dataset_name)
        best_param_hpo = run_hpo(dataset_name, best_architecture_params)
        final_train(dataset_name=dataset_name,
            best_architecture_params=best_architecture_params,
            best_hpo_params=best_param_hpo)
    
      
    else:
        print("[DECISION] Using DARTS search strategy + Optuna HPO")
        best_architecture_params=darts_arch_search(dataset_name) # Finding best architecture using hard coded hyperparams
        best_param_hpo= run_darts_hpo() # Finding best hyperparams for the full training
        full_train(dataset_name=dataset_name,
            best_architecture_params=best_architecture_params,
            best_hpo_params=best_param_hpo)

        #from darts_train_search import main as darts_search
        #darts_search(dataset_name)

if __name__ == "__main__":
    main()
