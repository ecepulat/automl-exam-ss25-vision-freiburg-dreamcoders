import argparse
import sys

from data_analyze import analyze_dataset
from optuna_hpo import run_hpo
from optuna_train_search import optuna_arch_search
from optuna_train import final_train
# Add src/ to sys.path to resolve the darts folder 
sys.path.append(str(Path(__file__).resolve().parents[1]))
from darts.darts_train_search import darts_arch_search
from darts.darts_hpo import run_darts_hpo
from darts.darts_train import full_train
from checkpoints import save_checkpoint, load_checkpoint
import time
import json

import os 
def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
def load_hpo_params(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract params if nested
    if "params" in data:
        return data["params"]
    return data
def main():
    parser = argparse.ArgumentParser(description="AutoML pipeline entrypoint")

    parser.add_argument(
        '--datasetname', type=str, required=True,
        help='Name of the dataset folder inside the ./data directory'
    )
    parser.add_argument("--test", action="store_true", help="If set, treat test set as unlabeled")
    
    args = parser.parse_args()
    dataset_name = args.datasetname
    test_flag = args.test
    architecture_time = hpo_time = final_train_time = 0
    total_start = time.time()
    #Analyze the dataset to get resolution
    metadata = analyze_dataset(dataset_name)
    metadata_path = Path(f"dataset_analysis_{dataset_name}.json")
    class_count = metadata["num_classes"]


    width, height = metadata.get("image_resolution", (0, 0))
    resolution_product = width * height

    print(f"[INFO] Resolution for '{dataset_name}': {width}x{height} → {resolution_product}")

    if resolution_product > 250 * 250:
        print("[DECISION] Using Optuna + Hyperband (multi-fidelity HPO)")
        start = time.time()
        best_architecture_params = optuna_arch_search(dataset_name)
        architecture_time = time.time() - start
      
        start = time.time()
        best_param_hpo = run_hpo(dataset_name, best_architecture_params)

        hpo_time = time.time() - start

        start = time.time()


        trainer= final_train(dataset_name=dataset_name,
            best_architecture_params=best_architecture_params,
            best_hpo_params=best_param_hpo, test_mode = test_flag)
        final_train_time = time.time() - start
       
    else:
        print("[DECISION] Using DARTS search strategy")
        
        # Architecture Search
        start = time.time()
        best_architecture_params = darts_arch_search(dataset_name)
        architecture_time = time.time() - start
        print(f"[DEBUG] Got genotype: {best_architecture_params}")
        print(f"[TIMING] Architecture search took {architecture_time:.2f} seconds")

        # HPO
        start = time.time()
        best_param_hpo = run_darts_hpo(dataset_name, best_architecture_params, metadata_path)
        hpo_time = time.time() - start
        print(f"[DEBUG] Got HPO params: {best_param_hpo}")
        print(f"[TIMING] HPO took {hpo_time:.2f} seconds")
        
        print("[DEBUG] Starting full_train()...")
        start = time.time()
        full_train(
            dataset_name=dataset_name,
            best_architecture_params=best_architecture_params,
            best_hpo_params=best_param_hpo,
            class_count=class_count
        )
        final_train_time = time.time() - start
        print(f"[TIMING] Final training took {final_train_time:.2f} seconds")


    total_time = time.time() - total_start
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data", "exam_dataset")

    os.makedirs(data_dir, exist_ok=True)
    if test_flag:
        trainer.generate_test_predictions(

            save_path = os.path.join(data_dir, "predictions.npy")
        )

    with open("time_pipeline.log", "w") as f:
        f.write("Time Pipeline Summary:\n")
        f.write(f"   • Architecture Search: {format_time(architecture_time)}\n")
        f.write(f"   • Hyperparameter Opt.: {format_time(hpo_time)}\n")
        f.write(f"   • Final Training:      {format_time(final_train_time)}\n")
        f.write(f"   • Total Runtime:       {format_time(total_time)}\n")



if __name__ == "__main__":

    main()
