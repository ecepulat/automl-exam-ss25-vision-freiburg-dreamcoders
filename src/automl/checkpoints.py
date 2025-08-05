import json
from pathlib import Path

CHECKPOINT_PATH = Path("darts_pipeline_checkpoint.json")

def save_checkpoint(stage, genotype=None, hpo_params=None):
    checkpoint = {
        "last_stage": stage,
        "genotype": genotype,
        "hpo_params": hpo_params,
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return None
