import json
import argparse
import os
import numpy as np
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS


BASE_EXP_DIR = "/mnt/amlfs-01/shared/robocasa_benchmark/diffusion_policy/data/outputs"

def compute_stats(checkpoint_path):
    stats = dict(
        train=dict(),
        test=dict(),
    )

    assert os.path.exists(os.path.join(BASE_EXP_DIR, checkpoint_path))

    for split in ["train", "test"]:
        split_dir = os.path.join(BASE_EXP_DIR, checkpoint_path, split)
        if not os.path.exists(split_dir):
            continue
        for task_name in os.listdir(split_dir):
            task_dir = os.path.join(split_dir, task_name)
            stats_path = os.path.join(task_dir, "eval_log.json")
            if not os.path.exists(stats_path):
                continue
            with open(stats_path, 'r') as f:
                this_data = json.load(f)
            
            sr_key = f"success_rate/{task_name}"
            if sr_key in this_data:
                stats[split][task_name] = int(this_data[sr_key] * 100)

    all_task_names = set(list(stats["train"].keys()) + list(stats["test"].keys()))
    atomic_task_names = [task for task in all_task_names if task in list(SINGLE_STAGE_TASK_DATASETS.keys())]
    composite_task_names = [task for task in all_task_names if task in list(MULTI_STAGE_TASK_DATASETS.keys())]

    print("ATOMIC TASK EVALS")
    train_vals = []
    test_vals = []
    for task_name in sorted(atomic_task_names):
        train_sr = None
        test_sr = None
        if "train" in stats:
            train_sr = stats["train"].get(task_name)
        if "test" in stats:
            test_sr = stats["test"].get(task_name)
        str_to_print = f"{task_name}: {train_sr} / {test_sr}"
        print(str_to_print)

        if train_sr is not None:
            train_vals.append(train_sr)
        if test_sr is not None:
            test_vals.append(test_sr)
    print("AVG:", np.mean(train_vals), "/", np.mean(test_vals))


    print()
    print()

    print("COMPOSITE TASK EVALS")
    train_vals = []
    test_vals = []
    for task_name in sorted(composite_task_names):
        train_sr = stats["train"].get(task_name)
        test_sr = stats["test"].get(task_name)
        str_to_print = f"{task_name}: {train_sr} / {test_sr}"
        print(str_to_print)

        if train_sr is not None:
            train_vals.append(train_sr)
        if test_sr is not None:
            test_vals.append(test_sr)
    print("AVG:", np.mean(train_vals), "/", np.mean(test_vals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="relative path to eval dir, eg. 2025.07.23/20.11.51_train_diffusion_transformer_hybrid_human_45atomic_78composite/evals/epoch=0300-test_mean_score=0.400",
    )
    args = parser.parse_args()
    compute_stats(args.dir)